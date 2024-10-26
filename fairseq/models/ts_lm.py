# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field, fields
from omegaconf import DictConfig
from argparse import Namespace
from typing import Optional

import re
import torch
import fairseq
from fairseq import distributed_utils, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from fairseq.modules import PositionalEmbedding
from omegaconf import II
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, upgrade_state_for_langs_difference
from fairseq.file_io import PathManager

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from .language_modeling import LanguageConfig, LanguageModel, LMDecoder

DEFAULT_MAX_TARGET_POSITIONS = 1024
logger = logging.getLogger(__name__)

_NAME_PARSER = r"(teacher)_(.*)"

@dataclass
class TeacherStudentLanguageConfig(LanguageConfig):
    teacher_checkpoint_path: str = field(
        default="",
    )


@register_model("ts_lm", dataclass=TeacherStudentLanguageConfig)
class TeacherStudentLanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder):
        self.args = args
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):

        def _build_decoder(args):
            if getattr(args, "max_target_positions", None) is None:
                args.max_target_positions = getattr(
                    args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
                )

            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_embed_dim
            )

            embed_positions = (
                PositionalEmbedding(
                    args.max_target_positions,
                    args.decoder_embed_dim,
                    task.dictionary.pad(),
                    learned=args.decoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )

            if args.share_decoder_input_output_embed:
                output_projection = torch.nn.Linear(
                    embed_tokens.weight.shape[1],
                    embed_tokens.weight.shape[0],
                    bias=False,
                )
                output_projection.weight = embed_tokens.weight
            else:
                output_projection = torch.nn.Linear(
                    args.decoder_embed_dim, len(task.dictionary), bias=False
                )
                torch.nn.init.normal_(
                    output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
                )

            if getattr(args, "moe_freq", 0) > 0 and (
                getattr(args, "fp16", False)
                and not getattr(args, "memory_efficient_fp16", False)
                and getattr(args, "ddp_backend", None) != "fully_sharded"
            ):
                assert (
                    args.fp16_no_flatten_grads
                ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

            args.ddp_rank = distributed_utils.get_data_parallel_rank()

            config = DecoderConfig()
            config.override(args)

            decoder = LMDecoder(
                config,
                embed_tokens,
                embed_positions,
                output_projection,
                is_encoder_decoder=False,
                dictionary=task.dictionary,
            )

            return decoder

        student = _build_decoder(args)
        # teacher = _build_decoder(args.teacher)

        student.teacher = None
        if getattr(args, "teacher_checkpoint_path", "") != "":
            logger.info(
                f"Loading Teacher Model from {args.teacher_checkpoint_path}"
            )
            teacher, _teacher_cfg = load_model_ensemble(
                [args.teacher_checkpoint_path], strict=False,
            )
            teacher = teacher[0]
            for p in teacher.parameters():
                p.requires_grad = False
            student.teacher = teacher
            logger.info(
                f"Loaded Teacher Model from {args.teacher_checkpoint_path}"
            )

        return cls(args, student)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        strict = False
        assert not strict # only load student model, teacher model is loaded in def build_model

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)

        # Remove keys related to the teacher model
        # Note: The prefix 'teacher' is assumed to be where the teacher model parameters are stored.
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if 'teacher' not in k}

        missing_keys = torch.nn.Module.load_state_dict(self, filtered_state_dict, strict=strict)

        filter_missing_keys = []
        for item in missing_keys[0]:
            if "teacher" not in item:
                filter_missing_keys.append(item)

        logger.info(f"After removing teacher model, missing keys are {filter_missing_keys}")
        logger.info(f"Unexpected keys are {missing_keys[1]}")

        return missing_keys
    

@register_model_architecture("ts_lm", "ts_lm_base")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("ts_lm", "ts_gpt_small")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("ts_lm", "ts_gpt_tiny")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("ts_lm", "ts_gpt_xl")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("ts_lm", "ts_gpt_xl_v2")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5460)
    args.moe_ffn_dim = getattr(args, "moe_ffn_dim", 5460)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.n_kv_heads = getattr(args, "n_kv_heads", 32)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("ts_lm", "ts_gpt_large_v2")
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.moe_ffn_dim = getattr(args, "moe_ffn_dim", 5460)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.n_kv_heads = getattr(args, "n_kv_heads", 16)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "silu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


def load_model_ensemble(
    filenames,
    arg_overrides = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
    is_moe=False,
):
    logger.info("load_model_ensemble_and_task is_moe={}".format(is_moe))
    assert state is None or len(filenames) == 1

    from fairseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None

    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"

            if not PathManager.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides, is_moe=is_moe)
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"Neither args nor cfg exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            # build model for ensemble
            model = task.build_model(cfg.model)

            if (
                hasattr(cfg.model, "langs")
                and hasattr(task, "langs")
                and cfg.model.langs != task.langs
            ):
                upgrade_state_for_langs_difference(state, cfg.model, task)

            missing_keys = model.load_state_dict(state["model"], strict=strict, model_cfg=cfg.model)

            filter_missing_keys = []
            for item in missing_keys[0]:
                if "move" not in item:
                    filter_missing_keys.append(item)

            logger.info(f"After removing all move bias, missing keys are {filter_missing_keys}")
            logger.info(f"Unexpected keys are {missing_keys[1]}")

            # reset state so it gets loaded for the next model in ensemble
            state = None

        ensemble.append(model)
    return ensemble, cfg