# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert OPT checkpoint."""


import argparse
from pathlib import Path

import torch

# from transformers import LlamaConfig
# from quip_sharp.model.llama import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.utils import logging
from quip_sharp.dictionary import Dictionary
import sentencepiece


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model" in sd.keys():
        sd = torch.load(checkpoint_path, map_location="cpu")["model"]

    # pop unnecessary weights
    keys_to_delete = [
        "decoder.version",
        "decoder.output_projection.weight",
    ]
    for key in keys_to_delete:
        if key in sd:
            sd.pop(key)

    keys_to_rename = {
        "decoder.embed_tokens.weight": "embed_tokens.weight",
        "decoder.layer_norm.weight": "norm.weight",
    }
    for old_key, new_key in keys_to_rename.items():
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

    keys = list(sd.keys())
    for key in keys:
        if ".qkv_proj." in key:
            value = sd[key]
            # We split QKV in separate Q,K,V

            q_name = key.replace(".qkv_proj.", ".q_proj.")
            k_name = key.replace(".qkv_proj.", ".k_proj.")
            v_name = key.replace(".qkv_proj.", ".v_proj.")

            depth = value.shape[0]
            assert depth % 3 == 0
            # `SequeuceParallelTransformerBlock` has QKV weight is separated in K,V,Q despite the naming:
            # https://cs.github.com/facebookresearch/metaseq/blob/51871bd73cd04c038f239ea2a26db1d7f6b37927/metaseq/modules/sequence_parallel_transformer_layer.py#L97
            k, v, q = torch.split(value, depth // 3, dim=0)

            sd[q_name] = q
            sd[k_name] = k
            sd[v_name] = v
            del sd[key]

        if ".ffn." in key:
            value = sd[key]
            new_name = key.replace(".ffn.", ".mlp.")
            if ".fc3" in new_name:
                new_name = new_name.replace(".fc3", ".gate_proj")
            elif ".fc1" in new_name:
                new_name = new_name.replace(".fc1", ".up_proj")
            elif ".fc2" in new_name:
                new_name = new_name.replace(".fc2", ".down_proj")
            else:
                print(new_name)
            sd[new_name] = value
            del sd[key]

        if ".out_proj" in key:
            value = sd[key]
            new_name = key.replace(".out_proj", ".o_proj")
            sd[new_name] = value
            del sd[key]

        if ".self_attn_layer_norm" in key:
            value = sd[key]
            new_name = key.replace(".self_attn_layer_norm", ".input_layernorm")
            sd[new_name] = value
            del sd[key]

        if ".final_layer_norm" in key:
            value = sd[key]
            new_name = key.replace(".final_layer_norm", ".post_attention_layernorm")
            sd[new_name] = value
            del sd[key]

    keys = list(sd.keys())
    for key in keys:
        if "decoder." in key:
            value = sd[key]
            new_name = key.replace("decoder.", "")
            sd[new_name] = value
            del sd[key]

    return sd


@torch.no_grad()
def convert_llama_checkpoint(checkpoint_path, pytorch_dump_folder_path, spm_model, arch=None, subln=False):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    state_dict = load_checkpoint(checkpoint_path)

    EOL_SYMBOL = "</line>"
    logger.info("load dictionary directly from spm_model")
    dictionary = Dictionary.load(spm_model, from_SPM=True)
    dictionary.add_symbol(EOL_SYMBOL)
    dictionary.pad_to_multiple_(2)

    # if subln:
    #     from magneto.configuration_llama_subln import LlamaConfig
    #     from magneto.modeling_llama_subln import LlamaModel
    # else:
    from transformers import LlamaConfig
    from quip_sharp.model.modeling_llama import LlamaModel

    # if subln:
    #     if arch == "xl":
    #         config = LlamaConfig(
    #             vocab_size=len(dictionary),
    #             hidden_size=2048,
    #             intermediate_size=5504,
    #             num_hidden_layers=24,
    #             num_attention_heads=32,
    #             num_key_value_heads=None,
    #             hidden_act="silu",
    #             max_position_embeddings=2048,
    #             initializer_range=0.02,
    #             rms_norm_eps=1e-6,
    #             use_cache=True,
    #             pad_token_id=dictionary.pad(),
    #             bos_token_id=dictionary.bos(),
    #             eos_token_id=dictionary.eos(),
    #             pretraining_tp=1,
    #             tie_word_embeddings=True,
    #             rope_theta=10000.0,
    #             rope_scaling=None,
    #             attention_bias=False,
    #             subln=subln,
    #         )
    #     elif arch == "3b":
    #         config = LlamaConfig(
    #             vocab_size=len(dictionary),
    #             hidden_size=3200,
    #             intermediate_size=8640,
    #             num_hidden_layers=26,
    #             num_attention_heads=32,
    #             num_key_value_heads=None,
    #             hidden_act="silu",
    #             max_position_embeddings=2048,
    #             initializer_range=0.02,
    #             rms_norm_eps=1e-6,
    #             use_cache=True,
    #             pad_token_id=dictionary.pad(),
    #             bos_token_id=dictionary.bos(),
    #             eos_token_id=dictionary.eos(),
    #             pretraining_tp=1,
    #             tie_word_embeddings=True,
    #             rope_theta=10000.0,
    #             rope_scaling=None,
    #             attention_bias=False,
    #             subln=subln,
    #         )
    # else:
    if arch == "xl":
        config = LlamaConfig(
            vocab_size=len(dictionary),
            hidden_size=2048,
            intermediate_size=5460,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=dictionary.pad(),
            bos_token_id=dictionary.bos(),
            eos_token_id=dictionary.eos(),
            pretraining_tp=1,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
        )
    elif arch == "3b":
        config = LlamaConfig(
            vocab_size=len(dictionary),
            hidden_size=3200,
            intermediate_size=8640,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=dictionary.pad(),
            bos_token_id=dictionary.bos(),
            eos_token_id=dictionary.eos(),
            pretraining_tp=1,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
        )
    elif arch == "large":
        config = LlamaConfig(
            vocab_size=len(dictionary),
            hidden_size=1536,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=dictionary.pad(),
            bos_token_id=dictionary.bos(),
            eos_token_id=dictionary.eos(),
            pretraining_tp=1,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
        )

    model = LlamaModel(config).half()
    model.load_state_dict(state_dict, strict=True)

    # Check results
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        type=str,
        default="",
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--spm-model", default=None, type=str, help="")
    parser.add_argument("--arch", default=None, type=str, help="")
    parser.add_argument("--subln", default=False, type=bool, help="")
    args = parser.parse_args()
    convert_llama_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, spm_model=args.spm_model, arch=args.arch, subln=args.subln)
