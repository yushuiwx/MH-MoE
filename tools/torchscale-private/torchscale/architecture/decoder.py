# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper, wrap

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.component.xmoe.tutel_moe_layer import MOELayer
from torchscale.component.xmoe.tutel_routing import TopkGate
# from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    from torch.nn import LayerNorm
    from torchscale.component.rms_norm import RMSNorm
try:
    from fairseq.model_parallel.megatron.mpu import copy_to_model_parallel_region, gather_from_model_parallel_region
except:
    print("No Megatron-LM")

class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
        is_encoder_decoder=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before
        
        if args.rms_norm:
            Layernorm = RMSNorm
        else:
            Layernorm = LayerNorm

        self.self_attn_layer_norm = Layernorm(self.embed_dim, eps=args.layernorm_eps)

        if not is_encoder_decoder:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = Layernorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.args,
            )
        else:
            self.mhmoe_heads_number = args.mhmoe_heads_number
            assert self.embed_dim % self.mhmoe_heads_number == 0
            # if args.moe_top1_expert:
            #     gate = Top1Gate(
            #         args,
            #         self.embed_dim // self.mhmoe_heads_number,
            #         args.moe_expert_count,
            #         use_fp32=args.moe_gating_use_fp32,
            #         moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
            #         use_xmoe=args.use_xmoe,
            #     )
            # else:
            #     gate = Top2Gate(
            #         args,
            #         self.embed_dim // self.mhmoe_heads_number,
            #         args.moe_expert_count,
            #         args.moe_gating_use_fp32,
            #         args.moe_second_expert_policy,
            #         args.moe_normalize_gate_prob_before_dropping,
            #         args.moe_eval_capacity_token_fraction,
            #         use_xmoe=args.use_xmoe,
            #     )

            gate = TopkGate(
                args,
                self.embed_dim // self.mhmoe_heads_number,
                args.moe_expert_count,
                use_fp32=args.moe_gating_use_fp32,
                moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                use_xmoe=args.use_xmoe,
            )

            moe_ffn_dim = getattr(args, "moe_ffn_dim", -1)
            if moe_ffn_dim == -1:
                moe_ffn_dim = self.ffn_dim
            experts = make_experts(args, self.embed_dim // self.mhmoe_heads_number, moe_ffn_dim, is_moe_layer)
            self.moe_layer = MOELayer(gate, experts, args)

        self.final_layer_norm = Layernorm(self.embed_dim, eps=args.layernorm_eps)

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = math.pow(3.0 * args.decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0
        
    #     self.beta = math.pow(2 * (depth + 1), 0.5)
    #     self.fixup()

    # def fixup(self):
    #     self.ffn.fc1.weight.data.div_(self.beta)
    #     self.ffn.fc2.weight.data.div_(self.beta)
    #     self.ffn.fc3.weight.data.div_(self.beta)

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            args,
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
            encoder_decoder_attention=True,
            subln=args.subln,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        self_attn_rel_pos=None,
        cross_attn_rel_pos=None,
        position_ids=None, # pass position_ids for rotary embedding
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
            rel_pos=self_attn_rel_pos,
            is_causal=True,
            position_ids=position_ids, # pass position_ids for rotary embedding
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=None,
                rel_pos=cross_attn_rel_pos,
            )
            x = self.dropout_module(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None, l_aux


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        embed_tokens=None,
        embed_positions=None,
        output_projection=None,
        is_encoder_decoder=False,
        mp_rank=-1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args
        self.mp_rank = mp_rank

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if args.rms_norm:
            Layernorm = RMSNorm
        else:
            Layernorm = LayerNorm

        if (
            output_projection is None
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = Layernorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = Layernorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        self.self_attn_relative_position = None
        self.cross_attn_relative_position = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.self_attn_relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.decoder_attention_heads,
            )
            if is_encoder_decoder:
                self.cross_attn_relative_position = RelativePositionBias(
                    num_buckets=args.rel_pos_buckets,
                    max_distance=args.max_rel_pos,
                    n_heads=args.decoder_attention_heads,
                )

        if args.bert_init:
            self.apply(init_bert_params)

        # if args.deepnorm:
        # if is_encoder_decoder:
        #     init_scale = math.pow(12.0 * args.decoder_layers, 0.25)
        # else:
        # init_scale = math.pow(4.0 * args.decoder_layers, 0.5)
        # for name, p in self.named_parameters():
        #     if (
        #         "fc1" in name
        #         or "fc2" in name
        #         or "fc3" in name
        #         or "out_proj" in name
        #         or "v_proj" in name
        #     ):
        #         p.data.div_(init_scale)

        # init_scale = math.pow(2.0 * args.decoder_layers, 0.5)
        # for name, p in self.named_parameters():
        #     if (
        #         "fc1" in name
        #         or "fc2" in name
        #         or "fc3" in name
        #         or "out_proj" in name
        #         or "q_proj" in name
        #         or "k_proj" in name
        #         or "v_proj" in name
        #     ):
        #         p.data.div_(init_scale)

        # if args.subln:
        # if is_encoder_decoder:
        #     init_scale = math.sqrt(math.log(args.decoder_layers * 3))
        # else:
        # init_scale = math.sqrt(math.log(args.decoder_layers * 2))
        # for name, p in self.named_parameters():
        #     if "encoder_attn" in name:
        #         continue
        #     if (
        #         "fc1" in name
        #         or "fc2" in name
        #         or "fc3" in name
        #         or "out_proj" in name
        #         or "v_proj" in name
        #     ) and not name.endswith("clip_val"):
        #         p.data.mul_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )
        return output_projection

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def forward(
        self,
        prev_output_tokens,
        self_attn_padding_mask=None,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs
    ):
        # embed tokens and positions
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )

        # relative position
        self_attn_rel_pos_bias = None
        slen = prev_output_tokens.size(1)
        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(0), qlen=slen, klen=slen
            )
            if incremental_state is not None:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[-1:, :, :]
        cross_attn_rel_pos_bias = None
        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(0),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(1),
            )
            if incremental_state is not None:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[-1:, :, :]

        # decoder layers
        inner_states = [x]

        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        # generate position_ids for rotary embedding, 
        # follow transformers.models.llama.modeling_llama.prepare_inputs_for_generation
        seq_length = prev_output_tokens.size(1)
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if incremental_state is not None:
            past_key_values_length = incremental_state[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        device = prev_output_tokens.device 
        # Usually, to be consistent with Fairseq, we have to start from 2 in position embeddings
        # but this `position_ids` is for rotary embedding which needs to starts from 0, so we do NOT +2 here.
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if not self.args.flash_attention:
            self_attn_mask = torch.triu(
                torch.zeros([x.size(1), x.size(1)])
                .float()
                .fill_(float("-inf"))
                .type_as(x),
                1,
            )
        else:
            self_attn_mask = None

        for idx, layer in enumerate(self.layers):
            if incremental_state is not None:
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                position_ids=position_ids # pass position_ids for rotary embedding
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)
        
        if self.args.model_parallel_size > 1 and not self.training:
            x = gather_from_model_parallel_region(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }

    def output_layer(self, features):
        if self.args.model_parallel_size > 1:
            features = copy_to_model_parallel_region(features)

        return self.output_projection(features)
