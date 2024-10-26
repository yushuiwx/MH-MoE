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
from argparse import Namespace

import torch

# from transformers import OPTConfig
# from gpt.modeling_opt import OPTModel
from transformers.utils import logging
from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    logger.info(f"Load Llama Model from {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu")

    keys_to_rename = {
        "model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "model.norm.weight": "decoder.layer_norm.weight",
        "lm_head.weight": "decoder.output_projection.weight"
    }
    for old_key, new_key in keys_to_rename.items():
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

    keys = list(sd.keys())
    for key in keys:
        if ".mlp" in key:
            value = sd[key]
            new_name = key.replace(".mlp", ".ffn").replace("model.", "decoder.")
            if ".gate_proj" in new_name:
                new_name = new_name.replace(".gate_proj", ".fc3")
            elif ".up_proj" in new_name:
                new_name = new_name.replace(".up_proj", ".fc1")
            elif ".down_proj" in new_name:
                new_name = new_name.replace(".down_proj", ".fc2")
            else:
                raise NotImplementedError
            sd[new_name] = value
            del sd[key]

        if ".o_proj" in key:
            value = sd[key]
            new_name = key.replace(".o_proj", ".out_proj").replace("model.", "decoder.")
            sd[new_name] = value
            del sd[key]

        if ".input_layernorm" in key:
            value = sd[key]
            new_name = key.replace(".input_layernorm", ".self_attn_layer_norm").replace("model.", "decoder.")
            sd[new_name] = value
            del sd[key]

        if ".post_attention_layernorm" in key:
            value = sd[key]
            new_name = key.replace(".post_attention_layernorm", ".final_layer_norm").replace("model.", "decoder.")
            sd[new_name] = value
            del sd[key]

    keys = list(sd.keys())
    for key in keys:
        if "model." in key:
            value = sd[key]
            new_name = key.replace("model.", "decoder.")
            sd[new_name] = value
            del sd[key]

    keys = list(sd.keys())
    for key in keys:
        if "decoder." not in key:
            value = sd[key]
            new_name = "decoder." + key
            sd[new_name] = value
            del sd[key]

    return sd


@torch.no_grad()
def convert_fairseq_checkpoint(template_path, checkpoint_path, pytorch_dump_folder_path):

    template_sd = torch.load(template_path)

    state_dict = load_checkpoint(checkpoint_path)

    print(set(template_sd["model"].keys() - set(state_dict.keys())))
    print(set(state_dict.keys() - set(template_sd["model"].keys())))

    # assert set(template_sd["model"].keys()) == set(state_dict.keys())
    # # debug
    # for key in template_sd["model"].keys():
    #     if not torch.equal(template_sd["model"][key], state_dict[key]):
    #         print(key)
    # print("success")


    template_sd['model'] = state_dict

    torch.save(template_sd, pytorch_dump_folder_path)

    logger.warning(f"Converted Fairseq Model saved at {pytorch_dump_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = options.get_training_parser()
    parser.add_argument(
        "--template-fairseq-path", 
        default=None, 
        type=str, 
        help="Only replace the model of template checkpoint with huggingface model",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default="",
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )

    # args = options.parse_args_and_arch(parser)
    # cfg = convert_namespace_to_omegaconf(args)
    args = parser.parse_args()
    convert_fairseq_checkpoint(
        args.template_fairseq_path, args.hf_path, args.pytorch_dump_folder_path, 
    )
