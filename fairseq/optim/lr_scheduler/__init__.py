# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

# register dataclass
LR_SCHEDULER_DATACLASS_REGISTRY = {}
LR_SCHEDULER_REGISTRY = {}
LR_SCHEDULER_CLASS_NAMES = set()


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("optim.lr_scheduler." + file_name)

        # # extra `model_parser` for sphinx
        # if file_name in LR_SCHEDULER_REGISTRY:
        #     parser = argparse.ArgumentParser(add_help=False)
        #     group_archs = parser.add_argument_group("Named architectures")
        #     group_archs.add_argument(
        #         "--lr-scheduler", choices=ARCH_MODEL_INV_REGISTRY[file_name]
        #     )
        #     group_args = parser.add_argument_group("Additional command-line arguments")
        #     LR_SCHEDULER_REGISTRY[file_name].add_args(group_args)
        #     globals()[file_name + "_parser"] = parser
