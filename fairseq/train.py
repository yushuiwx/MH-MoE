# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# flake8: noqa
import models
import tasks
from fairseq_cli.train import cli_main

if __name__ == "__main__":
    # from torchscale.component.xmoe.global_groups import get_moe_group
    # print("import get_moe_group sucess ")
    # print("print get_moe_group", get_moe_group)
    # assert hasattr(get_moe_group, "_moe_groups") # need to init groups first
    
    cli_main()
