# X-MoE
```
description: base_moe_8experts

target:
  service: amlk8s
  # name: gcrprojvc1
  name: itphyperdgx2cl1
  vc: msrhyper
environment:
  # image: chizewen/pytorch:1.12.1-mpi
  image: nvidia/22.10:v2
  registry: shumingdocker.azurecr.io
  username: shumingdocker

code:
  local_dir: /mnt1/msranlpintern/amlt_exp/wuxun/MoE_code/mhmoe_standard_code/xmoe_baseline

storage:
  conversationhub:
    storage_account_name: conversationhub
    container_name: unilm
  msranlp:
    storage_account_name: msranlp
    container_name: unilm

# search:
jobs:
  - name: test1
    sku: G16
    command:
      - bash blob/mount.sh
      - pip3 install Cython
      - pip install Cython
      - pip install tensorboard
      - bash language_modeling/setup_gpt_baseline_flash_attn.sh
      - echo 'pip ok'
      - cd fairseq
      - bash train_base_moe.sh 8
      - echo "done"
    submit_args:
      env:
        NCCL_DEBUG: INFO
        MKL_NUM_THREADS: 1
        OMP_NUM_THREADS: 1
    priority: high
    mpi: True
    process_count_per_node: 1
```
# CPKT PATH
```
/mnt1/msranlpintern/wuxun/MoE/MoE_results/NEW_RESULTS/mhmoe_standard_code/base_moe/small-baseline-redstone_v2-flash_attn-8experts
```
```
