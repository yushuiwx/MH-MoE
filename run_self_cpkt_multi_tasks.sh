TASKs=("harness_arc_easy" "harness_arc_challenge" "harness_openbookqa" "harness_piqa" "harness_rte")
CPKTs=(290000 280000 270000 250000 240000 200000 180000 150000 100000 50000)
for cpkt in "${CPKTs[@]}"
do
    for task in "${TASKs[@]}"
    do
        echo ${task}
        bash scripts/nl-eval-mp1-modified.sh ${task} /mnt1/msranlpintern/wuxun/MoE/MoE_results/NEW_RESULTS/mhmoe_standard_code/base_moe/small-baseline-redstone_v2-flash_attn-64experts/checkpoint_1_${cpkt}.pt /mnt1/msranlpintern/wuxun/MoE/MoE_results/EVALUATION_RESULTS/base_moe/small-baseline-redstone_v2-flash_attn-64experts/checkpoint_1_${cpkt} 0 '--arch gpt_small --model-parallel-size 1 --required-batch-size-multiple 1 --criterion harness_eval  --pad-to-max-length 2048 --subln --xpos-rel-pos --no-token-positional-embeddings --tiktoken-model cl100k_base --dict-path /mnt/msranlp/shaohanh/exp/unigpt_exp/data/tiktoken/cl100k_w_code_dict.txt  --moe-expert-count 64 --moe-freq 2 --use-xmoe --moe-gating-use-fp32 --moe-second-expert-policy random --moe-normalize-gate-prob-before-dropping --moe-eval-capacity-token-fraction 2 '
    done
done
