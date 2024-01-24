DATA_PATH=/mnt/msranlp/shaohanh/temp/xmoe-gpt/
MODEL_PATH=/mnt1/msranlpintern/wuxun/MoE/MoE_results/16GPUs/base_moe/small-baseline-redstone_v2-flash_attn-8experts/checkpoint_1_35000-shared.pt
python -m fairseq_cli.eval_lm \
$DATA_DIR \
--path $MODEL_PATH \
--gen-subset valid \
--sample-break-mode none \
--tokens-per-sample 2048 \
--batch-size 1 \
--fp16 \
--output-word-probs \
--is-moe \
--distributed-world-size 4 \
--model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.05}"