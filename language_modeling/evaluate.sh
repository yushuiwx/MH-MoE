CKPT_PATH=/mnt1/msranlpintern/wuxun/MoE/MoE_results/16GPUs/base_moe/small-baseline-redstone_v2-flash_attn-8experts/checkpoint_1_35000-shared.pt
DATA_DIR=/mnt/msranlp/shaohanh/temp/xmoe-gpt/
OUTPUT_PATH=/mnt1/msranlpintern/wuxun/MoE/MoE_test_results/base_moe/small-baseline-redstone_v2-flash_attn-8experts
mkdir -p $OUTPUT_PATH

cd fairseq/

# hellaswag
python evaluate_gpt.py $DATA_DIR \
        --task fs_eval \
        --criterion fs_eval \
        --arch gpt_small \
        --tokens-per-sample 2048 \
        --reset-dataloader \
        --all-gpt-emb \
        --batch-size 4 \
        --log-format simple \
        --log-interval 4 \
        --lr-scheduler polynomial_decay \
        --optimizer adam \
        --adam-betas '(0.9,0.98)' \
        --adam-eps 1e-6 \
        --clip-norm 2.0 \
        --warmup-updates 0 \
        --total-num-update 1 \
        --max-update 0 \
        --fp16 \
        --restore-file $CKPT_PATH \
        --fp16-init-scale 4 --fp16-scale-window 256 --min-loss-scale 0.0001 \
        --eval-data hellaswag --seed 1 --k 0 --temp-index 0 \
        --train-num 5000 --valid-num 1000 \
        --required-batch-size-multiple 1 \
        --subln  --flash-attention --use-xmoe \
        --dict-path /mnt/msranlp/shumma/data/16g/dict.txt \
        --spm-model /mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model \
        --reset-optimizer |& tee $OUTPUT_PATH/hellaswag.log

# # winogrande
# python evaluate_gpt.py $DATA_DIR \
#         --task fs_eval \
#         --criterion fs_eval \
#         --arch gpt_small \
#         --tokens-per-sample 2048 \
#         --reset-dataloader \
#         --all-gpt-emb \
#         --batch-size 2 \
#         --log-format simple \
#         --log-interval 4 \
#         --lr-scheduler polynomial_decay \
#         --optimizer adam \
#         --adam-betas '(0.9,0.98)' \
#         --adam-eps 1e-6 \
#         --clip-norm 2.0 \
#         --warmup-updates 0 \
#         --total-num-update 1 \
#         --max-update 0 \
#         --fp16 \
#         --restore-file $CKPT_PATH \
#         --fp16-init-scale 4 --fp16-scale-window 256 --min-loss-scale 0.0001 \
#         --eval-data winogrande --seed 1 --k 0 --temp-index 0 \
#         --train-num 5000 --valid-num 1000 \
#         --required-batch-size-multiple 1 \
#         --subln --flash-attention \
#         --dict-path /mnt/msranlp/shumma/data/16g/dict.txt \
#         --spm-model /mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model \
#         --reset-optimizer |& tee $OUTPUT_PATH/winogrande.log

# # piqa
# python evaluate_gpt.py $DATA_DIR \
#         --task fs_eval \
#         --criterion fs_eval \
#         --arch gpt_small \
#         --tokens-per-sample 2048 \
#         --reset-dataloader \
#         --all-gpt-emb \
#         --batch-size 2 \
#         --log-format simple \
#         --log-interval 4 \
#         --lr-scheduler polynomial_decay \
#         --optimizer adam \
#         --adam-betas '(0.9,0.98)' \
#         --adam-eps 1e-6 \
#         --clip-norm 2.0 \
#         --warmup-updates 0 \
#         --total-num-update 1 \
#         --max-update 0 \
#         --fp16 \
#         --restore-file $CKPT_PATH \
#         --fp16-init-scale 4 --fp16-scale-window 256 --min-loss-scale 0.0001 \
#         --eval-data piqa --seed 1 --k 0 --temp-index 0 \
#         --train-num 5000 --valid-num 1000 \
#         --required-batch-size-multiple 1 \
#         --subln --flash-attention \
#         --dict-path /mnt/msranlp/shumma/data/16g/dict.txt \
#         --spm-model /mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model \
#         --reset-optimizer |& tee $OUTPUT_PATH/piqa.log