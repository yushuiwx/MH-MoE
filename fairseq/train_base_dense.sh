PORT=29334
heads_number=$1
OUTPUT_FOLDER='/mnt1/msranlpintern/wuxun/MoE/MoE_results/NEW_RESULTS'
OUTPUT_PATH=$OUTPUT_FOLDER/mhmoe_standard_code/base_dense_${heads_number}/small-baseline-redstone_v2-flash_attn

TB_PATH=$OUTPUT_PATH/tb-logs

echo $OUTPUT_PATH
mkdir -p $OUTPUT_FOLDER
mkdir -p $TB_PATH

python -m torch.distributed.launch --nproc_per_node=16 --master_port $PORT train.py /mnt/msranlp/shaohanh/data/redstone_v2_1_config/    \
    --num-workers 2 \
    --share-decoder-input-output-embed \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --arch gpt_small \
    --task gpt \
    --sample-break-mode none \
    --tokens-per-sample 2048 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 2.0 \
    --lr 6e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 375 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 2 \
    --decoder-attention-heads 12 \
    --update-freq 8 \
    --required-batch-size-multiple 1 \
    --total-num-update 300000 \
    --max-update 300000 \
    --seed 1 \
    --ddp-backend=c10d \
    --save-dir $OUTPUT_PATH \
    --tensorboard-logdir $TB_PATH \
    --log-format simple      --log-interval 50     --disable-validation      \
    --subln    --xpos-rel-pos  --no-token-positional-embeddings \
    --tiktoken-model cl100k_base \
    --dict-path /mnt/msranlp/shaohanh/exp/unigpt_exp/data/tiktoken/cl100k_w_code_dict.txt >> $OUTPUT_PATH/train.log