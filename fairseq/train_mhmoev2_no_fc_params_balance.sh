expert=$1
PORT=29001
heads_number=$2
OUTPUT_FOLDER='/mnt1/msranlpintern/wuxun/MoE/MoE_results/NEW_RESULTS'
OUTPUT_PATH=$OUTPUT_FOLDER/mhmoe_standard_code/mhmoe_v2_no_fc_params_balance_2_has_fc_params_balance_more_pb/small-baseline-redstone_v2-flash_attn-${expert}experts-${heads_number}heads

TB_PATH=$OUTPUT_PATH/tb-logs

echo $OUTPUT_PATH
mkdir -p $OUTPUT_FOLDER
mkdir -p $TB_PATH

python -m torch.distributed.launch --nproc_per_node=16 --master_port $PORT train.py /mnt/msranlp/shaohanh/data/redstone_v2_1_config/  \
--num-workers 2  --share-decoder-input-output-embed  --save-interval-updates 500  --no-epoch-checkpoints  \
--memory-efficient-fp16  --fp16-init-scale 4  --arch gpt_small  --task gpt  --sample-break-mode none  --tokens-per-sample 2048  \
--optimizer adam --adam-betas "(0.9, 0.98)"  --adam-eps 1e-06  --clip-norm 2.0  --lr 6e-4  --lr-scheduler polynomial_decay  \
--warmup-updates 375  --dropout 0.1  --attention-dropout 0.1  --weight-decay 0.01  --batch-size 1 --update-freq 16  \
--required-batch-size-multiple 1  --total-num-update 300000  --max-update 300000  --seed 1  --ddp-backend=no_c10d  \
--decoder-attention-heads 12 \
--save-dir $OUTPUT_PATH --tensorboard-logdir $TB_PATH  --log-format simple   \
--log-interval 50  --disable-validation   --subln  --xpos-rel-pos  --no-token-positional-embeddings  --tiktoken-model cl100k_base  \
--dict-path /mnt/msranlp/shaohanh/exp/unigpt_exp/data/tiktoken/cl100k_w_code_dict.txt --pad-to-max-len  --moe-expert-count $expert --moe-freq 2  \
--moe-gating-use-fp32 --moe-second-expert-policy random --moe-normalize-gate-prob-before-dropping  --moe-eval-capacity-token-fraction -1.0  --criterion moe_cross_entropy \
--moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum  --use-xmoe --use-mhmoe --mhmoe-heads-number ${heads_number} >> $OUTPUT_PATH/train.log