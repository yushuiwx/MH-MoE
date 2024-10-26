set -ex

NGPUs=$1
NNodes=$2
save_dir=$3
BSZ=$4
LR=$5
MOE=$6
TOPN=$7
FREQ=${8}
OPTIONS=${9}

ARCH=gpt_small_v2

UPDATE_FREQ=4
MODEL_PARALLEL=1

# Put your pre-training data here
DATADIR=DATAPATH

mkdir -p $save_dir
unset CUDA_VISIBLE_DEVICES

LOCAL_BSZ=$(awk "BEGIN {print int($BSZ / ($NGPUs * $NNodes * $UPDATE_FREQ / $MODEL_PARALLEL))}")

if [ $NNodes -eq 1 ]; then
    GPU_OPTIONS=""
else
    GPU_OPTIONS="--node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT"
fi

let TOTAL_GPUs=$NGPUs*$NNodes

if [ $TOTAL_GPUs -eq 1 ]; then
    DIST_OPTIONS=""
else
    DIST_OPTIONS="-m torch.distributed.launch --nproc_per_node=$1 --nnodes=$2"
fi

DIST_METHOD=no_c10d

MOE_TOPN_OPTIONS=""
if [ $TOPN -eq 1 ]; then
    MOE_TOPN_OPTIONS="--moe-top1-expert"
fi


cd fairseq/
python ${DIST_OPTIONS} ${GPU_OPTIONS} train.py $DATADIR \
    --num-workers 2 \
    --activation-fn silu \
    --share-decoder-input-output-embed \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --checkpoint-activations \
    --fp16-init-scale 4 \
    --arch $ARCH \
    --input-quant-method absmax_per_token \
    --weight-quant-method absmean \
    --input-bits 32 --weight-bits 32 \
    --task gpt \
    --sample-break-mode none \
    --tokens-per-sample 2048 \
    --scale-length 2048 \
    --log-format simple --log-interval 100 --disable-validation \
    --optimizer adam --adam-betas "(0.9, 0.95)" \
    --adam-eps 1e-06 \
    --clip-norm 2.0 \
    --lr $LR \
    --end-learning-rate 3e-5 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 375 \
    --dropout 0.0 \
    --attention-dropout 0.0 \
    --weight-decay 0.1 \
    --batch-size $LOCAL_BSZ \
    --update-freq $UPDATE_FREQ \
    --required-batch-size-multiple $MODEL_PARALLEL \
    --total-num-update 100000 \
    --max-update 100000 \
    --seed 1 \
    --save-dir ${save_dir} \
    --tensorboard-logdir ${save_dir}/tb-logs \
    --ddp-backend=$DIST_METHOD \
    --moe-expert-count $MOE \
    --moe-freq $FREQ \
    --moe-top-k $TOPN \
    --pad-to-max-len $MOE_TOPN_OPTIONS \
    --moe-gating-use-fp32 \
    --moe-normalize-gate-prob-before-dropping \
    --moe-eval-capacity-token-fraction -1.0 \
    --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
    --flash-attention \
    --spm-model ../sentencepiece.bpe.model \
    --reset-dataloader \
    --rotary-embed \
    --no-token-positional-embeddings \
    --no-scale-embedding \
    --no-bias \
    --rms-norm \
    --batch-read-ahead 100 ${OPTIONS}