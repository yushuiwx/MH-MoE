NGPUs=$1
NNodes=$2
DATADIR=/mnt/msranlp/shaohanh/data/tnlg_config/
save_dir=/mnt/msranlp/shumma/exp/gpt_unilm/$3
mkdir -p $save_dir

cd fairseq/
python -m torch.distributed.launch --nproc_per_node=$1 --nnodes=$2 \
    --node_rank=$PSSH_NODENUM --master_addr=10.0.0.4 --master_port=12345 \
    train.py \
    $DATADIR \
    --num-workers 2 \
    --activation-fn gelu \
    --share-decoder-input-output-embed \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --memory-efficient-fp16 \
    --fp16-init-scale 4 \
    --arch gpt_small \
    --decoder-layers 32 \
    --decoder-embed-dim 4096 \
    --decoder-ffn-embed-dim 16384 \
    --decoder-attention-heads 32 \
    --criterion cross_entropy \
    --task gpt \
    --sample-break-mode none \
    --tokens-per-sample 2048 \
    --scale-length 2048 \
    --log-format simple --log-interval 100 --disable-validation \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --clip-norm 2.0 \
    --lr 1.2e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 750 \
    --dropout 0.0 \
    --attention-dropout 0.0 \
    --weight-decay 0.01 \
    --batch-size 4 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --total-num-update 300000 \
    --max-update 300000 \
    --seed 1 \
    --save-dir ${save_dir} \
    --tensorboard-logdir ${save_dir}/tb-logs \
    --ddp-backend=c10d \
    --subln --xpos-rel-pos --flash-attention --checkpoint-activations \
    --dict-path /mnt/msranlp/shumma/data/16g/dict.txt \
    --spm-model /mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model \
    --reset-dataloader \
    --batch-read-ahead 100