set -ex

# pip install -U datasets
# hellaswag, cb, boolq, coqa, piqa, winograd, winogrande, storycloze, 
TASK=$1
MODEL=$2
OUTPUT_FOLDER=$3
SHOT=$4
extra_args=$5

BSZ=4
LENGTH=256
if [ "$TASK" = "cb" ]
then
BSZ=12
fi
if [ "$TASK" = "boolq" ]
then
LENGTH=512
fi
if [ "$TASK" = "harness_anli_r1" ]
then
BSZ=6
fi
if [ "$TASK" = "harness_anli_r2" ]
then
BSZ=6
fi
if [ "$TASK" = "harness_anli_r3" ]
then
BSZ=6
fi

OUTPUT_PATH=$OUTPUT_FOLDER/${TASK}_${SHOT}

echo OUTPUT_PATH
mkdir -p $OUTPUT_FOLDER


python -m torch.distributed.launch --nproc_per_node=1  fairseq/validate.py /mnt/msranlp/shaohanh/data/tnlg_cl100k_dict_config/ --task fs_eval --criterion fs_eval   \
  --tokens-per-sample 2048     --reset-dataloader --all-gpt-emb     --log-format simple --log-interval 4  \
  --lr-scheduler polynomial_decay --optimizer adam --adam-betas '(0.9,0.98)'     --adam-eps 1e-6 --clip-norm 2.0 --warmup-updates 0   \
  --total-num-update 1 --max-update 0 --fp16     --restore-file ${MODEL}      --fp16-init-scale 4 --fp16-scale-window 256 --min-loss-scale 0.0001       \
  --train-num 5000 --valid-num 1000     --required-batch-size-multiple 1   --spm-model /mnt/msranlp/cube/llama/sentencepiece.bpe.model  \
  --reset-optimizer  --fp16-no-flatten-grads \
  --eval-data ${TASK} --seed 1 --k ${SHOT} --temp-index 0  --batch-size ${BSZ} --pad-to-max-length ${LENGTH} --eval-data ${TASK} ${extra_args} \
  > $OUTPUT_PATH 2>&1