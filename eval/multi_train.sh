#!bash -ve

PREFIX=/home/gijs/Work/vacuum-cleaner/
PYTHON=${PREFIX}/.venv3/bin/python
SCRIPT=${PREFIX}/vacuum/trainer.py
OUTPUT_PREFIX=/scratch/vacuum-cleaner/final_eval_lr/gan_psf/run
INPUT_DIR=/scratch/datasets/astro_deconv_2019/train

for i in 1 2 3 4 5 6 7 8 9 10
do
    ${PYTHON} \
    ${SCRIPT} \
    --output_dir ${OUTPUT_PREFIX}${i}/train \
    --max_steps 100000 \
    --input_dir ${INPUT_DIR} \
    --train_start 0 \
    --train_end 9400 \
   --scale_size=256 \
   --separable_conv
done
