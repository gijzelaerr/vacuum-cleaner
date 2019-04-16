#!/usr/bin/env bash

output_prefix=/scratch/vacuum-cleaner/eval/astro_deconv_2019_res_gan_noscale/
checkpoint_prefix=/scratch/vacuum-cleaner/train/astro_deconv_2019_res_gan_noscale/model-
input_dir=/scratch/datasets/astro_deconv_2019
test_start=9400
test_end=9700


indexes=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000)

python=../.venv3/bin/python
script=../vacuum/test.py

for index in "${indexes[@]}"; do
    output_dir=${output_prefix}/${index}
    checkpoint=${checkpoint_prefix}${index}

    ${python} ${script} \
        --input_dir ${input_dir} \
        --output_dir ${output_dir} \
        --checkpoint ${checkpoint} \
        --test_start ${test_start} \
        --test_end ${test_end}
done