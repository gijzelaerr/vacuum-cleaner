#!/usr/bin/env bash -ve

experiment=gan_psf_res  # gan_psf  gan gan_psf  gan_psf_res
runs=(1 2 3 4 5 6 7 8 9 10)
for run in "${runs[@]}"; do
    output_prefix=/scratch/vacuum-cleaner/final_eval_lr/${experiment}/test/run${run}
    checkpoint_prefix=/scratch/vacuum-cleaner/final_eval_lr/${experiment}/run${run}/train/model-
    input_dir=/scratch/datasets/astro_deconv_2019/test
    test_start=9400
    test_end=9700

    indexes=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000)

    python=../.venv2/bin/python
    script=../vacuum/test.py

    for index in "${indexes[@]}"; do
        mkdir -p output_prefix
        output_dir=${output_prefix}/${index}
        checkpoint=${checkpoint_prefix}${index}

        ${python} ${script} \
            --input_dir ${input_dir} \
            --output_dir ${output_dir} \
            --checkpoint ${checkpoint} \
            --test_start ${test_start} \
            --test_end ${test_end} \
            --separable_conv
            #--disable_psf
    done
done