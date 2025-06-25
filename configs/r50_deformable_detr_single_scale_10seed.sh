#!/usr/bin/env bash
set -x

SEED=(0)
SHOT=(1)

#PTH_DIR=exps/r50_deformable_detr_single_scale_weights/checkpoint0049.pth
PTH_DIR=/DATA3/flaya/iFSD/exps/checkpoint.pth
PY_ARGS=${@:1}

for shot in ${SHOT[*]}
do
    for seed in ${SEED[*]}
    do
        EXP_DIR=./$shot$seed
        python -u main.py \
            --num_feature_levels 1 \
            --output_dir ${EXP_DIR} \
            --dataset_seed $seed \
            --shot $shot \
            --i \
            --ifsd ${PTH_DIR} \
            ${PY_ARGS} | tee ./test_single_${shot}shot_seed${seed}
    done
done 