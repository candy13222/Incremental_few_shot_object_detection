#!/usr/bin/env bash
set -x

SEED=(0 1 2)
SHOT=(5 10)
PTH_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_weights/checkpoint0049.pth
PY_ARGS=${@:1}
for shot in ${SHOT[*]} 
do
    for seed in ${SEED[*]}
    do
        EXP_DIR=./$seed
        python -u main.py \
            --output_dir ${EXP_DIR} \
            --with_box_refine \
            --two_stage \
            --dataset_seed $seed \
            --shot $shot \
            --i \
            --ifsd ${PTH_DIR} \
            ${PY_ARGS} | tee ./test_plusplus_${shot}shot_seed${seed}
    done
done
