# !/usr/bin/env bash

SEED=(6 7 8 9)
SHOT=(5)
PY_ARGS=${@:1}

for shot in ${SHOT[*]}
do
    for seed in ${SEED[*]}
    do
    python -u coco_llm_gemini.py \
        --dataset_seed $seed \
        --shot $shot \
        ${PY_ARGS}
        
    done    
done