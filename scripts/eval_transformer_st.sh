#!/bin/bash

experiment_path=$1
lang_pair=$2

path_to_ckpt=${experiment_path}/ckpts/avg_best_10_checkpoint.pt

best_ckpts="$(find "$experiment_path"/ckpts/checkpoint.best*.pt | tr '\n' ' ' )"
python "$PERCEIVER_ROOT"/fairseq/scripts/average_checkpoints.py \
    --inputs $best_ckpts \
    --output "$path_to_ckpt"

if [ $lang_pair == en-de ]; then
    lenpen=1.5
else
    lenpen=1.0
fi

fairseq-generate "${MUSTC_ROOT}/${lang_pair}" \
--config-yaml config_st.yaml \
--gen-subset tst-COMMON_st \
--task speech_to_text \
--path "$path_to_ckpt" \
--max-tokens 360_000 \
--batch-size 360 \
--beam 5 \
--lenpen $lenpen \
--scoring sacrebleu \
--seed 42