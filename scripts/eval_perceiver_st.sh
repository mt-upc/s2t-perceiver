#!/bin/bash

experiment_path=$1
lang_pair=$2
dla_inf_k=$3

path_to_ckpt=${experiment_path}/ckpts/avg_best_10_checkpoint.pt

best_ckpts="$(find "$experiment_path"/ckpts/checkpoint.best*.pt | tr '\n' ' ' )"
python "$PERCEIVER_ROOT"/fairseq/scripts/average_checkpoints.py \
    --inputs $best_ckpts \
    --output "$path_to_ckpt"

if [ "$dla_inf_k" == 2048 ]; then
    batch_size=64
    max_tokens=64_000
elif [ "$dla_inf_k" == 1024 ]; then
    batch_size=112
    max_tokens=224_000
elif [ "$dla_inf_k" == 512 ]; then
    batch_size=216
    max_tokens=280_000
elif [ "$dla_inf_k" == 384 ]; then
    batch_size=360
    max_tokens=260_000
elif [ "$dla_inf_k" == 256 ]; then
    batch_size=448
    max_tokens=448_000
elif [ "$dla_inf_k" == 128 ]; then
    batch_size=720
    max_tokens=720_000
elif [ "$dla_inf_k" == 64 ]; then
    batch_size=1_200
    max_tokens=1_200_000
fi

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
--max-tokens $max_tokens \
--batch-size $batch_size \
--beam 5 \
--lenpen $lenpen \
--scoring sacrebleu \
--seed 42 \
--model-overrides "{'dla_inf_num_latents': $dla_inf_k}"