#!/bin/bash

experiment_path=$1
lang_pair=$2
dla_inf_k=$3

path_to_ckpt=${experiment_path}/ckpts/avg_best_10_checkpoint.pt

python "$PERCEIVER_ROOT"/scripts/find_best_ckpts.py \
    "$experiment_path"/ckpts 10 min
inputs=$(head -n 1 "${experiment_path}"/ckpts/best_10.txt)

python "$PERCEIVER_ROOT"/fairseq/scripts/average_checkpoints.py \
    --inputs "$inputs" \
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
    batch_size=960
    max_tokens=960_000
elif [ "$dla_inf_k" == 64 ]; then
    batch_size=1_504
    max_tokens=1_504_000
fi

fairseq-train "${MUSTC_ROOT}/${lang_pair}" \
--config-yaml config_st.yaml \
--gen-subset tst-COMMON_st \
--task speech_to_text \
--path "$path_to_ckpt" \
--max-tokens $max_tokens \
--batch-size $batch_size \
--beam 5 \
--scoring sacrebleu \
--seed 42 \
--model-overrides "{'dla_inf_num_latents': $dla_inf_k}"