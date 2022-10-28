#!/bin/bash

n=$1
k=$2

experiment_name=asr_perceiver_${n}_DLA-k${k}
experiment_path=${OUTPUT_ROOT}/asr/"$experiment_name"
mkdir -p "$experiment_path"

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

if [ $k == 128 ]; then
    base_update_freq=4
    batch_size=128
    max_tokens=100_000
elif [ $k == 256 ] ; then
    base_update_freq=7
    batch_size=64
    max_tokens=100_000
elif [ $k == 384 ] ; then
    base_update_freq=9
    batch_size=48
    max_tokens=96_000
elif [ $k == 512 ]; then
    base_update_freq=14
    batch_size=32
    max_tokens=64_000
elif [ $k == 768 ]; then
    base_update_freq=18
    batch_size=24
    max_tokens=48_000
elif [ $k == 1024 ]; then
    base_update_freq=28
    batch_size=16
    max_tokens=32_000
fi

fairseq-train "${MUSTC_ROOT}"/en-de \
--save-dir "${experiment_path}"/ckpts/ \
--config-yaml config_asr.yaml \
--train-subset train_asr \
--valid-subset dev_asr \
--num-workers $((n_cpus / 2)) \
--batch-size $batch_size \
--max-tokens $max_tokens \
--max-update 100_000 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--arch s2t_perceiver_s \
--optimizer adam \
--lr 0.001 \
--lr-scheduler inverse_sqrt \
--warmup-updates 5_000 \
--clip-norm 10.0 \
--seed 42 \
--fp16 \
--no-scale-embedding \
--update-freq $((base_update_freq / n_gpus)) \
--no-epoch-checkpoints \
--keep-best-checkpoints 10 \
--patience 10 \
--best-checkpoint-metric nll_loss \
--num-latents "$n" \
--dla-train-num-latents "$k" \
--dropout 0.1 \
--conv-stride 1 \
--data-buffer-size 100