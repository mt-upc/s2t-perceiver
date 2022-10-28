#!/bin/bash

experiment_name=asr_transformer
experiment_path=${OUTPUT_ROOT}/asr/${experiment_name}
mkdir -p "${experiment_path}"

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-train "${MUSTC_ROOT}"/en-de \
--save-dir "${experiment_path}"/ckpts/ \
--config-yaml config_asr.yaml \
--train-subset train_asr \
--valid-subset dev_asr \
--num-workers $((n_cpus / 2)) \
--max-tokens 54_000 \
--max-update 100_000 \
--batch-size 114 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--arch s2t_transformer_s \
--optimizer adam \
--lr 0.001 \
--lr-scheduler inverse_sqrt \
--warmup-updates 5000 \
--clip-norm 10.0 \
--seed 42 \
--fp16 \
--no-scale-embedding \
--activation-fn gelu \
--encoder-layers 13 \
--update-freq $((6 / n_gpus)) \
--no-epoch-checkpoints \
--keep-best-checkpoints 10 \
--patience 10 \
--best-checkpoint-metric nll_loss \
--data-buffer-size 100