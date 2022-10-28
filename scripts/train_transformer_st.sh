#!/bin/bash

lang_pair=$1

### Average best asr checkpoints
asr_experiment_name=asr_transformer
asr_experiment_path=${OUTPUT_ROOT}/asr/${asr_experiment_name}
asr_checkpoint=${asr_experiment_path}/ckpts/avg_best_10_checkpoint.pt

python "$PERCEIVER_ROOT"/scripts/find_best_ckpts.py \
    "$asr_experiment_path"/ckpts 10 min
inputs=$(head -n 1 "${asr_experiment_path}"/ckpts/best_10.txt)

python "$PERCEIVER_ROOT"/fairseq/scripts/average_checkpoints.py \
    --inputs "$inputs" \
    --output "$asr_checkpoint"
###

experiment_name=st_${lang_pair}_transformer
experiment_path=${OUTPUT_ROOT}/st/${experiment_name}
mkdir -p "${experiment_path}"

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-train "${MUSTC_ROOT}/${lang_pair}" \
--save-dir "${experiment_path}"/ckpts/ \
--config-yaml config_st.yaml \
--train-subset train_st \
--valid-subset dev_st \
--num-workers $((n_cpus / 2)) \
--max-tokens 54_000 \
--max-update 100_000 \
--batch-size 114 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--arch s2t_transformer_s \
--optimizer adam \
--lr 0.002 \
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
--patience 15 \
--best-checkpoint-metric nll_loss \
--data-buffer-size 100 \
--load-pretrained-encoder-from "$asr_checkpoint"