# Efficient Speech Translation with Dynamic Latent Perceivers

TODO: add here arxiv link

The pre-print of this research is available [here](...).

## Abstract

<em>
Transformers have been the dominant architecture for Speech Translation in recent years, achieving significant improvements in translation quality. Since speech signals are longer than their textual counterparts, and due to the quadratic complexity of the Transformer, a down-sampling step is essential for its adoption in Speech Translation. Instead, in this research, we propose to ease the complexity by using a Perceiver encoder to map the speech inputs to a fixed-length latent representation. Furthermore, we introduce a novel way of training Perceivers, with Dynamic Latent Access (DLA), unlocking larger latent spaces without any additional computational overhead. Speech-to-Text Perceivers with DLA can match the performance of a Transformer baseline across three language pairs in MuST-C. Finally, a DLA-trained model is easily adaptable to DLA at inference, and can be flexibly deployed with various computational budgets, without significant drops in translation quality.
</em>

## Speech-to-Text Perceiver and Dynamic Latent Access

S2T-Perceiver |  Dynamic Latent Access
:-------------------------:|:-------------------------:
![](figures/s2t-perceiver.png)  |  ![](figures/dla.png)

## Citation

TODO (add arxiv citation info)

## Setup

TODO (empty roots)

```bash
export PERCEIVER_ROOT=~/repos/s2t-perceiver
export MUSTC_ROOT=$SPEECH_DATA/MUSTC_v2.0_spec
export OUTPUT_ROOT=$VEUSSD/s2t-perceiver
```

Clone this repository to `$PERCEIVER_ROOT`:

```bash
git clone https://github.com/mt-upc/s2t_perceiver.git ${PERCEIVER_ROOT}
```

Create a conda environment using the `environment.yml` file and activate it:
TODO (setup correct commit in fairseq as a submodule)

```bash
conda env create -f ${PERCEIVER_ROOT}/environment.yml && \
conda activate s2t_perceiver && \
pip install --editable ${PERCEIVER_ROOT}/fairseq/
```

To prepare the MuST-C data follow the instructions [here](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md#data-preparation). We used en-de (v2.0) and en-es, en-ru (v1.0).

## Train an S2T-Perceiver with Dynamic Latent Access (DLA-train)

To train the model first do the ASR pre-training step and then start the ST training with the pre-trained encoder.

- To train without DLA-train, set `$k_train = $n`.
- Example is for English-to-German (en-de).
- The suggested values for `base_update_freq` and `batch_size` are for an NVIDIA GeForce RTX 2080 Ti
, adjust them accordingly for other devices.
- For training with multiple devices make sure that `base_update_freq` is divisible by `n_gpus`.

```bash
# total number of latents
n=...
# number of latents for DLA-train
k_train=...

# ASR pre-training
bash ${PERCEIVER_ROOT}/scripts/train_perceiver_asr.sh $n $k_train

# ST training
bash ${PERCEIVER_ROOT}/scripts/train_perceiver_st.sh en-de $n $k_train
```

## Evaluate an S2T-Perceiver with Dynamic Latent Access (DLA-inf)

To evaluate without DLA-inf, set `$k_inf = $n`.

```bash
# path to the trained model ($path_to_exp/ckpts)
path_to_exp=...
# number of latents for DLA-inf
k_inf=...

bash ${PERCEIVER_ROOT}/scripts/eval_st.sh $path_to_exp en-de $k_inf
```
