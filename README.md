# Towards Unified Neural Decoding with Brain Functional Network Modeling
[![PyPI](https://img.shields.io/pypi/v/OpenBioSeq)](https://pypi.org/project/OpenBioSeq)
[![license](https://img.shields.io/badge/license-Apache--2.0-%23B7A800)](https://github.com/Westlake-AI/OpenBioSeq/blob/main/LICENSE)

## Introduction

The main branch works with **PyTorch 1.11** (required by some self-supervised methods) or higher (**PyTorch 1.12**). You can still use **PyTorch 1.8** for most cases.

### What does this repo do?

This is the official implementation of the paper **Towards Unified Neural Decoding with Brain Functional Network Modeling**.


## Installation

There are quick installation steps for develepment:

```shell
conda create -n openbioseq python=3.8 -y
conda activate openbioseq
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 # as an example
python setup.py develop
```

Installation from scratch typically takes less than 2 hours.

## Training Syllable Decoders

Please see [Getting Started](docs/GETTING_STARTED.md) for the basic usage of OpenBioSeq (based on OpenMixup and MMSelfSup). As an example, you can start a multiple GPUs training with a certain `CONFIG_FILE` using the following script: 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
Then, please see [tutorials](docs/tutorials) for more tech details (based on MMClassification).

CONFIG_FILE for training prediction models of syllable components (initial, tone, and final) are located in configs/benchmarks/classification

## License

This project is released under the [Apache 2.0 license](LICENSE).
