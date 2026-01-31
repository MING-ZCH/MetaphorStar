# TFQ-GRPO with EasyR1

This training code is a clean fork of the original [veRL](https://github.com/volcengine/verl) project to support vision language models (VLMs) training with our **TFQ-GRPO algorithm**. EasyR1 is efficient and scalable due to the design of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and the latest release of **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode.

## Quick Start

### Installation

Please follow the instructions below to install the required packages.

1.Clone this repository

```bash
git clone https://github.com/MING-ZCH/MetaphorStar.git
```

2.Install Package

```bash
conda create -n metaphorstar python=3.10 -y
conda activate metaphorstar
cd MetaphorStar/train
pip install -e .
```

### TFQ-GRPO Training

```bash
bash examples/qwen2_5_vl_7b_TFQ_Data_Lite_TFQ_GRPO.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```
