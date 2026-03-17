# CAHL

Official code for **Context-Aware Hierarchical Learning (CAHL)**: A Two-Step Paradigm towards Safer LLMs.

## Overview

CAHL implements a safety training and evaluation pipeline for tool-use LLM scenarios, with three model variants:

- StruQ 
- ISE 
- CAHL 

The code is mainly used for:

- Supervised fine-tuning on TCA datasets
- Inference-time evaluation on benign and attack test sets
- Attack-related metric reporting, with optional AlpacaEval capability evaluation

## Structure

The `tca/` includes Tool-Completion Attack and benchmark components:
The `src/` folder contains additional experimental code for attacks, training, and model on StruQ baseline pipeline.

## Installation

Python 3.10+ and CUDA are recommended.

```bash
cd tca
pip install -r requirements.txt
```

Notes:

- The training pipeline uses `torch` + `transformers` + `trl`

## Quick Start

### 1) Training

Example (`tca` pipeline):

```bash
cd tca/train
python train_tool.py --cfg cahl.yaml
```

Example (`src` pipeline):

```bash
cd src
python train/train.py train/training2.yaml
```

### 2) Evaluation

Evaluation entry: `tca/test/tcb_test.py`.

Edit `tca/test/tcb_test.yaml` and set at least:

- `model_name_or_path`: path to the model to evaluate
- `result_path`: output directory for evaluation results
- `model_type`: `struq` / `ise` / `cahl`

Run:

```bash
cd tca/test
python tcb_test.py --cfg tcb_test.yaml
```

Example (`src` attack/evaluation script):

```bash
cd src
python attack/test_ICAseqQformer.py \
    -m <MODEL_PATH> \
    -a none completion_real \
```

This script supports multiple attack modes (e.g., `none`, `naive`, `ignore`, `escape_separation`, `completion_real`).


## Citation

```bibtex
@inproceedings{ma2025contextaware,
    title={Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer {LLM}s},
    author={Tengyun Ma and Jiaqi Yao and Daojing He and Shihao Peng and YU LI and Shaohui Liu and Zhuotao Tian},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://arxiv.org/abs/2512.03720}
}
```
