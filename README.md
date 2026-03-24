# ESG Text Classification using Transformer (Karpathy Autoresearch)

This project demonstrates training a Transformer-based language model from scratch for **ESG text classification** (Environmental vs Non-Environmental), using a custom dataset and adapting a research repository to run on a local GPU setup.

---

## Objective

Classify text into:
- Environmental
- Non-Environmental

Using instruction-style prompting with a Transformer model.

---

## Tech Stack

- Python 3.10
- PyTorch
- rustbpe (Tokenizer)
- PyArrow (Parquet format)
- NVIDIA GPU (RTX 16GB)

---

## Installation (Anaconda)

```bash
# Create environment
conda create -n autoresearch_demo python=3.10 -y
conda activate autoresearch_demo

# Install PyTorch (CUDA version as per your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```
---
---
---

# Project Workflow

## 1. Data Preparation

    Original dataset format:
    ------------------------------
    text | label
    ------------------------------
----
    Converted into instruction-style format:
    Classify the following text:
    text....
    Answer: <label>
----
    Example:
    Classify the following text:
    We pursue sustainable transportation solutions.
    Answer: Environmental
----
----
## 2. Data Conversion Script
    Then converted into Parquet format for compatibility with the training pipeline.
----
    convert_csv_to_parquet.py
----
----
## 3. Tokenizer Training
```bash
python prepare.py --num-shards 0
```
----
    This generates:
    1. tokenizer.pkl
    2. token_bytes.pt
----
----

## 4. Model Training
```bash
python train.py
```
----
# Modifications Made (Key Highlight)
The original repo is designed for Linux + H100 GPUs.
To run locally on Windows + RTX GPU, the following changes were required:

    Note: refer train.ipynb file for complete code replacement
---
## 1. Flash Attention (Unsupported on Windows)
→ Replaced with:
```bash
torch.nn.functional.scaled_dot_product_attention
```
---
## 2. torch.compile (Requires Triton)
→ Disabled
```bash
torch.compile
```
---

## 3. Muon Optimizer (Triton-based)
→ Replaced with:
```bash
torch.optim.AdamW
```
---

## 4. Batch configuration too large
→ Tuned:
```bash
DEVICE_BATCH_SIZE

TOTAL_BATCH_SIZE
```
---

## 5. Dataset format mismatch
→ Converted classification → instruction format

---
## 6.Results

    Model size: ~26M parameters
    Training time: ~5 minutes
    Tokens processed: ~2.3M
    Validation BPB: ~1.17
---
## 6.Example Inference
    Input:
        Classify the following text:
        We are investing in renewable energy
    Answer:

    Output:
    Environmental
---
## 7. Challenges Faced

    1. Flash Attention not supported on Windows
    2. Triton dependency errors
    3. Optimizer incompatibility
    4. GPU memory constraints
    5. Batch size tuning
---
## 8. Future Improvements

    1. Train longer for better convergence
    2. Improve dataset size & balance
    3. Add evaluation metrics (accuracy, F1-score)
    4. Build inference API or UI
---
## 9. Project Structure \
├── train.py              # Modified training script \
├── train.ipynb              # Details of script replaced \
├── prepare.py            # Data + tokenizer (unchanged) \
├── convert_data.py       # Custom data conversion \
├── notebook/ \
│   └── experiment.ipynb  \
├── data/ \
│       └── sample_data.csv  \
├── requirements.txt  
└── README.md

---
## 10. How to Run
```bash
python convert_data.py
```
```bash
python prepare.py --num-shards 0
```
```bash
python train.py
```
---

## 11. Files Excluded

The following are not included in the repo:

    .cache/
    .parquet files (large)
    .pt, .pkl files
----
----
----

# 🤝 Acknowledgement

This project is based on:
    https://github.com/karpathy/autoresearch
----

