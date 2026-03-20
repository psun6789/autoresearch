# 🚀 ESG Text Classification using Transformer (Karpathy Autoresearch)

This project demonstrates training a Transformer-based language model from scratch for **ESG text classification** (Environmental vs Non-Environmental), using a custom dataset and adapting a research repository to run on a local GPU setup.

---

## 🧠 Objective

Classify text into:
- Environmental
- Non-Environmental

Using instruction-style prompting with a Transformer model.

---

## ⚙️ Tech Stack

- Python 3.10
- PyTorch
- rustbpe (Tokenizer)
- PyArrow (Parquet format)
- NVIDIA GPU (RTX 16GB)

---

## 🛠️ Installation (Anaconda)

```bash
# Create environment
conda create -n autoresearch_demo python=3.10 -y
conda activate autoresearch_demo

# Install PyTorch (CUDA version as per your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt