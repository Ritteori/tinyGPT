# TinyGPT — Minimal GPT-like Language Model from Scratch

This repository contains **TinyGPT**, a from-scratch implementation of a small GPT-style language model trained on the WikiText dataset using a **custom BPE tokenizer**, **causal self-attention**, and a **pure PyTorch training loop**.

The goal of this project is **educational and engineering-focused**: to understand and implement the full pipeline of an autoregressive language model without relying on high-level frameworks or pretrained components.

---

## 🔍 Project Highlights

* Transformer-based autoregressive language model (GPT-style)
* Causal multi-head self-attention with KV caching support
* Custom Byte Pair Encoding (BPE) tokenizer (implemented from scratch)
* End-to-end training loop in PyTorch
* Dataset preprocessing and token flattening
* TensorBoard logging and detailed debug diagnostics
* FastAPI-based inference server
* Docker support for reproducibility

---

## 📂 Repository Structure

```
.
├── all_steps_with_bpe.ipynb   # Full tokenizer + encoding walkthrough
├── train.ipynb               # Training notebook
├── test.ipynb                # Generation / evaluation notebook
├── model.py                  # TinyGPT model definition
├── tokenizer.py              # Custom BPE tokenizer implementation
├── dataset.py                # Dataset + DataLoader logic
├── utils.py                  # Training loop, generation utilities
├── inference.py              # FastAPI inference server
├── config.py                 # All hyperparameters and paths
├── base.html                 # Simple frontend for text generation
├── checkpoints/              # Saved model checkpoints
├── logs/                     # TensorBoard logs
├── data/                     # WikiText parquet files
├── saved_tokenizer/          # Saved vocab, merges, encoded tokens
├── Dockerfile                # Docker build file
└── __pycache__/
```

---

## 🧠 Model Architecture

**TinyGPT** follows a classic decoder-only Transformer design:

* Token embedding + positional embedding
* Stack of Transformer blocks:

  * LayerNorm → Causal Multi-Head Self-Attention → Residual
  * LayerNorm → Feed-Forward Network → Residual
* Final LayerNorm + Linear projection to vocabulary

### Key Parameters

```python
EMBED_DIM = 512
NUM_HEADS = 8
DEPTH = 3
MAX_SEQ_LEN = 256
HIDDEN_DIM_MULTIPLICATOR = 4
DROPOUT = 0.2
```

---

## 🔤 Tokenization

* Custom **Byte Pair Encoding (BPE)** tokenizer
* Trained from scratch on WikiText text
* Subword-level tokenization (character-based base vocabulary + merges)
* Encoded dataset is flattened into a single token stream for efficient training

Tokenizer artifacts are saved and reused:

```
saved_tokenizer/
├── vocab.pt
├── train_encoded_textes.pth
└── test_encoded_textes.pth
```

---

## 📊 Training Setup

### Hyperparameters

```python
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 7
MERGES = 2000
```

* Optimizer: Adam
* Scheduler: CosineAnnealingLR
* Loss: CrossEntropyLoss
* Mixed precision training with `torch.amp`

### Logged Metrics

* Training / validation loss
* Perplexity
* Token-level accuracy
* Gradient norm

TensorBoard logs are stored in:

```
logs/
```

---

## 📈 Training Results (WikiText-2)

After 7 epochs:

* **Train Perplexity:** ~23.3
* **Test Perplexity:** ~46.2
* **Test Accuracy:** ~30.8%

The model demonstrates:

* Stable convergence
* Meaningful short-text generation
* Correct usage of frequent linguistic patterns

(Example TensorBoard screenshots can be added here.)

---

## ✍️ Text Generation Examples

```text
Prompt: "Once upon a time"
Output: once upon a time . the episode was written by jim johnson and directed by jack johnson

Prompt: "The meaning of life is"
Output: the meaning of life is a real of the same name . the original name is derived from the original name of
```

---

## 🚀 Inference API

A FastAPI server is provided for inference:

```bash
python inference.py
```

Endpoint:

```
POST /generate
{
  "prompt": "Once upon a time",
  "max_tokens": 50
}
```

A simple HTML frontend (`base.html`) is included for browser-based interaction.

---

## 🐳 Docker Support

Build and run:

```bash
docker build -t tinygpt .
docker run -p 8000:8000 tinygpt
```

---

## 🎯 Project Goals

This project focuses on:

* Understanding Transformers at a low level
* Implementing tokenization, training, and inference manually
* Building ML systems without pretrained shortcuts
* Developing ML engineering discipline (logging, structure, reproducibility)

It is **not intended to compete with modern large language models**, but to serve as a strong foundation for deeper LLM work.

---

## 🔮 Future Improvements

* Training on WikiText-103
* Deeper Transformer stacks
* KV-cache optimized inference loop
* Sampling strategies (top-k, top-p, temperature)
* Checkpoint resume & experiment tracking

---

If you are reviewing this project: this implementation was intentionally built from scratch to demonstrate understanding of **language modeling fundamentals and ML engineering practices**.
