# Awesome LLM Quantization Guide & Notebooks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange.svg)](https://huggingface.co/)

A comprehensive reference guide and hands-on repository dedicated to deep learning model compression techniques. This repository features detailed conceptual breakdowns alongside practical implementations (such as using Intel's **AutoRound**, **GPTQ**, and **AWQ**) to optimize large language models (LLMs) for ultra-efficient local and enterprise deployment.

---

## ⚡ What is Quantization?

**Quantization** is the process of compressing a machine learning model's numerical representations from high-precision formats (like FP32 or FP16) into lower-bit formats (like INT8 or INT4). This drastically slashes memory usage, dramatically accelerates inference throughput, and minimizes computational costs with negligible loss in model accuracy.

> 💡 **Example Impact:** Converting a **9B parameter model** from standard `FP16` (~18 GB VRAM requirement) down to a 4-bit (`W4`) format reduces its footprint to roughly **4.5 GB**, democratizing the model to run on standard consumer-grade GPUs or even edge devices!

---

## 🧠 Core Components: Weights vs. Activations

Quantization strategies evaluate and compress two principal elements of a model during running execution:

| Component | Definition | Conceptual Analogy | Runtime Nature |
| :--- | :--- | :--- | :--- |
| **Weights** | Stored, static learned parameters of the network. | The model's permanent knowledge base. | Static (Saved in model file) |
| **Activations** | Dynamic, temporary intermediate values calculated during a forward pass. | The intermediate runtime calculations / thoughts. | Transient (Changes per input token) |

### 🔍 Numerical Precision Scale Example
* **FP32 (High Precision):** `0.7345281` *(Uses 32 bits, highly precise decimals)*
* **FP16 (Half Precision):** `0.7344` *(Uses 16 bits, fewer decimal places)*
* **INT4 (4-bit Quantization):** `7` *(Cannot store detailed decimals directly; values are scaled to discrete integers)*

---

## 📊 Numeric Precision Formats

Below is a technical landscape of precision formats widely applied throughout modern artificial intelligence training, optimization, and serving:

| Format | Bits per Number | Type | Typical Use Case | Pros | Cons |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **FP64** | 64 bits | Floating Point | Scientific computing & precision simulations | Extreme numerical safety | Extremely slow compute, massive VRAM |
| **FP32** | 32 bits | Floating Point | Standard machine learning model training | Excellent precision & stability | Heavy memory consumption |
| **FP16** | 16 bits | Floating Point | Traditional AI training & standard inference | Compact footprint, fast compute | Restricted dynamic range compared to BF16 |
| **BF16** | 16 bits | Floating Point | Modern LLM training (Google TPUs / NVIDIA GPUs) | Matches FP32 dynamic range; stable | Marginally less precise than FP16 |
| **TF32** | 19 bits | Tensor Float | Internal processing on NVIDIA Tensor Cores | Speeds up FP32 code seamlessly | Native/proprietary to NVIDIA architecture |
| **FP8** | 8 bits | Floating Point | Cutting-edge LLM serving & training (H100/B200) | Blazing throughput, tiny memory footprint | Requires specialized hardware frameworks |
| **INT16** | 16 bits | Integer | Digital Signal Processing (DSP) & audio models | Highly efficient fixed-point math | Incapable of representing raw fractions |
| **INT8** | 8 bits | Integer | Quantized AI inference production serving | High deployment speeds, broad hardware | Potential for minor accuracy degradation |
| **INT4** | 4 bits | Integer | Local LLM compression & edge deployment | Massive VRAM savings (~75% reduction) | Minor degradation in complex reasoning |
| **INT2** | 2 bits | Integer | Experimental ultra-low-bit research | Hyper-miniaturized model sizes | High risk of accuracy/output degradation |
| **Binary** | 1 bit | Integer | Extreme micro-hardware optimization research | Absolute lowest physical memory scale | Extremely low fidelity; training complexities |

---

## 🏷️ Bit-Width Nomenclature

When viewing modern model configurations or Hugging Face repositories, the following shorthand conventions identify the allocation of bits across weights (`W`) and activations (`A`):

* `W4`: Weights are compressed to 4-bit precision.
* `W8`: Weights are compressed to 8-bit precision.
* `A16`: Activations are preserved at 16-bit precision to maintain conversational quality.
* `W4A16`: 4-bit weights paired with 16-bit activations (The sweet spot for local LLMs).
* `W8A8`: Both weights and activations are quantized to 8-bit (Optimized for specialized INT8 matrix hardware accelerators).
* `FP16 / BF16`: Non-quantized baseline half-precision states.

---

## 🛠️ Advanced Quantization Methods & Algorithms

These industry-standard algorithms employ sophisticated mathematics to map parameters safely into low-bit configurations without snapping neural connections:

| Quantization Method | Full Form / Philosophy | Main Strategic Purpose | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **GPTQ** | Generative Pre-trained Transformer Quantization | Highly accurate Post-Training Quantization (PTQ) processing layers sequentially using calibration datasets. | Local LLM inference on consumer-grade desktop GPUs. |
| **AWQ** | Activation-aware Weight Quantization | Protects the critical 1% "salient weights" by tracking activation charts and scaling them separately. | High-throughput enterprise serving infrastructures. |
| **AutoRound** | Optimization-based Rounding Quantization | Leverages sign-gradient descent optimization to discover superior integer rounding paths over simple nearest-rounding. | Highly stable, granular low-bit (2-bit or 4-bit) weight tuning. |
| **BitsAndBytes (bnb)** | CUDA Quantization Engine | Dynamically isolates and separates activation outliers on-the-fly during operational runtime. | Hugging Face native integration, rapid prototyping, and **QLoRA**. |
| **QLoRA** | Quantized Low-Rank Adaptation | Injects low-rank adapters over a frozen 4-bit base model (`NF4`), backpropagating gradients through the quantized core. | Efficient fine-tuning of large models on budget consumer hardware. |
| **GGUF** | llama.cpp Quantization Format | Unified binary format configured for dual CPU/GPU resource splitting using block-wise quantization matrices. | Local desktop, laptop, and mobile device deployments via `llama.cpp`. |
| **TensorRT-LLM** | NVIDIA TensorRT Optimization Suite | Hardware-fused compiler framework providing aggressive graph optimizations and deep tensor routing. | Industrial, ultra-low latency enterprise GPU cluster scaling. |

---
