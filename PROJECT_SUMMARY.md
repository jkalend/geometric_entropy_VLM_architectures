# Geometric Entropy Across VLM Architectures — Project Summary

## Project Overview

This project implements the **HEDGE (Hallucination Estimation via Dense Geometric Entropy)** framework to evaluate cross-architecture transferability of geometric hallucination detection methods across Vision-Language Models (VLMs). The goal is to determine whether geometric stability metrics remain reliable when moving from dense-tokenization models (e.g., Qwen2.5-VL) to restricted-tokenization models (Med-Gemma).

### Key Concepts

- **HEDGE**: Reframes hallucination detection as measuring the stability of a model's embedding manifold under controlled visual perturbations.
- **Metrics**: Semantic Entropy (SE), RadFlag, and Vision-Amplified Semantic Entropy (VASE). VASE historically provides the most robust hallucination signal by measuring the stability gap between clean and noisy conditions.
- **Architectures**:
  - **Qwen2.5-VL**: Dense visual tokenization, unified-fusion — highest baseline signal.
  - **Qwen3-VL-8B**: Successor to Qwen2.5-VL; enhanced visual reasoning.
- **Qwen3-VL-30B-A3B**: High-performance MoE architecture; state-of-the-art visual perception. Uses **4-bit quantization** (BitsAndBytes) to fit in consumer VRAM.
  - **Med-Gemma**: Restricted tokenization, compressed visual features — lowest baseline signal.

---

## Technical Steps Taken

### 1. Environment Setup

- **Platform**: Windows with Python venv.
- **PyTorch**: Installed with CUDA 13.0 (cu130) for RTX 4090 (24GB VRAM).
- **Dependencies**: transformers, accelerate, datasets, scikit-learn, albumentations, sentence-transformers, optuna, opencv-python, pandas, tqdm, scipy, bitsandbytes, unsloth (optional).
- **Rationale**: vLLM (used by hedge-bench) does not support Windows; a transformers-based inference pipeline was implemented instead.

### 2. Dataset Preparation

- **VQA-RAD**: Medical VQA with images (flaviagiammarino/vqa-rad). Primary dataset for multimodal evaluation.
- **MedHallu**: Medical QA (UTAustin-AIHealth/MedHallu). Text-heavy; uses gray placeholder images for VLM compatibility.
- **HaluEval-Wild**: Real-world queries (yushihu/halueval-wild-nocr). Adversarially filtered; uses placeholder images.
- **Rationale**: MedHal and HaluEval-Wild are text-only; placeholder images allow VLMs to focus on the text query while preserving the HEDGE perturbation structure.

### 3. Memory-Efficient Model Loading

- **Precision**: bfloat16 for all models (~14–16GB VRAM for 7B).
- **Execution**: Sequential model loading; one model at a time to respect 32GB RAM and 24GB VRAM limits.
- **Models**: Qwen2.5-VL-7B (primary), MedGemma-4B-IT.
- **Unsloth** (optional): Experimental support for optimized inference.

### 4. HEDGE Pipeline Implementation

- **Distortion**: Albumentations pipeline (affine, color jitter, Gaussian/shot noise) to generate perturbed images.
- **Generation**: Hugging Face transformers for VLM inference (clean + distorted, low/high temperature).
- **Clustering**: Embedding-based clustering with medical model (embeddinggemma-300m-medical) or general (all-MiniLM-L6-v2). Optuna tunes the similarity threshold.
- **Metrics**: SE, RadFlag, VASE computed from clustering results.
- **Hallucination labels**: Ollama LLM judge (glm-4.7-flash, gpt-oss-20b) or simple string-match heuristic.

### 5. Project Structure

```text
geometric-entropy/
├── src/
├──   ├── hedge_algorithms.py   # SE, RadFlag, VASE, clustering
├──   ├── data_loader.py        # VQA-RAD, MedHallu, HaluEval-Wild
├──   ├── distortion.py         # Visual perturbation
├──   ├── expert_routing.py     # Expert Routing Dynamics (ERD, EPV) for MoE models
├──   ├── label_judge.py        # Ollama / simple hallucination labels
├──   ├── layer_dynamics.py     # LVD, LVS, layer-wise metrics
├──   ├── model_inference.py    # Transformers-based VLM inference
├──   └── pipeline.py           # Full HEDGE pipeline
├── scripts/
├──   └── prepare_data.py       # Dataset download/prep
├── run_evaluation.py         # Main entry point
├── run_cross_arch.py         # Cross-architecture evaluation
├── setup.ps1                 # Windows setup (venv + cu130)
├── requirements.txt
└── PROJECT_SUMMARY.md        # This file
```

---

## How to Run

### Setup

```powershell
.\setup.ps1
# Or manually:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Single-Model Evaluation

```powershell
# Full-scale (100 samples, 12 distortions, 12 answers; medical embeddings, Optuna tuning)
python run_evaluation.py --max-samples 100 --num-distortions 12 --n-answers-high 12 --output results.json

# Quick run
python run_evaluation.py --dataset vqa_rad --max-samples 10 --model qwen2.5-vl-7b --output results.json

# MedGemma evaluation
python run_evaluation.py --model medgemma-4b-it --max-samples 10 --output results_medgemma.json
```

### Layer-Wise Semantic Dynamics

```powershell
python run_evaluation.py --layer-dynamics --dataset vqa_rad --max-samples 10 --output results_layer.json
```

Uses internal hidden states across LLM layers instead of output-level clustering. Metrics: **LVD** (Layer Variance Delta), **LVS** (Layer Variance Spike), **LVD_middle** (middle layers only). Supports **Qwen2.5-VL**, **Qwen3-VL-30B**, and **MedGemma**.

### Cross-Architecture Evaluation

```powershell
python run_cross_arch.py
```

### Data Preparation

```powershell
python scripts/prepare_data.py
```

---

## Results

### Full-Scale Evaluation (Qwen2.5-VL-7B, VQA-RAD)

Run: **100 samples**, **12 distortions**, **12 n_answers_high**. Ollama judge (glm-4.7-flash), medical embeddings, Optuna threshold tuning (20 trials).

| Metric | ROC AUC | Interpretation |
|--------|---------|----------------|
| SE | 0.585 | Above random; moderate predictive power |
| RadFlag | 0.617 | Best-performing metric |
| VASE | 0.593 | Similar to SE |

- **Best embedding threshold**: ~0.887 (Optuna-tuned; default 0.90).
- All metrics above 0.5 → geometric stability correlates with hallucination status.
- RadFlag (fraction of clean samples in cluster 0) is the strongest signal for this setup.

### Cross-Architecture

The current implementation supports Qwen2.5-VL and Med-Gemma.

---

## Architectural Choices and Rationale

| Choice | Reason |
|--------|--------|
| Transformers instead of vLLM | vLLM does not support Windows; transformers works on Windows with CUDA. |
| Embedding clustering over NLI | Lower memory and compute; suitable for 32GB RAM. |
| Placeholder images for text-only datasets | Allows VLMs to process text queries while keeping the HEDGE perturbation structure. |
| bfloat16 | Fits 7B models in 24GB VRAM with headroom for context. |
| Sequential model evaluation | Avoids loading multiple VLMs simultaneously. |
| Unsloth (optional) | Experimental support for optimized inference. |

---

## Layer-Wise Semantic Dynamics

- **Purpose**: Analyze geometric evolution of hidden states across layers to identify where hallucination begins.
- **Metrics**: LVD (mean layer variance delta), LVS (max variance ratio), LVD_middle (middle layers).
- **Research question**: Do layer-wise geometric patterns transfer more robustly across datasets (HaluEval vs. MedHallu) than output-level detectors?
- **Usage**: `python run_evaluation.py --layer-dynamics`

---

## Future Work

- Produce quantitative comparison of VASE across Qwen2.5 and Med-Gemma.
- **Expert Routing**: Wire `output_router_logits` through DeepSeek-VL2 forward to enable full ERD/EPV computation.
