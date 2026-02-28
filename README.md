# Geometric Entropy Across VLM Architectures

This project implements the **HEDGE (Hallucination Estimation via Dense Geometric Entropy)** framework to evaluate the stability of Vision-Language Models (VLMs) under controlled visual perturbations. It specifically investigates how geometric hallucination detection methods transfer between different VLM architectures, such as dense-tokenization models (Qwen2.5-VL) and restricted-tokenization models (MedGemma).

## Key Features

- **HEDGE Framework**: Detects hallucinations by measuring the stability of a model's embedding manifold.
- **Multi-Architecture Support**: Native support for **Qwen2.5-VL**, **Qwen3-VL-8B**, **Qwen3-VL-30B**, and **MedGemma**.
- **Layer-Wise Semantic Dynamics**: Analyzes internal hidden states across LLM layers to identify where hallucinations originate. Supports **Qwen2.5-VL**, **Qwen3-VL-30B**, and **MedGemma**.
- **Medical Domain Focus**: Evaluates performance on medical VQA datasets (VQA-RAD, MedHallu).
- **Windows Optimized**: Custom transformers-based inference pipeline designed to run on Windows (where vLLM is unsupported).

## Project Structure

```text
geometric-entropy/
├── src/
│   ├── hedge_algorithms.py   # SE, RadFlag, VASE, clustering
│   ├── data_loader.py        # VQA-RAD, MedHallu, HaluEval-Wild
│   ├── distortion.py         # Visual perturbation
│   ├── label_judge.py        # Ollama / simple hallucination labels
│   ├── layer_dynamics.py     # LVD, LVS, layer-wise metrics
│   ├── model_inference.py    # Transformers-based VLM inference
│   └── pipeline.py           # Full HEDGE pipeline
├── scripts/
│   └── prepare_data.py       # Dataset download/prep
├── run_evaluation.py         # Main entry point for HEDGE and Layer Dynamics
├── run_cross_arch.py         # Cross-architecture evaluation script
├── setup.ps1                 # Windows setup script (venv + CUDA 13.0)
└── requirements.txt          # Project dependencies
```

## Getting Started

### Prerequisites

- Windows OS
- Python 3.10+
- NVIDIA GPU (RTX 4090 recommended for 24GB VRAM)
- [Ollama](https://ollama.com/) (optional, for LLM-based hallucination judging)

### Installation

1. Clone the repository.
2. Run the setup script to create a virtual environment and install dependencies:
   ```powershell
   .\setup.ps1
   ```
   *Note: This script installs PyTorch with CUDA 13.0 support.*

3. Prepare the datasets:
   ```powershell
   python scripts/prepare_data.py
   ```

## Usage

### Running HEDGE Evaluation

To run a standard HEDGE evaluation on VQA-RAD:

```powershell
python run_evaluation.py --dataset vqa_rad --max-samples 10 --model qwen2.5-vl-7b --output results.json
```

For a full-scale run with Optuna threshold tuning:

```powershell
python run_evaluation.py --max-samples 100 --num-distortions 12 --n-answers-high 12 --output results_full.json
```

### Running Layer-Wise Dynamics

To analyze internal layer stability instead of output-level clustering:

```powershell
python run_evaluation.py --layer-dynamics --dataset vqa_rad --max-samples 10 --output results_layer.json
```

### Using MedGemma

```powershell
python run_evaluation.py --model medgemma-4b-it --max-samples 10 --output results_medgemma.json
```

### Hallucination Judging with Ollama

The pipeline can use an Ollama-hosted model to judge if a generated answer is a hallucination:

```powershell
python run_evaluation.py --label-method ollama --ollama-model gpt-oss:20b
```

## Metrics

- **SE (Semantic Entropy)**: Measures the diversity of generated answers.
- **RadFlag**: Measures the fraction of clean samples that fall into the primary cluster.
- **VASE (Vision-Amplified Semantic Entropy)**: Measures the stability gap between clean and noisy visual conditions.
- **LVD (Layer Variance Delta)**: (Layer Dynamics) Measures the change in variance across layers.

## License

[MIT License](LICENSE)
