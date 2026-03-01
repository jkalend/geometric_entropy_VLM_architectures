"""
Cross-architecture evaluation: run HEDGE on Qwen2.5-VL, Qwen3-VL-8B, Qwen3-VL-30B, and Med-Gemma.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_hedge_pipeline

MODELS = ["qwen2.5-vl-7b", "qwen3-vl-8b", "qwen3-vl-30b", "medgemma-4b-it"]


def main():
    results = {}
    for model in MODELS:
        print(f"\n=== Evaluating {model} ===")
        try:
            out = run_hedge_pipeline(
                dataset="vqa_rad",
                max_samples=5,
                num_distortions=3,
                n_answers_high=3,
                model_name=model,
            )
            results[model] = out["aucs"]
        except Exception as e:
            print(f"  Error: {e}")
            results[model] = {"error": str(e)}

    Path("cross_arch_results.json").write_text(json.dumps(results, indent=2))
    print("\nResults:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
