"""
Main entry point for HEDGE evaluation across VLM architectures.
Usage: python run_evaluation.py [--dataset vqa_rad] [--max-samples 10] [--model qwen2.5-vl-7b]
       python run_evaluation.py --layer-dynamics [--dataset medhallu] ...
"""

import argparse
import json
from pathlib import Path

from src.pipeline import run_hedge_pipeline, run_layer_dynamics_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dynamics", action="store_true",
                        help="Use layer-wise semantic dynamics (internal hidden states) instead of output-level HEDGE")
    parser.add_argument("--dataset", default="vqa_rad", choices=["vqa_rad", "medhallu", "halueval_wild"])
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--num-distortions", type=int, default=5)
    parser.add_argument("--n-answers-high", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        help="VLM: qwen2.5-vl-7b, qwen3-vl-8b, qwen3-vl-30b, medgemma-4b-it")
    parser.add_argument("--use-unsloth", action="store_true", default=False,
                        help="Use Unsloth (if available) for optimized inference")
    parser.add_argument("--embed-model", default="medical", choices=["general", "medical"],
                        help="Embedding model (HEDGE only): medical or general")
    parser.add_argument("--embed-threshold", type=float, default=None,
                        help="Fixed embedding threshold (HEDGE only; default: tune with Optuna)")
    parser.add_argument("--no-tune-threshold", action="store_true",
                        help="Disable Optuna threshold tuning (HEDGE only)")
    parser.add_argument("--tune-trials", type=int, default=20,
                        help="Optuna trials for threshold tuning (HEDGE only)")
    parser.add_argument("--label-method", default="ollama", choices=["simple", "ollama"],
                        help="Hallucination labels: simple (string-match) or ollama (LLM judge)")
    parser.add_argument("--ollama-model", default="gpt-oss:20b",
                        help="Ollama model for judge (e.g. glm-4.7-flash, gpt-oss:20b, qwen3.5:27b)")
    parser.add_argument("--ollama-timeout", type=int, default=120)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if args.use_unsloth:
        print("Warning: Unsloth support is experimental for the remaining models.")
    if args.label_method == "ollama":
        from src.label_judge import check_ollama_available
        if not check_ollama_available():
            raise SystemExit("Ollama not reachable at localhost:11434. Start Ollama or use --label-method simple")

    if args.layer_dynamics:
        print(f"Running Layer Dynamics on {args.dataset} with {args.model} ({args.max_samples} samples, labels: {args.label_method})")
        result = run_layer_dynamics_pipeline(
            dataset=args.dataset,
            max_samples=args.max_samples,
            num_distortions=args.num_distortions,
            n_answers_high=args.n_answers_high,
            model_name=args.model,
            label_method=args.label_method,
            ollama_model=args.ollama_model,
            ollama_timeout=args.ollama_timeout,
        )
        out = {
            "mode": "layer_dynamics",
            "dataset": args.dataset,
            "model": args.model,
            "label_method": args.label_method,
            "judge_model": args.ollama_model if args.label_method == "ollama" else None,
            "aucs": result["aucs"],
        }
    else:
        tune_threshold = not args.no_tune_threshold
        print(f"Running HEDGE on {args.dataset} with {args.model} ({args.max_samples} samples, labels: {args.label_method}, embed: {args.embed_model}, tune: {tune_threshold}, unsloth: {args.use_unsloth})")
        result = run_hedge_pipeline(
            dataset=args.dataset,
            max_samples=args.max_samples,
            num_distortions=args.num_distortions,
            n_answers_high=args.n_answers_high,
            model_name=args.model,
            use_unsloth=args.use_unsloth,
            embed_model=args.embed_model,
            embed_threshold=args.embed_threshold,
            tune_threshold=tune_threshold,
            tune_trials=args.tune_trials,
            label_method=args.label_method,
            ollama_model=args.ollama_model,
            ollama_timeout=args.ollama_timeout,
        )
        out = {
            "mode": "hedge",
            "dataset": args.dataset,
            "model": args.model,
            "label_method": args.label_method,
            "judge_model": args.ollama_model if args.label_method == "ollama" else None,
            "embed_model": args.embed_model,
            "embed_threshold": result.get("embed_threshold"),
            "aucs": result["aucs"],
        }

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Results saved to {args.output}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
