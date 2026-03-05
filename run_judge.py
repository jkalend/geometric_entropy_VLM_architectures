"""
Rerun hallucination judge on existing results.json file and recompute metrics.
Usage: python run_judge.py results.json [--label-method ollama] [--ollama-model gpt-oss:20b]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from src.label_judge import add_hallucination_labels, check_ollama_available
from src.pipeline import compute_roc_aucs, apply_embed_clustering_df, EMBED_MODELS
from src.layer_dynamics import apply_layer_dynamics_metrics, compute_layer_roc_aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to results.json file")
    parser.add_argument("--label-method", default="ollama", choices=["simple", "ollama"])
    parser.add_argument("--ollama-model", default="gpt-oss:20b")
    parser.add_argument("--ollama-timeout", type=int, default=120)
    parser.add_argument("--output", help="Path to save updated results (default: overwrite input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File {args.input} not found.")
        return

    data = json.loads(input_path.read_text())
    if "samples" not in data:
        print("Error: No 'samples' found in the input file. Make sure it was generated with the latest version.")
        return

    df = pd.DataFrame(data["samples"])
    mode = data.get("mode", "hedge")

    print(f"Rerunning judge ({args.label_method}) on {len(df)} samples...")
    
    if args.label_method == "ollama":
        if not check_ollama_available():
            print("Error: Ollama not reachable at localhost:11434.")
            return

    # Update labels
    df = add_hallucination_labels(
        df,
        method=args.label_method,
        ollama_model=args.ollama_model,
        ollama_timeout=args.ollama_timeout,
    )

    # Recompute metrics based on mode
    if mode == "layer_dynamics":
        print("Recomputing Layer Dynamics metrics...")
        # Note: Layer dynamics metrics (LVD, etc.) don't depend on the label, 
        # but the ROC AUC does.
        df = apply_layer_dynamics_metrics(df)
        aucs = compute_layer_roc_aucs(df)
        data["aucs"] = aucs
    else:
        print("Recomputing HEDGE metrics...")
        embed_model = data.get("embed_model", "medical")
        threshold = data.get("embed_threshold", 0.90)
        
        from sentence_transformers import SentenceTransformer
        model_id = EMBED_MODELS.get(embed_model, embed_model)
        model = SentenceTransformer(model_id)
        
        cache = {}
        def embed_fn(x):
            if x not in cache:
                cache[x] = model.encode(x, convert_to_numpy=True)
            return cache[x]
            
        df = apply_embed_clustering_df(df, embed_fn, threshold=threshold)
        aucs = compute_roc_aucs(df)
        data["aucs"] = aucs

    # Update data
    data["label_method"] = args.label_method
    if args.label_method == "ollama":
        data["judge_model"] = args.ollama_model
    
    # Convert df back to samples (stripping any temporary columns like 'metrics_embed' or 'metrics_layer')
    # We only want to keep the original sample columns + the new hallucination_label
    cols_to_keep = ["idx_img", "question", "image", "true_answer", "description", 
                    "original_high_temp", "distorted_high_temp", "original_low_temp", 
                    "variant_name", "hallucination_label"]
    
    # Filter columns that exist
    final_cols = [c for c in cols_to_keep if c in df.columns]
    data["samples"] = df[final_cols].to_dict(orient="records")

    output_path = Path(args.output) if args.output else input_path
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Updated results saved to {output_path}")


if __name__ == "__main__":
    main()
