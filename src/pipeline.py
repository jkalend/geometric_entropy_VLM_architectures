"""
HEDGE pipeline: distortion, generation, clustering, metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .hedge_algorithms import (
    cluster_terms_by_embedding,
    radflag,
    sentence_semantic_entropy,
    vase,
)
from .distortion import generate_distortions
from .label_judge import add_hallucination_labels
from .model_inference import generate_answers_transformers, generate_answers_with_layer_dynamics
from .data_loader import load_vqa_rad, load_medhallu, load_halueval_wild, to_vqa_dict
from .layer_dynamics import apply_layer_dynamics_metrics, compute_layer_roc_aucs

EMBED_MODELS = {
    "general": "all-MiniLM-L6-v2",
    "medical": "sentence-transformers/embeddinggemma-300m-medical",
}


def make_seq_for_clustering(row, alpha=1.0, append_question=False):
    """Build clustering input from a result row."""
    p = (row["question"] + " ") if append_question else ""
    normal = [p + d["ans"] for d in row["original_high_temp"]]
    noisy = [p + d["ans"] for d in row["distorted_high_temp"]]
    logn = [np.mean(d["logprob"]) for d in row["original_high_temp"]]
    logd = [np.mean(d["logprob"]) for d in row["distorted_high_temp"]]
    seq = [p + row["original_low_temp"]["ans"]] + normal + noisy
    n = len(normal)
    return {"n": n, "seq_input": seq, "normal": normal, "noisy": noisy, "logn": logn, "logd": logd, "alpha": alpha}


def apply_embed_clustering_df(df, embedding_fn, threshold=0.90, append_question=False, show_progress=True):
    """Apply embedding clustering and compute metrics."""
    df = df.copy()
    df["clustering_input"] = df.apply(make_seq_for_clustering, append_question=append_question, axis=1)
    all_sequences = [x["seq_input"] for x in df["clustering_input"]]
    ids_embds = []
    iterator = tqdm(all_sequences, desc="Embedding clustering") if show_progress else all_sequences
    for seq in iterator:
        ids_embd = cluster_terms_by_embedding(seq, embedding_fn, threshold=threshold)
        ids_embds.append(ids_embd)
    df["cluster_embed"] = ids_embds
    df["metrics_embed"] = df.apply(
        lambda row: _compute_metrics(
            n=row["clustering_input"]["n"],
            cluster_ids=row["cluster_embed"],
            normal_logs=row["clustering_input"]["logn"],
            noisy_logs=row["clustering_input"]["logd"],
            alpha=row["clustering_input"]["alpha"],
        ),
        axis=1,
    )
    df = df.drop(columns=["clustering_input"])
    return df


def _compute_metrics(n, cluster_ids, normal_logs, noisy_logs, alpha=1.0):
    """Compute SE, RadFlag, VASE from clustering results."""
    ent_clean, dist_clean = sentence_semantic_entropy(normal_logs, cluster_ids[1 : 1 + n])
    ent_noisy, dist_noisy = sentence_semantic_entropy(noisy_logs, cluster_ids[1 + n :])
    return {
        "SE": float(ent_clean),
        "RadFlag": radflag(cluster_ids, n),
        "VASE": vase(n, cluster_ids, dist_clean, dist_noisy, alpha),
    }


def compute_roc_aucs(df):
    """
    Compute ROC AUC for each metric.
    We predict hallucination: higher SE/VASE = more hallucination; RadFlag inverted.
    Invert labels so hallucination (0) becomes 1 = positive class for sklearn default.
    """
    if "hallucination_label" not in df.columns:
        raise KeyError("DataFrame must contain 'hallucination_label' column.")
    
    aucs = {}
    for variant_name, group in df.groupby("variant_name"):
        aucs[variant_name] = {}
        # Invert: 1=hallucination (positive), 0=correct — sklearn uses greater label as positive
        y_true = 1 - group["hallucination_label"]
        
        # Guard: ROC AUC requires at least two unique classes
        if len(np.unique(y_true)) < 2:
            metrics_df = pd.json_normalize(group["metrics_embed"])
            for k in metrics_df.columns:
                aucs[variant_name][k] = np.nan
            continue

        metrics_df = pd.json_normalize(group["metrics_embed"])
        for k, v in metrics_df.items():
            score = 1 - v if k == "RadFlag" else v
            aucs[variant_name][k] = roc_auc_score(y_true, score)
    return aucs


def _tune_threshold(df: pd.DataFrame, embed_fn, n_trials: int = 20) -> float:
    """Use Optuna to find the embedding threshold that maximizes mean AUC."""
    def objective(trial: optuna.Trial) -> float:
        thresh = trial.suggest_float("threshold", 0.75, 0.98)
        df_clustered = apply_embed_clustering_df(
            df.copy(), embed_fn, threshold=thresh, show_progress=False
        )
        aucs = compute_roc_aucs(df_clustered)
        
        # Use first available variant if "default" is missing
        variant_key = "default" if "default" in aucs else next(iter(aucs)) if aucs else None
        if variant_key is None or not aucs[variant_key]:
            return 0.0
            
        mean_auc = np.mean([v for v in aucs[variant_key].values() if not np.isnan(v)])
        return mean_auc if not np.isnan(mean_auc) else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params["threshold"]


def run_hedge_pipeline(
    dataset: str = "vqa_rad",
    max_samples: int = 20,
    num_distortions: int = 5,
    n_answers_high: int = 5,
    model_name: str = "qwen2.5-vl-7b",
    embed_model: str = "medical",
    embed_threshold: Optional[float] = None,
    tune_threshold: bool = True,
    tune_trials: int = 20,
    label_method: str = "simple",
    ollama_model: str = "gpt-oss-20b",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: int = 120,
) -> dict:
    """Run full HEDGE pipeline and return metrics."""
    if dataset == "vqa_rad":
        samples = load_vqa_rad(split="test", max_samples=max_samples)
    elif dataset == "medhallu":
        samples = load_medhallu(split="pqa_labeled", max_samples=max_samples)
    elif dataset == "halueval_wild":
        samples = load_halueval_wild(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    vqa_dict = to_vqa_dict(samples)

    distorted = generate_distortions(
        vqa_dict,
        num_samples=num_distortions,
        dataset_id=dataset,
    )

    answers = generate_answers_transformers(
        distorted,
        model_name=model_name,
        n_answers_high=n_answers_high,
    )

    df = pd.DataFrame(answers)
    df = add_hallucination_labels(
        df,
        method=label_method,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        ollama_timeout=ollama_timeout,
    )

    from sentence_transformers import SentenceTransformer
    model_id = EMBED_MODELS.get(embed_model, embed_model)
    model = SentenceTransformer(model_id)
    cache = {}

    def embed_fn(x):
        if x not in cache:
            cache[x] = model.encode(x, convert_to_numpy=True)
        return cache[x]

    if tune_threshold:
        print("Tuning embedding threshold with Optuna...")
        threshold = _tune_threshold(df, embed_fn, n_trials=tune_trials)
        print(f"Best threshold: {threshold:.3f}")
    else:
        threshold = embed_threshold if embed_threshold is not None else 0.90

    df = apply_embed_clustering_df(df, embed_fn, threshold=threshold)
    aucs = compute_roc_aucs(df)
    return {"aucs": aucs, "df": df, "embed_threshold": threshold}


def run_layer_dynamics_pipeline(
    dataset: str = "vqa_rad",
    max_samples: int = 20,
    num_distortions: int = 5,
    n_answers_high: int = 5,
    model_name: str = "qwen2.5-vl-7b",
    label_method: str = "simple",
    ollama_model: str = "glm-4.7-flash",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: int = 120,
) -> dict:
    """Run layer-wise semantic dynamics pipeline. Uses internal hidden states instead of output clustering.
    Supports Qwen2.5-VL and InternVL 2.5-8B. Returns LVD, LVS, LVD_middle metrics and ROC AUCs."""
    if dataset == "vqa_rad":
        samples = load_vqa_rad(split="test", max_samples=max_samples)
    elif dataset == "medhallu":
        samples = load_medhallu(split="pqa_labeled", max_samples=max_samples)
    elif dataset == "halueval_wild":
        samples = load_halueval_wild(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    vqa_dict = to_vqa_dict(samples)

    distorted = generate_distortions(
        vqa_dict,
        num_samples=num_distortions,
        dataset_id=dataset,
    )

    answers = generate_answers_with_layer_dynamics(
        distorted,
        model_name=model_name,
        n_answers_high=n_answers_high,
    )

    df = pd.DataFrame(answers)
    df = add_hallucination_labels(
        df,
        method=label_method,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        ollama_timeout=ollama_timeout,
    )

    df = apply_layer_dynamics_metrics(df)
    aucs = compute_layer_roc_aucs(df)
    return {"aucs": aucs, "df": df}
