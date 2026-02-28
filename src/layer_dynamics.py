"""
Layer-wise Semantic Dynamics for hallucination detection.
Analyzes geometric evolution of hidden states across LLM layers to identify
where hallucination begins to form. Metrics: layer variance delta, variance spike.
"""

from __future__ import annotations

import numpy as np
import torch


def _extract_layer_states(runs: list[dict]) -> list[list[torch.Tensor] | None]:
    """Extract layer hidden states from a list of run dicts (each has 'layer_hidden_states')."""
    return [r.get("layer_hidden_states") for r in runs]


def _layer_variance(states_per_run: list[torch.Tensor], eps: float = 1e-8) -> float:
    """Compute variance of last-token hidden states across runs at one layer.
    states_per_run: list of (hidden_dim,) tensors. Returns scalar variance."""
    valid = [s for s in states_per_run if s is not None]
    if len(valid) < 2:
        return 0.0
    stacked = torch.stack(valid)
    return float(stacked.var(dim=0).mean().item()) + eps


def compute_layer_metrics(
    clean_runs: list[dict],
    noisy_runs: list[dict],
    middle_layer_frac: tuple[float, float] = (1 / 3, 2 / 3),
) -> dict[str, float]:
    """
    Compute layer-wise geometric metrics from clean vs distorted hidden states.
    Higher values indicate more instability under perturbation (potential hallucination).

    Returns:
        LVD: Layer Variance Delta - mean over layers of (var_noisy - var_clean)
        LVS: Layer Variance Spike - max over layers of var_noisy / var_clean
        LVD_middle: LVD restricted to middle layers (default: 1/3 to 2/3)
    """
    clean_states = _extract_layer_states(clean_runs)
    noisy_states = _extract_layer_states(noisy_runs)

    # Filter to runs that have valid layer states
    clean_valid = [s for s in clean_states if s is not None and len(s) > 0]
    noisy_valid = [s for s in noisy_states if s is not None and len(s) > 0]

    if not clean_valid or not noisy_valid:
        return {"LVD": 0.0, "LVS": 0.0, "LVD_middle": 0.0}

    num_layers = min(len(clean_valid[0]), len(noisy_valid[0]))
    if num_layers == 0:
        return {"LVD": 0.0, "LVS": 0.0, "LVD_middle": 0.0}

    deltas = []
    ratios = []
    for L in range(num_layers):
        clean_layer = [r[L] for r in clean_valid if L < len(r)]
        noisy_layer = [r[L] for r in noisy_valid if L < len(r)]
        if len(clean_layer) < 2 or len(noisy_layer) < 1:
            continue
        var_clean = _layer_variance(clean_layer)
        var_noisy = _layer_variance(noisy_layer)
        deltas.append(var_noisy - var_clean)
        ratios.append(var_noisy / var_clean if var_clean > 0 else 0.0)

    if not deltas:
        return {"LVD": 0.0, "LVS": 0.0, "LVD_middle": 0.0}

    lvd = float(np.mean(deltas))
    lvs = float(np.max(ratios)) if ratios else 0.0

    # Middle layers
    lo = int(num_layers * middle_layer_frac[0])
    hi = int(num_layers * middle_layer_frac[1])
    lo, hi = max(0, lo), min(num_layers, hi)
    middle_deltas = deltas[lo:hi] if hi > lo else deltas
    lvd_middle = float(np.mean(middle_deltas)) if middle_deltas else lvd

    return {"LVD": lvd, "LVS": lvs, "LVD_middle": lvd_middle}


def apply_layer_dynamics_metrics(df):
    """Add layer dynamics metrics to a dataframe with original_high_temp and distorted_high_temp.
    Also adds expert routing metrics (ERD, EPV) when runs contain router data (e.g. DeepSeek-VL2 MoE)."""
    import pandas as pd

    def _row_metrics(row):
        return compute_layer_metrics(
            row["original_high_temp"],
            row["distorted_high_temp"],
        )

    df = df.copy()
    df["metrics_layer"] = df.apply(_row_metrics, axis=1)
    try:
        from .expert_routing import apply_expert_routing_metrics
        df = apply_expert_routing_metrics(df)
    except ImportError:
        pass
    return df


def compute_layer_roc_aucs(df) -> dict:
    """
    Compute ROC AUC for layer dynamics metrics.
    Higher LVD/LVS/LVD_middle = more hallucination. Invert labels for sklearn.
    Also includes expert routing metrics (ERD, EPV) when metrics_expert_routing is present.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    assert "hallucination_label" in df.columns
    aucs = {}
    for variant_name, group in df.groupby("variant_name"):
        aucs[variant_name] = {}
        y_true = 1 - group["hallucination_label"]
        metrics_df = pd.json_normalize(group["metrics_layer"])
        for k in ["LVD", "LVS", "LVD_middle"]:
            if k in metrics_df.columns:
                aucs[variant_name][k] = roc_auc_score(y_true, metrics_df[k])
        if "metrics_expert_routing" in group.columns:
            expert_df = pd.json_normalize(group["metrics_expert_routing"])
            for k in ["ERD", "EPV"]:
                if k in expert_df.columns and expert_df[k].abs().sum() > 0:
                    aucs[variant_name][k] = roc_auc_score(y_true, expert_df[k])
    return aucs
