"""
Expert Routing Dynamics for MoE-based hallucination detection.

Measures whether hallucinated generations trigger different expert paths compared to
factually grounded ones in Mixture-of-Experts models.

Metrics:
- ERD (Expert Routing Delta): Mean over layers of routing distribution divergence (clean vs noisy)
- EPV (Expert Path Variance): Variance in selected experts across runs under perturbation

Extraction of router_logits requires model support (output_router_logits=True in forward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _extract_router_states(runs: list[dict]) -> list[list[torch.Tensor] | None]:
    """Extract router logits or expert indices from runs (each has 'router_logits' or 'expert_indices')."""
    return [r.get("router_logits") or r.get("expert_indices") for r in runs]


def _router_divergence(clean_probs: torch.Tensor, noisy_probs: torch.Tensor, eps: float = 1e-8) -> float:
    """KL divergence or L1 distance between clean and noisy routing distributions."""
    clean = clean_probs.clamp(min=eps)
    noisy = noisy_probs.clamp(min=eps)
    return float((noisy * (noisy.log() - clean.log())).sum(dim=-1).mean().item())


def compute_expert_routing_metrics(
    clean_runs: list[dict],
    noisy_runs: list[dict],
) -> dict[str, float]:
    """
    Compute expert routing metrics from clean vs distorted runs.

    Expects each run to have 'router_logits' (per-layer) or 'expert_indices'.
    Returns zeros if router data is not available.

    Returns:
        ERD: Expert Routing Delta - mean routing distribution divergence (clean vs noisy)
        EPV: Expert Path Variance - variance in expert selection under perturbation
    """
    clean_router = _extract_router_states(clean_runs)
    noisy_router = _extract_router_states(noisy_runs)
    clean_valid = [r for r in clean_router if r is not None and len(r) > 0]
    noisy_valid = [r for r in noisy_router if r is not None and len(r) > 0]
    if not clean_valid or not noisy_valid:
        return {"ERD": 0.0, "EPV": 0.0}

    num_layers = min(len(clean_valid[0]), len(noisy_valid[0]))
    if num_layers == 0:
        return {"ERD": 0.0, "EPV": 0.0}

    divergences = []
    variances = []
    for L in range(num_layers):
        clean_layer = [r[L] for r in clean_valid if L < len(r)]
        noisy_layer = [r[L] for r in noisy_valid if L < len(r)]
        if len(clean_layer) < 1 or len(noisy_layer) < 1:
            continue
        
        # Convert all runs in layer to probabilities
        c_probs_list = [x.float().softmax(dim=-1) if (x.dim() >= 2 and x.shape[-1] > 1) else x.float() for x in clean_layer]
        n_probs_list = [x.float().softmax(dim=-1) if (x.dim() >= 2 and x.shape[-1] > 1) else x.float() for x in noisy_layer]
        
        # Compute mean divergence across all paired runs (clean[0] vs each noisy[i])
        # or aggregate across all pairs if multiple clean runs exist.
        # Here we follow the suggestion: aggregate across all runs.
        layer_divs = []
        for cp in c_probs_list:
            for np_ in n_probs_list:
                if cp.dim() >= 2 and cp.shape[-1] > 1:
                    # Validate shapes before computing divergence
                    if cp.shape != np_.shape:
                        # Align/truncate to common prefix length for sequence-first dims
                        if cp.dim() >= 2 and np_.dim() >= 2:
                            # Handle sequence dimension: use min sequence length
                            seq_dim = -2 if cp.dim() >= 2 else None
                            if seq_dim is not None and cp.shape[seq_dim] != np_.shape[seq_dim]:
                                min_seq_len = min(cp.shape[seq_dim], np_.shape[seq_dim])
                                cp = cp[..., :min_seq_len, :] if cp.dim() == 2 else cp[..., :min_seq_len, :]
                                np_ = np_[..., :min_seq_len, :] if np_.dim() == 2 else np_[..., :min_seq_len, :]
                            # If still mismatched after sequence alignment, skip
                            if cp.shape != np_.shape:
                                import warnings
                                warnings.warn(f"Skipping KL computation: shape mismatch {cp.shape} vs {np_.shape}")
                                continue
                        else:
                            import warnings
                            warnings.warn(f"Skipping KL computation: shape mismatch {cp.shape} vs {np_.shape}")
                            continue
                    layer_divs.append(_router_divergence(cp, np_))
        
        if layer_divs:
            divergences.append(float(np.mean(layer_divs)))

        # Variance of expert selection across noisy runs
        if len(noisy_layer) >= 2:
            stacked = torch.stack([x.float() for x in noisy_layer])
            variances.append(float(stacked.var(dim=0).mean().item()))

    return {
        "ERD": float(np.mean(divergences)) if divergences else 0.0,
        "EPV": float(np.mean(variances)) if variances else 0.0,
    }


def apply_expert_routing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add expert routing metrics to rows that have router data in original_high_temp/distorted_high_temp."""
    def _row_metrics(row):
        return compute_expert_routing_metrics(
            row["original_high_temp"],
            row["distorted_high_temp"],
        )

    df = df.copy()
    df["metrics_expert_routing"] = df.apply(_row_metrics, axis=1)
    return df
