"""
HEDGE algorithms: semantic entropy, clustering, and hallucination metrics.
Adapted from hedge-bench (https://github.com/simula/HEDGE) for Windows/transformers pipeline.
"""

import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def sentence_semantic_entropy(mean_log_liks, semantic_ids, eps=1e-38):
    """Compute semantic entropy from log-likelihoods and cluster IDs."""
    logls = torch.as_tensor(mean_log_liks, dtype=torch.float32)
    semids = torch.as_tensor(semantic_ids, dtype=torch.long)
    weight = torch.exp(logls - logls.max())
    _, invix = torch.unique(semids, return_inverse=True, sorted=True)
    groups = torch.zeros(invix.max() + 1, dtype=logls.dtype, device=logls.device).scatter_add(
        0, invix, weight
    )
    log_group_probs = torch.log(groups.clamp_min(eps)) - torch.log(weight.sum().clamp_min(eps))
    group_probs = log_group_probs.exp()
    entropy = -(group_probs * log_group_probs).sum()
    return entropy, group_probs


def cluster_terms_by_embedding(seq, embedding_cached_fn, threshold=0.90):
    """Cluster sequences by embedding similarity using connected components."""
    if not seq:
        return []
    S = [s[:500] for s in seq]
    E = np.stack([np.array(embedding_cached_fn(s)).flatten() for s in S])
    if E.shape[0] == 1:
        return [0]
    nn = NearestNeighbors(n_neighbors=len(E), metric="cosine", algorithm="brute")
    nn.fit(E)
    dists, indices = nn.kneighbors(E)
    # Pairs within threshold (excluding self)
    r, c = np.where(
        (dists[:, 1:] <= (1 - threshold)) & (indices[:, 1:] != np.arange(len(E))[:, None])
    )
    if len(r) == 0:
        return list(range(len(E)))
    row = r
    col = indices[r, c + 1]
    G = coo_matrix(
        (np.ones(len(row)), (row, col)), shape=(len(E), len(E)), dtype=np.float32
    )
    G = G + G.T
    _, labels = connected_components(G, directed=False)
    return labels.tolist()


def normalize_nli_output(out) -> list[dict]:
    """Normalize NLI output to a consistent list of dicts with 'label' and 'score'."""
    if not out:
        return []
    # Handle single item vs nested list
    items = out[0] if isinstance(out[0], list) else out
    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append({
                "label": str(item.get("label", "")).upper(),
                "score": float(item.get("score", 0.0))
            })
    return normalized


def cluster_terms_by_nli(S, nli, batch_size=64, max_len=200):
    """Cluster by NLI entailment (slower, used optionally)."""
    if not S:
        return []
    S = [s[:max_len] for s in S]
    n = len(S)
    ids = [-1] * n
    next_id = 0
    for i in range(n):
        if ids[i] != -1:
            continue
        ids[i] = next_id
        for j in range(i + 1, n):
            if ids[j] != -1:
                continue
            batch = [[S[i], S[j]]]
            try:
                raw_out = nli(batch, batch_size=batch_size, truncation=True)
                normalized = normalize_nli_output(raw_out)
                entail = any(
                    o["label"] == "ENTAILMENT" and o["score"] > 0.5
                    for o in normalized
                )
                if entail:
                    ids[j] = ids[i]
            except Exception as e:
                import logging
                logging.error(f"NLI error in cluster_terms_by_nli: {e}")
        next_id += 1
    return ids


def get_nli_labels(all_sequences, nli_model, B=64):
    """Get NLI labels for clustering (placeholder for full NLI pipeline)."""
    results = []
    for seq in tqdm(all_sequences, desc="NLI"):
        if len(seq) <= 1:
            results.append([{"entailment": True}])
            continue
        labels = []
        for i in range(len(seq)):
            row = []
            for j in range(len(seq)):
                if i == j:
                    row.append(1)
                else:
                    try:
                        raw_out = nli_model([[seq[i], seq[j]]], batch_size=B, truncation=True)
                        normalized = normalize_nli_output(raw_out)
                        ent = any(
                            o["label"] == "ENTAILMENT" and o["score"] > 0.5
                            for o in normalized
                        )
                        row.append(1 if ent else 0)
                    except Exception as e:
                        import logging
                        logging.error(f"NLI error in get_nli_labels: {e}")
                        row.append(0)
            labels.append(row)
        results.append(labels)
    return results


def cluster_from_nli_labels(nli_labels):
    """Convert NLI label matrix to cluster IDs via connected components."""
    clusters = []
    for labels in nli_labels:
        if not labels or (isinstance(labels[0], dict) and "entailment" in labels[0]):
            clusters.append(list(range(len(labels))) if labels else [])
            continue
        n = len(labels)
        if n <= 1:
            clusters.append(list(range(n)))
            continue
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if labels[i][j] == 1:
                    G[i, j] = 1
        G = np.maximum(G, G.T)
        _, comp = connected_components(G, directed=False)
        clusters.append(comp.tolist())
    return clusters


def radflag(semantic_ids, n):
    """RadFlag: fraction of clean samples in cluster 0."""
    semantic_subset = semantic_ids[1 : 1 + n]
    return semantic_subset.count(0) / n if n > 0 else 0.0


def vase(n, semantic_ids, SeDist, SeDist_noisy, alpha=1.0):
    """Vision-Amplified Semantic Entropy (VASE)."""
    all_ids = torch.as_tensor(semantic_ids, dtype=torch.long)
    SeDist = torch.as_tensor(SeDist, dtype=torch.float32)
    SeDist_noisy = torch.as_tensor(SeDist_noisy, dtype=torch.float32)
    max_id = int(max(semantic_ids)) + 1
    dist_vec = torch.zeros(max_id)
    u_clean = all_ids[1 : n + 1].unique()
    u_noisy = all_ids[n + 1 : 2 * n + 1].unique()
    if len(u_clean) != len(SeDist) or len(u_noisy) != len(SeDist_noisy):
        return 0.0
    dist_clean = dist_vec.clone().scatter(0, u_clean, SeDist)
    dist_noisy = dist_vec.clone().scatter(0, u_noisy, SeDist_noisy)
    ViSeDist = torch.softmax(dist_clean + alpha * (dist_clean - dist_noisy), 0)
    return -(ViSeDist * (ViSeDist + 1e-10).log()).sum().item()
