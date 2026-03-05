"""
Hallucination label judges: simple heuristic, Ollama-based LLM judge.
Ollama runs out-of-process (zero memory in this script).
"""

import json
import re
import urllib.request
import urllib.error
from typing import Literal

import pandas as pd
from tqdm import tqdm

LabelMethod = Literal["simple", "ollama"]

JUDGE_PROMPT = """You are a strict evaluator for medical/visual question answering.

Question: {question}
Ground truth answer: {true_answer}
Model's answer: {model_answer}

Is the model's answer correct? Consider it correct if it matches, is consistent with, or appropriately elaborates on the ground truth. Consider it wrong (hallucination) if it contradicts the ground truth, invents facts, or is irrelevant.

Respond with ONLY a JSON object: {{"correct": 1}} or {{"correct": 0}}
No other text."""


def _score_simple(row: pd.Series) -> int:
    """String-match heuristic: 1 if answer matches, else 0."""
    pred = str(row["original_low_temp"]["ans"]).strip().lower()
    true = str(row["true_answer"]).strip().lower()
    if not true:
        return 1
    return 1 if true in pred or pred in true else 0


def _call_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434", timeout: int = 120) -> str:
    """Call Ollama chat API. Returns raw response content."""
    url = f"{base_url}/api/chat"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data.get("message", {}).get("content", "").strip()


def _parse_judge_response(text: str) -> int:
    """Parse judge response to 1 (correct) or 0 (hallucination). Default 0 on parse failure."""
    text = text.strip()
    # Try JSON parse
    try:
        # Extract JSON object if wrapped in markdown
        match = re.search(r"\{[^{}]*\"correct\"[^{}]*\}", text)
        if match:
            obj = json.loads(match.group())
            val = obj.get("correct", obj.get("correctness", -1))
            if val in (1, True, "1", "yes"):
                return 1
            if val in (0, False, "0", "no"):
                return 0
        obj = json.loads(text)
        val = obj.get("correct", obj.get("correctness", -1))
        if val in (1, True, "1", "yes"):
            return 1
        if val in (0, False, "0", "no"):
            return 0
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: look for explicit yes/no
    lower = text.lower()
    if '"correct":1' in lower or '"correct": 1' in lower:
        return 1
    if '"correct":0' in lower or '"correct": 0' in lower:
        return 0
    if "yes" in lower[:50] or "correct" in lower[:50]:
        return 1
    if "no" in lower[:50] or "incorrect" in lower[:50] or "hallucin" in lower[:50]:
        return 0
    return 0


def _score_ollama(row: pd.Series, model: str, base_url: str, timeout: int) -> int:
    """Use Ollama to judge correctness. Returns 1 (correct) or 0 (hallucination)."""
    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        true_answer=str(row["true_answer"]),
        model_answer=str(row["original_low_temp"]["ans"]),
    )
    try:
        resp = _call_ollama(prompt, model, base_url, timeout)
        return _parse_judge_response(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        # Fallback to simple on network/API errors
        return _score_simple(row)


def add_hallucination_labels(
    df: pd.DataFrame,
    method: LabelMethod = "simple",
    ollama_model: str = "gpt-oss:20b",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: int = 120,
) -> pd.DataFrame:
    """
    Add hallucination labels: 1 = correct, 0 = hallucination.

    Methods:
      - simple: String-match heuristic (true in pred or pred in true).
      - ollama: LLM judge via local Ollama API. Zero memory in this process.
                Use glm-4.7-flash for speed, gpt-oss:20b or qwen3.5:27b for accuracy.
    """
    df = df.copy()
    if method == "simple":
        df["hallucination_label"] = df.apply(_score_simple, axis=1)
        return df

    if method == "ollama":
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Ollama judge"):
            labels.append(_score_ollama(row, ollama_model, ollama_base_url, ollama_timeout))
        df["hallucination_label"] = labels
        return df

    raise ValueError(f"Unknown label method: {method}")


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
