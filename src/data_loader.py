"""
Dataset preparation for HEDGE: VQA-RAD, MedHallu, HaluEval-Wild.
Maps text-heavy queries to visual prompts using placeholder images where needed.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from PIL import Image


HEDGE_CACHE_PATH = ".cache_HEDGE"


def _hash_img(img: Image.Image | np.ndarray) -> str:
    arr = np.array(img) if isinstance(img, Image.Image) else img
    return hashlib.md5(arr.tobytes()).hexdigest()


def load_vqa_rad(split: str = "test", max_samples: int | None = None) -> list[dict]:
    """Load VQA-RAD (medical VQA with images)."""
    ds = load_dataset("flaviagiammarino/vqa-rad", split=split)
    out = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        out.append({
            "idx": i,
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
            "description": None,
        })
    return out


def load_medhallu(split: str = "pqa_labeled", max_samples: int | None = None) -> list[dict]:
    """Load MedHallu (medical QA). Uses gray placeholder image for VLM compatibility."""
    ds = load_dataset("UTAustin-AIHealth/MedHallu", split)
    placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
    out = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        question = sample.get("question", sample.get("query", ""))
        answer = sample.get("answer", sample.get("ref_answer", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        out.append({
            "idx": i,
            "image": placeholder,
            "question": question,
            "answer": str(answer),
            "description": sample.get("context", None),
        })
    return out


def load_halueval_wild(max_samples: int | None = None) -> list[dict]:
    """Load HaluEval-Wild (real-world queries). Uses gray placeholder for VLM."""
    ds = load_dataset("yushihu/halueval-wild-nocr", split="train")
    placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
    out = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        s = dict(sample) if not isinstance(sample, dict) else sample
        query = s.get("query", s.get("question", ""))
        ref = s.get("ref_answer", s.get("answer", ""))
        if isinstance(ref, (list, dict)):
            ref = ref[0] if isinstance(ref, list) and ref else str(ref)
        out.append({
            "idx": i,
            "image": placeholder,
            "question": str(query),
            "answer": str(ref),
            "description": None,
        })
    return out


def to_vqa_dict(samples: list[dict]) -> list[dict]:
    """Convert to HEDGE vqa_dict format: idx, image, question, answer, description."""
    return [
        {
            "idx": s["idx"],
            "image": s["image"],
            "question": s["question"],
            "answer": s["answer"],
            "description": s.get("description"),
        }
        for s in samples
    ]
