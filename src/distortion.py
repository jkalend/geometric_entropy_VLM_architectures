"""
Visual distortion generation for HEDGE perturbation pipeline.
"""

import hashlib
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm


def distort_image(h: int, w: int) -> A.Compose:
    """Create albumentations pipeline for geometric + color + noise perturbations."""
    affine = A.Affine(
        rotate=random.choice([(-10, -2), (2, 10)]),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        fit_output=True,
        border_mode=cv2.BORDER_CONSTANT,
    )
    return A.Compose([
        affine,
        A.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.95, 1.05),
            hue=(-0.02, 0.02),
        ),
        A.GaussNoise(std_range=(0.07, 0.07), mean_range=(0.0, 0.0), p=1.0),
        A.ShotNoise(scale_range=(0.014, 0.014), p=1.0),
    ])


def generate_distortions(
    vqa_dict: list[dict],
    num_samples: int = 10,
    cache_dir: str | Path = ".cache_HEDGE/datasets",
    dataset_id: str = "default",
    force_regenerate: bool = False,
) -> list[dict]:
    """
    Generate and cache distorted images. Returns list of dicts with:
    idx, image_path, question, answer, description, distorted_image_paths
    """
    root = Path(cache_dir) / dataset_id.replace(" ", "_").replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)

    def _process(entry: dict) -> dict:
        img = entry["image"]
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
        h, w = arr.shape[:2]
        img_hash = hashlib.md5(arr.tobytes()).hexdigest()
        d = root / f"img_{img_hash}"
        d.mkdir(parents=True, exist_ok=True)
        orig_path = d / "original.png"
        if not orig_path.exists():
            Image.fromarray(arr).save(orig_path)
        distorted_paths = []
        for k in range(num_samples):
            out_path = d / f"distorted_{k}.png"
            if not out_path.exists() or force_regenerate:
                transform = distort_image(h, w)
                distorted = transform(image=arr)["image"]
                Image.fromarray(distorted).save(out_path)
            distorted_paths.append(str(out_path))
        return {
            "idx": entry["idx"],
            "image_path": str(orig_path),
            "question": entry["question"],
            "answer": entry["answer"],
            "description": entry.get("description"),
            "distorted_image_paths": distorted_paths[:num_samples],
        }

    results = []
    for entry in tqdm(vqa_dict, desc="Generating distortions"):
        results.append(_process(entry))
    return results
