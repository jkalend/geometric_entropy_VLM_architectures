"""Download and prepare datasets for HEDGE evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_vqa_rad, load_medhallu, load_halueval_wild

def main():
    print("Loading VQA-RAD...")
    vqa = load_vqa_rad(split="test", max_samples=50)
    print(f"  Loaded {len(vqa)} samples")

    print("Loading MedHallu...")
    try:
        med = load_medhallu(split="pqa_labeled", max_samples=50)
        print(f"  Loaded {len(med)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    print("Loading HaluEval-Wild...")
    try:
        halu = load_halueval_wild(max_samples=50)
        print(f"  Loaded {len(halu)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
