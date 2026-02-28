"""Download and prepare datasets for HEDGE evaluation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_vqa_rad, load_medhallu, load_halueval_wild

def main():
    datasets = {}
    
    print("Loading VQA-RAD...")
    try:
        vqa = load_vqa_rad(split="test", max_samples=50)
        print(f"  Loaded {len(vqa)} samples")
        datasets["vqa_rad"] = vqa
    except Exception as e:
        print(f"  Failed: {e}")

    print("Loading MedHallu...")
    try:
        med = load_medhallu(split="pqa_labeled", max_samples=50)
        print(f"  Loaded {len(med)} samples")
        datasets["medhallu"] = med
    except Exception as e:
        print(f"  Failed: {e}")

    print("Loading HaluEval-Wild...")
    try:
        halu = load_halueval_wild(max_samples=50)
        print(f"  Loaded {len(halu)} samples")
        datasets["halueval_wild"] = halu
    except Exception as e:
        print(f"  Failed: {e}")

    # Persist datasets to disk
    output_dir = Path("data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in datasets.items():
        # Remove PIL images from serialization if present
        serializable_data = []
        for sample in data:
            s = sample.copy()
            if "image" in s:
                del s["image"]
            serializable_data.append(s)
            
        output_path = output_dir / f"{name}.json"
        output_path.write_text(json.dumps(serializable_data, indent=2))
        print(f"  Persisted {name} to {output_path}")

    print("Data preparation complete.")
    return datasets

if __name__ == "__main__":
    main()
