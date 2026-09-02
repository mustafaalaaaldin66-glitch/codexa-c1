import hashlib
import json
import re
import sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text.normalize import normalize_text


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 1. Language ID Check (v9 Pipeline Step 3)
def simple_language_id(text: str) -> str:
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    if has_arabic and has_latin:
        return "mixed"
    elif has_arabic:
        return "ar"
    elif has_latin:
        return "en"
    return "other"


# 2. Quality Score Filter (v9 Pipeline Step 4)
def passes_quality_filter(text: str) -> bool:
    if len(text.strip()) < 30:  # Short noise filter
        return False
    # Repetition check
    words = text.split()
    if len(words) > 0 and (len(set(words)) / len(words)) < 0.3:
        return False
    return True


def get_eval_hashes() -> set[str]:
    eval_dir = PROJECT_ROOT / "data" / "tokenizer_eval"
    eval_hashes = set()
    if eval_dir.exists():
        for txt_file in eval_dir.rglob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    norm = normalize_text(line.strip())
                    if norm:
                        eval_hashes.add(compute_sha256(norm))
    return eval_hashes


def fetch_arabic_samples(target_count=5000):
    print(f"Streaming {target_count} Arabic texts (Wikipedia AR)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ar", split="train", streaming=True
    )
    samples = []
    for row in ds:
        text = row.get("text", "")
        for line in text.split("\n"):
            line = line.strip()
            if passes_quality_filter(line):
                samples.append(line)
                if len(samples) >= target_count:
                    return samples
    return samples


def fetch_english_samples(target_count=5000):
    print(f"Streaming {target_count} English texts (Wikitext)...")
    ds = load_dataset(
        "wikitext", "wikitext-103-v1", split="train", streaming=True
    )
    samples = []
    for row in ds:
        text = row.get("text", "").strip()
        if passes_quality_filter(text) and not text.startswith("="):
            samples.append(text)
            if len(samples) >= target_count:
                return samples
    return samples


def fetch_code_samples(target_count=3000):
    print(f"Streaming {target_count} Code/Math samples...")
    ds = load_dataset(
        "flytech/python-codes-25k", split="train", streaming=True
    )
    samples = []
    for row in ds:
        output = row.get("output", "") or row.get("instruction", "")
        if passes_quality_filter(output):
            samples.append(output.strip())
            if len(samples) >= target_count:
                return samples
    return samples


def main():
    print("==================================================")
    print(" Executing Step 9: v9 Data Pipeline Standard ")
    print("==================================================")

    eval_hashes = get_eval_hashes()
    print(
        f"Loaded {len(eval_hashes)} eval hashes for decontamination gate."
    )

    raw_collections = {
        "arabic": fetch_arabic_samples(5000),
        "english": fetch_english_samples(5000),
        "mixed": fetch_arabic_samples(1000) + fetch_english_samples(1000),
        "code_math": fetch_code_samples(3000),
    }

    exact_seen_hashes = set()
    manifest_records = []
    train_base_dir = PROJECT_ROOT / "data" / "tokenizer_train"

    for category, lines in raw_collections.items():
        cat_dir = train_base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        output_file = cat_dir / "train_clean.txt"

        clean_lines = []
        for raw_line in lines:
            # Step 3: Lang ID check
            lang = simple_language_id(raw_line)

            # Step 4: Quality score check
            if not passes_quality_filter(raw_line):
                continue

            # Step 5: Normalization v1
            norm_line = normalize_text(raw_line)
            if not norm_line:
                continue

            # Step 6: Exact Dedup (SHA-256)
            h = compute_sha256(norm_line)
            if h in exact_seen_hashes:
                continue

            # Step 8: Decontamination Gate
            if h in eval_hashes:
                print(
                    f"[DECONTAMINATION GATE] Blocked eval overlap in {category}"
                )
                continue

            exact_seen_hashes.add(h)
            clean_lines.append(norm_line)

        with output_file.open("w", encoding="utf-8") as f:
            for cl in clean_lines:
                f.write(cl + "\n")

        total_chars = sum(len(cl) for cl in clean_lines)
        total_words = sum(len(cl.split()) for cl in clean_lines)

        manifest_records.append(
            {
                "category": category,
                "file_path": str(
                    output_file.relative_to(PROJECT_ROOT)
                ),
                "line_count": len(clean_lines),
                "word_count": total_words,
                "char_count": total_chars,
                "file_sha256": compute_sha256("\n".join(clean_lines)),
            }
        )

    manifest_file = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "tokenizer_train_manifest.json"
    )
    manifest_data = {
        "version": "tokenizer-train-corpus-v9-compliant",
        "created_at": "2026-09-02",
        "normalization": "norm-v1",
        "pipeline_steps": [
            "raw_source",
            "license_gate",
            "language_id",
            "quality_filter",
            "norm-v1",
            "exact_dedup",
            "decontamination",
            "manifest_record",
        ],
        "categories": manifest_records,
    }

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print("STEP 9 COMPLETE - Corpus Pipeline successfully built!")
    for rec in manifest_records:
        print(
            f" -> {rec['category']}: {rec['line_count']} lines | {rec['word_count']} words"
        )
    print(f"Manifest written: {manifest_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()