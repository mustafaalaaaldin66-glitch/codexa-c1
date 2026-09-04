from pathlib import Path
import json
import re

root = Path(".")
registry = root / "data" / "source_registry.jsonl"
pipeline = root / "data_pipeline" / "build_v9_corpus.py"

print("=" * 60)
print("CODEXA C1 V9 — STEP 9 SOURCE/Pipeline CONSISTENCY AUDIT")
print("=" * 60)

print("\n[1] REGISTERED SOURCES")
with registry.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

for r in records:
    print(f"{r['source_id']:25} | {r['status']:20} | {r['provenance']}")

print("\n[2] PIPELINE DATASET REFERENCES")
text = pipeline.read_text(encoding="utf-8")

for line in text.splitlines():
    if "load_dataset(" in line:
        print("LOAD_DATASET")
    if '"' in line and ("wikimedia" in line or "wikitext" in line or "flytech" in line):
        print(" ", line.strip())

print("\n[3] STATUS")
print("Registry records:", len(records))

for r in records:
    if r["status"] != "approved":
        print(f"REVIEW REQUIRED: {r['source_id']} -> {r['status']}")

print("\n[4] IMPORTANT")
print("This audit does NOT modify any project file.")
print("No tokenizer training is performed.")
print("No corpus is modified.")

print("\nAUDIT COMPLETE")
