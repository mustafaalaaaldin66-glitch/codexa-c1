from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

print("=" * 60)
print("CODEXA C1 V9 — PRODUCTION DATA AUDIT")
print("=" * 60)

checks = [
    ("source_registry", ROOT / "data" / "source_registry.jsonl"),
    ("license_policy", ROOT / "data" / "licenses" / "README.md"),
    ("eval_manifest", ROOT / "data" / "tokenizer_eval" / "manifest.json"),
    ("local_manifest", ROOT / "data" / "manifests" / "tokenizer_train_local_manifest.json"),
]

for name, path in checks:
    print(f"\n[{name}]")
    print("exists :", path.exists())
    if path.exists():
        print("size   :", path.stat().st_size)
        print("path   :", path)

registry = ROOT / "data" / "source_registry.jsonl"

if registry.exists():
    print("\n" + "=" * 60)
    print("SOURCE REGISTRY")
    print("=" * 60)

    lines = [
        x.strip()
        for x in registry.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    print("records:", len(lines))

    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            print(f"\n--- RECORD {i} ---")
            print(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"INVALID JSONL RECORD {i}: {e}")

print("\n" + "=" * 60)
print("LICENSE POLICY")
print("=" * 60)

license_file = ROOT / "data" / "licenses" / "README.md"

if license_file.exists():
    print(license_file.read_text(encoding="utf-8"))

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
