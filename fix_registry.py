from pathlib import Path
import json

path = Path("data/source_registry.jsonl")

records = [
    {
        "source_id": "fineweb2_arb_arab",
        "language": "ar",
        "role": "arabic_web",
        "license": "ODC-By-1.0",
        "terms": "Common Crawl ToU",
        "status": "approved",
        "provenance": "HuggingFace HuggingFaceFW/fineweb-2",
        "config": "arb_Arab",
        "notes": "Primary high-quality Arabic web dataset"
    },
    {
        "source_id": "arabic_wikipedia",
        "language": "ar",
        "role": "arabic_encyclopedic",
        "license": "CC-BY-SA-4.0",
        "terms": "Wikimedia Terms of Use",
        "status": "conditional_review",
        "provenance": "Wikimedia Foundation Dump",
        "notes": "Formal Modern Standard Arabic corpus; requires license governance review"
    },
    {
        "source_id": "fineweb_edu_english",
        "language": "en",
        "role": "english_educational",
        "license": "ODC-By-1.0",
        "terms": "Common Crawl ToU",
        "status": "approved",
        "provenance": "HuggingFace HuggingFaceFW/fineweb-edu",
        "notes": "High-quality educational English text"
    },
    {
        "source_id": "open_code_math",
        "language": "code_math",
        "role": "technical_code",
        "license": "MIT",
        "terms": "Open Source Licenses",
        "status": "pending_provenance",
        "provenance": "UNVERIFIED",
        "notes": "Do not ingest until exact dataset/repository provenance and license are recorded"
    }
]

with path.open("w", encoding="utf-8", newline="\n") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("SOURCE REGISTRY REWRITTEN")
print(f"records = {len(records)}")
print("Wikipedia = conditional_review")
print("open_code_math = pending_provenance")
