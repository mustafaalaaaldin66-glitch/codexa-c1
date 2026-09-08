"""Registry-gated, disk-conscious Hugging Face parquet ingestion.

This data-only tool intentionally has no torch dependency.  It downloads only
approved records from data/source_registry.jsonl and never adds raw data to Git.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if "torch" in sys.modules:
    raise RuntimeError("torch must not be imported by data_pipeline ingestion")

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "source_registry.jsonl"
RAW = ROOT / "data" / "raw"
FINEWEB2 = "HuggingFaceFW/fineweb-2"

# These are the only source/config identities accepted by this ingress point.
SOURCES = {
    "fineweb2_arb_arab": (FINEWEB2, "arb_Arab"),
    "fineweb2_arz_arab": (FINEWEB2, "arz_Arab"),
    "fineweb_edu_english": ("HuggingFaceFW/fineweb-edu", "fineweb_edu_english"),
}
MIN_FREE_BYTES = 2 * 1024**3


def approved_sources() -> dict[str, dict]:
    records: dict[str, dict] = {}
    with REGISTRY.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            source_id = record["source_id"]
            if source_id in records:
                raise ValueError(f"duplicate source_id at line {number}: {source_id}")
            records[source_id] = record
    return {
        source_id: record
        for source_id, record in records.items()
        if source_id in SOURCES and record.get("status") == "approved"
    }


def ensure_space(destination: Path, required_bytes: int = MIN_FREE_BYTES) -> None:
    free = shutil.disk_usage(destination).free
    if free < required_bytes:
        raise RuntimeError(f"insufficient disk at {destination}: {free} bytes free")


def download_arz() -> Path:
    """Download the complete, approximately 1GB arz_Arab configuration."""
    destination = RAW / "fineweb2_arz_arab"
    destination.mkdir(parents=True, exist_ok=True)
    ensure_space(destination)
    snapshot_download(
        repo_id=FINEWEB2,
        repo_type="dataset",
        allow_patterns=["data/arz_Arab/*.parquet", "data/arz_Arab/**/*.parquet"],
        local_dir=destination,
    )
    return destination


def download_arb_sample() -> Path:
    """Download exactly one local MSA parquet shard; never the 99GB corpus."""
    destination = RAW / "fineweb2_arb_arab_sample"
    destination.mkdir(parents=True, exist_ok=True)
    ensure_space(destination, 6 * 1024**3)
    cached = hf_hub_download(
        repo_id=FINEWEB2,
        repo_type="dataset",
        filename="data/arb_Arab/train/000_00000.parquet",
        local_dir=destination,
    )
    return Path(cached).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id", choices=sorted(SOURCES), nargs="?")
    parser.add_argument("--source-id", dest="source_id_option", choices=sorted(SOURCES))
    parser.add_argument("--sample-one", action="store_true")
    args = parser.parse_args()
    if args.source_id and args.source_id_option:
        parser.error("use either source_id or --source-id, not both")
    source_id = args.source_id_option or args.source_id
    allowed = approved_sources()
    requested = [source_id] if source_id else [
        "fineweb2_arz_arab", "fineweb2_arb_arab", "fineweb_edu_english"
    ]
    if args.sample_one and requested != ["fineweb2_arb_arab"]:
        parser.error("--sample-one is supported only for fineweb2_arb_arab")

    for source_id in requested:
        if source_id not in allowed:
            raise PermissionError(f"source is not approved in registry: {source_id}")
        if source_id == "fineweb2_arz_arab":
            print(download_arz())
        elif source_id == "fineweb2_arb_arab":
            print(download_arb_sample())
        else:
            # English is opt-in: do not discover or download unspecified shards.
            print(f"{source_id}: approved; no local shard requested")


if __name__ == "__main__":
    main()
