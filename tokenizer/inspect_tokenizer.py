"""Read-only inspector for Codexa tokenizer artifacts.

Compares the committed vs. working-copy bpe32k artifact and reports the
basic properties of any tokenizer.json: vocab size, special tokens and
round-trip on canonical samples.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\inspect_tokenizer.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer

try:  # Make Arabic sample output printable on the Windows console.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - best effort
    pass

SAMPLES = (
    "أنا أحب البرمجة",
    "إزاي الحال النهاردة يا صاحبي",
    "I love programming",
    "أنا أحب programming و PyTorch",
)


def describe(label: str, path: Path) -> None:
    print(f"\n=== {label}")
    print(f"    path: {path}")
    if not path.exists():
        print("    MISSING")
        return
    print(f"    bytes: {path.stat().st_size:,}")
    tok = Tokenizer.from_file(str(path))
    print(f"    vocab size: {tok.get_vocab_size()}")
    specials = [t.content for t in tok.get_added_tokens_decoder().values()]
    print(f"    added tokens ({len(specials)}): {specials}")
    unk = tok.token_to_id("<unk>")
    print(f"    unk id: {unk}")
    for sample in SAMPLES:
        enc = tok.encode(sample)
        dec = tok.decode(enc.ids)
        status = "OK" if dec == sample else f"DIFF {dec!r}"
        n_unk = sum(1 for i in enc.ids if i == unk) if unk is not None else 0
        print(f"    {sample!r} -> {len(enc.ids)} ids, unk={n_unk}, rt={status}")


def describe_committed(rel: str) -> None:
    import json

    print(f"\n=== committed HEAD:{rel}")
    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"    git error: {exc}")
        return
    print(f"    bytes: {len(raw):,}")
    data = json.loads(raw.decode("utf-8"))
    model = data.get("model", {})
    print(f"    model vocab entries: {len(model.get('vocab', {}))}")
    merges = data.get("model", {}).get("merges", [])
    print(f"    merges: {len(merges)}")
    print(f"    added tokens: {len(data.get('added_tokens', []))}")


if __name__ == "__main__":
    artifacts = PROJECT_ROOT / "tokenizer" / "artifacts"
    describe("working bpe32k", artifacts / "bpe32k" / "tokenizer.json")
    describe("working bpe32k_v10", artifacts / "bpe32k_v10" / "tokenizer.json")
    describe_committed("tokenizer/artifacts/bpe32k/tokenizer.json")
