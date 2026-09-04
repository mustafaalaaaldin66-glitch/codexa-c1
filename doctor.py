import sys
import os
import subprocess
from pathlib import Path

def check_status(label, condition_fn):
    try:
        res, msg = condition_fn()
        if res:
            print(f"  [PASS] {label}{f' ({msg})' if msg else ''}")
            return True
        else:
            print(f"  [FAIL] {label}{f' -> {msg}' if msg else ''}")
            return False
    except Exception as e:
        print(f"  [FAIL] {label} -> {e}")
        return False

def main():
    print("\n==================================================")
    print("     CODEXA C1 V9 AUTOMATED DIAGNOSTIC SYSTEM     ")
    print("==================================================")

    root = Path(__file__).parent.resolve()
    results = []

    # 1. Environment & Python
    print("\n[1] Core Environment Check:")
    results.append(check_status("Python Version >= 3.10", lambda: (sys.version_info >= (3, 10), f"{sys.version_info.major}.{sys.version_info.minor}")))

    # 2. Key Libraries
    print("\n[2] Required Dependencies Check:")
    def check_imports():
        import numpy
        import tokenizers
        import datasets
        import pyarrow
        import torch
        return True, "All modules imported successfully"
    results.append(check_status("Core Modules (torch, numpy, tokenizers, datasets, pyarrow)", check_imports))

    # 3. Project Directory Structure
    print("\n[3] Project Structure Integrity:")
    def check_paths():
        required = ["config.py", "data", "tokenizer", "data_pipeline"]
        missing = [p for p in required if not (root / p).exists()]
        if missing:
            return False, f"Missing paths: {', '.join(missing)}"
        return True, "All critical folders/files exist"
    results.append(check_status("Project Directories & Config", check_paths))

    # 4. Pytest Suite
    print("\n[4] Test Suite Execution:")
    def run_pytest():
        res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, capture_output=True, text=True)
        if res.returncode == 0:
            return True, "All pytest checks passed"
        return False, res.stdout.strip().split('\n')[-1] if res.stdout else "Tests failed"
    results.append(check_status("Pytest Verification", run_pytest))

    # Summary Result
    print("\n==================================================")
    passed = sum(results)
    total = len(results)
    print(f"DIAGNOSTIC SUMMARY: {passed}/{total} Checks Passed.")
    if passed == total:
        print("STATUS: SAFE TO CONTINUE [GREEN LIGHT]")
    else:
        print("STATUS: BLOCKED - RESOLVE ISSUES FIRST [RED LIGHT]")
    print("==================================================\n")

if __name__ == "__main__":
    main()
