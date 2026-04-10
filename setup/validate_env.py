"""
Run this to confirm every dependency is correctly installed.
Usage: python setup/validate_env.py
All items should print PASS before proceeding to phase1_baseline/benchmark.py
"""

import importlib, sys

LIBS = [
    ("torch",           "2.0.0"),
    ("transformers",    "4.40.0"),
    ("tokenizers",      None),
    ("huggingface_hub", None),
    ("datasets",        None),
    ("bitsandbytes",    "0.43.0"),
    ("peft",            "0.10.0"),
    ("trl",             "0.8.0"),
    ("accelerate",      None),
    ("gradio",          None),
]

def check(name, min_ver):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        if min_ver:
            from packaging.version import Version
            ok = Version(ver) >= Version(min_ver)
        else:
            ok = True
        return ok, ver
    except ImportError:
        return False, "NOT INSTALLED"

print("\n" + "="*55)
print("  QuantLlama — Environment Validation")
print("="*55)

all_ok = True
for name, min_ver in LIBS:
    ok, ver = check(name, min_ver)
    tick = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    print(f"  [{tick}] {name:<20} {ver:<15} [{status}]")
    all_ok = all_ok and ok

print()
import torch
cuda_ok = torch.cuda.is_available()
tick = "✓" if cuda_ok else "✗"
gpu = torch.cuda.get_device_name(0) if cuda_ok else "NOT FOUND"
vram = round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if cuda_ok else 0
print(f"  [{tick}] {'CUDA GPU':<20} {gpu} ({vram} GB) [{'PASS' if cuda_ok else 'FAIL'}]")
all_ok = all_ok and cuda_ok

print()
if all_ok:
    print("  ✓ All checks passed — ready for phase1_baseline/benchmark.py")
else:
    print("  ✗ Fix failures above before proceeding.")
print("="*55 + "\n")
