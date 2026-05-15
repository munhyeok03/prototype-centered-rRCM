"""
Environment verification script for Margin-Aware rRCM project.
Run this FIRST before the preliminary experiment to confirm CUDA is ready.

Usage (from project root, with .venv active):
    .venv\Scripts\python.exe verify_cuda_env.py
"""
import sys, platform
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Environment Verification — Margin-Aware rRCM")
print("=" * 60)
print(f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python       : {sys.version}")
print(f"Platform     : {platform.platform()}")

try:
    import numpy as np
    print(f"numpy        : {np.__version__}  OK")
except ImportError:
    print("numpy        : NOT FOUND  (pip install numpy)")

try:
    import scipy
    print(f"scipy        : {scipy.__version__}  OK")
except ImportError:
    print("scipy        : NOT FOUND  (pip install scipy)")

try:
    import matplotlib
    print(f"matplotlib   : {matplotlib.__version__}  OK")
except ImportError:
    print("matplotlib   : NOT FOUND  (pip install matplotlib)")

try:
    import torch
    print(f"torch        : {torch.__version__}  OK")
    print(f"  cuda.available     : {torch.cuda.is_available()}")
    print(f"  torch.version.cuda : {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  GPU name           : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory         : {mem:.2f} GB")
        print(f"  CUDA OK -- ready for GPU experiment")
    else:
        print(f"  WARNING: CUDA not available in this PyTorch build.")
        print(f"  To install CUDA PyTorch (CUDA 11.8, works with driver 12.6):")
        print(f"    .venv\\Scripts\\python.exe -m pip install --force-reinstall ^")
        print(f"      torch torchvision torchaudio ^")
        print(f"      --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print("torch        : NOT FOUND")
    print("  To install: .venv\\Scripts\\python.exe -m pip install torch")
    sys.exit(1)

try:
    import torchvision
    print(f"torchvision  : {torchvision.__version__}  OK")
except ImportError:
    print("torchvision  : not installed (not required for run_preliminary_experiment.py)")

data_dir = Path("data/cifar10/cifar-10-batches-py")
if data_dir.exists() and list(data_dir.glob("data_batch_*")):
    print(f"CIFAR-10 data: found at {data_dir}  OK")
else:
    print(f"CIFAR-10 data: NOT FOUND at {data_dir}")
    print("  Download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    print("  Extract to: data/cifar10/")

print("\n" + "=" * 60)
import torch
if torch.cuda.is_available():
    print("RESULT: CUDA ready. Run with --device cuda")
    print("  .venv\\Scripts\\python.exe run_preliminary_experiment.py --device cuda")
else:
    print("RESULT: CPU only. Run with --device cpu")
    print("  .venv\\Scripts\\python.exe run_preliminary_experiment.py --device cpu")
print("=" * 60)
