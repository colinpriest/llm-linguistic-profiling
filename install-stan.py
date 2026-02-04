import os
import shutil
from cmdstanpy import install_cmdstan

# Adjust to your conda env root (this uses the currently active env)
CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if not CONDA_PREFIX:
    raise RuntimeError("CONDA_PREFIX not set. Activate your conda env before running.")

mingw_bin = os.path.join(CONDA_PREFIX, "Library", "mingw-w64", "bin")
library_bin = os.path.join(CONDA_PREFIX, "Library", "bin")

# Ensure both are on PATH (DLLs often live in Library\\bin)
os.environ["PATH"] = mingw_bin + ";" + library_bin + ";" + os.environ.get("PATH", "")

# Force cmdstanpy to use this exact make
make_exe = os.path.join(mingw_bin, "mingw32-make.exe")
os.environ["MAKE"] = make_exe

print("CONDA_PREFIX:", CONDA_PREFIX)
print("MAKE:", os.environ["MAKE"])
print("which mingw32-make (python):", shutil.which("mingw32-make"))

# Pin version for reproducibility
install_cmdstan(version="2.35.0")
