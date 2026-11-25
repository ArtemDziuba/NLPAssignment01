"""Capture the native build provenance inside the Linux image."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import kenlm

root = Path("/opt/kenlm")
metadata = {
    "kenlm_release": "0.3.0",
    "source_sha256": hashlib.sha256((root / "source.tar.gz").read_bytes()).hexdigest(),
    "cmake_options": ["CMAKE_BUILD_TYPE=Release", "KENLM_MAX_ORDER=3"],
    "compiler": subprocess.check_output(["c++", "--version"], text=True),
    "cmake": subprocess.check_output(["cmake", "--version"], text=True),
    "system_packages": subprocess.check_output(["dpkg-query", "-W"], text=True),
    "python": platform.python_version(),
    "kenlm_binding": kenlm.__file__,
}
(root / "toolchain.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
