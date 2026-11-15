"""Environment Provenance and Reproducibility Metadata Collector.

Captures system platform, Git commit, package versions, random seeds,
and training hyperparameters for report audit trails.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.reporting.models import ProvenanceInfo, RunRecord


def get_git_commit(cwd: Path | str | None = None) -> str:
    """Safely obtain the current Git commit hash, or 'Not available'."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "Not available"


def get_package_version(pkg_name: str) -> str:
    """Safely retrieve installed version of a package."""
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        try:
            mod = __import__(pkg_name)
            return getattr(mod, "__version__", "Not available")
        except Exception:
            return "Not available"


def collect_provenance(
    source_dirs: Sequence[Path | str] | None = None,
    records: Sequence[RunRecord] | None = None,
    data_sources: dict[str, str] | None = None,
) -> ProvenanceInfo:
    """Collect comprehensive provenance and metadata for reproducibility."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    git_hash = get_git_commit()
    py_ver = f"{platform.python_version()} ({platform.python_implementation()})"
    plat_str = platform.platform()

    np_ver = get_package_version("numpy")
    gym_ver = get_package_version("gymnasium")
    sb3_ver = get_package_version("sb3_contrib")
    if sb3_ver == "Not available":
        sb3_ver = get_package_version("stable_baselines3")
    torch_ver = get_package_version("torch")
    mpl_ver = get_package_version("matplotlib")
    jinja_ver = get_package_version("jinja2")

    # Extract seeds from records if present
    seeds_found: list[int] = []
    timesteps_val = 10000
    reward_scale_val = 100.0

    if records:
        for r in records:
            if r.seed not in seeds_found:
                seeds_found.append(r.seed)
            if r.train_timesteps and r.train_timesteps > 0:
                timesteps_val = r.train_timesteps

    seeds_found.sort()
    if not seeds_found:
        seeds_found = [42, 43, 44]

    src_str_list = [str(Path(d).resolve()) for d in (source_dirs or [])]

    return ProvenanceInfo(
        timestamp=ts,
        git_commit=git_hash,
        python_version=py_ver,
        platform=plat_str,
        numpy_version=np_ver,
        gymnasium_version=gym_ver,
        sb3_version=sb3_ver,
        torch_version=torch_ver,
        matplotlib_version=mpl_ver,
        jinja2_version=jinja_ver,
        seeds=seeds_found,
        timesteps=timesteps_val,
        reward_scale=reward_scale_val,
        alpha=1.0,
        beta=1.0,
        source_dirs=src_str_list,
        data_sources=data_sources or {},
    )
