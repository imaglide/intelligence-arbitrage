"""
Shared CSV writer with overwrite protection and provenance tracking.

Every result CSV gets a .meta.json sidecar recording when, how, and with
what software versions the file was produced.  By default, writing to an
existing path raises FileExistsError — pass force=True to overwrite.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_run_metadata(script=None, **kwargs):
    """Build a metadata dict for the current run."""
    try:
        import dspy
        dspy_version = getattr(dspy, "__version__", "unknown")
    except ImportError:
        dspy_version = "not installed"

    meta = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "dspy_version": dspy_version,
        "python_version": sys.version.split()[0],
        "git_sha": _git_sha(),
        "script": script,
        "args": kwargs,
    }
    return meta


def safe_write_csv(df, output_path, metadata=None, force=False, index=False):
    """
    Write a DataFrame to CSV with overwrite protection.

    Raises FileExistsError if output_path already exists and force is False.
    Always writes a .meta.json sidecar alongside the CSV.
    Pass index=True for pivot tables where the index contains data.
    """
    if os.path.exists(output_path) and not force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=index)

    meta_path = output_path + ".meta.json"
    if metadata is None:
        metadata = get_run_metadata()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
