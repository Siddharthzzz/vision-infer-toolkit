"""Provenance for every results row (methodology §8): run_id, git_commit, env_id."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import subprocess
import uuid
from pathlib import Path

from .env import collect, write_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# Fields that change from minute to minute; recorded in the snapshot but excluded from the
# fingerprint. The enforced power limit is dynamic on this laptop (it differed by several
# watts between snapshots taken a minute apart on 2026-11-12; quantified in the M2 pilot),
# so it cannot identify a setup.
VOLATILE = (
    "timestamp_utc",
    "git.commit",
    "gpu.temperature.gpu",
    "gpu.pstate",
    "gpu.enforced.power.limit",
)


def git_commit() -> str:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout

    sha = run("rev-parse", "--short", "HEAD").strip() or "none"
    dirty = run("status", "--porcelain", "--untracked-files=no").strip()
    return sha + ("-dirty" if dirty else "")


def env_id() -> str:
    """Fingerprint of software + hardware + power config. Snapshot saved once per id."""
    info = collect()
    stable = {k: v for k, v in info.items() if k not in VOLATILE}
    h = hashlib.sha256(repr(sorted(stable.items())).encode()).hexdigest()[:12]
    path = RESULTS / f"env_{h}.csv"
    if not path.exists():
        write_csv(info, path)
    return h


def new_run() -> dict[str, str]:
    return {
        "run_id": uuid.uuid4().hex[:12],
        "git_commit": git_commit(),
        "env_id": env_id(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows; header written on first use. Column set must stay identical."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    new = not path.exists()
    if not new:
        with path.open() as f:
            existing = next(csv.reader(f))
        if existing != fields:
            raise ValueError(f"{path}: column mismatch\n file: {existing}\n rows: {fields}")
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
