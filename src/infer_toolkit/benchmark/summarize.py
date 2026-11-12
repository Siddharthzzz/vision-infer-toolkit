"""Derive per-configuration summary tables from raw trial rows (methodology §7).

Never hand-typed: every number in a summary comes from results/<milestone>_*.csv.

python -m infer_toolkit.benchmark.summarize --milestone m2
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..runinfo import RESULTS

KEYS = ["model", "backend", "precision", "batch", "input_hw"]


def _flagged(df: pd.DataFrame) -> pd.Series:
    return df["flags"].fillna("").astype(str).str.len() > 0


def summarize_latency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS, sort=False):
        ok = g[g["status"] == "ok"]
        clean = ok[~_flagged(ok)]
        r = dict(zip(KEYS, key, strict=True))
        r.update(n_trials=len(ok), n_flagged=len(ok) - len(clean), status=g["status"].iloc[0])
        if len(clean):
            for c in ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]:
                r[c] = clean[c].median()
                r[f"{c}_min"], r[f"{c}_max"] = clean[c].min(), clean[c].max()
            r["cv_median"] = clean["cv"].median()
            r["sm_clock_med_mhz"] = clean["sm_clock_med_mhz"].median()
            r["temp_max_c"] = clean["temp_max_c"].max()
            r["build_s"] = clean["build_s"].iloc[0]
            r["trial_spread"] = (clean.p50_ms.max() - clean.p50_ms.min()) / clean.p50_ms.median()
        r["run_ids"] = ";".join(sorted(g["run_id"].unique()))
        r["git_commits"] = ";".join(sorted(g["git_commit"].astype(str).unique()))
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_throughput(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS, sort=False):
        ok = g[g["status"] == "ok"]
        clean = ok[~_flagged(ok)]
        r = dict(zip(KEYS, key, strict=True))
        r.update(n_trials=len(ok), n_flagged=len(ok) - len(clean), status=g["status"].iloc[0])
        if len(clean):
            r["images_per_s"] = clean["images_per_s"].median()
            r["images_per_s_min"] = clean["images_per_s"].min()
            r["images_per_s_max"] = clean["images_per_s"].max()
        r["run_ids"] = ";".join(sorted(g["run_id"].unique()))
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default="m2")
    a = ap.parse_args()
    for kind, fn in [("latency", summarize_latency), ("throughput", summarize_throughput)]:
        src = RESULTS / f"{a.milestone}_{kind}.csv"
        if src.exists():
            out = fn(pd.read_csv(src))
            out.to_csv(
                RESULTS / f"{a.milestone}_{kind}_summary.csv", index=False, float_format="%.5g"
            )
            print(f"wrote {a.milestone}_{kind}_summary.csv ({len(out)} configs)")


if __name__ == "__main__":
    main()
