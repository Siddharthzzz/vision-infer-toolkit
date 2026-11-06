"""Print the environment snapshot and save it to results/env.csv."""

from pathlib import Path

from infer_toolkit.env import collect, write_csv

if __name__ == "__main__":
    info = collect()
    width = max(map(len, info))
    for k, v in sorted(info.items()):
        print(f"{k:<{width}}  {v}")
    out = Path(__file__).resolve().parents[1] / "results" / "env.csv"
    write_csv(info, out)
    print(f"\nwrote {out}")
