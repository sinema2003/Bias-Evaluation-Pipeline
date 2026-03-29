import os
import json
import argparse
import pandas as pd

from src.io_utils import read_jsonl, write_jsonl
from src.metrics import Evaluators, add_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to outputs/runs/<run_id>.jsonl")
    ap.add_argument("--out_dir", default="outputs/summaries")
    ap.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_jsonl(args.run)
    evals = Evaluators(device=args.device)
    rows = add_metrics(rows, evals)

    df = pd.DataFrame(rows)

    # Aggregate stats
    grp = df.groupby(["model", "decoding", "category", "attribute"])
    summary = grp.agg(
        n=("toxicity", "count"),
        tox_mean=("toxicity", "mean"),
        tox_std=("toxicity", "std"),
        sent_mean=("sentiment", "mean"),
        sent_std=("sentiment", "std"),
    ).reset_index()

    # Bias gap: for each (model, decoding, category), compare max-min across attributes
    gaps = []
    for (m, d, c), g in summary.groupby(["model", "decoding", "category"]):
        if g["attribute"].nunique() < 2:
            continue
        tox_gap = float(g["tox_mean"].max() - g["tox_mean"].min())
        sent_gap = float(g["sent_mean"].max() - g["sent_mean"].min())
        gaps.append({"model": m, "decoding": d, "category": c, "tox_gap": tox_gap, "sent_gap": sent_gap})

    gaps_df = pd.DataFrame(gaps)

    run_id = df["run_id"].iloc[0]
    sum_csv = os.path.join(args.out_dir, f"{run_id}.summary.csv")
    gap_csv = os.path.join(args.out_dir, f"{run_id}.gaps.csv")
    enriched_jsonl = os.path.join(args.out_dir, f"{run_id}.enriched.jsonl")

    summary.to_csv(sum_csv, index=False)
    gaps_df.to_csv(gap_csv, index=False)
    write_jsonl(enriched_jsonl, rows)

    meta = {
        "run_id": run_id,
        "summary_csv": sum_csv,
        "gaps_csv": gap_csv,
        "enriched_jsonl": enriched_jsonl,
    }
    with open(os.path.join(args.out_dir, f"{run_id}.summary.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print(" -", sum_csv)
    print(" -", gap_csv)
    print(" -", enriched_jsonl)

if __name__ == "__main__":
    main()