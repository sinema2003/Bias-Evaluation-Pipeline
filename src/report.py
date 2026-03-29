import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--gaps_csv", required=True)
    ap.add_argument("--out_dir", default="outputs/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    gaps = pd.read_csv(args.gaps_csv)

    # Plot 1: Toxicity mean by category & decoding (averaged over attributes)
    s1 = summary.groupby(["category", "decoding"]).agg(tox_mean=("tox_mean", "mean")).reset_index()
    for cat in s1["category"].unique():
        tmp = s1[s1["category"] == cat]
        plt.figure()
        plt.bar(tmp["decoding"], tmp["tox_mean"])
        plt.title(f"Toxicity by decoding - {cat}")
        plt.xlabel("decoding")
        plt.ylabel("tox_mean")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"tox_by_decoding_{cat}.png"))
        plt.close()

    # Plot 2: Bias gaps (tox_gap) per category & decoding
    if len(gaps):
        for cat in gaps["category"].unique():
            tmp = gaps[gaps["category"] == cat]
            plt.figure()
            plt.bar(tmp["decoding"], tmp["tox_gap"])
            plt.title(f"Toxicity gap by decoding - {cat}")
            plt.xlabel("decoding")
            plt.ylabel("tox_gap (max-min across attributes)")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"tox_gap_{cat}.png"))
            plt.close()

    print(f"Saved plots to: {args.out_dir}")

if __name__ == "__main__":
    main()