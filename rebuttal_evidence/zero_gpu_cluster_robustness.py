import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

d = pd.read_csv("zero_gpu_80cell_data.csv")
rng = np.random.default_rng(20261114)

def analyse(z, label, B=2000):
    print("\n" + "="*78)
    print(label)
    print("="*78)

    x = z["mean_abs_alpha_shift"].to_numpy()
    y = z["mse_gain_pct"].to_numpy()

    print("Cells:", len(z))
    print(f"Pearson:  r={pearsonr(x,y).statistic:.6f}, p={pearsonr(x,y).pvalue:.8g}")
    print(f"Spearman: r={spearmanr(x,y).statistic:.6f}, p={spearmanr(x,y).pvalue:.8g}")

    # Dataset-level leave-one-out
    print("\nLEAVE-ONE-DATASET-OUT")
    vals = []

    for ds in sorted(z.dataset.unique()):
        q = z[z.dataset != ds]
        r = pearsonr(
            q["mean_abs_alpha_shift"],
            q["mse_gain_pct"]
        ).statistic

        vals.append((ds, r))
        print(f"remove {ds:20s} r={r:.6f}")

    # Cluster bootstrap at dataset x model level
    zz = z.copy()
    zz["cluster"] = zz["dataset"].astype(str) + "|" + zz["model"].astype(str)
    clusters = zz["cluster"].unique()

    boot = []

    for _ in range(B):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)

        parts = []
        for j, c in enumerate(sampled):
            g = zz[zz.cluster == c].copy()
            # unique bootstrap cluster id so duplicated groups remain duplicated
            g["_boot_cluster"] = j
            parts.append(g)

        q = pd.concat(parts, ignore_index=True)

        if q["mean_abs_alpha_shift"].std() == 0 or q["mse_gain_pct"].std() == 0:
            continue

        r = pearsonr(
            q["mean_abs_alpha_shift"],
            q["mse_gain_pct"]
        ).statistic

        if np.isfinite(r):
            boot.append(r)

    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("\nDATASET x MODEL CLUSTER BOOTSTRAP")
    print("Clusters:", len(clusters))
    print("Valid bootstrap samples:", len(boot))
    print(f"95% CI: [{lo:.6f}, {hi:.6f}]")
    print(f"P(r <= 0): {np.mean(np.array(boot) <= 0):.6f}")

    return vals, lo, hi

out = []

vals, lo, hi = analyse(d, "ALL 80 CELLS")
for ds, r in vals:
    out.append(["all", ds, r])
out_ci = [["all", lo, hi]]

q = d[d.dataset != "exchange_rate"].copy()
vals, lo, hi = analyse(q, "WITHOUT EXCHANGE RATE")
for ds, r in vals:
    out.append(["without_exchange_rate", ds, r])
out_ci.append(["without_exchange_rate", lo, hi])

pd.DataFrame(
    out,
    columns=["analysis","dataset_removed","pearson_r"]
).to_csv(
    "zero_gpu_leave_one_dataset_out.csv",
    index=False
)

pd.DataFrame(
    out_ci,
    columns=["analysis","cluster_boot_ci_low","cluster_boot_ci_high"]
).to_csv(
    "zero_gpu_cluster_bootstrap_ci.csv",
    index=False
)

print("\nWROTE:")
print("zero_gpu_leave_one_dataset_out.csv")
print("zero_gpu_cluster_bootstrap_ci.csv")
