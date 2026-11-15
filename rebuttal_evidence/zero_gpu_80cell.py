import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

np.random.seed(20261114)

f = "final_analysis_3200/alpha_performance.csv"
d = pd.read_csv(f)

print("=" * 78)
print("80-CELL ALPHA MOVEMENT VS ADAPTIVE-FIXED GAP")
print("=" * 78)

print("Input rows:", len(d))

# Aggregate 10 seeds -> dataset x model x horizon
cells = (
    d.groupby(["dataset", "model", "horizon"], as_index=False)
     .agg(
         n_seeds=("seed", "nunique"),
         mean_alpha=("alpha", "mean"),
         mean_alpha_shift=("alpha_shift", "mean"),
         mean_abs_alpha_shift=("absolute_alpha_shift", "mean"),
         adaptive_mse=("adaptive_mse", "mean"),
         fixed_mse=("fixed_mse", "mean"),
         adaptive_mae=("adaptive_mae", "mean"),
         fixed_mae=("fixed_mae", "mean"),
     )
)

# Positive = Adaptive better than Fixed.
cells["mse_gain"] = cells["fixed_mse"] - cells["adaptive_mse"]
cells["mse_gain_pct"] = (
    100 * cells["mse_gain"] / cells["fixed_mse"]
)

cells["mae_gain"] = cells["fixed_mae"] - cells["adaptive_mae"]
cells["mae_gain_pct"] = (
    100 * cells["mae_gain"] / cells["fixed_mae"]
)

print("Cells:", len(cells))
print("Seeds per cell:", sorted(cells.n_seeds.unique()))

assert len(cells) == 80, f"Expected 80 cells, got {len(cells)}"
assert cells.n_seeds.min() == 10, "Some cells have fewer than 10 seeds"


def bootstrap_corr(df, x, y, kind="pearson", B=20000):
    arr = df[[x, y]].dropna().to_numpy()
    n = len(arr)
    vals = []

    rng = np.random.default_rng(20261114)

    for _ in range(B):
        z = arr[rng.integers(0, n, n)]

        if np.std(z[:,0]) == 0 or np.std(z[:,1]) == 0:
            continue

        if kind == "pearson":
            r = pearsonr(z[:,0], z[:,1]).statistic
        else:
            r = spearmanr(z[:,0], z[:,1]).statistic

        if np.isfinite(r):
            vals.append(r)

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def report(label, z):
    print("\n" + "-" * 78)
    print(label)
    print("-" * 78)
    print("N CELLS:", len(z))

    for x in ["mean_abs_alpha_shift", "mean_alpha_shift"]:
        print(f"\nX = {x}")

        for y in ["mse_gain_pct", "mae_gain_pct"]:
            pr = pearsonr(z[x], z[y])
            sr = spearmanr(z[x], z[y])

            pci = bootstrap_corr(z, x, y, "pearson")
            sci = bootstrap_corr(z, x, y, "spearman")

            print(f"\nY = {y}")
            print(
                f"Pearson  r={pr.statistic:.6f}, "
                f"p={pr.pvalue:.6g}, "
                f"bootstrap95=[{pci[0]:.6f}, {pci[1]:.6f}]"
            )
            print(
                f"Spearman r={sr.statistic:.6f}, "
                f"p={sr.pvalue:.6g}, "
                f"bootstrap95=[{sci[0]:.6f}, {sci[1]:.6f}]"
            )


report("ALL 80 CELLS", cells)

without_exchange = cells[cells.dataset != "exchange_rate"].copy()
report("WITHOUT EXCHANGE RATE", without_exchange)


print("\n" + "=" * 78)
print("DATASET DESCRIPTIVES")
print("=" * 78)

ds = (
    cells.groupby("dataset")
         .agg(
             cells=("mse_gain_pct", "size"),
             mean_abs_alpha_shift=("mean_abs_alpha_shift", "mean"),
             mean_mse_gain_pct=("mse_gain_pct", "mean"),
             median_mse_gain_pct=("mse_gain_pct", "median"),
             mse_win_rate=("mse_gain_pct", lambda x: 100*(x > 0).mean()),
         )
)

print(ds.round(6).to_string())

cells.to_csv("zero_gpu_80cell_data.csv", index=False)
ds.to_csv("zero_gpu_80cell_dataset_summary.csv")

print("\nWROTE:")
print("zero_gpu_80cell_data.csv")
print("zero_gpu_80cell_dataset_summary.csv")
