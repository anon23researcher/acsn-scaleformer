import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

study1 = pd.read_csv("final_analysis_3200/results.csv")
tier1 = pd.read_csv("tier1_all_runs.csv")

keys = ["dataset", "model", "horizon", "seed"]

# Original Study 1 adaptive runs
a = study1[study1["mode"] == "adaptive"].copy()
a = a[keys + ["mse", "mae"]].rename(
    columns={
        "mse": "adaptive_mse",
        "mae": "adaptive_mae"
    }
)

# New matched alpha=0.5 frozen control
f = tier1[tier1["variant"] == "frozen05"].copy()
f = f[keys + ["mse", "mae"]].rename(
    columns={
        "mse": "frozen_mse",
        "mae": "frozen_mae"
    }
)

p = a.merge(f, on=keys, how="inner")

print("=" * 78)
print("STUDY 1 ADAPTIVE vs MATCHED FROZEN ALPHA=0.5")
print("=" * 78)
print("Matched pairs:", len(p))

if len(p) != 240:
    print("WARNING: expected 240 matched pairs")

# Positive = Adaptive better
p["mse_imp_pct"] = 100 * (
    p["frozen_mse"] - p["adaptive_mse"]
) / p["frozen_mse"]

p["mae_imp_pct"] = 100 * (
    p["frozen_mae"] - p["adaptive_mae"]
) / p["frozen_mae"]

# Log ratio: positive = Adaptive better
p["mse_log_gain"] = np.log(
    p["frozen_mse"] / p["adaptive_mse"]
)

p["mae_log_gain"] = np.log(
    p["frozen_mae"] / p["adaptive_mae"]
)

print("\nOVERALL")
print(f"MSE mean improvement   : {p.mse_imp_pct.mean():.4f}%")
print(f"MSE median improvement : {p.mse_imp_pct.median():.4f}%")
print(f"MSE win rate           : {100*(p.mse_imp_pct > 0).mean():.2f}%")
print(f"MAE mean improvement   : {p.mae_imp_pct.mean():.4f}%")
print(f"MAE median improvement : {p.mae_imp_pct.median():.4f}%")
print(f"MAE win rate           : {100*(p.mae_imp_pct > 0).mean():.2f}%")

print("\nMEAN RAW ERRORS")
print(f"Adaptive MSE : {p.adaptive_mse.mean():.8f}")
print(f"Frozen05 MSE : {p.frozen_mse.mean():.8f}")
print(f"Adaptive MAE : {p.adaptive_mae.mean():.8f}")
print(f"Frozen05 MAE : {p.frozen_mae.mean():.8f}")

print("\nPAIRED TESTS")
t = ttest_rel(p["adaptive_mse"], p["frozen_mse"])
print(f"MSE paired t-test: t={t.statistic:.6f}, p={t.pvalue:.8g}")

try:
    w = wilcoxon(p["adaptive_mse"], p["frozen_mse"])
    print(f"MSE Wilcoxon: W={w.statistic:.6f}, p={w.pvalue:.8g}")
except Exception as e:
    print("Wilcoxon unavailable:", e)

print("\nBY DATASET")
print(
    p.groupby("dataset")["mse_imp_pct"]
     .agg(
         n="count",
         mean="mean",
         median="median",
         win_rate=lambda x: 100*(x > 0).mean()
     )
     .round(4)
     .to_string()
)

print("\nBY MODEL")
print(
    p.groupby("model")["mse_imp_pct"]
     .agg(
         n="count",
         mean="mean",
         median="median",
         win_rate=lambda x: 100*(x > 0).mean()
     )
     .round(4)
     .to_string()
)

print("\nBY DATASET / MODEL")
print(
    p.groupby(["dataset","model"])["mse_imp_pct"]
     .agg(
         n="count",
         mean="mean",
         median="median",
         win_rate=lambda x: 100*(x > 0).mean()
     )
     .round(4)
     .to_string()
)

print("\nBY DATASET / HORIZON")
print(
    p.groupby(["dataset","horizon"])["mse_imp_pct"]
     .agg(
         n="count",
         mean="mean",
         median="median",
         win_rate=lambda x: 100*(x > 0).mean()
     )
     .round(4)
     .to_string()
)

# Bootstrap mean improvement CI
rng = np.random.default_rng(20261114)

for metric in ["mse_imp_pct", "mae_imp_pct"]:
    x = p[metric].to_numpy()
    means = np.empty(20000)

    for i in range(20000):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()

    lo, hi = np.percentile(means, [2.5, 97.5])

    print(
        f"\nBootstrap 95% CI mean {metric}: "
        f"[{lo:.4f}%, {hi:.4f}%]"
    )

p.to_csv("zero_gpu_adaptive_vs_frozen05_pairs.csv", index=False)

print("\nWROTE:")
print("zero_gpu_adaptive_vs_frozen05_pairs.csv")
