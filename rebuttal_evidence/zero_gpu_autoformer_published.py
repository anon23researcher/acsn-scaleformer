import pandas as pd

ours = pd.read_csv("zero_gpu_fixed_absolute.csv")

published = {
    ("electricity", 96):  (0.201, 0.317),
    ("electricity", 192): (0.222, 0.334),
    ("electricity", 336): (0.231, 0.338),
    ("electricity", 720): (0.254, 0.361),

    ("exchange_rate", 96):  (0.197, 0.323),
    ("exchange_rate", 192): (0.300, 0.369),
    ("exchange_rate", 336): (0.509, 0.524),
    ("exchange_rate", 720): (1.447, 0.941),

    ("traffic", 96):  (0.613, 0.388),
    ("traffic", 192): (0.616, 0.382),
    ("traffic", 336): (0.622, 0.337),
    ("traffic", 720): (0.660, 0.408),

    ("weather", 96):  (0.266, 0.336),
    ("weather", 192): (0.307, 0.367),
    ("weather", 336): (0.359, 0.395),
    ("weather", 720): (0.419, 0.428),
}

x = ours[
    (ours["model"] == "AutoformerMS") &
    (ours["dataset"].isin(["electricity", "exchange_rate", "traffic", "weather"]))
].copy()

x["published_mse"] = [
    published[(d, h)][0]
    for d, h in zip(x.dataset, x.horizon)
]

x["published_mae"] = [
    published[(d, h)][1]
    for d, h in zip(x.dataset, x.horizon)
]

x["mse_diff_pct"] = (
    100 * (x["mse_mean"] - x["published_mse"])
    / x["published_mse"]
)

x["mae_diff_pct"] = (
    100 * (x["mae_mean"] - x["published_mae"])
    / x["published_mae"]
)

x["source"] = "Autoformer, NeurIPS 2021"
x["protocol_status"] = "MATCHED_CORE_PROTOCOL"

print("=" * 78)
print("FIXED ALPHA=1 vs PUBLISHED AUTOFORMER")
print("=" * 78)

print(
    x[
        [
            "dataset", "horizon",
            "mse_mean", "published_mse", "mse_diff_pct",
            "mae_mean", "published_mae", "mae_diff_pct"
        ]
    ].round(3).to_string(index=False)
)

x.to_csv(
    "zero_gpu_autoformer_vs_published.csv",
    index=False
)

print("\nWROTE:")
print("zero_gpu_autoformer_vs_published.csv")
