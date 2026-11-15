import pandas as pd

d = pd.read_csv("final_analysis_3200/results.csv")

fixed = d[d["mode"] == "fixed"].copy()

table = (
    fixed.groupby(
        ["dataset", "model", "horizon"],
        as_index=False
    )
    .agg(
        n=("seed", "nunique"),
        mse_mean=("mse", "mean"),
        mse_sd=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
    )
)

print("=" * 78)
print("STUDY 1 FIXED ALPHA=1 ABSOLUTE PERFORMANCE")
print("=" * 78)

print("RUNS:", len(fixed))
print("CELLS:", len(table))
print("SEEDS PER CELL:", sorted(table["n"].unique()))

print("\nFULL TABLE")
print(table.round(6).to_string(index=False))

assert len(fixed) == 800
assert len(table) == 80
assert table["n"].min() == 10
assert table["n"].max() == 10

table.to_csv(
    "zero_gpu_fixed_absolute.csv",
    index=False
)

print("\nWROTE:")
print("zero_gpu_fixed_absolute.csv")
