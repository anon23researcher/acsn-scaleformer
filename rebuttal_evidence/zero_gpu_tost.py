import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t

p = pd.read_csv("zero_gpu_adaptive_vs_frozen05_pairs.csv")

def tost_log_ratio(adaptive, frozen, margin_pct, label):
    # Positive means Adaptive has lower error.
    x = np.log(frozen / adaptive)
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)

    # Symmetric multiplicative equivalence limits.
    low = np.log(1 / (1 + margin_pct / 100))
    high = np.log(1 + margin_pct / 100)

    # H0 lower: mean <= low; H1: mean > low
    t_lower = (mean - low) / se
    p_lower = 1 - t.cdf(t_lower, df=n - 1)

    # H0 upper: mean >= high; H1: mean < high
    t_upper = (mean - high) / se
    p_upper = t.cdf(t_upper, df=n - 1)

    # 90% CI is standard for alpha=.05 TOST.
    crit = t.ppf(0.95, df=n - 1)
    ci_low = mean - crit * se
    ci_high = mean + crit * se

    print(f"\n{label}: equivalence margin +/-{margin_pct}%")
    print(f"N: {n}")
    print(f"Mean log gain: {mean:.8f}")
    print(
        "Approximate multiplicative gain: "
        f"{100*(np.exp(mean)-1):.4f}%"
    )
    print(
        "90% CI as percentage gain: "
        f"[{100*(np.exp(ci_low)-1):.4f}%, "
        f"{100*(np.exp(ci_high)-1):.4f}%]"
    )
    print(f"Lower-bound test p: {p_lower:.8g}")
    print(f"Upper-bound test p: {p_upper:.8g}")
    print(
        "TOST p-value:",
        f"{max(p_lower, p_upper):.8g}"
    )
    print(
        "Equivalent:",
        "YES" if max(p_lower, p_upper) < 0.05 else "NO"
    )

for metric in ["mse", "mae"]:
    adaptive = p[f"adaptive_{metric}"].to_numpy()
    frozen = p[f"frozen_{metric}"].to_numpy()

    for margin in [1, 2]:
        tost_log_ratio(
            adaptive,
            frozen,
            margin,
            metric.upper()
        )
