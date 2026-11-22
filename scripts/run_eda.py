"""EDA report: reads features from the Feature Store and saves charts + findings.

Run:  python -m scripts.run_eda
Output: reports/eda/*.png, reports/eda/summary_stats.csv, reports/eda/findings.md
(The dashboard computes the same EDA live from the Feature Store.)
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eda import analysis as A  # noqa: E402
from src.store import get_store  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "eda"


def main() -> int:
    feats = get_store().read("hourly_features")
    if feats.empty:
        print("No features found - run the backfill first.")
        return 1
    df = A.prepare(feats)
    OUT.mkdir(parents=True, exist_ok=True)
    vars_ = [v for v in A.EDA_VARS if v in df]

    A.summary_stats(df).to_csv(OUT / "summary_stats.csv")
    A.outliers_iqr(df).to_csv(OUT / "outliers.csv", index=False)
    A.correlations(df).to_csv(OUT / "correlations.csv")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, v in zip(axes.ravel(), vars_):
        df[v].dropna().hist(ax=ax, bins=40)
        ax.set_title(f"Distribution: {A.LABELS[v]}")
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png", dpi=110)

    prof = A.hourly_profile(df)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, v in zip(axes.ravel(), vars_):
        prof[v].plot(ax=ax, marker="o")
        ax.set_title(f"Mean by local hour: {A.LABELS[v]}")
    fig.tight_layout()
    fig.savefig(OUT / "hourly_patterns.png", dpi=110)

    daily = A.daily_series(df)
    fig, axes = plt.subplots(len(vars_), 1, figsize=(14, 2.4 * len(vars_)), sharex=True)
    for ax, v in zip(axes, vars_):
        daily[v].plot(ax=ax)
        ax.set_ylabel(v)
    axes[0].set_title("Daily trends (precipitation = daily total)")
    fig.tight_layout()
    fig.savefig(OUT / "daily_trends.png", dpi=110)

    corr = A.correlations(df)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(OUT / "correlations.png", dpi=110)

    lines = ["# EDA findings (auto-generated from the Feature Store)", ""]
    lines += [f"- {f}" for f in A.findings(df)]
    lines += ["", "## Missing values (% of hours, after imputation)", "", "```",
              A.missing_report(df).to_string(), "```"]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
