"""Exploratory figures and summary tables for the report.

Outputs: reports/figures/eda_*.png and results/eda_summary.csv
"""
import numpy as np
import pandas as pd

from style import (BLUE_RAMP, NEGATIVE, ROOT, SERIES, TEXT_2, label_bars, pct, plt, save)

TABLE = ROOT / "data/processed/model_table.csv"
NGX = ROOT / "data/processed/ngx_listings.csv"
RESULTS = ROOT / "results"
BUCKETS = ["overpriced", "fair", "underpriced", "heavily_underpriced"]
BUCKET_LABELS = ["Overpriced\n(< 0%)", "Fair\n(0–10%)", "Underpriced\n(10–30%)", "Heavily underpriced\n(> 30%)"]


def by_group(df, col, order=None):
    g = df.groupby(col, observed=True).agg(n=("first_day_return", "size"),
                                           mean_return=("first_day_return", "mean"),
                                           median_return=("first_day_return", "median"),
                                           pop_rate=("pop", "mean"))
    return g.reindex(order) if order is not None else g


def bars(ax, labels, values, color=SERIES[0], fmt="{:.0%}"):
    b = ax.bar(labels, values, color=[NEGATIVE if v < 0 else color for v in values], width=0.6)
    label_bars(ax, b, fmt)
    pct(ax)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    return b


def yearly(df):
    g = by_group(df, "ipo_year")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(g.index, g["n"], color=SERIES[0], width=0.7)
    ax.set_title("US IPOs in the clean sample, by year")
    ax.set_ylabel("IPOs")
    save(fig, "eda_ipos_per_year")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(g.index, g["mean_return"], color=SERIES[0], marker="o", label="Mean")
    ax.plot(g.index, g["median_return"], color=SERIES[1], marker="o", label="Median")
    ax.axvspan(2020.5, g.index.max() + 0.5, color="#f1f0ec", zorder=0)
    ax.text(2020.7, ax.get_ylim()[1] * 0.92, "test years", color=TEXT_2, fontsize=8)
    for name, col, color in (("Mean", "mean_return", SERIES[0]), ("Median", "median_return", SERIES[1])):
        ax.annotate(name, (g.index[-1], g[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT_2, fontsize=9, va="center")
    ax.set_title("First-day return by year: the 2000 bubble, then 10–25% a year")
    pct(ax)  # lines are labelled at their right end, so no legend box
    save(fig, "eda_return_by_year")
    return g


def revision(df):
    d = df.dropna(subset=["price_revision"]).copy()
    d["rev"] = np.select([d["below_range"] == 1, d["above_range"] == 1], ["Below range", "Above range"],
                         "Within range")
    order = ["Below range", "Within range", "Above range"]
    g = by_group(d, "rev", order)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bars(axes[0], order, g["mean_return"].to_numpy())
    axes[0].set_title("Mean first-day return")
    bars(axes[1], order, g["pop_rate"].to_numpy())
    axes[1].set_title("Share that pop (> 0%)")
    for ax in axes:
        ax.set_xticks(range(3), [f"{o}\n(n={n})" for o, n in zip(order, g["n"])])
    fig.suptitle("Offer price vs. the initial filing range", x=0.01, ha="left", fontweight="bold")
    save(fig, "eda_price_revision")
    return g


def underwriter(df):
    d = df.dropna(subset=["lead_uw_rank"]).copy()
    d["uw"] = pd.cut(d["lead_uw_rank"], [-1, 6.999, 8.499, 10], labels=["Below 7", "7 – 8.5", "Above 8.5 (top tier)"])
    g = by_group(d, "uw")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars(ax, [f"{i}\n(n={n})" for i, n in zip(g.index, g["n"])], g["mean_return"].to_numpy())
    ax.set_title("Mean first-day return by lead underwriter rank")
    save(fig, "eda_underwriter")
    return g


def tone(df, col, title, name):
    d = df.dropna(subset=[col]).copy()
    if d.empty:
        return None
    d["q"] = pd.qcut(d[col], 5, labels=["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    g = by_group(d, "q")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars(ax, list(g.index), g["mean_return"].to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Quintile")
    save(fig, name)
    return g


def buckets(df):
    split = np.select([df["ipo_year"] <= 2017, df["ipo_year"] <= 2020], ["Train 2000–17", "Validate 2018–20"],
                      "Test 2021–25")
    t = pd.crosstab(split, df["bucket"], normalize="index")[BUCKETS]
    t = t.reindex(["Train 2000–17", "Validate 2018–20", "Test 2021–25"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x, w = np.arange(len(BUCKETS)), 0.26
    for i, (name, row) in enumerate(t.iterrows()):
        b = ax.bar(x + (i - 1) * (w + 0.02), row.to_numpy(), width=w, color=SERIES[i], label=name)
        label_bars(ax, b)
    ax.set_xticks(x, BUCKET_LABELS)
    pct(ax)
    ax.set_ylim(0, t.to_numpy().max() * 1.25)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Return buckets by split")
    save(fig, "eda_buckets")
    return t


def nigeria():
    n = pd.read_csv(NGX).dropna(subset=["adjusted_return"]).sort_values("adjusted_return")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(n) + 1.2))
    y = np.arange(len(n))
    ax.hlines(y, n["day1_return"], n["adjusted_return"], color="#c9c8c2", linewidth=2)
    ax.scatter(n["day1_return"], y, color=SERIES[1], s=50, zorder=3, label="Day-1 return (capped by the limit)")
    ax.scatter(n["adjusted_return"], y, color=SERIES[0], s=50, zorder=3,
               label="Adjusted: to the first session inside the limit")
    labels = [f"{c}{' *' if t != 'ipo' else ''}" for c, t in zip(n["company"], n["listing_type"])]
    ax.set_yticks(y, labels)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    pct(ax, "x")
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.legend(loc="lower right")
    ax.set_title("NGX: the 10% limit hides most of the first-day move")
    labels_short = [l.replace(" Plc", "").replace(" Communications", "").replace(" Development Company", "")
                    for l in labels]
    ax.set_yticks(y, labels_short)
    fig.text(0.01, 0.01, "* listed by introduction (reference price, no public offer)", fontsize=8, color=TEXT_2)
    save(fig, "eda_nigeria", rect=(0, 0.04, 1, 1))
    return n[["company", "listing_type", "day1_return", "adjusted_return", "censored"]]


def main() -> None:
    df = pd.read_csv(TABLE, low_memory=False)
    RESULTS.mkdir(exist_ok=True)
    parts = {"year": yearly(df), "price_revision": revision(df), "underwriter": underwriter(df),
             "lm_uncertainty": tone(df, "lm_risk_factors_uncertainty",
                                    "Mean first-day return by uncertain wording in Risk Factors",
                                    "eda_lm_uncertainty"),
             "finbert_tone": tone(df, "fb_risk_factors_net", "Mean first-day return by FinBERT tone of Risk Factors",
                                  "eda_finbert_tone")}
    rows = [g.assign(table=k).reset_index(names="group") for k, g in parts.items() if g is not None]
    pd.concat(rows).to_csv(RESULTS / "eda_summary.csv", index=False)
    buckets(df).to_csv(RESULTS / "eda_buckets.csv")
    nigeria().to_csv(RESULTS / "eda_nigeria.csv", index=False)
    print(f"figures -> reports/figures; tables -> results/eda_*.csv")
    print(parts["price_revision"].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
