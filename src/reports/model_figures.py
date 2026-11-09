"""Figures for the modelling results: model comparison, the text effect, the bucket confusion matrix.

Outputs: reports/figures/model_*.png
"""
import numpy as np
import pandas as pd

from style import BLUE_RAMP, ROOT, SERIES, TEXT, TEXT_2, label_bars, plt, save

RESULTS = ROOT / "results"
MODEL_LABELS = {"majority": "Majority class", "price_revision_only": "Price revision only",
                "logistic_l1": "L1 logistic", "random_forest": "Random forest", "lightgbm": "LightGBM"}
TARGET_LABELS = {"pop": "Pop (> 0%)", "pop10": "Pop > 10%", "bucket": "Bucket"}
BUCKETS = ["overpriced", "fair", "underpriced", "heavily_underpriced"]
BUCKET_SHORT = ["Overpriced", "Fair", "Underpriced", "Heavily\nunderpriced"]


def comparison(m: pd.DataFrame, target: str, metric: str) -> None:
    d = m[m["target"] == target]
    models = ["logistic_l1", "random_forest", "lightgbm"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x, w = np.arange(len(models)), 0.36
    for i, (fs, name) in enumerate((("no_text", "Without text"), ("text", "With prospectus text"))):
        vals = [d[(d["features"] == fs) & (d["model"] == mo)][f"test_{metric}"].iloc[0] for mo in models]
        b = ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=SERIES[i], label=name)
        label_bars(ax, b, "{:.3f}")
    for name, style in (("price_revision_only", "--"), ("majority", ":")):
        row = d[d["model"] == name]
        if len(row):
            v = row[f"test_{metric}"].iloc[0]
            ax.axhline(v, color=TEXT_2, linestyle=style, linewidth=1, label=f"Baseline: {MODEL_LABELS[name]} ({v:.3f})")
    ax.set_xticks(x, [MODEL_LABELS[mo] for mo in models])
    lo = min(0.45, d[f"test_{metric}"].min() - 0.03)
    ax.set_ylim(lo, max(d[f"test_{metric}"].max() * 1.12, 0.6))
    ax.set_ylabel(f"Test {metric.upper() if metric == 'auc' else metric.replace('_', '-')} (2021–2025)")
    ax.set_title(f"{TARGET_LABELS[target]}: test-set performance by model")
    ax.legend(loc="upper left", ncol=2)
    save(fig, f"model_comparison_{target}")


def text_effect(t: pd.DataFrame) -> None:
    t = t.copy()
    t["name"] = t["target"].map(TARGET_LABELS) + " · " + t["model"].map(MODEL_LABELS)
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.hlines(y, t["ci_low"], t["ci_high"], color=SERIES[0], linewidth=2)
    ax.scatter(t["gain"], y, color=SERIES[0], s=40, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    ax.set_yticks(y, t["name"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test-set gain from adding text (AUC; macro-F1 for buckets), 95% interval")
    ax.set_title("Does reading the prospectus help?")
    save(fig, "model_text_effect")


def confusion(cm: pd.DataFrame) -> None:
    v = cm.to_numpy().astype(float)
    share = v / v.sum(axis=1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f3f0", *BLUE_RAMP])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{share[i, j]:.0%}\n({int(v[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(4), BUCKET_SHORT)
    ax.set_yticks(range(4), BUCKET_SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title("Bucket model on the test years (row shares)")
    save(fig, "model_confusion_bucket")


def main() -> None:
    m = pd.read_csv(RESULTS / "metrics.csv")
    comparison(m, "pop", "auc")
    comparison(m, "pop10", "auc")
    comparison(m, "bucket", "macro_f1")
    text_effect(pd.read_csv(RESULTS / "text_effect.csv"))
    confusion(pd.read_csv(RESULTS / "confusion_bucket.csv", index_col=0))
    print("figures -> reports/figures/model_*.png")


if __name__ == "__main__":
    main()
