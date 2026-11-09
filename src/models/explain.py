"""SHAP explanations for the chosen models.

`contributions(bundle, X)` gives per-feature SHAP values for any row(s); the web API uses it for "why this
prediction". Run as a script, it explains each saved model on the test years:

Outputs: results/shap_importance_<target>.csv   mean |SHAP| per feature and per feature group
         reports/figures/shap_<target>_features.png and shap_<target>_groups.png
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "reports"))
from build_features import FEATURE_GROUPS, label  # noqa: E402

MODELS = ROOT / "models"
TABLE = ROOT / "data/processed/model_table.csv"
RESULTS = ROOT / "results"
TEST_FROM = 2021
GROUP_LABELS = {"deal": "Deal terms", "firm": "Firm", "underwriter": "Underwriter", "market": "Market conditions",
                "text_size": "Prospectus length", "text_lm": "LM dictionary tone", "text_finbert": "FinBERT tone",
                "text_relative": "Tone vs. recent IPOs"}
FEATURE_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def _explainer(bundle, background: pd.DataFrame):
    """(explainer, transform, output feature names) for a LightGBM model or an sklearn pipeline."""
    model, cols = bundle["model"], bundle["features"]
    if hasattr(model, "steps"):
        prep, last = model[:-1], model[-1]
        names = list(prep.get_feature_names_out(cols))
        bg = prep.transform(background[cols])
        if bundle["model_name"] == "logistic_l1":
            return shap.LinearExplainer(last, bg), prep.transform, names
        return shap.TreeExplainer(last), prep.transform, names
    return shap.TreeExplainer(model), (lambda X: X), cols


def _source(name: str) -> str:
    """Missing-value indicator columns count toward the feature they flag."""
    return name.removeprefix("missingindicator_")


def contributions(bundle, X: pd.DataFrame, background: pd.DataFrame | None = None,
                  class_index: int | np.ndarray | None = None) -> pd.DataFrame:
    """SHAP value per original feature, one row per input row.

    Binary models: contribution to the positive class. Multiclass: to `class_index` (default: the predicted class).
    """
    cols = bundle["features"]
    background = background if background is not None else X
    explainer, transform, names = _explainer(bundle, background.sample(min(len(background), 300), random_state=0))
    raw = explainer.shap_values(transform(X[cols]))
    # some explainers return one array per class
    vals = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    if vals.ndim == 3:  # (rows, features, classes)
        if bundle["kind"] == "binary":
            vals = vals[:, :, 1]
        else:
            if class_index is None:
                class_index = bundle["model"].predict_proba(X[cols]).argmax(axis=1)
            idx = np.broadcast_to(np.asarray(class_index), (len(X),))
            vals = vals[np.arange(len(X)), :, idx]
    out = pd.DataFrame(vals, columns=names, index=X.index)
    return out.T.groupby(_source).sum().T.reindex(columns=cols, fill_value=0.0)


def importance(target: str) -> pd.DataFrame | None:
    path = MODELS / f"{target}.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    df = pd.read_csv(TABLE, low_memory=False)
    test = df[df["ipo_year"] >= TEST_FROM]
    shap_vals = contributions(bundle, test, background=df[df["ipo_year"] < TEST_FROM])
    imp = shap_vals.abs().mean().rename("mean_abs_shap").to_frame()
    imp["group"] = imp.index.map(FEATURE_GROUP)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    imp.to_csv(RESULTS / f"shap_importance_{target}.csv", index_label="feature")
    plot(imp, target, bundle)
    return imp


def plot(imp: pd.DataFrame, target: str, bundle) -> None:
    from style import SERIES, TEXT_2, plt, save

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([label(f) for f in top.index], top["mean_abs_shap"], color=SERIES[0], height=0.6)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    unit = "log-odds" if bundle["model_name"] != "random_forest" else "probability"
    ax.set_xlabel(f"Mean |SHAP value| on the test years ({unit})")
    ax.set_title(f"What drives the {target} model ({bundle['model_name'].replace('_', ' ')})")
    save(fig, f"shap_{target}_features")

    groups = imp.groupby("group")["mean_abs_shap"].sum()
    share = (groups / groups.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    b = ax.barh([GROUP_LABELS.get(g, g) for g in share.index], share.to_numpy(), color=SERIES[0], height=0.6)
    for bar, v in zip(b, share.to_numpy()):
        ax.annotate(f"{v:.0%}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Share of the {target} model's explanation, by feature group")
    save(fig, f"shap_{target}_groups")


def main() -> None:
    for target in ("pop", "pop10", "bucket"):
        imp = importance(target)
        if imp is not None:
            g = imp.groupby("group")["mean_abs_shap"].sum()
            print(f"{target}: " + ", ".join(f"{k} {v / g.sum():.0%}" for k, v in g.sort_values(ascending=False).items()))
            print("   top: " + ", ".join(imp.index[:8]))


if __name__ == "__main__":
    main()
