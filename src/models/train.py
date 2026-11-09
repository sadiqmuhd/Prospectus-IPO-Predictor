"""Train and evaluate first-day return models with a time split, with and without prospectus text.

Split: train 2000-2017, validate 2018-2020 (tuning), test 2021-2025 (reported once, at the end).
Targets: pop (> 0%), pop10 (> 10%), bucket (4 classes).
Feature sets: "no_text" (deal, firm, underwriter, market) and "text" (adds prospectus size, LM, FinBERT).
Models: baselines (majority class; logistic regression on price revision only), L1 logistic regression,
random forest, LightGBM. Hyperparameters are picked on the validation years, then each model is refit on
train + validation and scored on the test years.

Research question 1 (does reading the prospectus help?) is answered by a paired bootstrap of the test-set
difference between the text and no-text versions of each model.

Outputs: results/metrics.csv                 every model x feature set x target, validation and test scores
         results/text_effect.csv             test-set gain from text features, with 95% bootstrap intervals
         results/test_predictions.csv        test-set probabilities of the chosen models
         results/confusion_bucket.csv        confusion matrix of the chosen bucket model
         models/<target>.joblib              chosen model per target, refit on every year, for the web API
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from build_features import FEATURE_GROUPS, TEXT_GROUPS, features  # noqa: E402

TABLE = ROOT / "data/processed/model_table.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TRAIN_END, VAL_END = 2017, 2020
SEED = 42
BUCKETS = ["overpriced", "fair", "underpriced", "heavily_underpriced"]
TARGETS = {"pop": "binary", "pop10": "binary", "bucket": "multiclass"}
FEATURE_SETS = {
    "no_text": [g for g in FEATURE_GROUPS if g not in TEXT_GROUPS],
    "text": list(FEATURE_GROUPS),
}
BOOTSTRAP = 1000

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- models -------------------------------------------------------------------------------------------

def logistic(C: float, multiclass: bool):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=5000, random_state=SEED))


def forest(depth, leaf):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))


def gbm(leaves, lr, n, multiclass):
    return lgb.LGBMClassifier(num_leaves=leaves, learning_rate=lr, n_estimators=n, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              objective="multiclass" if multiclass else "binary", random_state=SEED,
                              verbose=-1)


GRIDS = {
    "logistic_l1": [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)],
    "random_forest": [{"depth": d, "leaf": l} for d, l in itertools.product((6, 10, None), (5, 20))],
    "lightgbm": [{"leaves": nl, "lr": lr, "n": n}
                 for nl, lr, n in itertools.product((7, 15, 31), (0.02, 0.05), (200, 400, 800))],
}


def build(name: str, params: dict, multiclass: bool):
    if name == "logistic_l1":
        return logistic(params["C"], multiclass)
    if name == "random_forest":
        return forest(params["depth"], params["leaf"])
    return gbm(params["leaves"], params["lr"], params["n"], multiclass)


# ---------- scoring ------------------------------------------------------------------------------------------

def score(kind: str, y, p) -> dict:
    if kind == "binary":
        pred = (p >= 0.5).astype(int)
        return {"auc": roc_auc_score(y, p), "f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred),
                "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}
    pred = p.argmax(axis=1)
    return {"macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1), labels=range(len(BUCKETS)))}


def main_metric(kind: str) -> str:
    return "auc" if kind == "binary" else "macro_f1"


def predict(model, X, kind):
    p = model.predict_proba(X)
    return p[:, 1] if kind == "binary" else p


def encode(y: pd.Series, kind: str) -> np.ndarray:
    return y.map({b: i for i, b in enumerate(BUCKETS)}).to_numpy() if kind == "multiclass" else y.to_numpy()


def strategy_return(first_day_return: np.ndarray, p_pop: np.ndarray) -> dict:
    """Mean first-day return from buying only the IPOs the model calls a pop, vs. buying every IPO."""
    picked = p_pop >= 0.5
    return {"strategy_mean_return": first_day_return[picked].mean() if picked.any() else np.nan,
            "strategy_share_bought": picked.mean(), "all_ipos_mean_return": first_day_return.mean()}


def paired_bootstrap(kind, y, p_text, p_none, seed=SEED) -> tuple[float, float, float]:
    """Test-set gain in the main metric from adding text, with a 95% interval."""
    rng = np.random.default_rng(seed)
    m = main_metric(kind)
    point = score(kind, y, p_text)[m] - score(kind, y, p_none)[m]
    diffs = []
    n = len(y)
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        if kind == "binary" and len(np.unique(y[i])) < 2:
            continue
        diffs.append(score(kind, y[i], p_text[i])[m] - score(kind, y[i], p_none[i])[m])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


# ---------- main ---------------------------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(TABLE, parse_dates=["ipo_date"], low_memory=False).sort_values("ipo_date")
    train = df[df["ipo_year"] <= TRAIN_END]
    val = df[(df["ipo_year"] > TRAIN_END) & (df["ipo_year"] <= VAL_END)]
    test = df[df["ipo_year"] > VAL_END]
    trval = pd.concat([train, val])
    print(f"train {len(train)} ({train.ipo_year.min()}-{train.ipo_year.max()}), "
          f"validate {len(val)}, test {len(test)} ({test.ipo_year.min()}-{test.ipo_year.max()})")
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    rows, test_preds, chosen = [], {"ipo_id": test["ipo_id"].to_numpy()}, {}
    for target, kind in TARGETS.items():
        multiclass = kind == "multiclass"
        y_tr, y_va, y_te, y_tv = (encode(d[target], kind) for d in (train, val, test, trval))
        m = main_metric(kind)
        best_by_set = {}
        for set_name, groups in FEATURE_SETS.items():
            cols = features(groups)
            Xtr, Xva, Xte, Xtv = (d[cols] for d in (train, val, test, trval))

            # baselines: no features, and price revision alone (the classic predictor)
            if set_name == "no_text":
                base = DummyClassifier(strategy="prior").fit(Xtr, y_tr)
                pr = logistic(1.0, multiclass).fit(train[["price_revision"]], y_tr)
                for name, mdl, Xv, Xt in (("majority", base, Xva, Xte),
                                          ("price_revision_only", pr, val[["price_revision"]],
                                           test[["price_revision"]])):
                    refit = (DummyClassifier(strategy="prior") if name == "majority"
                             else logistic(1.0, multiclass)).fit(trval[Xt.columns], y_tv)
                    rows.append({"target": target, "features": "baseline", "model": name, "params": "{}",
                                 **{f"val_{k}": v for k, v in score(kind, y_va, predict(mdl, Xv, kind)).items()},
                                 **{f"test_{k}": v for k, v in score(kind, y_te, predict(refit, Xt, kind)).items()}})

            for model_name, grid in GRIDS.items():
                best = None
                for params in grid:
                    mdl = build(model_name, params, multiclass).fit(Xtr, y_tr)
                    s = score(kind, y_va, predict(mdl, Xva, kind))
                    if best is None or s[m] > best[1][m]:
                        best = (params, s)
                params, val_scores = best
                final = build(model_name, params, multiclass).fit(Xtv, y_tv)
                p_te = predict(final, Xte, kind)
                test_scores = score(kind, y_te, p_te)
                row = {"target": target, "features": set_name, "model": model_name, "params": json.dumps(params),
                       **{f"val_{k}": v for k, v in val_scores.items()},
                       **{f"test_{k}": v for k, v in test_scores.items()}}
                if target == "pop":
                    row.update(strategy_return(test["first_day_return"].to_numpy(), p_te))
                rows.append(row)
                best_by_set[(set_name, model_name)] = (val_scores[m], params, p_te)
                print(f"{target:7s} {set_name:8s} {model_name:14s} val {m} {val_scores[m]:.3f}  "
                      f"test {m} {test_scores[m]:.3f}", flush=True)

        # text effect per model family, on the test years
        for model_name in GRIDS:
            _, _, p_text = best_by_set[("text", model_name)]
            _, _, p_none = best_by_set[("no_text", model_name)]
            point, lo, hi = paired_bootstrap(kind, y_te, p_text, p_none)
            rows_te = {"target": target, "model": model_name, "metric": m, "gain": point, "ci_low": lo,
                       "ci_high": hi}
            chosen.setdefault("_text_effect", []).append(rows_te)

        # the model for the web app: best validation score across families and feature sets
        (set_name, model_name), (_, params, p_te) = max(best_by_set.items(), key=lambda kv: kv[1][0])
        chosen[target] = {"features_set": set_name, "model": model_name, "params": params}
        if kind == "binary":
            test_preds[f"p_{target}"] = p_te
        else:
            for i, b in enumerate(BUCKETS):
                test_preds[f"p_{b}"] = p_te[:, i]
            cm = confusion_matrix(y_te, p_te.argmax(axis=1), labels=range(len(BUCKETS)))
            pd.DataFrame(cm, index=[f"true_{b}" for b in BUCKETS], columns=[f"pred_{b}" for b in BUCKETS]) \
                .to_csv(RESULTS / "confusion_bucket.csv")

        cols = features(FEATURE_SETS[set_name])
        full = pd.concat([trval, test])
        model = build(model_name, params, multiclass).fit(full[cols], encode(full[target], kind))
        test_row = next(r for r in rows if r["target"] == target and r["features"] == set_name
                        and r["model"] == model_name)
        joblib.dump({"model": model, "features": cols, "target": target, "kind": kind,
                     "classes": BUCKETS if multiclass else [0, 1], "model_name": model_name, "params": params,
                     "feature_set": set_name, "trained_years": [int(full.ipo_year.min()), int(full.ipo_year.max())],
                     # SHAP background for explaining single predictions in the web API
                     "background": full[cols].sample(300, random_state=SEED),
                     "test_scores": {k.removeprefix("test_"): v for k, v in test_row.items()
                                     if k.startswith("test_")}},
                    MODELS / f"{target}.joblib")
        print(f"-> {target}: chose {model_name} on {set_name} features")

    pd.DataFrame(rows).to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(chosen.pop("_text_effect")).to_csv(RESULTS / "text_effect.csv", index=False)
    preds = pd.DataFrame(test_preds).merge(test[["ipo_id", "ipo_date", "company", "first_day_return", "pop",
                                                 "pop10", "bucket"]], on="ipo_id")
    preds.to_csv(RESULTS / "test_predictions.csv", index=False)
    (RESULTS / "chosen_models.json").write_text(json.dumps(chosen, indent=2))
    print(pd.read_csv(RESULTS / "text_effect.csv").to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
