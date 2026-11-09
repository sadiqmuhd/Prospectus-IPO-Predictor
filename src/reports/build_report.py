"""Fill reports/report_template.md with the current results.

Every number and table in the report comes from the pipeline outputs, so the report is rebuilt whenever the
pipeline runs. Placeholders are {{name}}; an unknown or unfillable placeholder fails the step.

Output: reports/report.md
"""
import json
import re
from datetime import date

import pandas as pd

from style import ROOT

TEMPLATE = ROOT / "reports" / "report_template.md"
OUT = ROOT / "reports" / "report.md"
RESULTS = ROOT / "results"
TARGET_LABELS = {"pop": "Pop (> 0%)", "pop10": "Pop > 10%", "bucket": "Bucket"}
MODEL_LABELS = {"majority": "Majority class", "price_revision_only": "Price revision only",
                "logistic_l1": "L1 logistic", "random_forest": "Random forest", "lightgbm": "LightGBM"}
GROUP_LABELS = {"deal": "deal terms", "firm": "firm characteristics", "underwriter": "the underwriter",
                "market": "market conditions", "text_size": "prospectus length", "text_lm": "LM dictionary tone",
                "text_finbert": "FinBERT tone",
                "text_relative": "tone relative to recent IPOs"}


def pct(v, d=0) -> str:
    return "n/a" if pd.isna(v) else f"{v * 100:.{d}f}%"


def signed(v, d=1) -> str:
    return f"{v * 100:+.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def values() -> dict:
    v = {"generated": date.today().isoformat()}
    clean = pd.read_csv(ROOT / "data/processed/us_ipos_clean.csv", low_memory=False)
    allx = pd.read_csv(ROOT / "data/processed/us_ipos.csv", low_memory=False, usecols=["px_unreliable"])
    facts = pd.read_csv(ROOT / "data/interim/prospectus_facts.csv")
    v.update(n_clean=f"{len(clean):,}", n_prospectus=f"{len(facts):,}",
             mean_return=pct(clean["first_day_return"].mean(), 1), median_return=pct(clean["first_day_return"].median(), 1),
             pop_rate=pct(clean["pop"].mean()), pop10_rate=pct(clean["pop10"].mean()),
             n_px_unreliable=int(allx["px_unreliable"].sum()),
             cov_price_revision=pct(clean["price_revision"].notna().mean()),
             cov_price_revision_missing=pct(clean["price_revision"].isna().mean()),
             cov_uw=pct(clean["lead_uw_rank"].notna().mean()), cov_syndicate=pct(clean["n_syndicate"].notna().mean()),
             cov_risk=pct((clean["risk_factors_chars"] > 0).mean()))
    m = clean.dropna(subset=["doc_offer_price"])
    v["cov_price_agree"] = pct(((m["doc_offer_price"] / m["offer_price"] - 1).abs() < 0.01).mean(), 1)

    y = clean["ipo_year"]
    v.update(n_train=f"{(y <= 2017).sum():,}", n_val=f"{y.between(2018, 2020).sum():,}", n_test=f"{(y >= 2021).sum():,}")
    b = pd.read_csv(RESULTS / "eda_buckets.csv", index_col=0)
    v.update(bucket_heavy_train=pct(b.loc["Train 2000–17", "heavily_underpriced"]),
             bucket_heavy_test=pct(b.loc["Test 2021–25", "heavily_underpriced"]))

    eda = pd.read_csv(RESULTS / "eda_summary.csv")
    rev = eda[eda["table"] == "price_revision"].set_index("group")
    v.update(above_mean=signed(rev.loc["Above range", "mean_return"]), below_mean=signed(rev.loc["Below range", "mean_return"]))
    v["eda_revision_table"] = md_table(pd.DataFrame({
        "Offer price vs. initial range": rev.index, "IPOs": rev["n"].astype(int),
        "Mean first-day return": rev["mean_return"].map(lambda x: signed(x)),
        "Median": rev["median_return"].map(lambda x: signed(x)), "Share that pop": rev["pop_rate"].map(pct)}))

    # models
    met = pd.read_csv(RESULTS / "metrics.csv")
    chosen = json.loads((RESULTS / "chosen_models.json").read_text())
    rows = []
    for _, r in met.iterrows():
        main = "test_auc" if r["target"] != "bucket" else "test_macro_f1"
        rows.append({"Target": TARGET_LABELS[r["target"]], "Model": MODEL_LABELS[r["model"]],
                     "Features": {"baseline": "–", "no_text": "without text", "text": "with text"}[r["features"]],
                     "Test score": f"{r[main]:.3f} {'AUC' if r['target'] != 'bucket' else 'macro-F1'}",
                     "Accuracy": f"{r['test_accuracy']:.3f}"})
    v["metrics_table"] = md_table(pd.DataFrame(rows))
    pop = met[met["target"] == "pop"]
    best = pop[pop["features"] != "baseline"].sort_values("test_auc", ascending=False).iloc[0]
    v.update(best_pop_auc=f"{best['test_auc']:.3f}",
             best_pop_model=f"{MODEL_LABELS[best['model']]}, {'with' if best['features'] == 'text' else 'without'} text",
             pr_only_auc=f"{pop[pop['model'] == 'price_revision_only']['test_auc'].iloc[0]:.3f}")
    ch = pop[(pop["model"] == chosen["pop"]["model"]) & (pop["features"] == chosen["pop"]["features_set"])].iloc[0]
    v.update(strategy_return=signed(ch["strategy_mean_return"]), all_return=signed(ch["all_ipos_mean_return"]),
             strategy_share=pct(ch["strategy_share_bought"]))

    te = pd.read_csv(RESULTS / "text_effect.csv")
    v["n_boot"] = "1,000"
    v["text_effect_table"] = md_table(pd.DataFrame({
        "Target": te["target"].map(TARGET_LABELS), "Model": te["model"].map(MODEL_LABELS),
        "Metric": te["metric"].str.replace("_", "-"), "Gain from text": te["gain"].map(lambda x: f"{x:+.3f}"),
        "95% interval": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(te["ci_low"], te["ci_high"])]}))
    sig = te[(te["ci_low"] > 0) | (te["ci_high"] < 0)]
    pos = sig[sig["gain"] > 0]
    if sig.empty:
        verdict = (f"In none of the {len(te)} model-target comparisons does adding the prospectus text features "
                   "change test performance by more than chance: every 95% interval includes zero.")
    else:
        names = "; ".join(f"{TARGET_LABELS[r.target]} with {MODEL_LABELS[r.model]} ({r.gain:+.3f})" for r in pos.itertuples())
        verdict = (f"In {len(sig)} of {len(te)} model-target comparisons the 95% interval excludes zero"
                   + (f"; text helps for {names}" if len(pos) else "") + ". "
                   "Elsewhere the gain is indistinguishable from zero.")
    # why text struggles out of sample: document tone drifts with time
    table = pd.read_csv(ROOT / "data/processed/model_table.csv", low_memory=False)
    drift = table[["lm_doc_negative", "lm_doc_uncertainty"]].corrwith(table["ipo_year"], method="spearman")
    rf_ret = table["lm_risk_factors_uncertainty"].corr(table["first_day_return"], method="spearman")
    verdict += (f" Part of the reason is drift: whole-prospectus tone changes steadily over time (Spearman "
                f"correlation with the IPO year {drift.min():.2f} to {drift.max():.2f} for negative and uncertain "
                f"words), so a model trained on 2000–2017 meets shifted values in 2021–2025. Uncertain wording in Risk "
                f"Factors does move with the first-day return (Spearman {rf_ret:.2f}), but weakly.")
    v["text_verdict"] = verdict
    helps = len(pos) >= max(2, len(te) // 3)
    v["text_headline"] = ("Prospectus text helps for some targets." if helps
                          else "Prospectus tone adds little on top.")
    chosen_sets = {t: chosen[t]["features_set"] for t in ("pop", "pop10", "bucket")}
    v["text_verdict"] += (" The models chosen on the validation years use "
                          + ", ".join(f"{'text' if s == 'text' else 'no text'} for {TARGET_LABELS[t]}"
                                      for t, s in chosen_sets.items()) + ".")

    shap = pd.read_csv(RESULTS / "shap_importance_pop.csv", index_col=0)
    g = shap.groupby("group")["mean_abs_shap"].sum()
    g = (g / g.sum()).sort_values(ascending=False)
    v["shap_deal_pop"] = pct(g.get("deal", 0))
    from build_features import label
    v["shap_text"] = ("For the chosen pop model, " + ", ".join(f"{GROUP_LABELS.get(k, k)} {pct(s)}" for k, s in g.items())
                      + " of the mean absolute SHAP value. The five most influential features are "
                      + ", ".join(label(f) for f in shap.index[:5]) + ".")

    ng = pd.read_csv(ROOT / "data/processed/ngx_listings.csv")
    nr = ng.dropna(subset=["adjusted_return"])
    v.update(ng_n=len(ng), ng_with_returns=len(nr), ng_max=signed(nr["adjusted_return"].max()))
    by = nr.set_index(nr["company"].str.split().str[0])
    v.update(ng_mtn=signed(by.loc["MTN", "adjusted_return"]), ng_bua=signed(by.loc["BUA", "adjusted_return"]),
             ng_legend=signed(by.loc["Legend", "adjusted_return"]))
    v["nigeria_table"] = md_table(pd.DataFrame({
        "Listing": nr["company"].str.replace(" Plc", ""), "Type": nr["listing_type"].str.replace("_", " "),
        "Listed": nr["listing_date"], "Day-1 return": nr["day1_return"].map(lambda x: signed(x)),
        "Adjusted return": nr["adjusted_return"].map(lambda x: signed(x)),
        "Sessions at the limit": nr["sessions_at_limit"].astype(int),
        "Censored": nr["censored"].map({True: "yes", False: "no"})}).sort_values("Adjusted return", key=lambda s: s.str.rstrip("%").astype(float), ascending=False))
    return v


def main() -> None:
    sys_path_features()
    text = TEMPLATE.read_text(encoding="utf-8")
    vals = values()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(vals))
    if missing:
        raise SystemExit(f"no value for: {', '.join(missing)}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(vals[m.group(1)]), text)
    OUT.write_text(out, encoding="utf-8")
    print(f"report -> {OUT.relative_to(ROOT)} ({len(out.split())} words)")


def sys_path_features() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src" / "features"))


if __name__ == "__main__":
    main()
