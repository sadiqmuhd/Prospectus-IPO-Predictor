# Reading the Prospectus: An NLP-Driven Machine Learning System for Predicting First-Day IPO Returns in the US and Nigeria

*Short-course data science project. Generated from the pipeline outputs on 2026-11-09; numbers update when
`src/pipeline.py` is rerun.*

## Summary

Firms going public tend to leave money on the table: US IPOs in our sample closed their first day
16.8% above the offer price on average (median 7.2%), and 69% of them "popped".
This project asks whether that first-day return can be predicted from information public before trading
starts, and in particular whether **reading the prospectus** with natural language processing adds to the
classic deal-level predictors.

We built a dataset of 3,642 US operating-company IPOs from 2000 to 2025, read 3,437 final
prospectuses from SEC EDGAR, scored their tone with the Loughran-McDonald financial dictionary and FinBERT, and
trained models with a strict time split (train 2000–2017, tune 2018–2020, test 2021–2025).

Key findings:

1. **First-day pops are predictable to a useful degree.** The best model reaches a test ROC-AUC of
   0.777 for pop vs. no pop (LightGBM, without text), against 0.703 for price revision alone and
   0.5 for guessing.
2. **The deal terms carry most of the signal.** By SHAP, deal terms explain 65% of the chosen pop
   model's predictions, led by the price revision: IPOs priced above their initial filing range rose
   +44.8% on average on day one, those priced below it +3.6%.
3. **Prospectus tone adds little on top.** In 1 of 9 model-target comparisons the 95% interval excludes zero; text helps for Pop > 10% with L1 logistic (+0.031). Elsewhere the gain is indistinguishable from zero. Part of the reason is drift: whole-prospectus tone changes steadily over time (Spearman correlation with the IPO year 0.45 to 0.51 for negative and uncertain words), so a model trained on 2000–2017 meets shifted values in 2021–2025. Uncertain wording in Risk Factors does move with the first-day return (Spearman 0.11), but weakly. The models chosen on the validation years use no text for Pop (> 0%), no text for Pop > 10%, no text for Bucket.
4. **Nigeria is different.** NGX's 10% daily price limit hides most of the first-day move: large listings
   closed exactly +10% on day one, but their return to the first unconstrained session reached +65.6%.

## 1. Research questions

1. Does information extracted from the prospectus (tone, length, syndicate) improve first-day return
   predictions beyond deal and market variables?
2. Which factors drive the predictions?
3. How do Nigerian listings compare, given the NGX daily price limit?

## 2. Data

| Source | Use |
|---|---|
| IPOScoop track record (2000–Sep 2020) | Offer price, first-day close, lead managers, including later-delisted firms |
| Nasdaq IPO calendar API (2000–2025) + Yahoo Finance | Deals and first-day prices after Sep 2020 |
| Jay Ritter's IPO data | Founding year, VC backing, dual class; Carter-Manaster underwriter ranks |
| SEC EDGAR | 3,437 final prospectuses (424B4/424B1) and earlier S-1/F-1 filings |
| Loughran-McDonald master dictionary (1993–2025) | Financial tone word lists |
| VIX, S&P 500, Nasdaq (Yahoo Finance) | Market conditions before each IPO |
| NGX Daily Official Lists, news reports | 29 Nigerian listings, 13 with returns |

SPACs, units, closed-end funds, ADRs and penny stocks (offer below $5) are excluded, as are 196
deals whose post-2020 Yahoo prices needed more than a 10× reverse-split adjustment. The yearly mean and median
first-day returns track Ritter's published statistics closely (correlation 0.85 for means, 0.88 for medians).

**Prospectus coverage** in the clean sample: price revision 81%, underwriter rank
93%, syndicate size 94%, Risk Factors text 93%. The offer price read from the
prospectus matches the dataset within 1% for 98.6% of documents.

![IPOs per year](figures/eda_ipos_per_year.png)
![First-day return by year](figures/eda_return_by_year.png)

### Targets

- `pop`: first-day return > 0 (69% of the sample)
- `pop10`: first-day return > 10% (44%)
- `bucket`: overpriced (< 0) · fair (0–10%) · underpriced (10–30%) · heavily underpriced (> 30%)

![Buckets by split](figures/eda_buckets.png)

The test years differ from the training years: heavily underpriced IPOs make up 16% of
2000–2017 but 28% of 2021–2025, a shift any model has to cope with.

## 3. Features

| Group | Features |
|---|---|
| Deal | Deal size, offer price, price revision vs. the initial filing range (above / below range, range width), Nasdaq listing |
| Firm | Age, VC backing, dual-class shares, internet firm, roll-up |
| Underwriter | Lead underwriter prestige (Carter-Manaster rank), syndicate size |
| Market | VIX, S&P 500 and Nasdaq returns over the prior month, number and mean first-day return of IPOs in the prior 30 days |
| Prospectus length | Whole document, Risk Factors section |
| LM tone | Share of negative, positive, uncertain, litigious, strong/weak modal and constraining words, and Fog readability, for the whole document, Summary, Risk Factors and Use of Proceeds |
| FinBERT tone | Mean positive, negative and net sentiment of 16 sentences sampled evenly through each of Summary, Risk Factors and Use of Proceeds |

Only information public before the first trade is used. The IPO year is not a feature (test years lie outside the
training range); Kaggle financials are not used because they exist only for firms that survived to 2018.

![Price revision](figures/eda_price_revision.png)
![Underwriter rank](figures/eda_underwriter.png)
![LM uncertainty](figures/eda_lm_uncertainty.png)
![FinBERT tone](figures/eda_finbert_tone.png)

| Offer price vs. initial range | IPOs | Mean first-day return | Median | Share that pop |
|---|---|---|---|---|
| Below range | 801 | +3.6% | +0.4% | 55% |
| Within range | 1398 | +11.7% | +7.2% | 69% |
| Above range | 747 | +44.8% | +33.3% | 95% |

## 4. Method

- **Split:** train 2000–2017 (2,707 IPOs), validation 2018–2020 (497), test 2021–2025 (438).
- **Models:** majority class and price-revision-only logistic regression as baselines; L1 logistic regression,
  random forest and LightGBM, each fit **with** and **without** the prospectus text features.
- **Tuning:** a small grid per model, chosen on validation ROC-AUC (macro-F1 for buckets); the chosen setting is
  refit on train + validation and scored once on the test years.
- **Does text help?** Paired bootstrap (1,000 resamples of the test set) of the score difference between
  the text and no-text version of each model.
- **Explanations:** SHAP values for the chosen models.

## 5. Results

### 5.1 Model performance (test years 2021–2025)

| Target | Model | Features | Test score | Accuracy |
|---|---|---|---|---|
| Pop (> 0%) | Majority class | – | 0.500 AUC | 0.676 |
| Pop (> 0%) | Price revision only | – | 0.703 AUC | 0.685 |
| Pop (> 0%) | L1 logistic | without text | 0.703 AUC | 0.689 |
| Pop (> 0%) | Random forest | without text | 0.767 AUC | 0.717 |
| Pop (> 0%) | LightGBM | without text | 0.777 AUC | 0.694 |
| Pop (> 0%) | L1 logistic | with text | 0.725 AUC | 0.719 |
| Pop (> 0%) | Random forest | with text | 0.747 AUC | 0.689 |
| Pop (> 0%) | LightGBM | with text | 0.765 AUC | 0.694 |
| Pop > 10% | Majority class | – | 0.500 AUC | 0.495 |
| Pop > 10% | Price revision only | – | 0.727 AUC | 0.721 |
| Pop > 10% | L1 logistic | without text | 0.731 AUC | 0.660 |
| Pop > 10% | Random forest | without text | 0.777 AUC | 0.717 |
| Pop > 10% | LightGBM | without text | 0.778 AUC | 0.712 |
| Pop > 10% | L1 logistic | with text | 0.762 AUC | 0.680 |
| Pop > 10% | Random forest | with text | 0.769 AUC | 0.724 |
| Pop > 10% | LightGBM | with text | 0.784 AUC | 0.726 |
| Bucket | Majority class | – | 0.088 macro-F1 | 0.215 |
| Bucket | Price revision only | – | 0.198 macro-F1 | 0.288 |
| Bucket | L1 logistic | without text | 0.324 macro-F1 | 0.326 |
| Bucket | Random forest | without text | 0.399 macro-F1 | 0.416 |
| Bucket | LightGBM | without text | 0.399 macro-F1 | 0.411 |
| Bucket | L1 logistic | with text | 0.354 macro-F1 | 0.402 |
| Bucket | Random forest | with text | 0.428 macro-F1 | 0.470 |
| Bucket | LightGBM | with text | 0.410 macro-F1 | 0.432 |

![Pop model comparison](figures/model_comparison_pop.png)
![Pop10 model comparison](figures/model_comparison_pop10.png)
![Bucket model comparison](figures/model_comparison_bucket.png)

The bucket model separates the extremes better than the middle:

![Bucket confusion matrix](figures/model_confusion_bucket.png)

**A simple economic check:** buying only the IPOs the chosen pop model favours (probability ≥ 0.5) would have
earned a mean first-day return of +25.5% on the test years, against +19.9% from buying every
IPO, while still buying 65% of deals. (First-day allocations are rationed in practice, so this
measures signal, not a tradable strategy.)

### 5.2 Does reading the prospectus help?

| Target | Model | Metric | Gain from text | 95% interval |
|---|---|---|---|---|
| Pop (> 0%) | L1 logistic | auc | +0.022 | [-0.007, +0.051] |
| Pop (> 0%) | Random forest | auc | -0.021 | [-0.053, +0.012] |
| Pop (> 0%) | LightGBM | auc | -0.012 | [-0.040, +0.017] |
| Pop > 10% | L1 logistic | auc | +0.031 | [+0.009, +0.053] |
| Pop > 10% | Random forest | auc | -0.008 | [-0.027, +0.012] |
| Pop > 10% | LightGBM | auc | +0.006 | [-0.011, +0.024] |
| Bucket | L1 logistic | macro-f1 | +0.030 | [-0.018, +0.080] |
| Bucket | Random forest | macro-f1 | +0.029 | [-0.018, +0.078] |
| Bucket | LightGBM | macro-f1 | +0.011 | [-0.037, +0.061] |

![Text effect](figures/model_text_effect.png)

In 1 of 9 model-target comparisons the 95% interval excludes zero; text helps for Pop > 10% with L1 logistic (+0.031). Elsewhere the gain is indistinguishable from zero. Part of the reason is drift: whole-prospectus tone changes steadily over time (Spearman correlation with the IPO year 0.45 to 0.51 for negative and uncertain words), so a model trained on 2000–2017 meets shifted values in 2021–2025. Uncertain wording in Risk Factors does move with the first-day return (Spearman 0.11), but weakly. The models chosen on the validation years use no text for Pop (> 0%), no text for Pop > 10%, no text for Bucket.

### 5.3 What drives the predictions

![SHAP pop groups](figures/shap_pop_groups.png)
![SHAP pop features](figures/shap_pop_features.png)

For the chosen pop model, deal terms 65%, market conditions 17%, firm characteristics 12%, the underwriter 6% of the mean absolute SHAP value. The five most influential features are Price vs. filing range, Priced above range, Offer price, Deal size, Firm age.

## 6. Nigeria

| Listing | Type | Listed | Day-1 return | Adjusted return | Sessions at the limit | Censored |
|---|---|---|---|---|---|---|
| MTN Nigeria Communications | introduction | 2019-05-16 | +10.0% | +65.6% | 5 | no |
| BUA Foods | introduction | 2022-01-05 | +10.0% | +65.0% | 4 | no |
| Legend Internet | introduction | 2025-04-24 | +9.9% | +60.1% | 4 | no |
| Transcorp Power | introduction | 2024-03-04 | +10.0% | +46.4% | 4 | no |
| Geregu Power | introduction | 2022-10-05 | +10.0% | +20.9% | 2 | no |
| Aradel Holdings | introduction | 2024-10-14 | +10.0% | +16.7% | 1 | no |
| Zichis Agro Allied Industries | introduction | 2026-01-20 | +9.9% | +9.9% | 1 | yes |
| Seplat Petroleum Development Company | ipo | 2014-04-14 | +5.0% | +5.0% | 0 | no |
| Dangote Sugar Refinery | ipo | 2007-03-08 | +5.0% | +5.0% | 1 | yes |
| The Initiates | introduction | 2016-10-25 | +4.7% | +4.7% | 0 | no |
| Dangote Cement | introduction | 2010-10-26 | +0.0% | +0.0% | 0 | no |
| Skyway Aviation Handling Company | ipo | 2019-04-23 | +0.0% | +0.0% | 0 | no |
| Airtel Africa | introduction | 2019-07-09 | +10.0% | -10.9% | 3 | no |

![Nigeria](figures/eda_nigeria.png)

There is no bulk source of Nigerian IPO data, so the sample was assembled listing by listing from the NGX Daily
Official Lists (from late 2014) and news reports. Most large recent listings were **introductions** (existing
shares admitted at a reference price, with no public offer), so they measure price discovery rather than
underpricing.

The NGX daily limit (5% before 2012, 10% since) censors the first-day return. Every large introduction from 2019
to 2025 closed exactly at the limit on day one, and most again on day two. Measured to the first session that
closed inside the limit, MTN Nigeria rose +65.6%, BUA Foods +65.0% and Legend Internet +60.1%. With
13 listings and different mechanics, the US models are not applied to Nigeria as a statistical
test; the web app flags Nigerian prospectuses and explains the limit instead.

## 7. The prospectus bot

The models are served by a web application: a FastAPI backend (`api/`) and a Next.js frontend (`web/`). Given an
EDGAR link or an uploaded PDF/HTML prospectus, it extracts the deal facts, finds the initial price range in the
company's earlier S-1 filings, scores the text, fetches market conditions for the IPO date, and returns the pop
probability, the bucket probabilities and the SHAP reasons. Deployment instructions are in `deploy/README.md`.

## 8. Limitations

- **Survivorship after September 2020:** Yahoo Finance has no history for delisted tickers, so the 2021–2025 test
  set holds only firms still trading; test returns are biased upward and the test set is smaller than the market.
- **Missing range data:** 19% of IPOs have no initial filing range, so the strongest
  predictor is missing for them.
- **FinBERT on a sample:** FinBERT scores 16 sentences per section on a CPU, not whole sections.
- **Nigeria:** too few listings for modelling; pre-2014 prices come from news reports only.
- **Not investment advice:** first-day allocations go mostly to institutions, and the models' accuracy is modest.

## 9. Reproducing

```bash
.venv/Scripts/python src/pipeline.py       # data → features → models → figures → this report
```
