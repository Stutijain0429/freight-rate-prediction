# Freight Rate Prediction – Solution

Predicts `posted_rate` for 12,000 validation loads (Nov–Dec 2025) from 48,000 labeled loads (Jan–Oct 2025).

## Run
```bash
python -m pip install -r requirements.txt
# put the provided files in data/: train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv
python train.py            # CV report -> fit -> validation_predictions.csv + outputs/december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
python make_report.py      # optional: regenerates figures + outputs/freight_rate_report.pdf
```

## Layout
- `features.py` – cleaning (negative/missing weight, missing market_index) + feature engineering
- `model.py` – two-stage model: robust linear model with time trend (extrapolates) + gradient boosting on residuals (geography)
- `train.py` – forward-chaining CV, final fit, predictions
- `exploration/` – scratch experiments used to make the modelling decisions
- `outputs/` – CV results, December predictions, report

## Approach (short)
- Validation is *after* the labeled period → forward-chaining CV (train on past, predict next 2 months), not random K-fold.
- ~1.4% of labels are gross outliers (×0.2–0.4 or ×2.3–5); they are excluded when fitting and the model predicts the conditional median (L1).
- Rates drift upward ≈8%/year beyond market_index; a linear trend term in stage 1 handles extrapolation into Nov–Dec.
- 8 validation cities are unseen in training → no city-ID features, only coordinates/distance; verified with a held-out-city test.
- Daily market aggregates come from feature columns only (no labels).

Rolling-origin CV (mean of 3 folds): MAE ≈ $98, MAPE 4.2%, median APE 1.5% (trimmed MAPE 1.85%).
