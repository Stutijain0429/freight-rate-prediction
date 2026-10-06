"""End-to-end run: CV report -> fit on all labeled data -> validation + December predictions."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from features import clean_and_engineer
from model import FreightModel

D = Path("data"); OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
train = pd.read_csv(D/"train_test.csv", parse_dates=["date"])
val = pd.read_csv(D/"validation.csv", parse_dates=["date"])
dec = pd.read_csv(D/"december_chart_inputs.csv", parse_dates=["date"])
tmpl = pd.read_csv(D/"validation_predictions_template.csv")

tr = clean_and_engineer(train); y = np.log(tr.posted_rate)

def metrics(yt, p):
    e = np.abs(p-yt)/yt
    return dict(MAE=float(np.mean(np.abs(p-yt))), RMSE=float(np.sqrt(np.mean((p-yt)**2))), MAPE=float(100*e.mean()),
                MdAPE=float(100*np.median(e)), MAPE_excl_top1_5pct=float(100*e[e < np.quantile(e, .985)].mean()))

# ---- 1. rolling-origin (forward-chaining) CV: always train on the past, predict the next 2 months ----
folds = [("2025-05-01","2025-07-01"),("2025-07-01","2025-09-01"),("2025-09-01","2025-11-01")]
variants = {"A linear (no trend)": dict(use_trend=False, use_gbm=False),
            "B linear + trend": dict(use_trend=True, use_gbm=False),
            "C trend-free linear + GBM": dict(use_trend=False, use_gbm=True),
            "D linear + trend + GBM (final)": dict(use_trend=True, use_gbm=True)}
cv = {}
for name, kw in variants.items():
    rows = []
    for a, b in folds:
        f_tr, f_te = tr[tr.date < a], tr[(tr.date >= a) & (tr.date < b)]
        m = FreightModel(**kw).fit(f_tr, np.log(f_tr.posted_rate))
        rows.append(metrics(f_te.posted_rate.values, m.predict(f_te)))
    cv[name] = pd.DataFrame(rows).mean().round(3).to_dict()
    print(f"{name:34s}", cv[name])
json.dump(cv, open(OUT/"cv_results.json", "w"), indent=2)

# ---- 2. final fit on all labeled data ----
final = FreightModel().fit(tr, y)
print(f"gross label outliers excluded from fit: {(~final.keep).sum()} ({(~final.keep).mean():.2%})")

# ---- 3. validation predictions (daily market aggregates come from validation feature columns only) ----
va = clean_and_engineer(val)
pred = pd.DataFrame({"load_id": val.load_id, "predicted_rate": np.round(final.predict(va), 2)})
pred = tmpl[["load_id"]].merge(pred, on="load_id", how="left")
assert pred.predicted_rate.notna().all() and len(pred) == 12000
pred.to_csv("validation_predictions.csv", index=False)

# ---- 4. December chart inputs: only the date varies; daily market_index / quote_signal come from
#         the December rows of validation.csv (feature columns, no labels) ----
coords = pd.concat([val[["pickup","pickup_lat","pickup_lon"]].set_axis(["c","lat","lon"], axis=1),
                    val[["delivery","delivery_lat","delivery_lon"]].set_axis(["c","lat","lon"], axis=1),
                    train[["pickup","pickup_lat","pickup_lon"]].set_axis(["c","lat","lon"], axis=1)]).drop_duplicates("c").set_index("c")
dd = dec.copy()
daily = va.groupby("date")[["market_index","quote_signal"]].mean()
dd["market_index"] = dd.date.map(daily.market_index); dd["quote_signal"] = dd.date.map(daily.quote_signal)
for side in ("pickup","delivery"):
    dd[f"{side}_lat"] = dd[side].map(coords.lat); dd[f"{side}_lon"] = dd[side].map(coords.lon)
dd = clean_and_engineer(dd)   # one row per date, so the 'daily mean' equals that date's value
dec["predicted_rate"] = np.round(final.predict(dd), 2)
dec["date"] = dec.date.dt.strftime("%Y-%m-%d")
dec.to_csv(OUT/"december_predictions.csv", index=False)
print(dec[["date","predicted_rate"]].describe())
