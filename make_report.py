import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import clean_and_engineer
from model import FreightModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

t = clean_and_engineer(pd.read_csv("data/train_test.csv", parse_dates=["date"])); y = np.log(t.posted_rate)
# figure 1: label outliers (in-sample residual from final model) + monthly drift
m = FreightModel().fit(t, y); res = y - m.predict_log(t)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].hist(res.clip(-2, 2), bins=120, color="#064A56"); ax[0].set_yscale("log")
ax[0].set_title("Label residual (log) – 1.4% gross outliers", fontsize=9); ax[0].set_xlabel("log(actual/pred)")
rpm = (t.posted_rate/t.distance); mo = t.groupby(t.date.dt.to_period("M")).agg(rpm=("posted_rate", lambda s: np.nan))
mm = t.assign(rpm=rpm).groupby(t.date.dt.to_period("M")).agg(rpm=("rpm", "median"), mi=("market_index", "mean"))
ax[1].plot(mm.index.astype(str), mm.rpm, marker="o", color="#064A56", label="median $/mile"); ax[1].set_title("Monthly median $/mile and market_index", fontsize=9)
ax1b = ax[1].twinx(); ax1b.plot(mm.index.astype(str), mm.mi, color="#E07A1F", marker="s", label="market_index"); ax[1].tick_params(axis="x", rotation=60, labelsize=7)
tt = t[res.abs() < .4]; r2 = (np.log(tt.posted_rate)-0)  # trend plot: residual of model without trend
m0 = FreightModel(use_trend=False, use_gbm=False).fit(t, y); r0 = (y - m0.predict_log(t))[res.abs() < .4]
dm = r0.groupby(t.date[res.abs() < .4]).mean()
ax[2].plot(dm.index, dm.values, lw=.8, color="#064A56"); ax[2].set_title("Daily mean residual w/o time trend → drift", fontsize=9)
ax[2].tick_params(axis="x", rotation=60, labelsize=7)
plt.tight_layout(); plt.savefig("outputs/fig_eda.png", dpi=160); plt.close()
# figure 2: CV
cv = json.load(open("outputs/cv_results.json"))
fig, ax = plt.subplots(figsize=(6.5, 2.8)); names = list(cv); mae = [cv[k]["MAE"] for k in names]
ax.barh(range(len(names)), mae, color=["#9DAFB3"]*3+["#064A56"]); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("MAE ($), rolling-origin CV mean"); [ax.text(v+1, i, f"{v:.0f}", va="center", fontsize=8) for i, v in enumerate(mae)]
plt.tight_layout(); plt.savefig("outputs/fig_cv.png", dpi=160); plt.close()

S = getSampleStyleSheet(); B = ParagraphStyle("b", parent=S["BodyText"], fontSize=9.5, leading=13.2, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=S["Heading1"], fontSize=17, textColor=colors.HexColor("#064A56"))
H2 = ParagraphStyle("h2", parent=S["Heading2"], fontSize=12, textColor=colors.HexColor("#064A56"), spaceBefore=8)
bul = lambda s: Paragraph("• " + s, ParagraphStyle("bl", parent=B, leftIndent=12, firstLineIndent=-8, spaceAfter=2))
doc = SimpleDocTemplate("outputs/freight_rate_report.pdf", pagesize=letter, leftMargin=.75*inch, rightMargin=.75*inch, topMargin=.7*inch, bottomMargin=.7*inch)
E = []
E += [Paragraph("Freight Rate Prediction – Approach &amp; Validation Report", H1),
      Paragraph("Candidate: Stuti · Role: Machine Learning Engineer · Code: see repository (README.md → <font face='Courier'>python train.py</font>)", B)]
E += [Paragraph("1. Data and key findings", H2),
 bul("<b>Labeled data</b>: 48,000 loads, 2025-01-01 → 2025-10-31, 64 cities. <b>Validation set</b>: 12,000 loads, 2025-11-01 → 2025-12-31, i.e. strictly <i>after</i> the labeled period."),
 bul("Rate is essentially <b>distance × a per-mile rate</b> (corr 0.91 with distance); the per-mile rate depends on equipment (Dry Van &lt; Flatbed &lt; Reefer), short-haul premium, lane/city geography, and a market level."),
 bul("<b>market_index</b> is a day-level signal measured with row-level noise (sd≈0.025 around the daily mean). The daily means explain most of the day-to-day rate movement; <b>quote_signal</b> adds a smaller, U-shaped effect (rates are higher when it is far from ≈2.0)."),
 bul("After removing market effects, rates still <b>drift upward ≈ +8% over the year</b> (about 0.022% per day). This matters because validation lies beyond the training date range."),
 bul("Noise floor: for ≈98.6% of rows, the best model's log-error sd is ≈2%; the remaining ≈1.4% of labels are gross outliers (rate ×0.2–0.4 or ×2.3–5, unrelated to any feature).")]
E += [Image("outputs/fig_eda.png", width=7*inch, height=1.98*inch)]
E += [Paragraph("2. Data-quality issues and handling", H2)]
dq = [["Issue", "Evidence", "Handling"],
 ["Negative weights", "292 train / 145 val rows; magnitudes look like normal weights", "Use |weight| (sign error); keep a flag feature"],
 ["Missing weight", "300 train / 165 val (0.6–1.4%)", "Median impute + missing flag"],
 ["Missing market_index", "374 train / 249 val", "Impute with same-day mean of other loads (features only) + flag"],
 ["Weight capped at 47,500", "1,204 train rows exactly at the cap", "Left as is (tree/linear handle it)"],
 ["Gross label outliers", "≈1.4% of rows: ×0.2–0.4 or ×2.3–5 off the model", "Excluded from fitting (robust Huber pass flags |resid|&gt;0.4 in log space); L1 loss in GBM"],
 ["Unseen cities in validation", "8 cities (≈12% of val rows) never appear in training", "No city-ID features; lat/lon + distance only. Verified with held-out-city simulation"],
 ["Distance vs coordinates", "Coordinates are jittered/synthetic: haversine ≠ distance; 70-mile floor", "Trust `distance` (consistent per lane, ±2%); coordinates only used as geography features"],
 ["No duplicate IDs / duplicate rows", "checked", "none needed"]]
tb = Table([[Paragraph(c, ParagraphStyle("c", parent=B, fontSize=8, leading=10, spaceAfter=0)) for c in r] for r in dq], colWidths=[1.35*inch, 2.5*inch, 3.1*inch])
tb.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9E6E9")), ("GRID", (0,0), (-1,-1), .4, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP")]))
E += [tb]
E += [Paragraph("3. Train / validation split strategy", H2),
 Paragraph("The final validation set is <b>later in time</b> (Nov–Dec) than every labeled row, so a random K-fold split would be optimistic: it would let the model see the same days/market conditions it is tested on. I therefore used <b>forward-chaining (rolling-origin) cross-validation</b>: train on all data before a cutoff and predict the next two months (the same horizon as the real task), for cutoffs 1 May, 1 Jul and 1 Sep. Metrics are averaged across the three folds. Because 12% of validation rows involve cities absent from training, I also ran a <b>held-out-city check</b>: train without 8 random cities, test on Sep–Oct. Error on rows touching unseen cities (MAE ≈ $98, MdAPE 1.57%) was essentially the same as on seen cities (MAE ≈ $99, MdAPE 1.51%).", B),
 Paragraph("Raw labels (including outliers) are used for evaluation because the real validation labels will contain such outliers too; I also report a trimmed MAPE (dropping the worst 1.5% of rows) to show model quality on ‘normal’ loads.", B)]
rows = [["Model (rolling-origin CV mean)", "MAE $", "RMSE $", "MAPE %", "MdAPE %", "trimmed MAPE %"]]
for k, v in cv.items(): rows.append([k, f"{v['MAE']:.1f}", f"{v['RMSE']:.0f}", f"{v['MAPE']:.2f}", f"{v['MdAPE']:.2f}", f"{v['MAPE_excl_top1_5pct']:.2f}"])
tb2 = Table(rows, colWidths=[2.6*inch, .7*inch, .75*inch, .75*inch, .8*inch, 1.1*inch]); tb2.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9E6E9")), ("GRID", (0,0), (-1,-1), .4, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 8), ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#EEF5F6"))]))
E += [tb2, Spacer(1, 4), Image("outputs/fig_cv.png", width=4.3*inch, height=1.85*inch)]
E += [Paragraph("4. Model and reasoning", H2),
 Paragraph("<b>Two-stage model in log-rate space.</b> Stage 1 is a robust (Huber-screened, then Ridge) linear model on log-distance (+quadratic), equipment, weight, market_index, quote_signal terms, day-of-week and an explicit <b>linear time trend</b>. This stage carries the market level and drift and, crucially, <i>extrapolates</i> into Nov–Dec, which tree models cannot do (they flat-line at the last observed level). Stage 2 is a HistGradientBoosting regressor (L1 loss, 300 iterations, lr 0.05) fit on the stage-1 residuals using geography (lat/lon, deltas), distance, equipment and the other load features; it captures lane/region premiums and non-linear distance effects. No raw date feature is given to the GBM. Predictions are exp(log-rate), i.e. a conditional median, which is the right target under the heavy-tailed label noise (median is optimal for MAE/MAPE-type metrics).", B),
 Paragraph("<b>Why not a plain GBM?</b> Without the trend stage the validation-like folds show a systematic under-prediction of ≈2% (Sep–Oct), and MAE of 110 vs 98 for the final model. Model comparison above: adding the trend cuts MdAPE 3.3 → 2.4 (linear), and adding GBM on top 2.4 → 1.5, which is already close to the ≈1.4–2% noise floor.", B),
 Paragraph("<b>Daily aggregates for the validation/December period</b> (daily mean market_index, quote_signal) are computed only from the <i>feature</i> columns of validation.csv – no labels are used. For the December chart (one row per date), market_index and quote_signal for each date are the December daily means from validation.csv; weight, distance, equipment and lane are the fixed inputs.", B)]
E += [Paragraph("5. Fixed December prediction chart (produced by the provided score.py)", H2),
 Image("scorer_results/candidate_december.png", width=6.9*inch, height=3.0*inch),
 Paragraph("Predicted Lexington → Fort Wayne (Dry Van, 360 mi, 32,000 lb) rate ranges ≈ $823–$846 with a weekly cycle (mid-week higher, weekend lower) and slightly rising market level; this is in line with this lane’s Jan–Oct history (≈ $780–$930 monthly medians).", B),
 Paragraph("6. Limitations / what I would do next", H2),
 bul("The time trend is estimated from only 10 months and extrapolated 2 months; if the drift is really seasonal rather than linear, Nov–Dec error could be ≈1–2% higher. A longer history would let me separate seasonality from trend."),
 bul("Outlier rows (≈1.4%) are inherently unpredictable and dominate RMSE; a classifier cannot find them from the features (outlier rate is flat across equipment/distance bins). If the scoring metric is RMSE, expect a large irreducible component."),
 bul("Possible extensions: lane-level target encodings with shrinkage for frequently-seen lanes, quantile outputs for prediction intervals, and monitoring for drift in production.")]
doc.build(E)
