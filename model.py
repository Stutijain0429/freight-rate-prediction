"""Two-stage model: (1) robust linear model in log space that carries time trend / market effects
(extrapolates beyond the training date range), (2) gradient boosting on its residuals for
geography / distance / equipment non-linearities (no raw time features, so no extrapolation problem)."""
import numpy as np, pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from features import clean_and_engineer

T0 = pd.Timestamp("2025-01-01")
def lin_matrix(d, use_trend=True, use_daily=True):
    X = pd.DataFrame(index=d.index)
    X["ld"] = d.log_dist; X["ld2"] = d.log_dist**2
    X["flat"] = d.eq_Flatbed; X["reef"] = d.eq_Reefer
    X["w"] = d.weight_clean/1e4; X["w_miss"] = d.weight_missing
    X["mi"] = d.mi; X["q"] = d.quote_signal; X["qa"] = d.q_abs_dev; X["qa2"] = d.q_abs_dev**2
    if use_daily:
        X["dq"] = d.day_quote_signal
    for k in range(7): X[f"dow{k}"] = (d.dow == k).astype(float)
    if use_trend: X["trend"] = (d.date - T0).dt.days/100.0
    return X

GBM_FEATS = ["log_dist","weight_clean","weight_missing","mi","quote_signal","q_abs_dev","dow",
             "pickup_lat","pickup_lon","delivery_lat","delivery_lon","dlat","dlon",
             "eq_DryVan","eq_Flatbed","eq_Reefer"]

class FreightModel:
    def __init__(self, use_trend=True, use_gbm=True, outlier_cut=0.4, gbm_params=None, gbm_loss="absolute_error"):
        self.use_trend, self.use_gbm, self.cut = use_trend, use_gbm, outlier_cut
        self.gbm_params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                               l2_regularization=1.0, random_state=0); self.gbm_params.update(gbm_params or {})
        self.gbm_loss = gbm_loss
    def fit(self, d, y):
        X = lin_matrix(d, self.use_trend)
        self.cols = X.columns
        # pass 1: Huber fit on everything, flag gross label outliers (|resid| > cut in log space)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        Z = (X-self.mu)/self.sd
        h = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(Z, y)
        res = y - h.predict(Z)
        self.keep = res.abs() < self.cut
        r = Ridge(alpha=1e-3).fit(Z[self.keep], y[self.keep]); self.lin = r
        res2 = y - r.predict(Z)
        if self.use_gbm:
            self.gbm = HGB(loss=self.gbm_loss, **self.gbm_params).fit(d.loc[self.keep, GBM_FEATS], res2[self.keep])
        return self
    def predict_log(self, d):
        Z = (lin_matrix(d, self.use_trend)[self.cols]-self.mu)/self.sd
        p = self.lin.predict(Z)
        if self.use_gbm: p = p + self.gbm.predict(d[GBM_FEATS])
        return p
    def predict(self, d): return np.exp(self.predict_log(d))
