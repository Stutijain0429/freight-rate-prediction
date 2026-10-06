import numpy as np, pandas as pd

def clean_and_engineer(df, ref_daily=None):
    """Return feature frame. Daily aggregates are computed from the feature columns of `df` itself
    (no labels), so they are available for any set of dates (incl. validation / December)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weight_missing"] = d["weight"].isna().astype(int)
    d["weight_was_negative"] = (d["weight"] < 0).astype(int)
    d["weight_clean"] = d["weight"].abs()            # negative values look like sign errors
    d["weight_clean"] = d["weight_clean"].fillna(d["weight_clean"].median())
    d["mi_missing"] = d["market_index"].isna().astype(int)
    daily = d.groupby("date")[["market_index", "quote_signal"]].mean().add_prefix("day_")
    daily["day_load_count"] = d.groupby("date").size()
    if ref_daily is not None:
        daily = ref_daily
    d = d.join(daily, on="date")
    d["mi"] = d["market_index"].fillna(d["day_market_index"])   # impute with same-day mean
    d["log_dist"] = np.log(d["distance"])
    d["dist_inv"] = 1.0 / d["distance"]
    d["dow"] = d["date"].dt.dayofweek
    d["mi_dev"] = d["mi"] - d["day_market_index"]
    d["q_dev"] = d["quote_signal"] - d["day_quote_signal"]
    d["q_abs_dev"] = (d["quote_signal"] - 2.05).abs()
    d["dlat"] = d["delivery_lat"] - d["pickup_lat"]
    d["dlon"] = d["delivery_lon"] - d["pickup_lon"]
    for e in ["Dry Van", "Flatbed", "Reefer"]:
        d["eq_" + e.replace(" ", "")] = (d["equipment"] == e).astype(int)
    return d

FEATURES = ["log_dist","distance","weight_clean","weight_missing","weight_was_negative","mi","mi_missing",
            "quote_signal","q_abs_dev","day_market_index","day_quote_signal","mi_dev","q_dev","dow",
            "pickup_lat","pickup_lon","delivery_lat","delivery_lon","dlat","dlon",
            "eq_DryVan","eq_Flatbed","eq_Reefer"]
