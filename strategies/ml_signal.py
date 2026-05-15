import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class MLSignalStrategy(BaseStrategy):
    """
    LightGBM classifier trained on lagged return features.
    Target: sign of next-day return. Output: P(up)*2-1 ∈ [-1, +1].
    No look-ahead: trained on [t-train_window, t-1], predicts t.
    """

    def __init__(self):
        super().__init__("ml_signal", STRATEGY_PARAMS["ml_signal"])

    def _build_features(self, returns: pd.Series) -> pd.DataFrame:
        feats = {f"ret_{lag}d": returns.shift(lag) for lag in self.params["feature_lookbacks"]}
        feats["vol_10d"]   = returns.rolling(10).std()
        feats["vol_21d"]   = returns.rolling(21).std()
        feats["skew_21d"]  = returns.rolling(21).skew()
        feats["vol_ratio"] = feats["vol_10d"] / feats["vol_21d"]
        return pd.DataFrame(feats)

    def generate_signals(self, close, returns, **_kwargs):
        p              = self.params
        tw             = p["train_window"]
        retrain_every  = p.get("retrain_every", 21)  # retrain monthly by default
        signals        = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for sym in close.columns:
            ret = returns[sym].dropna()
            if len(ret) < tw + 30:
                continue

            feats  = self._build_features(ret)
            # target[i] = 1 if ret[i+1] > 0: predict NEXT day's direction
            target = (ret.shift(-1) > 0).astype(int)

            model          = None
            last_trained_i = -1

            for i in range(tw, len(ret)):
                # Retrain whenever the model is stale or missing.
                # Training window is strictly [i-tw, i-1] — no future data.
                if model is None or (i - last_trained_i) >= retrain_every:
                    X_tr = feats.iloc[i - tw: i].dropna()
                    # Drop the last label row: target[i-1] = ret[i] which IS
                    # known at i, but we want the model to generalise, not fit
                    # on it directly. Drop rows where target is NaN (last bar).
                    y_tr = target.iloc[i - tw: i].loc[X_tr.index].dropna()
                    X_tr = X_tr.loc[y_tr.index]

                    if len(X_tr) >= 30:
                        try:
                            model = lgb.LGBMClassifier(
                                n_estimators  = p["n_estimators"] // 4,
                                max_depth     = p["max_depth"],
                                learning_rate = p["learning_rate"],
                                verbosity     = -1,
                                n_jobs        = 1,
                            )
                            model.fit(X_tr, y_tr)
                            last_trained_i = i
                        except Exception as e:
                            logger.debug(f"MLSignal retrain {sym} i={i}: {e}")
                            model = None

                if model is None:
                    continue

                # Predict on bar i using only data known at i (features are
                # all lagged — no same-bar target leakage).
                X_test = feats.iloc[[i]].dropna()
                if X_test.empty:
                    continue
                try:
                    prob = model.predict_proba(X_test)[0][1]
                    signals.iloc[i, close.columns.get_loc(sym)] = prob * 2 - 1
                except Exception:
                    pass

        return signals
