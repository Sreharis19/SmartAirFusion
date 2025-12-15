# detector_runtime.py
import numpy as np
import pandas as pd


class SmartAirFusionDetector:
    def __init__(
        self,
        model,
        feature_cols,
        resample_freq="10s",
        window="10min",
        step_rows=6,
        leak_threshold=0.5,
        valve_flow_q95=None,
        valve_drop_q95=None,
        instab_median=None,
        instab_iqr=None,
        instab_threshold=None,
    ):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.resample_freq = resample_freq
        self.window = window
        self.step_rows = int(step_rows)

        self.leak_threshold = float(leak_threshold)

        self.valve_flow_q95 = valve_flow_q95
        self.valve_drop_q95 = valve_drop_q95

        self.instab_median = instab_median or {}
        self.instab_iqr = instab_iqr or {}
        self.instab_threshold = instab_threshold

    def _resample_and_derive(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        # Fix typo if present
        if "DV_eletric" in df.columns and "DV_electric" not in df.columns:
            df = df.rename(columns={"DV_eletric": "DV_electric"})

        analog = [c for c in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Motor_current", "Oil_temperature"] if c in df.columns]
        digital = [c for c in ["COMP", "DV_electric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level"] if c in df.columns]
        pulse = [c for c in ["Caudal_impulses"] if c in df.columns]

        # types
        for c in digital:
            df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0.5).astype(np.int8)

        for c in analog + pulse:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # resample
        a = df[analog].resample(self.resample_freq).mean().interpolate("time") if analog else pd.DataFrame(index=df.resample(self.resample_freq).mean().index)
        d = df[digital].resample(self.resample_freq).max().ffill().fillna(0).astype(np.int8) if digital else pd.DataFrame(index=a.index)
        p = df[pulse].resample(self.resample_freq).sum().fillna(0) if pulse else pd.DataFrame(index=a.index)

        x = pd.concat([a, d, p], axis=1)

        # derived discrepancy
        if "TP3" in x.columns and "Reservoirs" in x.columns:
            x["TP3_minus_Reservoirs"] = x["TP3"] - x["Reservoirs"]

        return x

    def _make_features(self, x: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=x.index)

        core = [c for c in ["TP2", "TP3", "Reservoirs", "DV_pressure", "TP3_minus_Reservoirs", "Motor_current", "Oil_temperature"]
                if c in x.columns]

        for c in core:
            r = x[c].rolling(self.window, min_periods=10)
            feats[f"{c}_mean"] = r.mean()
            feats[f"{c}_std"] = r.std()
            feats[f"{c}_min"] = r.min()
            feats[f"{c}_max"] = r.max()

            # slope over window
            win_rows = int(pd.Timedelta(self.window) / pd.Timedelta(self.resample_freq))
            secs = pd.Timedelta(self.window).total_seconds()
            feats[f"{c}_slope"] = (x[c] - x[c].shift(win_rows)) / secs

        if "Caudal_impulses" in x.columns:
            feats["Caudal_sum"] = x["Caudal_impulses"].rolling(self.window, min_periods=10).sum()

        for c in [c for c in ["DV_electric", "COMP", "MPG", "LPS"] if c in x.columns]:
            feats[f"{c}_on_frac"] = x[c].rolling(self.window, min_periods=10).mean()
            feats[f"{c}_toggles"] = x[c].diff().abs().rolling(self.window, min_periods=10).sum()

        # downsample
        feats = feats.iloc[::self.step_rows]

        # CRITICAL FIX: do not drop everything; fill instead
        feats = feats.ffill().fillna(0.0)

        # ensure training columns
        for col in self.feature_cols:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[self.feature_cols]

        return feats

    def predict(self, df_raw: pd.DataFrame):
        x = self._resample_and_derive(df_raw)
        F = self._make_features(x)

        # still empty? return empty out
        if F is None or len(F) == 0:
            out = pd.DataFrame(columns=[
                "network_leak_score", "network_leak_pred",
                "valve_leak_flag",
                "pressure_instability_score", "pressure_instability_flag"
            ])
            return x, F, out

        # score
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(F)[:, 1]
        else:
            s = self.model.decision_function(F)
            p = (s - s.min()) / (s.max() - s.min() + 1e-9)

        out = pd.DataFrame(index=F.index)
        out["network_leak_score"] = p
        out["network_leak_pred"] = (out["network_leak_score"] >= self.leak_threshold).astype(int)

        # valve heuristic
        if all(c in F.columns for c in ["DV_electric_on_frac", "Caudal_sum", "Reservoirs_slope"]) and \
           (self.valve_flow_q95 is not None) and (self.valve_drop_q95 is not None):
            dv_off = F["DV_electric_on_frac"] < 0.2
            high_flow = F["Caudal_sum"] > self.valve_flow_q95
            strong_drop = (-F["Reservoirs_slope"]) > self.valve_drop_q95
            out["valve_leak_flag"] = (dv_off & high_flow & strong_drop).astype(int)
        else:
            out["valve_leak_flag"] = 0

        # instability
        inst_feats = [c for c in ["TP3_std", "Reservoirs_std", "DV_electric_toggles", "MPG_toggles"] if c in F.columns]
        if inst_feats and (self.instab_threshold is not None):
            z_parts = []
            for c in inst_feats:
                med = self.instab_median.get(c, 0.0)
                iqr = self.instab_iqr.get(c, 1.0)
                z = (F[c] - med) / (iqr + 1e-9)
                z_parts.append(z.clip(-10, 10))
            instab_score = sum(z_parts) / len(z_parts)
            out["pressure_instability_score"] = instab_score
            out["pressure_instability_flag"] = (instab_score >= self.instab_threshold).astype(int)
        else:
            out["pressure_instability_score"] = 0.0
            out["pressure_instability_flag"] = 0

        return x, F, out
