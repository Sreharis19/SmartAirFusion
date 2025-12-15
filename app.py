# app.py
from typing import List, Optional, Dict, Any
import __main__

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random

# IMPORTANT: make SmartAirFusionDetector available as __main__.SmartAirFusionDetector
from detector_runtime import SmartAirFusionDetector
__main__.SmartAirFusionDetector = SmartAirFusionDetector

MODEL_PATH = "smartairfusion_detector.joblib"

# Load detector once at startup
try:
    detector = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model file '{MODEL_PATH}': {e}")

app = FastAPI(title="SmartAirFusion Leak Detector API", version="1.0")


# ---------- Schemas ----------
class Event(BaseModel):
    timestamp: str

    TP2: Optional[float] = None
    TP3: Optional[float] = None
    H1: Optional[float] = None
    DV_pressure: Optional[float] = None
    Reservoirs: Optional[float] = None
    Oil_temperature: Optional[float] = None
    Motor_current: Optional[float] = None

    COMP: Optional[int] = None
    DV_electric: Optional[int] = None
    DV_eletric: Optional[int] = None  # accept typo too
    Towers: Optional[int] = None
    MPG: Optional[int] = None
    LPS: Optional[int] = None
    Pressure_switch: Optional[int] = None
    Oil_level: Optional[int] = None

    Caudal_impulses: Optional[float] = None

    class Config:
        extra = "allow"  # don't crash if user sends extra keys


class PredictRequest(BaseModel):
    events: List[Event] = Field(default_factory=list)
    event: Optional[Event] = None
    leak_threshold: Optional[float] = None


class PredictResponse(BaseModel):
    flag: int
    status: str
    details: Optional[Dict[str, Any]] = None


# ---------- Helpers ----------
def summarize(out_df: pd.DataFrame) -> Dict[str, Any]:
    # robust: handle missing columns
    valve = int(out_df.get("valve_leak_flag", 0).max()) == 1
    net = int(out_df.get("network_leak_pred", 0).max()) == 1
    press = int(out_df.get("pressure_instability_flag", 0).max()) == 1

    # priority: valve -> network -> instability -> none
    if valve:
        return {"flag": 1, "status": "valve leak detected"}
    if net:
        return {"flag": 1, "status": "network leak detected"}
    if press:
        return {"flag": 1, "status": "pressure instability detected"}
    return {"flag": 0, "status": "no leak detected"}


def events_to_df(events: List[Event]) -> pd.DataFrame:
    # pydantic v1
    df = pd.DataFrame([e.dict() for e in events])

    if "timestamp" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'timestamp' in events.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # normalize typo DV_eletric -> DV_electric
    if "DV_eletric" in df.columns and "DV_electric" not in df.columns:
        df = df.rename(columns={"DV_eletric": "DV_electric"})
    elif "DV_eletric" in df.columns and "DV_electric" in df.columns:
        df["DV_electric"] = df["DV_electric"].fillna(df["DV_eletric"])

    return df


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Build events list
    if req.events:
        events = req.events
    elif req.event is not None:
        events = [req.event]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'events' (list) or 'event' (single).")

    df_in = events_to_df(events)

    # You can keep this guard small; the real guard is "out empty"
    if len(df_in) < 2:
        return PredictResponse(
            flag=0,
            status="insufficient data",
            details={"received_points": int(len(df_in)), "hint": "Send >=10 minutes of events."}
        )

    # threshold override (temporary)
    old_thr = getattr(detector, "leak_threshold", None)
    if req.leak_threshold is not None and old_thr is not None:
        detector.leak_threshold = float(req.leak_threshold)

    try:
        _, F_feat, out = detector.predict(df_in)
    except Exception as e:
        if old_thr is not None:
            detector.leak_threshold = old_thr
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # restore threshold
    if old_thr is not None:
        detector.leak_threshold = old_thr

    # Key fix: if features are empty OR out is empty -> return clean message
    if F_feat is None or len(F_feat) == 0 or out is None or len(out) == 0:
        return PredictResponse(
            flag=0,
            status="insufficient data",
            details={
                "received_points": int(len(df_in)),
                "hint": "Not enough history for rolling-window features. Send longer window (>= 10–15 minutes).",
                "n_feature_rows": int(0 if F_feat is None else len(F_feat))
            }
        )

    summary = summarize(out)

    # Optional debugging for n8n
    details = {
        "max_network_leak_score": float(out["network_leak_score"].max()) if "network_leak_score" in out.columns else None,
        "any_valve_leak": int(out.get("valve_leak_flag", 0).max()),
        "max_pressure_instability_score": float(out["pressure_instability_score"].max()) if "pressure_instability_score" in out.columns else None,
        "last_timestamp": str(out.index.max()),
        "n_feature_rows": int(out.shape[0]),
        "leak_threshold_used": float(old_thr) if old_thr is not None else None
    }

    return PredictResponse(flag=summary["flag"], status=summary["status"], details=details)

def _gen_mock_events(payload_type: int, minutes: int = 15) -> Dict[str, Any]:
    """
    payload_type: 0 = no leak, 1 = leak
    minutes: number of 1-min events (default 15)
    """
    now = datetime.now()
    start = now - timedelta(minutes=minutes - 1)

    events = []
    # baseline starting values
    tp2 = random.uniform(8.55, 8.70)
    tp3 = random.uniform(8.15, 8.30)
    res = random.uniform(8.05, 8.20)

    oil = random.uniform(77.5, 78.5)
    motor = random.uniform(6.7, 7.1)

    # leak injection starts halfway
    leak_start_idx = minutes // 2

    for i in range(minutes):
        ts = start + timedelta(minutes=i)

        # small noise
        tp2 += random.uniform(-0.02, 0.02)
        tp3 += random.uniform(-0.02, 0.02)
        res += random.uniform(-0.02, 0.02)

        oil += random.uniform(0.00, 0.05)
        motor += random.uniform(-0.05, 0.05)

        # digital defaults
        dv_electric = 1
        comp = 0
        towers = 1
        mpg = 0
        lps = 0
        pressure_switch = 1
        oil_level = 1

        caudal = random.randint(1, 2)
        dv_pressure = random.uniform(0.04, 0.06)
        h1 = random.uniform(0.88, 0.92)

        # --- Leak behavior ---
        if payload_type == 1 and i >= leak_start_idx:
            # Make a strong leak signature:
            # Reservoirs drops fast, TP3 stays high/rises, flow increases, motor increases
            res -= random.uniform(0.20, 0.55)          # strong drop
            tp3 += random.uniform(0.02, 0.08)          # stays high
            tp2 += random.uniform(0.02, 0.08)

            caudal = random.randint(3, 8)
            motor += random.uniform(0.15, 0.35)
            dv_pressure = random.uniform(0.05, 0.08)

            # some system activity changes
            mpg = 1
            if i >= leak_start_idx + 2:
                lps = 1

        # keep values within reasonable bounds
        tp2 = float(max(0.0, tp2))
        tp3 = float(max(0.0, tp3))
        res = float(max(0.0, res))

        events.append({
            "timestamp": ts.isoformat(timespec="seconds"),
            "TP2": round(tp2, 2),
            "TP3": round(tp3, 2),
            "H1": round(h1, 2),
            "DV_pressure": round(dv_pressure, 2),
            "Reservoirs": round(res, 2),
            "Oil_temperature": round(oil, 1),
            "Motor_current": round(motor, 1),
            "COMP": comp,
            "DV_electric": dv_electric,
            "Towers": towers,
            "MPG": mpg,
            "LPS": lps,
            "Pressure_switch": pressure_switch,
            "Oil_level": oil_level,
            "Caudal_impulses": caudal
        })

    note = "normal payload (no leak)" if payload_type == 0 else "leak payload (Reservoirs drop + flow/motor rise)"
    return {"type": payload_type, "events": events, "note": note}


@app.get("/mock")
def mock_payload(type: int = 0, minutes: int = 15):
    """
    GET /mock?type=0 -> no leak
    GET /mock?type=1 -> leak
    Optional: &minutes=15
    """
    if type not in (0, 1):
        raise HTTPException(status_code=400, detail="type must be 0 (no leak) or 1 (leak)")
    if minutes < 5 or minutes > 120:
        raise HTTPException(status_code=400, detail="minutes must be between 5 and 120")

    return _gen_mock_events(type, minutes)