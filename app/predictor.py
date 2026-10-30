"""Load the irrigation model (from local file, or from the HF Hub if HF_MODEL_REPO is set)."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "models" / "irrigation_model.joblib"


@lru_cache(maxsize=1)
def load_bundle():
    repo = os.getenv("HF_MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "irrigation_model.joblib", revision=os.getenv("HF_MODEL_REVISION"))
    else:
        path = LOCAL
    return joblib.load(path)


def predict(soil_moisture, temperature, crop_age_months, rain_prob, et_mm=None) -> dict:
    b = load_bundle()
    if et_mm is None:  # rough ET estimate if the farmer doesn't supply one
        et_mm = round(0.18 * temperature * 0.9, 2)
    row = pd.DataFrame([{
        "soil_moisture": soil_moisture, "temperature": temperature,
        "crop_age_months": crop_age_months, "rain_prob": rain_prob, "et_mm": et_mm,
    }])[b["features"]]
    irrigate = bool(b["clf"].predict(row)[0])
    prob = float(b["clf"].predict_proba(row)[0][1])
    hours = float(round(max(b["reg"].predict(row)[0], 0.5), 1)) if irrigate else 0.0
    return {
        "irrigate": irrigate,
        "duration_hours": hours,
        "confidence": round(prob if irrigate else 1 - prob, 2),
        "rain_prob": rain_prob,
        "soil_moisture": soil_moisture,
        "model_version": b.get("version", "local"),
    }
