"""Generate SYNTHETIC sugarcane plot data for the irrigation advisory demo.

NOTE: This is NOT real KIAAR/GBL data. Labels come from simple agronomy-style
rules plus noise, so the model has something learnable. State this in your report.
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["soil_moisture", "temperature", "crop_age_months", "rain_prob", "et_mm"]


def crop_stage(age_months: float) -> str:
    if age_months < 3:
        return "germination"
    if age_months < 6:
        return "tillering"
    if age_months < 10:
        return "grand_growth"
    return "maturity"


def generate(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "soil_moisture": rng.uniform(8, 45, n),  # % volumetric
            "temperature": rng.uniform(18, 44, n),  # deg C
            "crop_age_months": rng.uniform(0.5, 12, n),
            "rain_prob": rng.uniform(0, 100, n),  # % in next 24h
        }
    )
    # evapotranspiration rises with temperature and with crop canopy (peaks in grand growth)
    canopy = np.interp(df.crop_age_months, [0.5, 3, 6, 9, 12], [0.4, 0.7, 1.0, 1.2, 0.8])
    df["et_mm"] = (0.18 * df.temperature * canopy + rng.normal(0, 0.3, n)).clip(0.5, 9)

    # threshold moves with crop stage: grand growth is most water hungry
    threshold = np.interp(df.crop_age_months, [0.5, 3, 6, 9, 12], [22, 25, 30, 32, 24])
    noise = rng.normal(0, 1.5, n)
    need = (df.soil_moisture + noise < threshold) & (df.rain_prob < 50)
    df["irrigate"] = need.astype(int)

    deficit = (threshold - df.soil_moisture).clip(lower=0)
    hours = 0.8 + 0.12 * deficit + 0.25 * df.et_mm + rng.normal(0, 0.15, n)
    df["duration_hours"] = np.where(need, hours.clip(0.5, 6), 0.0).round(2)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/irrigation_synthetic.csv")
    a = ap.parse_args()
    d = generate(a.n)
    d.to_csv(a.out, index=False)
    print(f"Wrote {len(d)} rows to {a.out}; irrigate rate = {d.irrigate.mean():.2f}")
