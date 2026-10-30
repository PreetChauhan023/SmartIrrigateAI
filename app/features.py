"""Shared deterministic feature transformation for training and inference."""
import numpy as np

def et_estimate(temperature, crop_age_months):
    canopy=float(np.interp(crop_age_months,[0.5,3,6,9,12],[0.4,0.7,1.0,1.2,0.8]))
    return round(float(np.clip(0.18*temperature*canopy,0.5,9)),3)
