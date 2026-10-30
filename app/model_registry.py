"""Local-first model registry; optional remote HF Hub download by selected version."""
import json,os
from functools import lru_cache
from pathlib import Path
import joblib,pandas as pd
from app.features import et_estimate
ROOT=Path(__file__).resolve().parents[1]
VERSIONS=ROOT/'models'/'versions'

def available_versions():
    local=sorted(p.stem for p in VERSIONS.glob('*.joblib'))
    return local or ['legacy']

@lru_cache(maxsize=12)
def load_version(version):
    if version=='legacy':return joblib.load(ROOT/'models'/'irrigation_model.joblib')
    if version not in ('v1_decision_tree','v2_random_forest','v3_gradient_boosting'):raise ValueError('Unknown model version')
    repo=os.getenv('HF_MODEL_REPO')
    if repo and os.getenv('USE_HF_MODELS')=='1':
        from huggingface_hub import hf_hub_download
        path=hf_hub_download(repo_id=repo,filename=f'versions/{version}.joblib',revision=os.getenv('HF_MODEL_REVISION'))
    else:path=VERSIONS/f'{version}.joblib'
    return joblib.load(path)  # Only load trusted model artifacts; joblib can execute code.

def predict_version(version,soil,temperature,age,rain):
    b=load_version(version)
    row=pd.DataFrame([{'soil_moisture':soil,'temperature':temperature,'crop_age_months':age,'rain_prob':rain,'et_mm':et_estimate(temperature,age)}])[b['features']]
    yes=bool(b['clf'].predict(row)[0]);prob=float(b['clf'].predict_proba(row)[0,1]);duration=max(0.5,float(b['reg'].predict(row)[0])) if yes else 0.
    return {'irrigate':yes,'duration_hours':round(duration,1),'confidence':round(prob if yes else 1-prob,3),'rain_prob':rain,'soil_moisture':soil,'model_version':version}

def comparison():
    p=VERSIONS/'comparison.json';return json.loads(p.read_text()) if p.exists() else []
