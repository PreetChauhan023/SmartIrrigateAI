"""Persistent local prediction logs and training-vs-serving drift monitoring."""
import csv,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'monitoring'/'predictions.csv'
FIELDS=['timestamp','model_version','soil_moisture','temperature','crop_age_months','rain_prob','et_mm','irrigate','duration_hours','confidence','latency_ms']

def log_prediction(version,inputs,pred,latency_ms):
    from app.features import et_estimate
    LOG.parent.mkdir(parents=True,exist_ok=True)
    row={'timestamp':datetime.now(timezone.utc).isoformat(),'model_version':version,'soil_moisture':inputs[0],
      'temperature':inputs[1],'crop_age_months':inputs[2],'rain_prob':inputs[3],
      'et_mm':et_estimate(inputs[1],inputs[2]),'irrigate':int(pred['irrigate']),
      'duration_hours':pred['duration_hours'],'confidence':pred['confidence'],'latency_ms':round(latency_ms,2)}
    with LOG.open('a',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if f.tell()==0:writer.writeheader()
        writer.writerow(row)

def read_logs():
    return pd.read_csv(LOG) if LOG.exists() and LOG.stat().st_size else pd.DataFrame(columns=FIELDS)

def psi(reference,current,bins=8):
    ref=np.asarray(reference,dtype=float);cur=np.asarray(current,dtype=float)
    ref=ref[np.isfinite(ref)];cur=cur[np.isfinite(cur)]
    if len(ref)<20 or len(cur)<20:return float('nan')
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3:return float('nan')
    edges[0]=-np.inf;edges[-1]=np.inf
    a=np.histogram(ref,bins=edges)[0]/len(ref);b=np.histogram(cur,bins=edges)[0]/len(cur)
    a=np.maximum(a,1e-5);b=np.maximum(b,1e-5)
    return float(np.sum((b-a)*np.log(b/a)))

def drift_report(logs,min_samples=20):
    path=ROOT/'models'/'versions'/'training_reference.csv'
    if not path.exists() or len(logs)<min_samples:return pd.DataFrame()
    train=pd.read_csv(path)
    rows=[]
    for feature in ['soil_moisture','temperature','crop_age_months','rain_prob','et_mm']:
        score=psi(train[feature],pd.to_numeric(logs[feature],errors='coerce').dropna())
        rows.append({'feature':feature,'psi':round(score,3) if math.isfinite(score) else None,
                     'status':'Insufficient variation' if not math.isfinite(score) else ('High drift' if score>=.25 else 'Moderate drift' if score>=.1 else 'Stable')})
    return pd.DataFrame(rows)
