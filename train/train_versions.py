"""Reproducible, leakage-free comparison on one shared synthetic split."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error
from data.generate_data import generate, FEATURES
from app.features import et_estimate

def train_all(n=3000,seed=42):
    df=generate(n,seed)
    # Ensure synthetic evaluation uses the same ET estimation available at inference.
    df['et_mm']=[et_estimate(t,a) for t,a in zip(df.temperature,df.crop_age_months)]
    ids_tr,ids_te=train_test_split(range(len(df)),test_size=.2,random_state=seed,stratify=df.irrigate)
    train,test=df.iloc[ids_tr],df.iloc[ids_te]
    candidates={
      'v1_decision_tree':(DecisionTreeClassifier(max_depth=7,random_state=seed),DecisionTreeRegressor(max_depth=7,random_state=seed)),
      'v2_random_forest':(RandomForestClassifier(n_estimators=100,max_depth=10,random_state=seed,n_jobs=-1),RandomForestRegressor(n_estimators=100,max_depth=10,random_state=seed,n_jobs=-1)),
      'v3_gradient_boosting':(GradientBoostingClassifier(random_state=seed),GradientBoostingRegressor(random_state=seed))}
    out=ROOT/'models'/'versions';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,(clf,reg) in candidates.items():
        clf.fit(train[FEATURES],train.irrigate)
        positive=train.irrigate==1
        reg.fit(train.loc[positive,FEATURES],train.loc[positive,'duration_hours'])
        predicted=clf.predict(test[FEATURES]);actual=test.irrigate
        positive_test=actual==1
        row={'version':name,'accuracy':round(accuracy_score(actual,predicted),4),
          'precision':round(precision_score(actual,predicted,zero_division=0),4),
          'recall':round(recall_score(actual,predicted,zero_division=0),4),
          'f1':round(f1_score(actual,predicted,zero_division=0),4),
          'duration_mae_hours':round(mean_absolute_error(test.loc[positive_test,'duration_hours'],reg.predict(test.loc[positive_test,FEATURES])),4),
          'train_rows':len(train),'test_rows':len(test),'synthetic':True}
        joblib.dump({'clf':clf,'reg':reg,'features':FEATURES,'version':name},out/f'{name}.joblib')
        results.append(row)
    (out/'comparison.json').write_text(json.dumps(results,indent=2))
    (out/'training_reference.csv').write_text(train[FEATURES].to_csv(index=False))
    (out/'test_reference.csv').write_text(test.to_csv(index=False))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=3000);p.add_argument('--seed',type=int,default=42);p.add_argument('--push',action='store_true');p.add_argument('--repo-id',default='');a=p.parse_args()
    results=train_all(a.n,a.seed);print(json.dumps(results,indent=2))
    if a.push:
        if not a.repo_id: p.error('--repo-id required for --push')
        from huggingface_hub import HfApi
        api=HfApi();api.create_repo(repo_id=a.repo_id,repo_type='model',exist_ok=True)
        for row in results:
            version=row['version'];path=ROOT/'models'/'versions'/f'{version}.joblib'
            api.upload_file(path_or_fileobj=str(path),path_in_repo=f'versions/{version}.joblib',repo_id=a.repo_id,repo_type='model',commit_message=f'Publish {version}')
        api.upload_file(path_or_fileobj=str(ROOT/'models'/'versions'/'comparison.json'),path_in_repo='versions/comparison.json',repo_id=a.repo_id,repo_type='model')
        print('Published versions to Hugging Face Hub')
