from app.features import et_estimate
from app.monitoring import psi
from app.rag_demo import DemoRetriever,ask
from app.model_registry import available_versions,predict_version,comparison

def test_feature():
    assert et_estimate(30,7)>et_estimate(30,1)

def test_registry():
    assert len(comparison())==3
    for version in available_versions():
        p=predict_version(version,22,32,7,20)
        assert 0<=p['confidence']<=1
        assert p['duration_hours']>=0

def test_psi():
    assert psi(list(range(100)),list(range(100)))<0.01
    assert psi(list(range(100)),list(range(100,200)))>.25

def test_rag():
    r=ask('Why should corrections be verified?',retriever=DemoRetriever())
    assert r['sources']
    assert 'DEMO-' in r['answer']
