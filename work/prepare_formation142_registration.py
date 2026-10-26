"""Bind finished synthetic checks and independent pre-reviews; no market data."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
logs={
 'features':('test_monthly_formation142_features.py','outputs/monthly_formation142_draft/FEATURE_TESTS_v3.log'),
 'selector':('test_monthly_formation142_selector.py','outputs/monthly_formation142_draft/SELECTOR_TESTS.log'),
 'runner':('test_monthly_formation142_runner.py','outputs/monthly_formation142_draft/RUNNER_TESTS_v3.log'),
 'prepare_review':('test_monthly_formation142_prepare_review.py','work/monthly_formation142_prepare_tests.log'),
 'outcome_review':('test_monthly_formation142_outcome_review.py','work/monthly_formation142_outcome_synthetic.log'),
 'registration':('test_register_monthly_formation142.py','outputs/monthly_formation142_draft/REGISTRATION_TESTS.log'),
 'completion':('test_complete_monthly_formation142.py','outputs/monthly_formation142_draft/COMPLETION_TESTS.log')}
for name in ('prepare','outcome'):
 p=ROOT/f'work/monthly_formation142_{name}_prereview.json';review=json.loads(p.read_text())
 assert review['passed'] is True and review['real_142_feature_or_return_calculations']==0
 for mapping in ('source_hashes','implementation_hashes','input_hashes'):
  for path,digest in review.get(mapping,{}).items():assert sha(ROOT/path)==digest,(name,path)
suites=[]
for name,(source,log) in logs.items():
 log=ROOT/log;text=log.read_text();matches=re.findall(r'Ran (\d+) tests? in ',text)
 assert matches and text.rstrip().endswith('OK'),name
 suites.append(dict(name=name,count=int(matches[-1]),passed=True,source=ref(ROOT/'work'/source),log=ref(log)))
value=dict(passed=True,synthetic_only=True,test_count=sum(s['count'] for s in suites),suites=suites,real_new_factor_or_return_calculations=0)
out=ROOT/'outputs/monthly_formation142';out.mkdir(exist_ok=True)
with (out/'TESTS.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out/'TESTS.json'),sha256=sha(out/'TESTS.json'),test_count=value['test_count'])))
