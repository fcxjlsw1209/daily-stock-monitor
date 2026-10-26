"""Collect existing successful synthetic logs before prospective registration."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_same_report144'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def main():
    names={'factor':('test_monthly_same_report144_factor.py','FACTOR_TESTS.log'),
           'runner':('test_monthly_same_report144_runner.py','RUNNER_TESTS.log'),
           'prepare_review':('test_monthly_same_report144_prepare_review.py','PREPARE_REVIEW_TESTS.log'),
           'outcome_review':('test_monthly_same_report144_outcome_review.py','OUTCOME_REVIEW_TESTS.log'),
           'registration':('test_register_monthly_same_report144.py','registration_tests.log'),
           'completion':('test_complete_monthly_same_report144.py','COMPLETION_TESTS.log')}
    suites=[]
    for name,(source,log) in names.items():
        p=OUT/log;text=p.read_text();counts=re.findall(r'Ran (\d+) tests? in ',text)
        if len(counts)!=1 or not text.rstrip().endswith('OK'):raise ValueError('Missing single successful suite '+name)
        suites.append(dict(name=name,count=int(counts[0]),passed=True,source=ref(ROOT/'work'/source),log=ref(p)))
    dest=OUT/'TESTS.json'
    with dest.open('x') as f:json.dump(dict(passed=True,synthetic_only=True,round=144,test_count=sum(x['count'] for x in suites),suites=suites),f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(dest),sha256=sha(dest),tests=sum(x['count'] for x in suites))))
if __name__=='__main__':main()
