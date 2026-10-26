"""Collect successful synthetic logs; distinguish reused144 factor proof."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_same_report145'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def main():
 names={
  'factor':('test_monthly_same_report144_factor.py',ROOT/'outputs/monthly_same_report144/FACTOR_TESTS.log'),
  'runner':('test_monthly_same_report145_runner.py',OUT/'RUNNER_TESTS_FROZEN.log'),
  'prepare_review':('test_monthly_same_report145_prepare_review.py',OUT/'PREPARE_REVIEW_TESTS.log'),
  'outcome_review':('test_monthly_same_report145_outcome_review.py',OUT/'OUTCOME_REVIEW_TESTS.log'),
  'registration':('test_register_monthly_same_report145.py',OUT/'REGISTRATION_TESTS_FROZEN.log'),
  'completion':('test_complete_monthly_same_report145.py',OUT/'COMPLETION_TESTS_FROZEN.log')}
 suites=[]
 for name,(source,log) in names.items():
  text=log.read_text();counts=re.findall(r'Ran (\d+) tests? in ',text)
  if len(counts)!=1 or not text.rstrip().endswith('OK'):raise ValueError('Missing single successful suite '+name)
  item=dict(name=name,count=int(counts[0]),passed=True,source=ref(ROOT/'work'/source),log=ref(log),executed_this_round=name!='factor')
  if name=='factor':item.update(reused_from_round=144,original_receipt=ref(ROOT/'outputs/monthly_same_report144/TESTS.json'),reason='Exact unchanged pure factor and test source; prior registered successful execution retained.')
  suites.append(item)
 result=dict(passed=True,synthetic_only=True,round=145,test_count=sum(x['count'] for x in suites),
  newly_executed_test_count=sum(x['count'] for x in suites if x['executed_this_round']),reused_test_count=sum(x['count'] for x in suites if not x['executed_this_round']),suites=suites)
 dest=OUT/'TESTS.json'
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(dest),sha256=sha(dest),tests=result['test_count'],new_tests=result['newly_executed_test_count'],reused=result['reused_test_count'])))
if __name__=='__main__':main()
