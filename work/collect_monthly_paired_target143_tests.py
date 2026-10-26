"""Collect existing successful synthetic test receipts, without market calculation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_paired_target143'
SUITES = [
 ('labels', 'after142_paired_target_labels_tests.py', 'TEST_LABELS.log'),
 ('model', 'test_monthly_paired_target143_model.py', 'TEST_MODEL.log'),
 ('runner', 'test_monthly_paired_target143_runner.py', 'TEST_RUNNER.log'),
 ('prepare_review', 'test_monthly_paired_target143_prepare_review.py', 'TEST_PREPARE_REVIEW.log'),
 ('outcome_review', 'test_monthly_paired_target143_outcome_review.py', 'TEST_OUTCOME_REVIEW.log'),
 ('completion', 'test_complete_monthly_paired_target143.py', 'TEST_COMPLETION.log'),
]
def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
rows=[]
for name, source, filename in SUITES:
    log=OUT/filename
    content=log.read_text()
    counts=re.findall(r'Ran (\d+) tests? in ', content)
    assert len(counts)==1 and content.rstrip().endswith('OK'), filename
    rows.append(dict(name=name,count=int(counts[0]),passed=True,source=ref(ROOT/'work'/source),log=ref(log)))
result=dict(passed=True,test_count=sum(r['count'] for r in rows),suites=rows,created_at=datetime.now(timezone.utc).isoformat(),synthetic_only=True,collector=ref(Path(__file__).resolve()))
with (OUT/'TESTS.json').open('x') as stream:
    json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(passed=True,test_count=result['test_count'],manifest=ref(OUT/'TESTS.json'))))
