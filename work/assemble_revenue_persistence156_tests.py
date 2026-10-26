"""Bind six successful fresh-process synthetic test logs before registration.

Each suite must be run separately through run_local.sh.  This assembler does
not import their modules: the independent prepare suite explicitly verifies
that no producer module has been imported into its process.
"""
from pathlib import Path
import hashlib
import json
import re
import monthly_revenue_persistence156_lifecycle as lifecycle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_revenue_persistence156'


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    if (OUT/'PLAN.json').exists() or (OUT/'TESTS.json').exists():
        raise ValueError('Preserve existing registration and tests')
    OUT.mkdir(exist_ok=True)
    rows=[]
    for name,filename in lifecycle.SUITES.items():
        path=OUT/(name.upper()+'_FINAL_TESTS.log')
        log=path.read_text()
        matches=re.findall(r'^Ran (\d+) tests? in ',log,re.MULTILINE)
        if len(matches)!=1 or int(matches[0])<=0 or not log.rstrip().endswith('\nOK'):
            raise ValueError(name+' requires one complete successful unskipped test run')
        rows.append(dict(name=name,count=int(matches[0]),passed=True,source=ref(ROOT/'work'/filename),log=ref(path)))
    doc=dict(passed=True,synthetic_only=True,process_isolation='one fresh run_local.sh process per suite',
             test_count=sum(r['count'] for r in rows),suites=rows)
    lifecycle.validate_tests(doc,lifecycle.old.Reader(ROOT))
    path=OUT/'TESTS.json'
    with path.open('x') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(**ref(path),test_count=doc['test_count'])))


if __name__=='__main__':main()
