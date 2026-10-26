"""Collect saved synthetic-test evidence; no real feature or outcome calculation."""
from pathlib import Path
import re
import basic_shares159_host as h

root = Path(__file__).resolve().parents[1]
evidence = {}
tests = []
for key, minimum in [('kernel',21),('host',22),('runner',24),('independent',22)]:
    path = root/'work'/('basic_shares159_'+key+'_validation.log')
    text = path.read_text()
    count = re.search(r'Ran (\d+) tests? in ',text)
    h.require(count is not None and int(count.group(1)) >= minimum and text.rstrip().endswith('OK'),
              'Incomplete synthetic test evidence: '+str(path))
    h.require('FAILED' not in text and 'Traceback' not in text,'Failed synthetic test evidence')
    evidence[str(path)] = h.sha(path)
    tests.append(dict(suite=key,tests=int(count.group(1)),log=str(path),sha256=h.sha(path)))
receipt = dict(passed=True,synthetic_only=True,generated_utc=h.now(),tests=tests,
    implementation_hashes={str(p):h.sha(p) for p in h.validation_paths()},
    evidence_hashes=evidence,collector_sha256=h.sha(__file__))
h.write(root/'work/basic_shares159_registration_tests.json',receipt)
print({'passed':True,'synthetic_tests':sum(t['tests'] for t in tests)})
