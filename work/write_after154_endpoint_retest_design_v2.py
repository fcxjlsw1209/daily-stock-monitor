"""Save only a novelty-metadata correction; no financial computations."""
from pathlib import Path
import copy
import hashlib
import json

root = Path(__file__).resolve().parents[1]
original = root / 'work/after154_endpoint_retest_design.json'
evidence = root / 'work/after154_terminal_loss_novelty.json'
out = root / 'work/after154_endpoint_retest_design_v2.json'
original_sha = '94c169ef28d8fce57b9a43084beda4b538b70ac5ed698995caba64541a8e5e56'
evidence_sha = '95b3bde9d26bcd1c46659000c619cdd12c663951b33dce229643d3064e24af64'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(original) == original_sha
assert sha(evidence) == evidence_sha
before = json.loads(original.read_text())
new = copy.deepcopy(before)
new['novelty'].pop('missing_packages_reported')
new['novelty'].pop('historical_scope')
new['novelty']['unresolved_rounds'] = [32, 116]
new['novelty']['source'] = str(evidence.relative_to(root))
new['novelty']['source_sha256'] = evidence_sha
new['input_hashes'][str(evidence.relative_to(root))] = evidence_sha
new['input_hashes'][str(original.relative_to(root))] = original_sha
assert new['candidate_contract'] == before['candidate_contract']
assert {k:v for k,v in new.items() if k not in ('novelty','input_hashes')} == {
    k:v for k,v in before.items() if k not in ('novelty','input_hashes')}
with out.open('x') as handle:
    json.dump(new, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write('\n')
assert sha(original) == original_sha
assert json.loads(out.read_text())['candidate_contract'] == before['candidate_contract']
print(json.dumps({'path':str(out.relative_to(root)), 'sha256':sha(out),
    'unresolved_rounds':new['novelty']['unresolved_rounds'],
    'candidate_contract_unchanged':True, 'original_preserved':True}))
