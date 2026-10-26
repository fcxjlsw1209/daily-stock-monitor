"""Read-only admission receipt and interface verification; no factor calls."""
import hashlib
import json
from pathlib import Path
import monthly_profitability141_runner as runner
R=Path(__file__).resolve().parent.parent
D=R/'outputs/monthly_profitability141_issuer_admission'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'RECEIPT.json').read_text());d=json.loads((D/'DATASET.json').read_text())
runner.validate_source_admission(r,dataset_sha=sha(D/'DATASET.json'),
 meta_sha=sha(R/'outputs/monthly_local_linear70/META.json'),
 sessions_sha=sha(R/'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'),
 proposal_sha=sha(R/'work/monthly_profitability141_proposal.json'))
for p,h in r['source_hashes'].items():
 path=R/p
 assert path.resolve()==path and sha(path)==h,p
assert len(d['events'])==90
nonempty={s:len(e) for s,e in d['events'].items() if e}
assert nonempty=={'MSFT':12,'NVDA':12}
assert all(e['admitted'] and e['first_public_version_verified'] is False for es in d['events'].values() for e in es)
print(json.dumps({'receipt_interface_passed':True,'source_bindings_verified':len(r['source_hashes']),'nonempty_event_feeds':nonempty,'total_securities':len(d['events']),'dataset_sha256':sha(D/'DATASET.json'),'receipt_sha256':sha(D/'RECEIPT.json'),'no_factor_calls':True}))
