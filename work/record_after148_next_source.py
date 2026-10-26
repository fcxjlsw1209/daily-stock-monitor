"""Record a source-only follow-up after independently completed round148."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import os
import tempfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
review=ROOT/'work/after148_next_mechanism_review.json'
assert sha(review)=='dc4237f5c4382f40800bb8cbb2b8268c069e80ba10703fac60d860408f70c1de'
done=ROOT/'outputs/monthly_session_cluster148/COMPLETION.json'
assert sha(done)=='3d8405758c531ee4c9c715a84dc8c71792e9c55b63a057d0e9ab46b13d3aeb03'
state=json.loads((ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json').read_text())
assert len(state['completed'])==148 and sum(r['trials'] for r in state['completed'])==946
assert state['registered_queue']==[]
p=ROOT/'migration/RESEARCH_RESUME.json'
raw=p.read_bytes(); prior=hashlib.sha256(raw).hexdigest(); resume=json.loads(raw)
assert resume['round']==148 and resume['target_70_reached'] is False
backup=ROOT/'migration'/('after148_source_resume_'+prior+'.json')
with backup.open('xb') as f: f.write(raw)
resume.update(next_candidate_review=dict(path=str(review.relative_to(ROOT)),sha256=sha(review),
 status='source_only_hypothesis_not_registered',new_field_admitted=False,new_strategy_trials=0),
 next_research_status='round148_complete_next_source_census_pending',
 next_step='Under standing research authorization, prepare a separate source-only plan to inspect at most44 current annual stock-compensation reconciliation cells from existing official reports; preserve all13 existing holds, audit exact period/unit/scope/availability, and do not calculate new factors or returns before separate strategy registration.',
 source_followup_authorization='The existing user-authorized research scope covers local source preparation; the review wording about separate authorization means a distinct registered source step, not a new user approval requirement.',
 next_mechanism_registered=False,updated_utc=datetime.now(timezone.utc).isoformat(),
 next_source_resume_checkpoint=str(backup.relative_to(ROOT)))
with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:
 staging=Path(f.name);f.write((json.dumps(resume,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode())
try:
 assert sha(p)==prior,'Concurrent resume update'
 os.replace(staging,p)
finally:
 if staging.exists():staging.unlink()
print(json.dumps(dict(round=resume['round'],next_research_status=resume['next_research_status'],resume_sha256=sha(p))))
