"""Metadata-only immutable training/query source validation; no arrays loaded."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
context=R/'work/monthly_identity_transfer206_context.json'
assert sha(context)=='29d7d503564aca5ac2ebe66a4180e519e93427e0497ba74ecebd15999ab7255b'
pins={}
def bind(name,h=None):
 p=Path(name);p=(p if p.is_absolute() else R/p).resolve()
 assert p.is_relative_to(R) or str(p)=='/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9'
 assert p!=R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
 actual=sha(p);assert h is None or actual==h,(str(p),'hash mismatch')
 assert str(p) not in pins or pins[str(p)]==actual
 pins[str(p)]=actual
prior=R/'outputs/monthly_local_year170/PLAN.json'
for p,h in read(prior)['input_sha256'].items():bind(p,h)
bind(prior);bind(context)
for p,h in read(context)['evidence_sha256'].items():bind(p,h)
loc=R/'outputs/monthly_local_linear70'
for name in ('PLAN.json','DESIGN.json','INPUT_HASHES.json','CACHE_HASHES.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json'):
 bind(loc/name)
raw=read(loc/'PLAN.json')['raw_source_hashes'];assert len(raw)==64
for p,h in raw.items():bind(p,h)
for p,h in read(loc/'PLAN.json')['hashes'].items():bind(p,h)
for p,h in read(loc/'PLAN.json')['source_hashes'].items():bind(p,h)
parity=read(R/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/PARITY.json')
assert parity['passed'] and parity['arm']=='control'
for k in ('dates','ends','event_dates','offsets','common_good'):
 assert parity['checks']['monthly'][k]['exact']
for k in ('train_plain','event_plain','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility'):
 assert parity['checks']['monthly'][k]['max_abs_error']==0
bind(Path(__file__).resolve())
out=R/'work/identity206_SOURCE_REVIEW.json'
receipt=dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),context_sha256=sha(context),source_sha256=pins,raw_source_hashes=raw,immutable_source_files=len(pins),source_kind='Original frozen four-target training cache and monthly query matrices; reuse exact hash-verified rebuild, actual parity repeated after registration.',prior_saved_training_rows=8820,prior_saved_query_rows=5978,prior_saved_feature_count=10,prior_parity_receipt_reverified=True,actual_features_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_source_arrays_loaded=False,new_data_acquired=False,strict_pit=False,limitation='Metadata/hash source review only. Original cached labels and proxy universe/vintages remain research assumptions; actual identity/value/cohort/maturity and independent model parity still required in registered prepare.')
with out.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=True,pins=len(pins),path=str(out),sha256=sha(out))))
