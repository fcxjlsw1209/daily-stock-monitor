"""Hash-verified reuse of207 source coverage; no new features, prices or returns."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'work/monthly_volume_stability208_context.json'
CTX_SHA='7d8e19ba13ce394168dbecb0a88b12f64d13ab5009ac3f7370ceaceefb2fa0cb'
PRIOR=ROOT/'work/downvolume207_SOURCE_REVIEW.json'
PRIOR_SHA='77218c4d8e4d2b0d9d64a17adb6370ced64b833a90493c02020e74bbaa1e9d04'
OUT=ROOT/'work/volume_stability208_SOURCE_REVIEW.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    require(not OUT.exists() and not (ROOT/'outputs/monthly_volume_stability208').exists(),'Writeonce before registration')
    pins={}
    def pin(path,digest=None):
        p=Path(path);p=(p if p.is_absolute() else ROOT/p).resolve()
        require(p.is_relative_to(ROOT) and p!=CENTRAL,'Only immutable local inputs')
        h=sha(p);require(digest is None or h==digest,'Changedsource '+str(p));require(str(p) not in pins or pins[str(p)]==h,'Conflictingpin');pins[str(p)]=h
    pin(__file__);pin(CTX,CTX_SHA);pin(PRIOR,PRIOR_SHA)
    ctx=read(CTX);require(ctx['round']==208 and ctx['integration']['dose']==-.025 and ctx['arms']==['incumbent_local67','down_volume_stability','all_volume_stability_control'],'Prospectivedefinition')
    require(all(ctx[k]==0 for k in ('pre_registration_actual_new_features','pre_registration_actual_orders','pre_registration_actual_returns')),'New208boundary')
    for p,h in ctx['evidence_sha256'].items():pin(p,h)
    old=read(PRIOR);require(old['passed'] and old['source_only'] and old['unresolved_blockers']==[] and old['strict_pit'] is False,'Original207certificate')
    for k in ('actual_features_computed','actual_down_flags_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed','actual_orders_computed','actual_returns_computed','actual_portfolio_returns_computed','observers_accessed'):
        require(old[k] is False,'Oldsource crossed boundary '+k)
    for mapping in (old['source_sha256'],old['raw_source_hashes']):
        for p,h in mapping.items():pin(p,h)
    raw=read(ROOT/'outputs/monthly_local_linear70/PLAN.json')['raw_source_hashes'];require(raw==old['raw_source_hashes'] and len(raw)==64,'Same64rawsource')
    c=old['source_window_coverage'];require(c['checked_cells']==5978 and c['common_available_cells']==5966 and c['unavailable_cells']==12,'Sourcecoverage')
    require(c['accepted_source_price_observations']==64*5966 and c['accepted_source_volume_observations']==63*5966,'Sourcecounts')
    require(ctx['formation']['price_count']==64 and ctx['formation']['volume_count']==63 and ctx['formation']['end_offset']==0,'Same sourcewindows')
    s=read(CENTRAL);require(s['current']=='monthly_down_volume207' and s['registered_queue']==[] and len(s['completed'])==205 and sum(x['trials'] for x in s['completed'])==1132,'Current207complete')
    result=dict(old)
    result.update(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),context_sha256=CTX_SHA,review_code_sha256=sha(__file__),source_sha256=pins,actual_source_coverage_computed=False,current_saved_source_coverage_reused=True,coverage_reuse=dict(path=str(PRIOR),sha256=PRIOR_SHA,original_computation_timestamp=old['created_utc'],all_registered_source_bytes_verified=True,exact_source_windows_and_domain_unchanged=True),observed_central_state=dict(completed_rounds=205,cumulative_trials=1132,registered_queue=[]),limitations=ctx['limitations']+['This receipt reuses exact207 sourcecoverage afterhash verification, does not claim freshvendorreconstruction or freshcoverage computation.','No208feature/delta/orders/returns computed. Prior207 featuresandoutcomesexist and informedthefixedreverse-signhypothesis.'],synthetic_source_checks_passed=0)
    for p,h in pins.items():require(sha(p)==h,'Changed during reuse review')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),reused_source_cells=5966,unavailable_cells=12,new_actual_features=0)))
if __name__=='__main__':main()
