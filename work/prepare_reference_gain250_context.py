"""Metadata-only prospective context; no bar, cache or strategy evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/monthly_reference_gain250_context.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert not OUT.exists(),'Context write-once'
    definition=ROOT/'work/after249_reference_gain250_definition_v2.json'
    assert sha(definition)=='acc0c601fe2b3d7b039c380715ae7f4137ecf7ca4acecd9612a3465cf1a641c0'
    d=read(definition);sourcepath=ROOT/'work/reference_gain250_SOURCE_CONTRACT.json';s=read(sourcepath)
    assert s['schema']=='monthly_reference_gain250_metadata_source_review_v1' and s['passed'] is True
    assert s['actual_cell_coverage']=='UNKNOWN_UNTIL_REGISTERED_PREPARE'
    central=read(ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    assert len(central['completed'])==245 and sum(x['trials'] for x in central['completed'])==1248 and central['registered_queue']==[]
    assert central['completed'][-1]['directory']=='monthly_volume_breadth249_v2'
    pins={}
    def bind(name,digest=None):
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        key=str(p.relative_to(ROOT));h=sha(p)
        if digest is not None:assert h==digest,key
        if key in pins:assert pins[key]==h,key
        pins[key]=h
    for name,h in d['input_sha256'].items():bind(name,h)
    for name,h in s['input_sha256'].items():bind(name,h)
    for path in [definition,sourcepath,Path(__file__),ROOT/'work/reference_gain250_DEFINITION_REVIEW.json',ROOT/'work/reference_gain250_MATH_REVIEW.json',ROOT/'work/reference_gain250_SOURCE_REVIEW.json']:
        bind(path)
    for name in ['reference_gain250_DEFINITION_REVIEW.json','reference_gain250_SOURCE_REVIEW.json']:
        r=read(ROOT/'work'/name);assert r['passed'] is True and r['unresolved_blockers']==[]
    context=dict(schema='monthly_reference_gain250_context_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis=d['hypothesis'],registered_trials=3,new_predictive_model_fits=0,arms=d['arms'],
        prospective_definition=d,definition_sha256=sha(definition),
        source=dict(review_path=str(sourcepath.relative_to(ROOT)),review_sha256=sha(sourcepath),panel_first_session=s['panel_first_session'],
            source_policy=s['source_policy'],price_count=253,volume_count=252,reference_volume_consumed=False,actual_new_window_support='UNKNOWN_UNTIL_REGISTERED_PREPARE'),
        exact_math=d['math'],funding=d['adapter'],selection=d['selection'],execution=d['execution'],validation=d['validation'],
        implementation_clarifications=dict(
            top_two='Top2 of funded pre-score among original eligible names stay locked by unchanged selector; not original unfunded top2.',
            eligible_counts='Existing selector:3 or more choose3 EW;2 chooseboth half;fewer than2 cash.',
            requested_episode='Active alpha*e0; inactive .05, where original cap yields e0. No alpha*.05 shortcut.',
            common_reference_support='Both raw-reference arms share identical253P/252V all49 support; equal-price control still falls back if requiredV unavailable.',
            source_fallback='All consumed cells scanned for hard errors before zero/missing/transition branches. Ref-day V is not a consumed cell.',
            exact_scaling='Only exact post-cast common multiples imply exact rescaling identity; arbitrary rounded transformations need not.',
            information_scope='Auxiliary skip42 invariants do not prohibit existing original current63 volatility computation.',
            normalization='G=1-R/P and P/R-1 are monotone equivalents for ranking on same positive domain; no new information claim.'),
        phase_order=[
            'Formal tests and independent code review then register immutable definition,source,code,input,runtime hashes before actualnew250 data/statistics.',
            'Verify original64 sources,27 arrays,original fullhistory core and original incumbent before new ranks or funded selection.',
            'Calculate both declared exactreferences on253P/252V; all49 source checks and exact incumbent fallback when inactive.',
            'Verify123cached prefixes and5physical prefixes with independent source,math,core and selection evidence.',
            'Use only first88 zero-fee outcomes and predetermined guards to freeze choice; then later34/full9paths/45slices/18traininggroups.',
            'Independent saved-result review,full21444protected audit,and append completed3trials; never mutate frozenobservers.'],
        coverage=d['coverage'],literature=d['literature'],
        limitations=d['limitations']+[
            'Candidate-minus-control equals negative price-volume covariance divided by meanV and referenceprice; this is historicalprice weighting, not causal investor cost or pure volume confirmation.',
            'Incumbent comparison combines auxiliary funding and newfactor; the equal-reference comparison is the matched weighting contrast.',
            'No250 availability masks or counts inherited from249; actual support must be checked in registeredprepare.',
            'Alltiedrank1/98 and rawzero remain active, so equal targets still fund oldauxiliaries and need not recoverincumbent.'],
        input_sha256=pins,builder_scope=dict(metadata_only=True,prices_loaded=False,volumes_loaded=False,npz_decoded=False,
            actual_features_orders_returns_computed=False,registered_strategy=False,central_modified=False,old_files_modified=False),
        previous_goal_turn_classification='progress',previous_goal_turn_evidence='249_v2 completed independent audit and closure;250 v1 prospective hypothesis recorded. User fee-clarification turn revalidated existing evidence only.')
    OUT.write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='CONTEXT_READY_NOT_REGISTERED',path=str(OUT.relative_to(ROOT)),sha256=sha(OUT),pins=len(pins))),flush=True)
if __name__=='__main__':main()
