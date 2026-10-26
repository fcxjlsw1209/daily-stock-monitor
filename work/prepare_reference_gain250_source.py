"""Write-once250 source contract. Metadata/calendar and SHA only; no bars/NPZ.

The full-history original-core gate and new253/252 numeric support remain
UNKNOWN until a separately registered prepare. Prior249 support is not reused.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/reference_gain250_SOURCE_CONTRACT.json'
EXPECTED={
 'work/after249_reference_gain250_definition_v2.json':'acc0c601fe2b3d7b039c380715ae7f4137ecf7ca4acecd9612a3465cf1a641c0',
 'work/reference_gain250_DEFINITION_REVIEW.json':'fd670979c37db2074efa69f3d97dfa114781af22a1564365b11e8cabb05e8411',
 'work/reference_gain250_MATH_REVIEW.json':'892d3b62d597679e8294027741383e693a7c38f61daa16c8e10527509e242655',
 'work/reference_gain250_SOURCE_REVIEW.json':'42c3cd0f5cb2548a154c8f53f6aa1c9a2a830d58647ea24dfa3951aacd5785d1',
 'work/volume_breadth249_SOURCE_REVIEW.json':'b07b6342161f2e90767dc08d059b5710915998b93412ce27877d6e903f2b47ea',
 'work/volume_breadth249_ZERO_POLICY_REVIEW.json':'f87370db418f1f46aeafe77a298ce19a3891af63149024bae8dad929bdd600a1',
 'work/after248_volume_breadth_TRANSITION_LEDGER.json':'4b35ee2b539e83e4502741fbba2edefa41c874b2082d37f06f927f6098146db8'}
def path(name):
    p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve()
    assert p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts
    return p

def sha(name):
    h=hashlib.sha256()
    with path(name).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(name):return json.loads(path(name).read_text())
def bind(pins,name,expected=None):
    p=path(name);name=str(p.relative_to(ROOT));assert p.name!='CONTINUATION_STATUS.json','No mutable central pin'
    digest=sha(p);assert expected is None or digest==expected,'Changed source '+name
    assert name not in pins or pins[name]==digest
    pins[name]=digest;return digest

def main():
    assert not OUT.exists(),'Write-once source contract'
    pins={}
    for name,digest in EXPECTED.items():bind(pins,name,digest)
    prior=read('work/volume_breadth249_SOURCE_REVIEW.json')
    assert prior['metadata_only'] is True and prior['source_admitted'] is False
    for name,digest in prior['input_sha256'].items():bind(pins,name,digest)
    definition_review=read('work/reference_gain250_DEFINITION_REVIEW.json')
    assert definition_review['passed'] is True and definition_review['unresolved_blockers']==[]
    assert read('work/reference_gain250_MATH_REVIEW.json')['mathematical_blockers']==[]
    assert read('work/reference_gain250_SOURCE_REVIEW.json')['passed'] is True
    bind(pins,'work/prepare_reference_gain250_source.py')
    meta=read(prior['metadata_path']);calendar=read(prior['calendar_path'])['sessions'];lookup={d:i for i,d in enumerate(calendar)}
    assert len(lookup)==len(calendar) and calendar==sorted(calendar)
    assert len(meta['dates'])==len(meta['symbols'])==122
    assert all(len(x)==len(set(x))==49 for x in meta['symbols'])
    plan=read('outputs/monthly_local_linear70/PLAN.json');raw={str(path(k)):v for k,v in plan['raw_source_hashes'].items()}
    assert len(raw)==64 and raw==prior['raw_source_hashes']
    for k,v in raw.items():bind(pins,k,v)
    origin=read(prior['panel_origin_evidence']['path'])
    assert prior['panel_first_session']==origin['actual_origin']=='2008-01-02'
    assert origin['raw_sources_exact'] is True and origin['raw_sources']==64
    assert prior['manifest_reported_start']==read(prior['panel_manifest_path'])['start']=='2010-01-04'
    clocks=[]
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=lookup[day];partial=day=='2016-08-17'
        assert t>=294 and t+1<len(calendar) and (partial or calendar[t+1][:7]!=day[:7])
        prices=calendar[t-294:t-41];volumes=calendar[t-294:t-42]
        assert len(prices)==253 and len(volumes)==252 and prices[:-1]==volumes
        clocks.append(dict(query_index=i,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=deepcopy(names),query_position=t,next_official_session=calendar[t+1],
          price_start=prices[0],price_end=prices[-1],price_count=253,volume_start=volumes[0],volume_end=volumes[-1],volume_count=252,historical_count=252,
          historical_price_start=prices[0],historical_price_end=prices[-2],reference_date=prices[-1],reference_volume_consumed=False))
    bounds=deepcopy(prior['listing_bounds']);by={b['symbol']:b for b in bounds};identities=[]
    assert len(by)==len(bounds)==2
    for old in prior['identity_cells']:
        x=deepcopy(old);q=x['query_index'];j=x['column_index'];e=clocks[q];b=by[x['symbol']];t=e['query_position']
        assert e['symbols'][j]==x['symbol'] and e['date']==x['date'] and all(x[k]==b[k] for k in ('issuer_key','available_from','not_before'))
        x.update({k:e[k] for k in ('price_start','price_end','volume_start','volume_end')})
        x.update(bound_visible=b['available_from']<=e['date'],price_window_dates_before_bound=sum(d<b['not_before'] for d in calendar[t-294:t-41]),volume_window_dates_before_bound=sum(d<b['not_before'] for d in calendar[t-294:t-42]))
        # Descriptive metadata only, never an actual support certificate.
        x['source_admission']='UNKNOWN_UNTIL_REGISTERED_PREPARE';identities.append(x)
    assert len(identities)==5 and {(x['query_index'],x['column_index']) for x in identities}=={(i,j) for i,names in enumerate(meta['symbols']) for j,s in enumerate(names) if s in by}
    ledger=read('work/after248_volume_breadth_TRANSITION_LEDGER.json')
    assert ledger['automatic_known_unresolved_transition_fallback_ids']==[] and len(ledger['records'])==5 and len(ledger['undated_known_loader_rules'])==3
    source={k:deepcopy(prior[k]) for k in ('panel_first_session','panel_axis_end','panel_axis_rows','panel_origin_evidence','manifest_reported_start','manifest_reported_start_is_panel_origin','panel_manifest_path','panel_manifest_sha256','calendar_path','calendar_sha256','metadata_path','metadata_sha256','listing_scope','unknown_other_listing_bounds','source_classes','source_lineage','upstream_lineage_sha_scope','source_scope')}
    policy=deepcopy(prior['source_policy'])
    policy.update(actual_common49_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',zero_volume_rule='unresolved_zero_volume_whole_event_incumbent_fallback',
      ordinary_missing='Original adjusted-price/core structure must first pass. Ordinary required missing in253 prices or252 historical volumes makes both candidates copy incumbent for the entire event after all49 scans. Reference-day volume is excluded. No imputation or survivor ranking.',
      invalid_source='All required source slots scanned before fallback. Boolean/non-real/Inf/failed or overflowing binary64 casts, price<=0, volume<0 or nonzeroReal casting to0 hardfail. Only native numeric volume0 has explicit unresolved_zero_volume status, preserving signedzero and raw bytes.',
      identity='Query-visible same-issuer positive price or volume before not_before hardfails. Scan P t-294..t-42 and V t-294..t-43 separately; reference volume is never required. Zero cannot supply prelisting history.',
      no_short_circuit='Complete253-price and252-volume scans across all49 names before zero, missing or eligible-transition fallback.',
      fallback_branch_precedence=['initial_partial','zero_volume','missing_source','known_transition','available'],
      zero_economic_semantics='UNKNOWN: native stored zero is neither certified no-trade nor proven missing placeholder.',
      reference_volume='V[t-42] is outside auxiliary source validation/statistics/rank/zero/gap/listing/transition scope. Whole-file digest protection is unchanged and is not a claim of outcome invariance.',
      unit_continuity='UNKNOWN',mode='DECLARED_ASSUMPTION_ARCHIVED_MIXED_STORED_VOLUME',strict_pit=False)
    source.update(schema='monthly_reference_gain250_metadata_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='METADATA_REVIEW_COMPLETE_NUMERIC_ADMISSION_PENDING',
      passed=True,metadata_only=True,source_admitted=False,strict_pit=False,registered_strategy=False,prices_loaded=False,volumes_loaded=False,npz_decoded=False,actual_features_orders_returns_computed=False,
      raw_source_count=64,raw_source_hashes=raw,raw_source_verification='Streamed original64 files for byte SHA only; no market rows or NPZ decoded.',events=122,width=49,formal_monthends=121,etfs=deepcopy(meta['etfs']),
      clock_events=clocks,price_sessions=253,volume_sessions=252,historical_sessions=252,reference_volume_consumed=False,listing_bounds=bounds,identity_cells=identities,
      transition_records=deepcopy(ledger['records']),undated_loader_rules=deepcopy(ledger['undated_known_loader_rules']),automatic_known_unresolved_transition_fallback_ids=[],empty_transition_list_certifies_continuity=False,
      source_policy=policy,actual_cell_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',actual_unit_continuity='UNKNOWN_DECLARED_ASSUMPTION',
      metadata_review_scope='Only saved provenance, official dates, inventory and byte hashes. Previous249315-volume availability counts are not250252-volume coverage.',
      limitations=['Stored-volume weighting of dividend/split-adjusted price is a research proxy, not investor cost, VWAP, cash turnover or certified flow.','Constant exact-representable volume units cancel, time-varying unit continuity remains UNKNOWN.','Only the two explicitly documented listing bounds are enforced; no all-security issuer-vintage or strict PIT certification.','Reference-volume exclusion is auxiliary-only; original current63 risk/core and full archive hash checks retain frozen scope.'])
    for field,name in [('definition','work/after249_reference_gain250_definition_v2.json'),('definition_review','work/reference_gain250_DEFINITION_REVIEW.json'),('math_review','work/reference_gain250_MATH_REVIEW.json'),('source_review','work/reference_gain250_SOURCE_REVIEW.json'),('zero_policy','work/volume_breadth249_ZERO_POLICY_REVIEW.json'),('transition_ledger','work/after248_volume_breadth_TRANSITION_LEDGER.json')]:
        source[field+'_path']=name;source[field+'_sha256']=pins[name]
    source['input_sha256']=pins
    with OUT.open('x') as f:json.dump(source,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(OUT.relative_to(ROOT)),sha256=sha(OUT),raw_sources=64,events=122,width=49,identity_cells=5,source_admitted=False)),flush=True)
if __name__=='__main__':main()
