"""Metadata-only volume-breadth preflight. Never imports a source loader.
Raw files are SHA-256 streamed only; only explicitly listed metadata JSON is decoded.
Run through ./run_local.sh; creates one new report exclusively.
"""
from pathlib import Path
from datetime import date, datetime, timezone
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after248_volume_breadth_source_preflight.json'
PINS = {
 'work/after248_volume_breadth_definition_draft.json':'c95851e4eb2c1d523b27241c543fa9a5cd587b866d947596b708e33201671e61',
 'work/calendar_clock248_SOURCE_REVIEW.json':'6fa0eb06b79d1a250d16629b8351cb7a0fd647f52245148d31eab80429c0a19f',
 'work/volume195_SOURCE_REVIEW.json':'ea5d8c96d8c340cf01d6a6db4ec424a1379c03a6545446039821d2e17828fe40',
 'outputs/monthly_stability_monitor/inputs.py':'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',
 'outputs/index_union/price_units.py':'353e2f039d022a03cf90be06028d680b8dd389fad7bb216727cc2ecbda226291',
 'work/volume195_source_review.py':'1d7b0f6c8f20acf90e1453cf028d427e29a3ba5f2f514c3ed54b245be4bb1d64',
 'work/monthly_volume_pair203_kernel.py':'c658b539519766be1f8fe8d284c101bd5e3b6e96918633e15235f8051020c8e6',
 'work/after247_exact_calendar_source_preflight.json':'42245d83d1d0192905b2675d7c8e963af706f6fc66e4c7e83d23dba921451419',
}
METADATA = {
 'outputs/monthly_local_linear70/PLAN.json',
 'outputs/monthly_local_linear70/META.json',
 'outputs/monthly_paired_target143/CALENDAR.json',
 'outputs/index_union/alias_resolved_inputs/MANIFEST.json',
 'outputs/index_union/input_diagnostics/MANIFEST.json',
 'work/calendar_clock248_SOURCE_REVIEW.json',
 'work/volume195_SOURCE_REVIEW.json',
}

def local(path):
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    p = p.resolve()
    if ROOT not in p.parents: raise ValueError('Nonlocal path')
    return p

def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def metadata(path):
    p = local(path); name = str(p.relative_to(ROOT))
    if name not in METADATA: raise ValueError('Not allowlisted metadata: '+name)
    return json.loads(p.read_text())

def main():
    if OUT.exists(): raise ValueError('Write-once output already exists')
    for p,digest in PINS.items():
        if sha(p)!=digest: raise ValueError('Changed definition/source pin: '+p)
    source = metadata('work/calendar_clock248_SOURCE_REVIEW.json')
    prior = metadata('work/volume195_SOURCE_REVIEW.json')
    plan = metadata('outputs/monthly_local_linear70/PLAN.json')
    meta = metadata('outputs/monthly_local_linear70/META.json')
    cal = metadata(source['calendar_path']); sessions = cal['sessions']
    if len(set(sessions))!=len(sessions) or sessions!=sorted(sessions): raise ValueError('Calendar axis')
    for d in sessions:
        if date.fromisoformat(d).isoformat()!=d: raise ValueError('Canonical date')
    if sha(source['calendar_path'])!=source['calendar_sha256'] or sha(source['metadata_path'])!=source['metadata_sha256']: raise ValueError('Metadata binding')
    raw = {str(local(p)):h for p,h in plan['raw_source_hashes'].items()}
    if len(raw)!=64 or raw!={str(local(p)):h for p,h in source['raw_source_hashes'].items()}: raise ValueError('Original64 closure')
    if raw!={str(local(p)):h for p,h in prior['raw_source_hashes'].items()}: raise ValueError('Prior volume closure mismatch')
    raw_bindings=[]
    for p,digest in sorted(raw.items()):
        if sha(p)!=digest: raise ValueError('Changed raw source: '+p)
        raw_bindings.append(dict(path=str(Path(p).relative_to(ROOT)),sha256=digest,bytes=Path(p).stat().st_size,verified_by='byte hash only; no numeric decoding'))
    if len(meta['dates'])!=122 or len(meta['symbols'])!=122 or any(len(s)!=49 or len(set(s))!=49 for s in meta['symbols']): raise ValueError('Original monthly inventory')
    names=sorted({s for row in meta['symbols'] for s in row})
    lineage=prior['source_lineage']
    if not set(names)<=set(lineage): raise ValueError('Missing inherited metadata lineage')
    # Old source report coverage/axes/value summaries are intentionally not used.
    source_chains={s:lineage[s] for s in names}
    operations=Counter(z['operation'] for chain in source_chains.values() for z in chain)
    for chain in source_chains.values():
        for z in chain:
            if 'path' in z and 'sha256' in z and str(local(z['path'])) in raw:
                if z['sha256']!=raw[str(local(z['path']))]: raise ValueError('Lineage/raw hash mismatch')
    aliases=metadata('outputs/index_union/alias_resolved_inputs/MANIFEST.json')
    integrated=metadata('outputs/index_union/input_diagnostics/MANIFEST.json')
    volume_field=next(z for z in aliases['fields'] if z['field']=='volume')
    base_volume=local(Path('outputs/index_union')/volume_field['file'])
    if raw.get(str(base_volume))!=volume_field['sha256']: raise ValueError('Alias volume field closure')
    positions={d:i for i,d in enumerate(sessions)};bounds=source['listing_bounds'];bound_by={z['symbol']:z for z in bounds}
    expected_id={(z['query_index'],z['column_index']):z for z in source['identity_cells']}
    events=[];identity_cells=[]
    for i,(day,ss) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=positions[day];partial=i==0 and day=='2016-08-17'
        if t<356 or t+1>=len(sessions) or (not partial and sessions[t+1][:7]==day[:7]): raise ValueError('Query clock')
        p=sessions[t-294:t-41];v=sessions[t-356:t-41];returns=sessions[t-293:t-41]
        if (len(p),len(v),len(returns))!=(253,315,252): raise ValueError('Exact source lengths')
        event=dict(query_index=i,date=day,initial_partial=partial,numeric_evaluation_required=not partial,symbols=ss,query_position=t,next_official_session=sessions[t+1],
          price_start=p[0],price_end=p[-1],price_count=253,return_start=returns[0],return_end=returns[-1],return_count=252,
          volume_start=v[0],volume_end=v[-1],volume_count=315,volume_extra_leading_sessions=62,
          first_return_volume=returns[0],first_baseline_start=v[0],first_baseline_end=sessions[t-294],
          last_return_volume=returns[-1],last_baseline_start=sessions[t-105],last_baseline_end=sessions[t-43],
          block_price_endpoints=[p[k] for k in range(0,253,21)],
          actual_complete49_source_support='UNKNOWN_NOT_SCANNED',within_window_unit_continuity='UNKNOWN_NOT_CERTIFIED')
        events.append(event)
        for j,symbol in enumerate(ss):
            if symbol not in bound_by: continue
            b=bound_by[symbol];old=expected_id.get((i,j))
            if old is None or old['symbol']!=symbol or old['date']!=day or old['issuer_key']!=b['issuer_key']: raise ValueError('Listing identity mismatch')
            visible=b['available_from']<=day
            identity_cells.append(dict(query_index=i,column_index=j,date=day,symbol=symbol,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=visible,
              price_start=p[0],price_end=p[-1],volume_start=v[0],volume_end=v[-1],
              price_window_dates_before_bound=sum(d<b['not_before'] for d in p) if visible else None,
              volume_window_dates_before_bound=sum(d<b['not_before'] for d in v) if visible else None,
              price_numeric_status='UNKNOWN_NOT_READ',volume_numeric_status='UNKNOWN_NOT_READ',
              required_later_action='Independently inspect each consumed positive source observation for visible identity contradiction; ordinary missing causes common event fallback, but cannot hide another invalid/contradictory member.'))
    if len(identity_cells)!=len(expected_id): raise ValueError('Complete known identity inventory')
    input_sha={p:sha(p) for p in sorted(set(PINS)|METADATA)}
    input_sha[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    report=dict(schema='after248_volume_breadth_metadata_source_preflight_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      status='METADATA_HASH_CLOCK_VERIFIED_NUMERIC_AND_VOLUME_ADMISSION_UNKNOWN',metadata_only=True,source_admitted=False,strict_pit=False,new_strategy_registered=False,
      original64=dict(count=64,all_sha256_match=True,plan_path='outputs/monthly_local_linear70/PLAN.json',plan_sha256=sha('outputs/monthly_local_linear70/PLAN.json'),
        equals_frozen248_source_map=True,equals_inherited195_volume_source_map=True,source_bindings=raw_bindings,
        limitation='Only byte identity verified. Does not prove numeric field coverage, unit continuity, volume admission or current frozen incumbent replay.'),
      source_lineage=dict(origin='Only source_lineage metadata from the hash-bound195 source receipt, checked against original64 where applicable. No inherited actual coverage counts reused.',
        unique_inventory_symbols=len(names),query_cells=122*49,operations=dict(sorted(operations.items())),chains=source_chains,
        upstream_paths='Upstream vendor/action files recorded in chains are metadata leads; only files in the original64 map were rehashed. Other recorded upstream SHA values are not newly certified here.',
        base_volume_manifest=volume_field,alias_manifest_sha256=sha('outputs/index_union/alias_resolved_inputs/MANIFEST.json'),integrated_manifest_sha256=sha('outputs/index_union/input_diagnostics/MANIFEST.json')),
      source_classes=[
        dict(source='Original alias signal panel',price='Dividend-adjusted adj signal series.',volume='Signal volume; documented raw_to_split_adjusted uses rawV*subsequent_split_multiplier while signal OHLC=rawOHLC/multiplier.',evidence='outputs/index_union/price_units.py:36-59',limit='Requires correct complete split/action history; metadata/code does not certify its completeness.'),
        dict(source='Yahoo/ETF/foreign manifest overlays',price='Stored Adj Close.',volume='Stored CSV Volume copied directly with no loader conversion.',evidence='outputs/monthly_stability_monitor/inputs.py:16-25',limit='Do not assume one uniform share basis across providers or inherited sources.'),
        dict(source='Published44/Top35 CSV additions',price='Stored Adj Close.',volume='Stored CSV Volume.',evidence='outputs/monthly_stability_monitor/inputs.py:28-35',limit='Source-receipt byte identity does not certify event/action clock or free-float turnover.'),
        dict(source='Early2016 extension',price='Only leading adjusted-price gaps are multiplied by overlap ratio.',volume='Stored Volume copied for the same gaps without multiplying by price overlap ratio.',evidence='outputs/monthly_stability_monitor/inputs.py:37-48',limit='Price splice equality cannot establish matching volume units at the boundary.'),
        dict(source='CHL/PTR overrides',price='Tiingo adjClose.',volume='Tiingo volume, not adjVolume; dates>=2021-01-01 masked.',evidence='outputs/monthly_stability_monitor/inputs.py:49-55',limit='Raw-volume override differs from split-only signal base; ADR/provider history continuity not certified.')],
      calendar=dict(path=source['calendar_path'],sha256=source['calendar_sha256'],provenance=cal['provenance'],session_count=len(sessions),first_session=sessions[0],last_session=sessions[-1],
        query_events=122,formal_events=121,width=49,etfs=meta['etfs'],events=events,
        formal_price_start_min=min(z['price_start'] for z in events[1:]),formal_price_end_max=max(z['price_end'] for z in events[1:]),
        formal_volume_start_min=min(z['volume_start'] for z in events[1:]),formal_volume_end_max=max(z['volume_end'] for z in events[1:]),
        significance='Calendar existence proves date selection only. It does not prove any security has an observed price or volume on those dates.'),
      listing=dict(bounds=bounds,identity_cells=identity_cells,unknown_other_security_bounds=True,
        limitation='Known5 cells are metadata inventory, not observed-source contradictions or availability. No E is read. Check t-356 volume independently from t-294 price; do not use future available_from evidence early. Broader issuance/delist/recycled-symbol/ADR certification remains incomplete.'),
      proposed_before_registration_policy=dict(status='REQUIRES_ROOT_APPROVAL_BEFORE_ANY_ACTUAL_FEATURES',
        nonnegotiable_gates=['Bind original64 bytes and exact original loader; no source substitution or silent source repair.','Require truly observed exact official-session fields; loader reindex must never count an invented/missing row as observed.','Scan all49 before whole-event ordinary-missing fallback so other invalid/listing contradictions are not hidden.','Use independent price253 andvolume315 visible-identity checks.','Keep declared source and unit-transition eligibility unchanged after any feature/result inspection.'],
        known_transition_intercept='Create a predeclared metadata ledger of documented provider/field/ADR/unit splice effective dates and certified basis where evidence exists. If an applicable consumed315-volume window crosses a documented unresolved unit/identity discontinuity, mark the whole event unavailable before ranking; do not drop one stock, truncate the window, rescale from adj/close or use outcome-guided repair. A documented split within one consistently normalized source is not automatically a discontinuity; its basis must be evidenced.',
        unknown_transition_handling='Absence of a transition record is not proof of continuity. Root must explicitly choose before registration either strict admission requiring a complete relevant action/transition ledger (currently unavailable), or declared-assumption stored-volume research with UNKNOWN continuity retained in audit and no strictPIT/true-turnover claim. This preflight does not choose between them.',
        declared_assumption_option='If explicitly approved, test the archived mixed stored-volume proxy exactly as supplied, acknowledging that within-window supplier/split/ADR conventions may drive H. Constant positive scale invariance is only a mathematical property, not source admission. Known contradictions still hardfail; known unresolved transitions obey the fixed whole-event rule.',
        no_free_float='No historical shares/free-float denominator is introduced or inferred. No same-day money flow or buyer/seller attribution is claimed.'),
      remaining_blockers=['All5978 query cells and315-volume/253-price observations remain unscanned; common49 event support UNKNOWN.','Per-security source field/action completeness and window-internal unit/ADR/provider continuity not certified; root must lock strict versus declared-assumption policy.','The new metadata transition ledger/gate is a proposed requirement, not an existing implemented/account-wide mechanism.','Known new-listing cells need independent numeric/source identity checks without hiding contradictions behind whole-event fallback.','Draft all-aux .01 proportional funding needs separate independent budget/adapter approval; existing funded helper is not a drop-in implementation.','Exact historical novelty remains UNKNOWN for six missing older volume/liquidity originals.'],
      work_performed=dict(raw_files_sha_only=True,decoded_price_volume_rows=0,npz_decoded=0,source_loader_imported=False,source_loader_executed=False,actual_features_computed=0,actual_orders_computed=0,actual_returns_computed=0,predictive_models_fitted=0,network_requests=0,old_files_modified=0,registered_trials=0,protected_archive_rehash=False),
      input_sha256=input_sha)
    with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],path=str(OUT),sha256=sha(OUT),source_files=64,metadata_clock_events=122,known_listing_cells=len(identity_cells),raw_price_volume_decoded=False)))

if __name__=='__main__':main()
