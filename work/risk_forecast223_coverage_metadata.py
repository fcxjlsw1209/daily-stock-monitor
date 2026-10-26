"""Round223 finite source/identity coverage metadata, never load price arrays."""
from pathlib import Path
from datetime import datetime, timezone, date
import json, hashlib
R=Path(__file__).resolve().parents[1]
OUT=R/'work/after222_risk_forecast_coverage.json'
PINS={}
def require(x,m):
    if not x:raise ValueError(m)
def path(p):
    p=Path(p);p=p if p.is_absolute() else R/p
    require(p.is_relative_to(R) and p.resolve().is_relative_to(R) and 'CloudStorage' not in str(p),'Nonlocal path')
    return p
def bind(p,expected=None):
    p=path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    v=h.hexdigest();require(expected is None or v==expected,'SHA mismatch '+str(p));PINS[str(p)]=v;return v
def read(p,expected=None):
    bind(p,expected);return json.loads(path(p).read_text())
require(not OUT.exists(),'Write-once report already exists')
lead=read('work/after222_next_research_note.json')
meta=read('outputs/monthly_local_linear70/META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
plan=read('outputs/monthly_local_linear70/PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read('outputs/monthly_local_linear70/INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
prior=read('work/idio_risk222_SOURCE_REVIEW.json','13c45f4009f5b580364f0584c3ad6427d323184fbd2f230d2854dc43d29a53f2')
raw=plan['raw_source_hashes'];require(raw==prior['raw_source_hashes'] and len(raw)==64,'Original64 map')
for p,h in raw.items():require(original.get(p)==h,'Original registered source missing');bind(p,h)
lineage=read('work/nested199_SOURCE_REVIEW.json','d12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911')
manifest=read('outputs/index_union/alias_resolved_inputs/MANIFEST.json')
dated=read('outputs/index_union/DATED_SYMBOL_IDENTITIES.json')
identity169=read('outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json','730cea9f8f58232ad7e9a81ad5e59afa267eb0c0edabce686215c220d9de1765')
calendar=read('outputs/monthly_paired_target143/CALENDAR.json','8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')
for p,h in calendar['provenance']['source_hashes'].items():bind(p,h)
ss=calendar['sessions'];require(ss==sorted(set(ss)) and len(ss)==7039,'Calendar identity')
for d in ss:require(date.fromisoformat(d).isoformat()==d,'ISO calendar')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Event axis')
require(meta['dates']==sorted(set(meta['dates'])) and all(len(r)==len(set(r))==49 for r in meta['symbols']),'Query width/date uniqueness')
u=sorted({s for r in meta['symbols'] for s in r});require(len(u)==90 and set(u)<=set(lineage['source_lineage']),'Canonical IDs and stored lineage')
explicit_quarantine=set('AGN LINTA LMCA QVCA STRZA GENZ NIHD KRFT KHC SOLS VSNT HONA EA research_equity_ERTS'.split())
records=[]
for sym in u:
    entries=lineage['source_lineage'][sym]
    proofs=[x for x in manifest['identity_proofs'] if x.get('symbol')==sym]
    records.append(dict(security_id=sym,is_theme17_etf=sym in meta['etfs'],base_manifest_issuer=manifest['issuers'].get(sym),base_identity_proof_basis=[x.get('basis') for x in proofs],source_operations=[{k:v for k,v in z.items() if k in ('operation','provider_symbol','path','sha256')} for z in entries],history_identity_certified=False))
by_month={}
for d in ss:by_month.setdefault(d[:7],[]).append(d)
pos={d:i for i,d in enumerate(ss)}
annual=[]
for year in sorted({int(d[:4]) for d in meta['dates']}):
    windows=[]
    for y in range(year-5,year):
        for m in range(1,13):
            month=f'{y:04d}-{m:02d}';days=by_month[month];j=pos[days[0]];prior_end=ss[j-1];prev=by_month[prior_end[:7]];q=pos[prev[0]]
            require(j>=64 and q>=1 and days[-1]<=f'{year-1:04d}-12-31','Historical full-month clock')
            windows.append(dict(realized_month=month,forecast_date=prior_end,forecast63_price_start=ss[j-64],forecast63_price_end=prior_end,forecast63_prices=64,previous_month=prior_end[:7],previous_month_price_start=ss[q-1],previous_month_price_end=prior_end,previous_month_returns=len(prev),realized_price_start=prior_end,realized_first_return_date=days[0],realized_price_end=days[-1],realized_returns=len(days)))
    require(len(windows)==60,'Five complete calendar years')
    annual.append(dict(query_year=year,cutoff=f'{year-1:04d}-12-31',first_realized_month=windows[0]['realized_month'],last_realized_month=windows[-1]['realized_month'],windows=windows))
source_texts=['outputs/monthly_return_vol_response70/features.py','work/monthly_own_error215_context.json','work/monthly_disagreement216_context.json','work/shell_pool_sensitivity168_identity.py','work/ptr169_rebuild.py','outputs/monthly_local_linear70/study.py','outputs/index_union/consolidate_security_aliases.py','outputs/index_union/research_scope.py','outputs/monthly_stability_monitor/inputs.py','outputs/monthly_path70_research/incumbent_adapter.py']
for p in source_texts:bind(p)
old=(R/source_texts[0]).read_text();require('lead = pearson(normalized[:-1], squared[1:])' in old and 'reverse = pearson(normalized[1:], squared[:-1])' in old,'Daily response definition')
unknown=[]
for name in ('monthly_garch70','monthly_ewvol70','monthly_session_variance70','monthly_squared_persistence70','monthly_variance_ratio70','monthly_year_risk70'):
    names=[f'outputs/{name}/PLAN.json',f'outputs/{name}/features.py',f'outputs/{name}/study.py',f'work/{name}.py']
    present=[p for p in names if (R/p).is_file()]
    require(not present,'Unexpected previously absent source; must inspect before coverage claim')
    unknown.append(dict(family=name,definition_status='UNKNOWN',bounded_paths_checked=names,reason='Completed family named in prior coverage; executable definition and PLAN not present at checked paths. No exhaustive novelty claim.'))
additional=[]
for p in ('outputs/monthly_published44_reconstruction/prices/SHEL.source.json','outputs/monthly_published44_reconstruction/prices/SKHY.source.json'):
    x=read(p);additional.append(dict(path=str(R/p),sha256=PINS[str(R/p)],metadata=x))
implementation=bind(Path(__file__))
report=dict(schema='risk_forecast223_finite_coverage_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_only=True,scope='Finite executable-definition and source/identity/calendar metadata review; not source-price coverage, not final strategy admission.',family_novelty=False,exact_novelty='NOT_ESTABLISHED_EXHAUSTIVELY',
 fixed_lead=dict(moment_definition='Uncentered arithmetic mean of squared daily simple returns; no ddof, centering, risk-free adjustment or annualization',candidate_forecast='Trailing63 returns ending previous month end',control_forecast='Previous complete natural-month daily-return second moment',realization='Complete following natural month including prior-month last close for first return',reliability='Negative mean squared symmetric error ((R-F)/(R+F))**2',annual_freeze='Prior December31; realized months JanuaryY-5 through DecemberY-1 inclusive',minimum_paired_months=18,common_support='Both forecasts and realization need all finite-positive source prices and both R+F denominators strictly positive; no filling or epsilon',integration_status='Root design: .01 existing auxiliary support replacement, original core unchanged, all states except initial partial; final context/source review pending'),
 known_coverage=[dict(name='monthly_return_vol_response70',path=source_texts[0],lines='89-147',definition='253 prices ->252 log returns; Pearson(r[:-1],r[1:] squared) versus reverse over251 adjacent-day pairs; .5 correlation effect, common support',difference='Daily lead/reverse association, not simple-return monthly second-moment forecast reliability or annual mature error freeze'),dict(name='monthly_own_error215',path=source_texts[1],definition='Own-security matured excess-return prediction errors; annual five-year18-month calibration; upper quantile versus median correction',difference='Expected centered/clipped log excess return errors, not future risk second-moment forecast errors'),dict(name='monthly_disagreement216',path=source_texts[2],definition='Expected excess-return error means conditioned on model disagreement; shrunk to unconditional error',difference='Return forecasts and disagreement-conditioned means, not risk forecast reliability'),dict(name='monthly_idio_risk222',path='work/idio_risk222_SOURCE_REVIEW.json',definition='Current63 centered OLS residual-risk denominator versus total-risk denominator',difference='Contemporaneous risk magnitude/core denominator, not mature monthly forecast error')],
 unknown_related_families=unknown,
 identity_policy=dict(key='Exact original canonical source column/security_id',no_new_ticker_alias_join=True,no_cross_column_issuer_join=True,no_historical_pool_membership_filter=True,requires_169_identity_metadata=False,reason_169='169 repeats membership/date/label-end/security string row identity for ML caches; it is not a permanent issuer identity master and would wrongly impose historical pool membership on own-price history.',source169_metadata_path='outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control/MONTHLY_META.json',source169_training_months=len(identity169['training']['dates']),base_alias_status=manifest['status'],conditional_continuity_assumption=True,strict_pit=False,full_historical_ticker_reuse_certificate=False,known_explicit_quarantine_overlap=sorted(set(u)&explicit_quarantine),current_SEC_corroboration_not_historical_certification=True,alias_column='research_equity_FB uses one chosen META provider history, not cross-company/ticker lookup; retrospective evidence dates not strict PIT',other_alias='BRK-B membership label maps to BRK.B; exact source ID thereafter',inherited_splice='Original2016 extension fills leading adjusted-price gaps only, scaled by first overlap; not a new223 fill, and not certified source-vintage identity',additional_source_metadata=additional,records=records),
 raw_source_count=64,raw_source_hashes=raw,raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata=dict(events=122,width=49,unique_ids=90,theme17=meta['etfs'],missing_base_issuer_non_etfs=[r['security_id'] for r in records if not r['is_theme17_etf'] and r['base_manifest_issuer'] is None]),calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',session_count=len(ss),annual_windows=annual,annual_cutoff_calendar_verified=True,actual_price_coverage_checked=False),
 limitations=['Current source column is a declared conditional research identity. Current SEC ticker correspondence and inherited vendor histories do not certify permanent security continuity over every earlier year.','No known explicit quarantine intersects the90 query IDs; this is not proof of no other ticker reuse or legal reorganization.','No price row was read; complete63-day/monthly source support or18 paired observations have not been established.','Adjusted prices, leading-gap normalization and retrospective alias decisions retain original assumptions and possible later corporate-action restatements.','The old GARCH/EWvol/sessionvariance/squaredpersistence executable definitions are unavailable at reviewed paths; exact overlap is UNKNOWN.'],
 input_sha256=dict(sorted(PINS.items())),implementation=dict(path=str(Path(__file__)),sha256=implementation),npz_decoded=False,actual_source_coverage_recomputed=False,new_features_computed=0,new_orders_computed=0,new_returns_computed=0,new_predictive_models_fit=0,new_trials_registered=0,original_sources_modified=False,observers_accessed=False)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=bind(OUT),implementation_sha256=implementation,pins=len(PINS)-1,unique_ids=len(u),unknown_base_issuer_non_etfs=report['metadata']['missing_base_issuer_non_etfs'],explicit_quarantine_overlap=report['identity_policy']['known_explicit_quarantine_overlap'],annual_years=len(annual),source_only=True)))
