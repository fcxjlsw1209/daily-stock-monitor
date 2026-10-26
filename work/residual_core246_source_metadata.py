"""Round246 metadata/hash-only source-clock review; run through run_local.sh."""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/residual_core246_SOURCE_REVIEW.json'
CONTEXT='work/monthly_residual_core246_context.json'
CONTEXT_SHA='1c8760a3a498bf169d621ea1d3be05024ff789729ed9e997c52c1e47441a05ef'
COVERAGE='work/after245_design_coverage.json'
COVERAGE_SHA='688f4c9f852b17229a372a83768c211a0cc081309024252f9bb3d1b94ef3d5ab'
PINS={}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def path(name):
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    require(p.is_relative_to(ROOT) and p.resolve().is_relative_to(ROOT) and 'CloudStorage' not in str(p),'Local source only')
    return p
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(name,expected=None):
    p=path(name);h=sha(p);require(expected is None or expected==h,'SHA mismatch '+str(p))
    key=str(p.relative_to(ROOT));require(key not in PINS or PINS[key]==h,'Conflicting pin')
    PINS[key]=h;return h
def read(name,expected=None):
    bind(name,expected);return json.loads(path(name).read_text())
def iso(s):
    require(type(s) is str and date.fromisoformat(s).isoformat()==s,'Canonical date');return s

require(not OUT.exists(),'Write-once source review')
c=read(CONTEXT,CONTEXT_SHA);coverage=read(COVERAGE,COVERAGE_SHA)
require(c['schema']=='monthly_residual_core246_context_v1' and c['integration']['blend']==.5,'Fixed context')
for name,h in c['input_sha256'].items():bind(name,h)
for name,h in coverage['input_sha256'].items():bind(name,h)
local='outputs/monthly_local_linear70/'
p=read(local+'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read(local+'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
prior=read('work/risk_consensus219_SOURCE_REVIEW.json','cbf98df0568cdfb67260e84b8126c72cf3b981ceee94ade50a34074d8cb01b5b')
raw=p['raw_source_hashes'];require(len(raw)==64 and raw==prior['raw_source_hashes'],'Original64 lineage')
for name,h in raw.items():
    require(original.get(name)==h,'Original source registry');bind(name,h)
meta=read(local+'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Original122 axis')
require(meta['dates']==sorted(set(meta['dates'])) and meta['dates'][0]=='2016-08-17','Original unique ordered dates')
require(len(set(meta['etfs']))==len(meta['etfs'])==17,'Original17ETF')
require(all(len(s)==len(set(s))==49 for s in meta['symbols']),'Original49 query names')
for s in meta['dates']+meta['ends']:iso(s)
cache=read(local+'CACHE_HASHES.json')
for name,h in cache.items():bind(local+name,h)
cal=read('outputs/monthly_paired_target143/CALENDAR.json','8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')
for name,h in cal['provenance']['source_hashes'].items():bind(name,h)
sessions=cal['sessions'];require(sessions==sorted(set(sessions)),'Official session ordering')
positions={iso(d):i for i,d in enumerate(sessions)};windows=[]
for day in meta['dates']:
    require(day in positions,'Query in official calendar');t=positions[day]
    require(t>=546 and t+1<len(sessions),'Source calendar range')
    require(day=='2016-08-17' or day[:7]!=sessions[t+1][:7],'Formal month-end')
    window=sessions[t-546:t-41];require(len(window)==505,'Full505 official sessions')
    windows.append(dict(date=day,price_start=window[0],first_return_date=window[1],price_end=window[-1]))
require([(x['date'],x['price_end']) for x in windows]==[(x['date'],x['price_end']) for x in prior['calendar']['windows']],'Same query and skip42 end, newly extended start')
policy=read('work/peer239_maturity_policy.json','3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862')
evidence=read('work/peer239_listing_gap_evidence.json',policy['evidence']['sha256'])
source=read('outputs/monthly_peer239_source_audit/REPORT.json','dd7c7f39ae529fc62ec15c82e7f736fe6f5667c76a10bfb4e5b0ef1fc5148b7c')
mature=read('outputs/monthly_peer239_maturity_audit/REPORT.json','5138025ced5531b6d3447cc9bb9835e886b3f45157b6b4e03a54e888fcc97f7c')
close=read('outputs/monthly_peer239_maturity_audit/COMPLETION.json','fdeaf5afdd7a2d2823d0ee66bcc00a13b4d767c88b1eded2f9b575f1f17b2dc8')
require(close['source_only'] is True and close['strict_pit'] is False,'Prior source scope')
for field,name in (('plan_sha256','SOURCE_PLAN.json'),('report_sha256','REPORT.json'),('independent_review_sha256','INDEPENDENT_REVIEW.json')):
    bind('outputs/monthly_peer239_maturity_audit/'+name,close[field])
ind=read('outputs/monthly_peer239_maturity_audit/INDEPENDENT_REVIEW.json')
require(ind['passed'] is True and [x['prefix'] for x in ind['physical_prefixes']]==[0,1,53,88,122],'Prior metadata independent receipt')
require(policy['rules']['source_dates']=='official t-252 and t-42','Do not reuse short source mask')
expected=[dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='spcx_nasdaq_new_issue'),dict(symbol='SKHY',issuer_key='https://companiesmarketcap.com/sk-hynix/marketcap/',not_before='2026-07-10',available_from='2026-07-11',kind='issuer_actual_adr_start',evidence_id='skhy_adr_listing')]
require(policy['securities']==expected,'Exact reviewed bounds')
primary={x['id']:x for x in evidence['primary_evidence']};listing_evidence=[]
for b in expected:
    e=primary[b['evidence_id']];require(iso(e['publication_date'])<b['available_from'],'Evidence publication clock')
    bind(e['raw_path'],e['raw_sha256'])
    for key,hkey in (('retrieval_receipt','retrieval_sha256'),('read_text_path','text_sha256')):
        if key in e:bind(e[key],e[hkey])
    listing_evidence.append(dict(bound=b,primary_evidence=e,clock_certification='Next-calendar-day availability is inherited declared assumption; first-publication byte vintage is not certified.'))
record_map={(r['query_index'],r['column_index']):r for r in source['records']}
mature_map={(r['query_index'],r['column_index']):r for r in mature['records']}
identity_cells=[]
for b in expected:
    cells=[]
    require(b['symbol'] not in meta['etfs'],'Bound instrument is a direct stock in original META')
    for i,names in enumerate(meta['symbols']):
        for j,symbol in enumerate(names):
            if symbol!=b['symbol']:continue
            r=record_map[(i,j)];m=mature_map[(i,j)]
            require(r['date']==m['date']==meta['dates'][i] and r['symbol']==m['symbol']==symbol and r['issuer_key']==m['issuer_key']==b['issuer_key'],'Full symbol/issuer/query identity')
            cells.append(dict(query_index=i,column_index=j,date=meta['dates'][i],symbol=symbol,issuer_key=b['issuer_key'],new_price_start=windows[i]['price_start'],new_price_end=windows[i]['price_end'],bound_visible=b['available_from']<=meta['dates'][i],start_precedes_bound=windows[i]['price_start']<b['not_before']))
    require(cells,'Reviewed bound has META identity evidence');identity_cells.extend(cells)
for name in ('work/peer239_maturity_sources.py','work/peer239_maturity_independent.py','work/peer239_maturity_host.py','work/peer239_source_contract.json','outputs/monthly_local_linear70/study.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_daily_episodes/episodes.py'):
    bind(name)
lineage=prior['source_lineage_reference'];bind(lineage['path'],lineage['sha256'])


# Every operation below is JSON identity/clock/schema checking or file-byte hashing.
# No strategy, source loader, price parser, numerical library or NPZ decoder is imported.
preflight=read('work/after245_residual_source_preflight.json','5374642c22a26b4985646605a67b12c79a14bccfc14f0276a243ca0074f29189')
require(preflight['metadata_and_hash_checks_passed'] is True and preflight['strict_pit'] is False,'Prior metadata scope')
for name,h in preflight['input_sha256'].items():bind(name,h)
prior245=read('work/endpoint_median245_SOURCE_REVIEW.json','972c0975c91e6e95141941c54aa13ad82882be048975589876bee09b85880ab9')
require(prior245['raw_source_hashes']==raw and prior245['listing_bounds']==expected,'Unchanged raw and listing identities')
ids=('query_index','column_index','date','symbol','issuer_key')
require([{k:x[k] for k in ids} for x in identity_cells]==[{k:x[k] for k in ids} for x in prior245['listing_identity_cells']],'Exact five inherited identity cells')
plan245=read('outputs/monthly_endpoint_median245/PLAN.json','38de3c4264bc86b0ad9e22f3cce970a6f61e9731a8435d6b427bbe21478e8b02')
clocks=[]
for i,w in enumerate(windows):
    t=positions[w['date']];base=t-546
    clocks.append(dict(query_index=i,date=w['date'],initial_partial=w['date']=='2016-08-17',query_position=t,
        source_start=sessions[base],source_end=sessions[t-42],source_prices=505,
        first_beta_first_return_date=sessions[base+1],first_beta_last_return_date=sessions[base+252],
        formation_start_price_date=sessions[base+252],first_formation_return_date=sessions[base+253],
        last_formation_return_date=sessions[base+504],last_beta_first_return_date=sessions[base+252],last_beta_last_return_date=sessions[base+503]))
require(clocks==preflight['calendar']['windows'],'Independent505 metadata clock rebuild')
source_close=read('outputs/monthly_peer239_source_audit/COMPLETION.json')
require(source_close['status']=='COMPLETED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED' and source_close['source_only'] is True,'Original full-inventory audit complete')
for field,name in [('plan_sha256','SOURCE_PLAN.json'),('report_sha256','REPORT.json'),('independent_review_sha256','INDEPENDENT_REVIEW.json')]:
    bind('outputs/monthly_peer239_source_audit/'+name,source_close[field])
source_plan=read('outputs/monthly_peer239_source_audit/SOURCE_PLAN.json')
original_ind=read('outputs/monthly_peer239_source_audit/INDEPENDENT_REVIEW.json')
require(original_ind['passed'] is True,'Original independent inventory metadata')
require(source_plan['raw_source_hashes']==raw and source_plan['source_only'] is True,'Original source-lineage identity')
contract=read('work/peer239_source_contract.json','f2821351c4fa1c4091cb70d93e944f605cfe1f54e6372720cd459739d47b55c1')
for name,h in contract['inputs_sha256'].items():bind(name,h)
feasibility=read('work/peer239_source_feasibility.json',source_plan['input_sha256'][str(ROOT/'work/peer239_source_feasibility.json')])
for name,h in feasibility['input_sha256'].items():bind(name,h)
repair_policy=read('work/ownsector238_repair_source_policy.json','bb743a4ca456c43219436f5e39f0387b3b721e65d2344012de699ba79d6ec7de')
require(repair_policy['communication_benchmark']['old']=='XLC' and repair_policy['communication_benchmark']['new']=='VOX','Fixed fullperiod communication proxy')
for rule in repair_policy['missing_mapping_rules']:
    require(iso(rule['public_date'])<iso(rule['valid_from']),'Dated repair not publication-day')
    if rule['valid_until_exclusive'] is not None:iso(rule['valid_until_exclusive'])
    require(rule['source_file'] in source_plan['input_sha256'] or str(ROOT/rule['source_file']) in source_plan['input_sha256'],'Mapping evidence registered')
    key=rule['source_file'] if rule['source_file'] in source_plan['input_sha256'] else str(ROOT/rule['source_file'])
    bind(rule['source_file'],source_plan['input_sha256'][key])
fields=c['sources']['mapping_fields']
require(fields==['query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence'],'Exact ten mapping keys')
require(len(source['records'])==len(mature['records'])==3904 and len(source['events'])==len(mature['events'])==122,'Full direct-stock mapping inventory')
records=[];repair_counts={};sector_counts={};unresolved=[]
for i,names in enumerate(meta['symbols']):
    stockcols=[j for j,symbol in enumerate(names) if symbol not in meta['etfs']]
    require(len(stockcols)==32,'32 directstocks each query')
    for report in (source,mature):
        e=report['events'][i]
        require(e['query_index']==i and e['date']==meta['dates'][i] and e['stock_indices']==stockcols,'Original event/stock inventory alignment')
    issuer_seen=set()
    for j in stockcols:
        r0=record_map[(i,j)];r1=mature_map[(i,j)];row={k:r1[k] for k in fields}
        require(row=={k:r0[k] for k in fields},'Maturity does not alter identity/mapping metadata')
        require(row['query_index']==i and row['column_index']==j and row['date']==meta['dates'][i] and row['symbol']==names[j],'Exact query identity')
        issuer=row['issuer_key'];require(type(issuer) is str and issuer and issuer==source_plan['issuer_keys'][row['symbol']],'Whole issuer exact registered identity')
        require(issuer not in issuer_seen,'No same-month duplicate issuer');issuer_seen.add(issuer)
        require(row['mapping_origin'] in ('query_date_sector_panel','frozen_FOREIGN_override'),'Original mapping origin')
        original_sector=row['original_sector']
        if original_sector=='XLC':wanted=('VOX','full_period_communication_proxy',None)
        elif original_sector is not None:wanted=(original_sector,'unchanged',None)
        else:
            rule=next((x for x in repair_policy['missing_mapping_rules'] if x['symbol']==row['symbol']),None)
            if rule and rule['valid_from']<=row['date'] and (rule['valid_until_exclusive'] is None or row['date']<rule['valid_until_exclusive']):
                wanted=(rule['benchmark'],'dated_business_proxy',rule['source_ids'])
            else:wanted=(None,'unresolved',None)
        require((row['sector'],row['mapping_repair'],row['mapping_evidence'])==wanted,'Exact dated repair metadata')
        records.append(row);repair_counts[row['mapping_repair']]=repair_counts.get(row['mapping_repair'],0)+1
        label=str(row['sector']);sector_counts[label]=sector_counts.get(label,0)+1
        if row['sector'] is None:unresolved.append(row)
require(len(records)==3904 and len(record_map)==len(mature_map)==3904,'All and only fullstock records, no duplicate cells')
require(records==[{k:r[k] for k in fields} for r in mature['records']],'Projection preserves canonical original record order')
require(repair_counts==preflight['domain']['metadata_mapping_counts'] and sector_counts==preflight['domain']['metadata_benchmark_counts'],'Previously checked metadata census')
require(unresolved==preflight['mapping']['unresolved_records'] and len(unresolved)==3,'Three unresolved SPCX mappings retained')
require(len(source_plan['issuer_keys'])==73 and all(r['symbol'] not in meta['etfs'] for r in records),'Direct-stock domain, not oldE or peer support')
# Acquired VOX prices are hashed as bytes only, never JSON-decoded here.
ac=read('work/ownsector238_vox_acquisition_acceptance.json','ecd695d81ebe56a1ae774e47a64e790618afc704b5324a21907a87cbca153106')
require(ac['symbol']=='VOX' and ac['acquired'] is True and ac['metadata_identity_review_passed'] is True and ac['source_admitted'] is False,'Acquisition scope distinct from source support')
for name,h in ac['input_sha256'].items():bind(name,h)
vox_receipt=read('outputs/monthly_ownsector238_vox_sources/prices.receipt.json',ac['input_sha256']['outputs/monthly_ownsector238_vox_sources/prices.receipt.json'])
require(vox_receipt['http_status']==200 and vox_receipt['raw_sha256']==ac['prices_raw_sha256'] and vox_receipt['binding']['symbol']=='VOX','Bound VOX response receipt')
require(vox_receipt['binding']['parameters']==dict(endDate='2026-09-24',format='json',resampleFreq='daily',startDate='2015-01-01'),'Fixed VOX request')
vox_path=ac['prices_raw_path'];vox_sha=bind(vox_path,ac['prices_raw_sha256'])
require(vox_path==preflight['vox']['raw_check']['path'] and vox_sha==preflight['vox']['raw_check']['sha256'],'Prior VOX raw byte identity')
for name in ('work/ownsector238_repair_sources.py','work/ownsector238_repair_host.py','work/monthly_ownsector_gate238_host.py','work/peer239_sources.py','work/peer239_source_host.py','work/peer239_maturity_host.py','work/monthly_peer239_host.py','work/monthly_risk_consensus219_host.py','outputs/monthly_stability_monitor/inputs.py','outputs/monthly_state_rank_research/state_rank.py'):
    bind(name)
# Bind repaired old endpoint certificate by metadata, without executing its loader/parser or claiming505 support.
repair_out='outputs/monthly_ownsector238_repaired_source_audit/'
require(path(repair_out+'COMPLETION.json').exists(),'Known repaired source metadata path')
repair_close=read(repair_out+'COMPLETION.json');repair_plan=read(repair_out+'SOURCE_PLAN.json',repair_close['plan_sha256'])
bind(repair_out+'REPORT.json',repair_close['report_sha256'])
require(repair_close['admitted'] is True and repair_close['source_only'] is True and repair_plan['raw_source_hashes']==raw,'Old repaired source lineage only')
for name in ('work/ownsector238_repair_RESULT_REVIEW.json','work/ownsector238_public_history_sources.json','work/ownsector238_vox_source_feasibility.json'):bind(name)
early=[w for w in clocks if w['source_start']<ac['start']]
earlycells=[row for row in records if row['sector']=='VOX' and clocks[row['query_index']]['source_start']<ac['start']]
require(early==preflight['vox']['earlier_than_receipt_windows'] and earlycells==preflight['vox']['affected_by_clock_only_vox_mapping_cells'],'Early VOX risk from request metadata only')
require(len(early)==8 and len(earlycells)==24,'Fixed prospective early-window metadata count')
used=sorted({r['sector'] for r in records if r['sector'] is not None})
require(used==preflight['mapping']['used_benchmarks'] and 'XLC' not in used and 'XLRE' not in used,'Current mapped factor identity scope')
factor_bounds=[dict(symbol=s,inception_date=d,available_from=None,evidence_clock_certified=False) for s,d in [('VOX','2004-09-23'),('XLC','2018-06-18'),('XLRE','2015-10-07')]]
require(preflight['etf_boundary_evidence']['vox_inception']['date']==factor_bounds[0]['inception_date'],'VOX existence evidence')
for b in factor_bounds[1:]:
    require(preflight['etf_boundary_evidence']['verified_dates'][b['symbol']]['inception']==b['inception_date'],'Unused ETF existence evidence')
    require(sector_counts.get(b['symbol'],0)==0,'Unused XLC/XLRE exact mapping census')
implementation=bind(Path(__file__))
report=dict(schema='residual_core246_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
 source_only=True,metadata_only=True,source_admission_metadata_only=True,context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
 preflight_sha256=PINS['work/after245_residual_source_preflight.json'],input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(c['input_sha256']),raw_source_hashes=raw,raw_source_count=64,
 raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,source_window_recomputed_505=True,listing_identity_metadata_verified=True,
 mapping_records=records,mapping_fields=fields,mapping_source_path='outputs/monthly_peer239_maturity_audit/REPORT.json',mapping_source_sha256=PINS['outputs/monthly_peer239_maturity_audit/REPORT.json'],
 mapping_metadata=dict(records=3904,events=122,stocks_per_event=32,distinct_stock_symbols=73,original_etf_count=17,repair_counts=repair_counts,benchmark_counts=sector_counts,used_benchmarks=used,unresolved_records=unresolved,old_endpoint_peer_E_support_flags_included=False,retrospective_query_known_fixed_proxy=True),
 listing_bounds=expected,listing_bound_evidence=listing_evidence,listing_identity_cells=identity_cells,
 calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',sha256=PINS['outputs/monthly_paired_target143/CALENDAR.json'],source_clock='t-546..t-42 inclusive',prices=505,total_simple_return_intervals=504,formation_returns=252,beta_prior_returns=252,windows=windows,fit_clocks=clocks,calendar_start=sessions[0],calendar_end=sessions[-1],session_count=len(sessions)),
 metadata=dict(events=122,width=49,stock_count_per_event=32,etf_count=17,stock_records=3904,known_listing_bound_securities=2,known_listing_identity_cells=5),
 vox_prices_path=vox_path,vox_prices_sha256=vox_sha,
 vox=dict(acquisition_path='work/ownsector238_vox_acquisition_acceptance.json',acquisition_sha256=PINS['work/ownsector238_vox_acquisition_acceptance.json'],request=vox_receipt['binding']['parameters'],provider_identity=ac['provider_identity'],raw_prices_decoded=False,parser_path='work/ownsector238_repair_sources.py',parser_sha256=PINS['work/ownsector238_repair_sources.py'],parser_function='parse_vox(rows, calendar)',parser_output='DataFrame one column VOX; exact official dates, finite positive adjClose, no fill/splice',parser_enforces_request_dates=True,early_windows_not_certified_by_request=early,early_mapped_cells_not_certified_by_request=earlycells,early_query_count=8,early_formal_query_count=7,early_mapped_cell_count=24,actual_505_support='UNKNOWN until registered prepare; request metadata alone is not decoded raw coverage'),
 old_repair_binding=dict(module='work/monthly_ownsector_gate238_host.py',function='source_admission_binding(raw)',module_sha256=PINS['work/monthly_ownsector_gate238_host.py'],source_plan_path=repair_out+'SOURCE_PLAN.json',source_plan_sha256=PINS[repair_out+'SOURCE_PLAN.json'],source_completion_path=repair_out+'COMPLETION.json',source_completion_sha256=PINS[repair_out+'COMPLETION.json'],source_report_path=repair_out+'REPORT.json',source_report_sha256=PINS[repair_out+'REPORT.json'],scope='Reusable immutable old repair/acquisition closure only; source_admitted means238 endpoint contract, not246505window admission',function_executed=False),
 factor_listing_bounds=factor_bounds,factor_listing_evidence=preflight['etf_boundary_evidence'],factor_listing_evidence_sha256={'work/ownsector238_public_history_sources.json':PINS['work/ownsector238_public_history_sources.json'],'work/ownsector238_vox_source_feasibility.json':PINS['work/ownsector238_vox_source_feasibility.json']},unused_factor_mapping_counts={'XLC':0,'XLRE':0},
 inherited_source_closure=dict(plan='outputs/monthly_endpoint_median245/PLAN.json',input_pins=len(plan245['input_sha256']),scope='246host inherits full245closure and binds this source receipt/pins. No repeated21444 scan here.'),
 host_required_checks_after_registration=[
 'Recompute505 official positive finite realnonbool closes for stock,SPY and assignedsector; all252 lagged fitwindows and formationdays use only t-546..t-42. Old273 or endpoint admission does not prove support.',
 'Preserve all3904 stock mapping records, alloriginal49 inventory and E; no oldmaturity/peer-count/endpoint/E flags enter newsource mask. ETFs retain originalcore and maytrade.',
 'Ownstock full505 source availability must be recorded independently of mapping/SPY/sector/fit masks. Match allfive exact query,column,date,symbol,issuer,start546,end42 proofs; visiblebound crossed by source-validhistory hardfails.',
 'Three nullSPCXsector records remain missing. NoSPYfallback, forwardfill, proxychange or oldsame-ticker source substitution. Query-known sector heldconstant across505 history is a declared nonPIT convention.',
 'VOX existingparser requires2015-01-01..2026-09-24 source dates, exact sessions, positiveadjClose. Early8queries/24VOX mappingcells are unproven505history. Botharms share unavailable/no-op, never shortened residuals or market-onlyfallback.',
 'Verify metadata-listed factor identity/existence evidence without invented historic first-public timestamps; XLC/XLRE currentlyunused, VOX benchmark composition transitions preserved. Any additional factor identity requires explicit source evidence.',
 'Actualcommonfit support and conditioning are new246 calculations: count descriptive market/joint rolling estimates separately; predictive fits zero. Use fixed independent numerical tolerance but exact masks/ranks/gates/orders; no postdata tolerance or support changes.'
 ],
 limitations=[
 'This is metadata and hash verification, not actual505 support, descriptive regression execution or strategy admission.',
 'Original32+17 and source64 retained; only directstocks receive newresidualtargets, no peer-support or originalE filter before ranking. Missingcoverage does not delete a trading name.',
 'ThreeSPCX mapping gaps and early24VOX mappingcells are not repaired or asserted usable. No literalrawpre2015absence inferred without parsing.',
 'Only two explicit newUSsecurity bounds/fiveoccurrences reviewed. Complete securitymaster, ticker/ADR/unit continuity and original2016leadinggap splice remain inherited assumptions.',
 'Query-date sector table/FOREIGN overrides and datedbusinessproxies are not comprehensive historicalGICS. Fixedqueryproxy throughpast505 days is retrospective research; VOX telecom/communication transition is not XLC-equivalent.',
 'KnownETF existence metadata is retrospective and does not supply certified first-public timestamps. RemainingusedSPDR inceptions are not newly exhaustively audited; actualpositiveprices must not be substituted for missing identity evidence.',
 'Current adjustedprice vintages, estimatedTop32/theme17 and repeateddevelopment history mean strictPITfalse. Hashes establish preservedbytes, not firstpublication accuracy.'
 ],
 source_lineage_reference=lineage,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
 actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,actual_model_fits_computed=False,actual_returns_computed=False,npz_decoded=False,source_npz_decoded=False,prices_loaded=False,loader_executed=False,original_sources_modified=False,observers_modified=False,strict_pit=False,host_not_pinned_to_avoid_circular_binding=True)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),implementation_sha256=implementation,input_pins=len(PINS),raw_source_files=64,metadata_events=122,mapping_records=len(records),listing_identity_cells=len(identity_cells),source_prices=505,unresolved_mappings=len(unresolved),early_vox_cells=len(earlycells))))
