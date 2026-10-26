"""262 source feasibility: metadata and SHA256 only; no price/NPZ decoding.

Run through ./run_local.sh. Default prints a bounded check summary. --write
creates the two owned JSON documents exactly once. An optional independently
reviewed early-listing policy may be bound before final prospective freezing.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'work/tracking_risk262_source_contract.json'
MEMO=ROOT/'work/tracking_risk262_source_feasibility.json'
DESIGN='work/after261_tracking_risk_design.json'
DESIGN_SHA='8cdebb40055af0ad22c5f2307335cab5a4bed17b331aea32c4c94f0163e564bf'
META='outputs/monthly_local_linear70/META.json'
CALENDAR='outputs/monthly_paired_target143/CALENDAR.json'
MAPPING='outputs/monthly_peer239_maturity_audit/REPORT.json'
PREVIOUS='work/residual_core246_SOURCE_REVIEW.json'
OLD_LISTING='work/peer239_maturity_policy.json'
LATE_LISTING='work/monthly_peer_path252_v2_LISTING_POLICY.json'
GAPS='work/monthly_peer_path252_SOURCE_GAPS.json'
ACQUISITION='work/ownsector238_vox_acquisition_acceptance.json'
FIELDS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence')
CLOCK_FIELDS=('query_index','date','initial_partial','numeric_evaluation_required','symbols','stock_indices','query_position','next_official_session','numerator_positions','numerator_dates','risk_positions','risk_dates','required_positions','required_dates')
PINS={}
def need(ok,message):
    if not ok:raise ValueError(message)
def local(raw):
    p=Path(raw);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
    need(p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Local research input only')
    need(p!=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','No mutable central binding')
    return p
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def bind(raw,wanted=None):
    p=local(raw);s=sha(p);need(wanted is None or wanted==s,'Source digest '+str(p));key=str(p.relative_to(ROOT))
    need(key not in PINS or PINS[key]==s,'Inconsistent pin');PINS[key]=s;return s
def read(raw,wanted=None):
    p=local(raw);need(p.suffix=='.json' and not p.name.endswith('.raw.json'),'Metadata JSON only')
    bind(p,wanted);return json.loads(p.read_text())
def iso(s):need(type(s) is str and date.fromisoformat(s).isoformat()==s,'ISO day');return s
def mapping_rule(row,policy):
    old=row['original_sector']
    if old=='XLC':return ('VOX','full_period_communication_proxy',None)
    if old is not None:return (old,'unchanged',None)
    rule=next((r for r in policy['missing_mapping_rules'] if r['symbol']==row['symbol']),None)
    if rule and rule['valid_from']<=row['date'] and (rule['valid_until_exclusive'] is None or row['date']<rule['valid_until_exclusive']):return (rule['benchmark'],'dated_business_proxy',rule['source_ids'])
    return (None,'unresolved',None)
def main(args):
    design=read(DESIGN,DESIGN_SHA)
    for path,h in design['input_sha256'].items():bind(path,h)
    meta=read(META);cal=read(CALENDAR);sessions=cal['sessions']
    need(sessions==sorted(set(sessions)) and all(iso(d) for d in sessions),'Official calendar')
    positions={d:i for i,d in enumerate(sessions)}
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and len(meta['etfs'])==len(set(meta['etfs']))==17,'Original122/17 metadata')
    need(meta['dates'][0]=='2016-08-17' and meta['dates']==sorted(set(meta['dates'])),'Original query order')
    prior=read(PREVIOUS);old=read(OLD_LISTING);late=read(LATE_LISTING);mapping=read(MAPPING)
    plan=read('outputs/monthly_local_linear70/PLAN.json');original=read('outputs/monthly_local_linear70/INPUT_HASHES.json')
    raw=plan['raw_source_hashes'];need(len(raw)==64 and raw==prior['raw_source_hashes'],'Unchanged original64 raw map')
    for path,h in raw.items():need(original[path]==h,'Original64 registry');bind(path,h)
    origin=read('work/calendar_clock248_v2_SOURCE_REVIEW.json','26b8f28a7c467788a55a891d5985945a8c1f5b4c82342dafad33e67501bbbac0')
    need(origin['panel_first_session']=='2008-01-02','Original panel origin')
    for k in ('diagnostic','failure_review'):bind(origin['origin_correction'][k+'_path'],origin['origin_correction'][k+'_sha256'])
    source_plan=read('outputs/monthly_peer239_source_audit/SOURCE_PLAN.json')
    closure={}
    for folder in ('monthly_peer239_source_audit','monthly_peer239_maturity_audit','monthly_ownsector238_repaired_source_audit'):
        base='outputs/'+folder+'/'
        c=read(base+'COMPLETION.json');need(c['source_only'] is True,'Old source-only certificate')
        bundle={'COMPLETION.json':PINS[base+'COMPLETION.json']}
        for field,name in (('plan_sha256','SOURCE_PLAN.json'),('report_sha256','REPORT.json'),('independent_review_sha256','INDEPENDENT_REVIEW.json')):
            if field in c:bundle[name]=bind(base+name,c[field])
        closure[folder]=dict(files=bundle,scope='Old identity/mapping/acquisition source evidence; not262 numeric coverage')
    repair=read('work/ownsector238_repair_source_policy.json')
    need(repair['communication_benchmark']['old']=='XLC' and repair['communication_benchmark']['new']=='VOX','Frozen fullperiod communication proxy')
    for r in repair['missing_mapping_rules']:
        need(iso(r['public_date'])<iso(r['valid_from']),'Dated repair availability')
        path=r['source_file'];h=source_plan['input_sha256'].get(str(ROOT/path),source_plan['input_sha256'].get(path));need(h is not None,'Repair provenance registered');bind(path,h)
    clocks=[];records=[];seen=set();expected_stock=[]
    for q,(d,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        need(len(names)==len(set(names))==49 and d in positions,'Original49/query identity')
        t=positions[d];partial=q==0;need(t>=252 and t+1<len(sessions),'Required official positions')
        need(partial or sessions[t+1][:7]!=d[:7],'Formal monthend')
        stock=[j for j,s in enumerate(names) if s not in meta['etfs']];need(len(stock)==32,'All32 beforeE');expected_stock.extend((q,j) for j in stock)
        ep=[] if partial else [t-252,t-42];risk=[] if partial else list(range(t-63,t+1));required=sorted(set(ep+risk))
        need(partial or len(ep)==2 and len(risk)==64 and len(required)==65 and required==[t-252]+risk,'65distinct exact official dates')
        event=dict(query_index=q,date=d,initial_partial=partial,numeric_evaluation_required=not partial,symbols=names,stock_indices=stock,query_position=t,next_official_session=sessions[t+1],numerator_positions=ep,numerator_dates=[sessions[k] for k in ep],risk_positions=risk,risk_dates=[sessions[k] for k in risk],required_positions=required,required_dates=[sessions[k] for k in required])
        need(tuple(event)==CLOCK_FIELDS,'Clock schema');clocks.append(event)
    for r in mapping['records']:
        row={k:r[k] for k in FIELDS};q,j=row['query_index'],row['column_index'];need((q,j) not in seen,'Unique mapping cell');seen.add((q,j))
        need(row['date']==meta['dates'][q] and row['symbol']==meta['symbols'][q][j] and (q,j) in expected_stock,'Mapping identity/column')
        need(row['issuer_key']==source_plan['issuer_keys'][row['symbol']],'Whole issuer key')
        need(row['mapping_origin'] in ('query_date_sector_panel','frozen_FOREIGN_override'),'Mapping provenance')
        need((row['sector'],row['mapping_repair'],row['mapping_evidence'])==mapping_rule(row,repair),'Dated mapping rules')
        records.append(row)
    need([(r['query_index'],r['column_index']) for r in records]==expected_stock and records==prior['mapping_records'] and len(records)==3904,'Exact projected mapping inventory')
    for q in range(122):need(len({r['issuer_key'] for r in records if r['query_index']==q})==32,'No duplicate issuer/representative substitution')
    sectors=dict(sorted(Counter(str(r['sector']) for r in records).items()));used=sorted({r['sector'] for r in records if r['sector'] is not None})
    need(used==['VOX','XLE','XLF','XLI','XLK','XLP','XLV','XLY'] and sectors.get('None')==3,'Frozen factor universe and unresolved count')
    listing_evidence=read('work/peer239_listing_gap_evidence.json',old['evidence']['sha256'])
    for x in listing_evidence['primary_evidence']:
        for key,hkey in (('raw_path','raw_sha256'),('retrieval_receipt','retrieval_sha256'),('read_text_path','text_sha256')):
            if key in x:bind(x[key],x[hkey])
    for p,h in late['input_sha256'].items():bind(p,h)
    ac=read(ACQUISITION)
    for p,h in ac['input_sha256'].items():bind(p,h)
    need(ac['symbol']=='VOX' and ac['start']=='2015-01-01' and ac['end']=='2026-09-24','Frozen originalVOX acquisition')
    need(ac['prices_raw_path']==prior['vox_prices_path'] and ac['prices_raw_sha256']==prior['vox_prices_sha256'],'No VOX sourceversion switch')
    receipt=read('outputs/monthly_ownsector238_vox_sources/prices.receipt.json')
    need(receipt['http_status']==200 and receipt['raw_sha256']==ac['prices_raw_sha256'],'VOX acquisition receipt binding')
    need(all(e['initial_partial'] or ac['start']<=e['required_dates'][0]<=e['required_dates'][-1]<=ac['end'] for e in clocks),'Required dates within request scope, not numeric coverage')
    for p,h in prior['factor_listing_evidence_sha256'].items():bind(p,h)
    for p in ('work/ownsector238_repair_sources.py','work/ownsector238_repair_independent.py','work/monthly_ownsector_gate238_host.py','work/peer239_sources.py','work/peer239_maturity_sources.py','work/peer239_maturity_independent.py','work/monthly_peer_path252_kernel_v2.py','work/monthly_peer_path252_independent_v2.py','outputs/monthly_state_rank_research/state_rank.py','outputs/monthly_stability_monitor/inputs.py','work/nested199_SOURCE_REVIEW.json'):bind(p)
    lineage=read('work/nested199_SOURCE_REVIEW.json','d12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911')
    gaps=read(GAPS,'aea867a84e91d739ec8182ba6b87366c0436afb844e8c5481b94fea9ca0a7e44')
    known_days=[x['date'] for x in gaps['unique_gaps']];need(known_days==['2026-06-08','2026-06-09','2026-06-10','2026-06-11'],'Known source gap evidence')
    early=None
    if args.early_policy:
        need(args.early_policy_sha256,'Explicit early-policy digest');early=read(args.early_policy,args.early_policy_sha256)
        need(early['source_only'] is True and early['passed'] is True and early['source_admitted'] is False,'Separate reviewed metadata policy, not numeric admission')
        b=early['path_bound_override'];need(b['symbol']=='SPCX' and b['issuer_key']==late['path_bound_override']['issuer_key'] and b['not_before']=='2026-06-12' and b['available_from']<='2026-06-30' and b['kind']=='issuer_actual_first_trade','Early visible actualfirsttrade evidence')
        need(early['original_history_bound']==late['original_history_bound'],'Old lowerbound preserved')
        need(iso(early['publication_date'])<iso(b['available_from']),'Conservative nextday source clock')
        for p,h in early['input_sha256'].items():bind(p,h)
    identity=[];blockers=[]
    for row in records:
        q=row['query_index'];oldbound=next((b for b in old['securities'] if b['symbol']==row['symbol'] and b['issuer_key']==row['issuer_key'] and b['available_from']<=row['date']),None)
        if oldbound is None:continue
        bound=oldbound;proof='original239_bound'
        latebound=late['path_bound_override']
        if row['symbol']==latebound['symbol'] and row['issuer_key']==latebound['issuer_key'] and latebound['available_from']<=row['date']:bound=latebound;proof='252_v2_late_confirmation'
        if early and row['symbol']==early['path_bound_override']['symbol'] and row['issuer_key']==early['path_bound_override']['issuer_key'] and early['path_bound_override']['available_from']<=row['date']:bound=early['path_bound_override'];proof='262_early_actual_confirmation'
        event=clocks[q];structural=bool(event['numerator_dates'] and event['numerator_dates'][0]<oldbound['not_before'])
        identity.append(dict(query_index=q,column_index=row['column_index'],date=row['date'],symbol=row['symbol'],issuer_key=row['issuer_key'],original_history_bound=oldbound,path_history_bound=bound,path_bound_proof=proof,structural_warmup=structural,required_dates_before_bound=[d for d in event['required_dates'] if d<bound['not_before']],required_dates_at_or_after_bound=[d for d in event['required_dates'] if d>=bound['not_before']]))
        if row['sector'] is None:need(structural,'No mature unknown mapping')
        if row['symbol']=='SPCX':
            conflict=[d for d in known_days if d in event['required_dates'] and d>=bound['not_before']]
            if conflict:blockers.append(dict(code='KNOWN_SOURCE_BLOCKER',query_index=q,date=row['date'],symbol=row['symbol'],required_missing_dates=conflict,reason='Previously saved missing prices fall after then-visible lowerbound; later actualtrade evidence cannot be backdated.',source_path=GAPS,source_sha256=PINS[GAPS]))
    need(len(identity)==5,'Five known listing cells')
    bind(Path(__file__))
    status='METADATA_POLICY_READY_NUMERIC_ADMISSION_PENDING' if not blockers else 'KNOWN_SOURCE_BLOCKER'
    contract=dict(schema='tracking_risk262_source_contract_v1',created_utc=datetime.now(timezone.utc).isoformat(),status=status,passed=not blockers,metadata_checks_passed=True,metadata_only=True,source_admitted=False,strict_pit=False,
      design_path=DESIGN,design_sha256=DESIGN_SHA,panel_first_session='2008-01-02',raw_source_count=64,raw_source_hashes=raw,raw_source_hashes_verified=True,
      metadata_path=META,metadata_sha256=PINS[META],calendar_path=CALENDAR,calendar_sha256=PINS[CALENDAR],metadata=dict(events=122,formal_events=121,width=49,stocks_per_event=32,etfs=meta['etfs'],stock_cells=3904),
      clock_fields=list(CLOCK_FIELDS),clock_events=clocks,mapping_fields=list(FIELDS),mapping_records=records,mapping_source_path=MAPPING,mapping_source_sha256=PINS[MAPPING],mapping_counts=sectors,used_sector_proxies=used,
      mapping_policy=dict(path='work/ownsector238_repair_source_policy.json',sha256=PINS['work/ownsector238_repair_source_policy.json'],query_known_fixed_through_lookback=True,whole_issuer_unique=True,etfs_never_new_target=True,no_peer_count_requirement=True,source_checks_before_E=True,not_complete_historical_GICS=True),
      listing_bounds=old['securities'],listing_policy252_v2=dict(path=LATE_LISTING,sha256=PINS[LATE_LISTING],policy=late),early_listing_policy=None if early is None else dict(path=str(local(args.early_policy).relative_to(ROOT)),sha256=args.early_policy_sha256,policy=early),listing_identity_cells=identity,
      factor_listing_bounds=prior['factor_listing_bounds'],factor_listing_evidence_sha256=prior['factor_listing_evidence_sha256'],factor_bound_clock='Only retrospective existence contradiction checks; available_from=null is NOT historical knowledge certification; unusedXLC/XLRE muststayunused.',
      vox_prices_path=ac['prices_raw_path'],vox_prices_sha256=ac['prices_raw_sha256'],vox=dict(acquisition_path=ACQUISITION,acquisition_sha256=PINS[ACQUISITION],request=receipt['binding']['parameters'],parser_path='work/ownsector238_repair_sources.py',parser_sha256=PINS['work/ownsector238_repair_sources.py'],parser_function='parse_vox(rows, calendar)',output_column='VOX',price_field='adjClose',full2015_source_version_preserved=True,extension2014_used=False),
      loader=dict(path='outputs/monthly_stability_monitor/inputs.py',sha256=PINS['outputs/monthly_stability_monitor/inputs.py'],entry='study.arch.old.panel_load(meta["etfs"]) -> panel,info',feature_frames=['adj','sectors'],own_and_nonVOX_sector_field='panel["adj"][canonical_symbol]',vox_field='separate parsedVOX["VOX"]',operation='Original frozen loader and exact64 source fingerprints; no new splice, fill or normalization.',original_existing_transforms=['Manifest CSV Adj Close reindex with no fill','Existing early-extension pre-overlap ratio adjustment remains frozen','CHL/PTR use dedicated Tiingo source before2021; PTR archivekey PTRCY','META display alias research_equity_FB; BRK-B canonical BRK.B'],lineage_path='work/nested199_SOURCE_REVIEW.json',lineage_sha256=PINS['work/nested199_SOURCE_REVIEW.json'],lineage_columns=sorted(lineage['source_lineage'])),
      source_taxonomy=dict(structure_hardfail=['missing/duplicate issuer or duplicate same-month issuer','changed query/column/symbol/mapping provenance','nonunique/nonmonotone/nonofficial consumeddate or timezone/nonmidnight axes','missing required postbound row or column'],ordinary_missing=['None','NaN','missing_row','missing_column'],price_hardfail=['bool','nonnumeric/non-real','positive_or_negative_inf','zero_or_negative','conversion_overflow_or_positive_underflow_to_zero','positive requiredprice before visible identitybound'],mature_missing='Hardfailure for any requiredslot; no deletion, compression, source substitution or fill.',prebound='Onlyordinarymissing permitted before visible sameissuer pathbound; presentmalformed/positivehardfail.',scan='Inspect all65 ownslots of every32formalstock independently of E/mapping/support; scan everyknownsector65slots evenifownwarmup; do not dereference nonexistent unknownsector.',early_failure_does_not_admit_partial_inventory=True),
      structural_policy=dict(warmup='Onlysameissuer query-visible original239 bound with t-252 strictlybefore bound permits shared structural unavailability. Keepall49/E; no oldpeer support.',missing_mapping='Deferredonlyfor suchvisibleIPOwarmup; otherwise hardfail. No map learned from prices/returns.',zero_risk='After allrequiredsource passes, genuinezeroeither total/trackingSD makes bothtargets unavailable. Negative/nonfinite or underflowfakezero hardfail. Mathcontract setscanonicalconstantbranch.',rank_domain='Identicalcommonvaliddirectstocks beforeE, no minimumpeer/support count: N0 no targets; N1 originalaveragepct gives+.5. Finitezero target isvalid.',partial='Named2016-08-17: identity/mapping metadata only, zero newpriceobservations/statistics, exactincumbent. n0 no new source consumption.',no_whole_event_missing_fallback=True),
      numerical_requirements=dict(numerator='Mi=Pi[t-42]/Pi[t-252]-1; Mb=Pb[t-42]/Pb[t-252]-1; R=(1+Mi)/(1+Mb)-1, originaloperationorder.',risk='63simpledailyreturns from64officialcloses t-63..t. Bothdenominators share samequery-assignedsector. Currentrisk uses recent42 deliberately.',data_dates_per_instrument=65,numerator_return_intervals=210,risk_return_intervals=63,original_core_parity='Separate fullpast originalcore/27array/P/F/order gates; this65slot contract doesnot replaceoriginalcontrol requirements.'),
      inherited_source_closure=closure,known_source_blockers=blockers,actual_source_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',actual_common_rank_support='UNKNOWN_UNTIL_REGISTERED_PREPARE',known_gap_evidence=dict(path=GAPS,sha256=PINS[GAPS],dates=known_days,prior_scope='Four same-instrument gaps saved under252source audit; no new raw numeric read.'),
      original_source_files_modified=False,prices_loaded=False,npz_decoded=False,loader_executed=False,actual_features_computed=False,actual_models_fitted=False,actual_orders_computed=False,actual_returns_computed=False,observers_modified=False,input_sha256=dict(sorted(PINS.items())))
    memo=dict(schema='tracking_risk262_source_feasibility_v1',created_utc=contract['created_utc'],status=status,metadata_checks_passed=True,known_source_blockers=blockers,
      scope='Onlycode, fixedJSONmetadata and streamedSHA256. No rawprices decoded, no NPZ, no261results, no source/model/strategy execution.',
      findings=['Original64 byte closure and122x49 metadata verified; new directstock inventory3904 retains every32stock beforeE.',
      'Eachformal instrument needs65distinct dates: t-252 pluscurrent64 t-63..t; t-42 alreadyinside riskpath. No requiredcontinuous253price path.',
      'Original2015VOXrequest spans allnewrequired dates bycalendar alone; no2014extension or later sourceversion isused. This isnot decodedcoverage.',
      '239/246 mappingsare inherited ten-field metadata, neveroldE/endpoint/peer/support admission.',
      'OldSPCXJune8 lowerbound andAug5actualfirsttrade publication remainseparate. June/Julycurrentrisk exposes fourknownpostboundmissingdays underoldpolicy; independentlyreviewed June15 issuerstatement withJune16declaredavailability resolves that timing-definition blocker without relaxingpostboundchecks. Numericcoverage remainsunknown.',
      'Noactualnumeric admission or complete PITcertification; prospectiveknownsourceblockers must be resolvedbeforestrategyregistration.'],
      data_clock_summary=dict(first_formal_query=clocks[1]['date'],first_long_start=clocks[1]['numerator_dates'][0],first_risk_start=clocks[1]['risk_dates'][0],last_query=clocks[-1]['date'],own_slot_inspections_if_registered=121*32*65,mapped_benchmark_cells=sum(r['query_index']>0 and r['sector'] is not None for r in records),known_listing_cells=5),
      mapping_counts=sectors,unresolved_mapping_records=[r for r in records if r['sector'] is None],early_listing_policy_bound=early is not None,
      actual_source_coverage=contract['actual_source_coverage'],strict_pit=False,prices_loaded=False,npz_decoded=False,new_features_orders_returns_computed=False,source_contract_path=str(CONTRACT.relative_to(ROOT)),builder_path=str(Path(__file__).relative_to(ROOT)),input_sha256=dict(sorted(PINS.items())))
    if args.write:
        need(not CONTRACT.exists() and not MEMO.exists(),'Writeonce owned262 documents')
        contract_bytes=json.dumps(contract,ensure_ascii=False,indent=2,allow_nan=False)+'\n';memo['source_contract_sha256']=hashlib.sha256(contract_bytes.encode()).hexdigest()
        CONTRACT.write_text(contract_bytes);MEMO.write_text(json.dumps(memo,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=status,written=args.write,metadata_checks_passed=True,known_source_blockers=blockers,source_files_pinned=len(PINS),calendar=memo['data_clock_summary'],contract_sha256=sha(CONTRACT) if args.write else None,memo_sha256=sha(MEMO) if args.write else None),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');p.add_argument('--early-policy');p.add_argument('--early-policy-sha256');main(p.parse_args())
