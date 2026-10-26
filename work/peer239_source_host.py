"""Write-once full-Top32 peer source admission. No strategy arithmetic or returns."""
from pathlib import Path
from datetime import datetime,timezone
import argparse, hashlib, importlib.util, io, json, unittest
import pandas as pd
import monthly_ownsector_gate238_host as previous
from monthly_peer_earnings139_inputs import issuer_map
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_peer239_source_audit'
CONTRACT=ROOT/'work/peer239_source_contract.json'
CONTRACT_SHA='f2821351c4fa1c4091cb70d93e944f605cfe1f54e6372720cd459739d47b55c1'
PRIOR=ROOT/'outputs/monthly_ownsector_gate238'
LOCAL=ROOT/'outputs/monthly_local_linear70'
CAP=ROOT/'outputs/monthly_top35_extension/CAP_COLLECTION.json'
SEC=ROOT/'outputs/dynamic_top100/priority_sec/CHL_submissions.json'
MAPPING_EVIDENCE=ROOT/'work/ownsector238_foreign_mapping_sources.json'
POLICY=ROOT/'work/ownsector238_repair_source_policy.json'
CENTRAL=previous.CENTRAL
TESTS=ROOT/'work/peer239_source_TESTS.json'
REVIEW=ROOT/'work/peer239_source_CODE_REVIEW.json'
PREFIXES=(0,1,53,88,122)
sha,read,dump,require=previous.sha,previous.read,previous.dump,previous.require

def now():return datetime.now(timezone.utc).isoformat()

def identity_keys(meta,records,sec,evidence):
    keys=issuer_map(records)
    require(str(sec.get('cik')).zfill(10)=='0001117795' and sec.get('name')=='CHINA MOBILE LTD /ADR/', 'CHL SEC issuer identity differs')
    companies=[c for c in evidence['companies'] if c.get('symbol')=='CHL']
    require(len(companies)==1 and companies[0]['business_evidence_public_by']=='2016-04-27'
            and companies[0]['source_ids']==['chl_2015_20f','chl_20160427_issuer_release'], 'CHL issuer evidence differs')
    require('CHL' not in keys,'Unexpected CAP issuer for CHL: needs explicit reconciliation')
    require(all(day>='2016-04-28' for day,syms in zip(meta['dates'],meta['symbols']) if 'CHL' in syms), 'CHL issuer evidence not available by query')
    key='sec_cik:0001117795'
    require(key not in keys.values(),'CHL key collision')
    keys['CHL']=key
    return keys

def validate_identity_scope(meta,supplement):
    require(supplement.get('issuer_key')=='sec_cik:0001117795'
            and supplement.get('valid_from_assumption')=='2016-04-28', 'CHL supplement semantics changed')
    actual=[day for day,syms in zip(meta['dates'],meta['symbols']) if 'CHL' in syms]
    require(actual==supplement.get('observed_query_dates') and len(set(actual))==len(actual),
            'CHL scope must equal evidence-reviewed original query dates')
    require(all(day>=supplement['valid_from_assumption'] for day in actual), 'CHL scope before evidence')


def files():
    names=('peer239_source_host.py','test_peer239_source_host.py','peer239_sources.py',
           'test_peer239_sources.py','peer239_source_independent.py','test_peer239_source_independent.py',
           'peer239_source_contract.json','peer239_source_feasibility.json','peer239_design_review.json',
           'ownsector238_saved_result_review.json','ownsector238_saved_result_reader.py',
           'monthly_peer_earnings139_inputs.py','ownsector238_source_audit.py',
           'ownsector238_source_independent.py','ownsector238_repair_sources.py')
    return [ROOT/'work'/n for n in names]

def tests():
    require(not TESTS.exists(),'Test receipt is write-once')
    names=('test_peer239_source_host.py','test_peer239_sources.py','test_peer239_source_independent.py')
    mods=[previous.module('peer239_tests_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in mods])
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=previous.ev.runtime(),
                   code_sha256={str(p):sha(p) for p in files()}))

def pin_sources():
    require(sha(CONTRACT)==CONTRACT_SHA,'Prospective contract changed')
    p=read(PRIOR/'PLAN.json');c=read(PRIOR/'COMPLETION.json')
    require(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['observers_modified'] is False
            and c['plan_sha256']==sha(PRIOR/'PLAN.json'),'238 closure incomplete')
    pins=dict(p['input_sha256'])
    for q in files()+[CAP,SEC,MAPPING_EVIDENCE,POLICY,TESTS,REVIEW,PRIOR/'PLAN.json',PRIOR/'COMPLETION.json']:
        previous.bind(pins,q)
    for k,d in read(CONTRACT)['inputs_sha256'].items():previous.bind(pins,ROOT/k,d)
    for k,d in read(ROOT/'work/peer239_source_feasibility.json')['input_sha256'].items():previous.bind(pins,ROOT/k,d)
    # Preserve original local source chain, with CHL metadata-only issuer repair.
    require(p['raw_source_hashes']==read(LOCAL/'PLAN.json')['raw_source_hashes'],'Original64 raw chain differs')
    require(read(TESTS)['code_sha256']=={str(p):sha(p) for p in files()},'Code changed since formal tests')
    require(read(TESTS)['passed'] is True and read(TESTS)['synthetic_only'] is True
            and read(TESTS)['runtime']==previous.ev.runtime(),'Formal synthetic tests required')
    review=read(REVIEW)
    require(review.get('passed') is True and review.get('unresolved_blockers')==[], 'Source code review required')
    require(review.get('code_sha256')=={str(p.relative_to(ROOT)):sha(p) for p in files()[:6]}, 'Code changed after review')
    require(review.get('contract_sha256')==CONTRACT_SHA and review.get('actual_source_values_loaded') is False,'Review scope differs')
    for k,d in review.get('input_sha256',{}).items():previous.bind(pins,ROOT/k,d)
    previous.verify(pins)
    return pins,p['raw_source_hashes']

def register():
    require(not OUT.exists(),'Fresh source directory required')
    state=read(CENTRAL)
    require(state['current']=='monthly_ownsector_gate238' and state['registered_queue']==[]
            and len(state['completed'])==234 and sum(r['trials'] for r in state['completed'])==1215,'Prior state differs')
    pins,raw=pin_sources()
    # Metadata only: no source price load, NPZ decoding or factor computation here.
    meta=read(LOCAL/'META.json');validate_identity_scope(meta,read(ROOT/'work/peer239_source_feasibility.json')['CHL_identity_supplement'])
    keys=identity_keys(meta,read(CAP)['records'],read(SEC),read(MAPPING_EVIDENCE))
    require(len(meta['dates'])==122 and all(len(s)==49 for s in meta['symbols']),'Original monthly axes')
    stocks={s for row in meta['symbols'] for s in row if s not in meta['etfs']}
    require(stocks<=keys.keys(),'Full-stock issuer coverage incomplete')
    OUT.mkdir()
    dump(OUT/'SOURCE_PLAN.json',dict(schema='peer239_source_plan_v1',status='REGISTERED_SOURCE_ONLY',created_utc=now(),
        contract_sha256=CONTRACT_SHA,runtime=previous.ev.runtime(),input_sha256=pins,raw_source_hashes=raw,
        issuer_keys={s:keys[s] for s in sorted(stocks)},issuer_repair=dict(symbol='CHL',key=keys['CHL'],valid_from='2016-04-28',
            scope='Issuer-key evidence only; not independent historical price, listing or ADR-unit certification'),
        physical_prefixes=list(PREFIXES),prior_central_sha256=sha(CENTRAL),source_only=True,strategy_trials=0,
        actual_source_values_loaded_before_registration=False,actual_strategy_math_computed=False,observers_modified=False))
    print(json.dumps(dict(status='REGISTERED_SOURCE_ONLY',plan_sha256=sha(OUT/'SOURCE_PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'SOURCE_PLAN.json')==expected,'Explicit SOURCE_PLAN hash required')
    p=read(OUT/'SOURCE_PLAN.json');pins,raw=pin_sources()
    require(p['input_sha256']==pins and p['raw_source_hashes']==raw and p['runtime']==previous.ev.runtime()
            and p['contract_sha256']==CONTRACT_SHA,'Source plan changed')
    require(sha(CENTRAL)==p['prior_central_sha256'],'Central changed during source-only lifecycle')
    meta=read(LOCAL/'META.json');validate_identity_scope(meta,read(ROOT/'work/peer239_source_feasibility.json')['CHL_identity_supplement'])
    keys=identity_keys(meta,read(CAP)['records'],read(SEC),read(MAPPING_EVIDENCE))
    require(p['issuer_keys']=={s:keys[s] for s in p['issuer_keys']},'Issuer identity changed')
    return p

def prefix(panel,meta,calendar,n):
    mm=previous.meta_prefix(meta,n)
    cc=previous.feature_prefix_calendar(calendar,meta,n)
    pp={k:f.loc[:pd.Timestamp(meta['dates'][n-1])].copy() if n else f.iloc[:0].copy() for k,f in panel.items()}
    return pp,mm,cc

def assert_source_report(r,meta):
    c=read(CONTRACT)
    require(set(r)==set(c['report_keys']) and r['schema']==c['report_schema'],'Report schema differs')
    require(set(r['counts'])==set(c['counts_keys']),'Report count schema differs')
    require(all(type(v) is int and v>=0 for v in r['counts'].values()),'Nonnegative integer source counts required')
    require(all(r[k]==v for k,v in c['flags'].items()),'Source-only scope changed')
    require(r['counts']['events']==len(meta['dates'])==len(r['events']),'Event count differs')
    require(r['counts']['stock_cells']==len(r['records']),'Stock count differs')
    require(r['gaps']==[x for x in r['records'] if x['gap_reasons']],'Gap coverage differs')
    require(r['admitted']==(not r['gaps']) and r['status']==('ADMITTED' if not r['gaps'] else 'SOURCE_GAPS'),'Admission mismatch')
    for rec in r['records']:
        require(set(rec)==set(c['records_keys']),'Record schema differs')
        i,j=rec['query_index'],rec['column_index']
        require(rec['date']==meta['dates'][i] and rec['symbol']==meta['symbols'][i][j] and rec['symbol'] not in meta['etfs'],'Record identity differs')
        require(rec['gap_reasons']==sorted(set(rec['gap_reasons'])),'Gap reasons not canonical')
    for ev in r['events']:require(set(ev)==set(c['events_keys']),'Event schema differs')

def require_independent(check,report,oracle):
    require(all(check.get(k) is True for k in oracle.FLAGS), 'Independent source check incomplete')
    require(check.get('counts')==report['counts'] and check.get('status')==report['status']
            and check.get('admitted') is report['admitted'], 'Independent source scope differs')
    require(check.get('checked_records')==len(report['records'])
            and check.get('checked_endpoints')==report['counts']['endpoint_checks'], 'Independent coverage differs')
    require(check.get('producer_modules_imported') is False and check.get('real_source_files_loaded') is False,
            'Independent module must consume supplied sources without producer imports')
    require(check.get('issuer_publication_evidence_checked_by_this_api') is False
            and check.get('issuer_evidence_clock_is_host_responsibility') is True, 'Do not overclaim issuer publication validation')
    require(all(check.get(k) is False for k in ('momentum_computed','rankings_computed','orders_computed','returns_computed'))
            and type(check.get('model_fits')) is int and check['model_fits']==0, 'Strategy computations forbidden')
    require(check.get('numerical_tolerance')==0. and check.get('frozen_helper_sha256')==sha(ROOT/'work/ownsector238_source_independent.py')
            and check.get('mapping_source_sha256')==sha(ROOT/'outputs/monthly_strict_relative/strict_study.py'), 'Independent evidence pin mismatch')


def audit(expected):
    plan=gate(expected)
    require(not (OUT/'REPORT.json').exists() and not (OUT/'COMPLETION.json').exists(),'Source audit write-once')
    import peer239_sources as producer
    import peer239_source_independent as oracle
    meta=read(LOCAL/'META.json');calendar=read(previous.CALENDAR)['sessions'];policy=read(POLICY)
    study=previous.local() # Define frozen functions only; do not call strategy.load or choices.
    source,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Loaded raw map changed');previous.verify(raw)
    panel={'adj':source['adj'],'sectors':source['sectors']}
    keys=plan['issuer_keys'];result=producer.inspect(panel,meta,calendar,keys,policy)
    assert_source_report(result,meta)
    independent=oracle.check_sources(panel,meta,calendar,keys,policy,result)
    require_independent(independent,result,oracle)
    checks=[]
    for n in PREFIXES:
        pp,mm,cc=prefix(panel,meta,calendar,n)
        r=producer.inspect(pp,mm,cc,keys,policy);ind=oracle.check_sources(pp,mm,cc,keys,policy,r)
        assert_source_report(r,mm);require_independent(ind,r,oracle)
        logical=producer.inspect(panel,meta,calendar,keys,policy,limit=n)
        require(logical==r,'Physical/logical prefix differs')
        require(r['events']==result['events'][:n] and r['records']==[x for x in result['records'] if x['query_index']<n]
                and r['gaps']==[x for x in result['gaps'] if x['query_index']<n],'Future metadata/source dependency')
        checks.append(dict(prefix=n,counts=r['counts'],independent_exact=True,physical_logical_exact=True,
            source_rows={k:len(v) for k,v in pp.items()},source_last_session={k:str(v.index[-1].date()) if len(v) else None for k,v in pp.items()},
            calendar_rows=len(cc),calendar_last_session=cc[-1] if cc else None,independent=ind))
    gate(expected) # Inputs and central state still unchanged; source-only report never edits either.
    dump(OUT/'REPORT.json',result)
    dump(OUT/'INDEPENDENT_REVIEW.json',dict(passed=True,full_record_equality=True,full_counts_equality=True,
        checked_stock_cells=len(result['records']),physical_prefixes=checks,source_only=True,full_independent=independent,
        source_oracle_sha256=sha(ROOT/'work/peer239_source_independent.py'),strategy_math_computed=False))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',created_utc=now(),
        admitted=result['admitted'],counts=result['counts'],plan_sha256=expected,report_sha256=sha(OUT/'REPORT.json'),
        independent_review_sha256=sha(OUT/'INDEPENDENT_REVIEW.json'),source_only=True,strategy_trials=0,
        central_unchanged=True,observers_modified=False,strict_pit=False,protected_full_archive_rehashed=False,
        protection_scope='All pinned consumed sources verified before/after; previous238 full21444 closure retained. Source-only new files, offline wrapper denies observer writes.'))
    print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','audit'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:
        require(bool(a.plan_sha256),'Explicit plan hash required');audit(a.plan_sha256)
