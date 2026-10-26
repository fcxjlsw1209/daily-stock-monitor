"""Write-once registration and offline audit for declared own-sector source repairs."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
import ownsector238_source_audit as old
import ownsector238_repair_sources as repair

ROOT = old.ROOT
OUT = ROOT/'outputs/monthly_ownsector238_repaired_source_audit'
POLICY = ROOT/'work/ownsector238_repair_source_policy.json'
POLICY_SHA = 'bb743a4ca456c43219436f5e39f0387b3b721e65d2344012de699ba79d6ec7de'
ORIGINAL_SHA = 'dc63b2232c6f2b68b93833c4d67d86558c42c85cbb09bbf522462803363e03e7'
ORIGINAL_PLAN = '7b7e5ab9e0796edfc62185c0be6fa1fa727ea6d46d4293152a6e8bfa7d1f4ecf'
ACQUISITION = ROOT/'outputs/monthly_ownsector238_vox_sources'
ACCEPTANCE = ROOT/'work/ownsector238_vox_acquisition_acceptance.json'
TESTS = ROOT/'work/ownsector238_repair_TESTS.json'
REVIEW = ROOT/'work/ownsector238_repair_CODE_REVIEW.json'


def code_files():
    return [ROOT/'work'/n for n in ('ownsector238_repair_sources.py','test_ownsector238_repair_sources.py',
        'ownsector238_repair_independent.py','test_ownsector238_repair_independent.py','ownsector238_repair_host.py')]


def tests():
    old.require(not TESTS.exists(), 'Formal test receipt is write-once')
    base = old.previous()
    modules = [base.module('repair238_test_'+str(i),p) for i,p in enumerate(code_files()[1:4:2])]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    old.require(result.wasSuccessful(),'Source repair tests failed')
    old.dump(TESTS,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=base.ev.runtime(),
             policy_sha256=old.sha(POLICY),code_sha256={str(p):old.sha(p) for p in code_files()}))


def accepted_inputs():
    """Verify reviewed acquisition records and their hashes, without decoding prices."""
    receipt = old.read(ACCEPTANCE)
    old.require(receipt.get('schema')=='ownsector238_vox_acquisition_acceptance_v1'
        and receipt.get('metadata_identity_review_passed') is True
        and receipt.get('acquired') is True and receipt.get('source_admitted') is False
        and receipt.get('strategy_math_computed') is False, 'Explicit acquisition review required')
    old.require(receipt.get('symbol')=='VOX' and receipt.get('start')=='2015-01-01'
        and receipt.get('end')=='2026-09-24','Acquisition scope differs')
    old.previous().verify({str(ROOT/p):h for p,h in receipt['input_sha256'].items()})
    raw = ROOT/receipt['prices_raw_path']
    old.require(raw.parent==ACQUISITION and raw.is_file() and not raw.is_symlink(), 'Dedicated raw source required')
    old.require(old.sha(raw)==receipt['prices_raw_sha256'], 'Raw source SHA differs')
    return receipt


def register():
    old.require(not OUT.exists(), 'Fresh repaired-source output directory required')
    old.require(old.sha(POLICY)==POLICY_SHA,'Prospective policy changed')
    repair.check_policy(old.read(POLICY))
    plan = old.gate(ORIGINAL_PLAN)
    old.require(old.sha(old.OUT/'REPORT.json')==ORIGINAL_SHA,'Original report differs')
    test = old.read(TESTS); base = old.previous()
    old.require(test.get('passed') is True and test.get('synthetic_only') is True
        and test.get('runtime')==base.ev.runtime() and test.get('policy_sha256')==POLICY_SHA
        and test.get('code_sha256')=={str(p):old.sha(p) for p in code_files()},'Formal tests/code/runtime mismatch')
    review = old.read(REVIEW)
    old.require(review.get('passed') is True and review.get('unresolved_blockers')==[]
        and review.get('policy_sha256')==POLICY_SHA and review.get('tests_sha256')==old.sha(TESTS)
        and review.get('code_sha256')=={str(p):old.sha(p) for p in code_files()}
        and review.get('actual_repaired_audit_executed') is False,'Repair source review incomplete')
    acceptance = accepted_inputs()
    pins = dict(plan['input_sha256'])
    for p in code_files()+[POLICY,TESTS,REVIEW,ACCEPTANCE,old.OUT/'REPORT.json',old.OUT/'COMPLETION.json',old.OUT/'SOURCE_PLAN.json']:
        base.bind(pins,p)
    for mapping in (old.read(POLICY)['source_input_sha256'],acceptance['input_sha256'],review['input_sha256']):
        for p,h in mapping.items():base.bind(pins,ROOT/p,h)
    base.verify(pins); before=old.CENTRAL.read_bytes(); protected=old.read(old.PROTECTED)
    old.require(len(protected)==21444,'Full archive protection required');base.verify(protected)
    old.require(old.CENTRAL.read_bytes()==before,'Central changed')
    OUT.mkdir()
    old.dump(OUT/'SOURCE_PLAN.json',dict(schema='ownsector238_repaired_source_plan_v1',status='REGISTERED_SOURCE_ONLY',created_utc=old.now(),
        input_sha256=pins,runtime=base.ev.runtime(),policy_sha256=POLICY_SHA,original_report_sha256=ORIGINAL_SHA,
        original_plan_sha256=ORIGINAL_PLAN,raw_source_hashes=plan['raw_source_hashes'],
        acquisition_acceptance_sha256=old.sha(ACCEPTANCE),physical_prefixes=list(old.PREFIXES),
        protected_manifest_sha256=old.sha(old.PROTECTED),protected_files=21444,
        source_only=True,strategy_trials=0,actual_strategy_math_computed=False,original_npz_decoded=False,
        central_modified=False,observers_modified=False,strict_pit=False))
    print(json.dumps(dict(status='REGISTERED_SOURCE_ONLY',plan_sha256=old.sha(OUT/'SOURCE_PLAN.json'))),flush=True)


def gate(expected):
    old.require(isinstance(expected,str) and old.sha(OUT/'SOURCE_PLAN.json')==expected,'Explicit plan SHA mismatch')
    plan=old.read(OUT/'SOURCE_PLAN.json');base=old.previous()
    old.require(plan['schema']=='ownsector238_repaired_source_plan_v1' and plan['status']=='REGISTERED_SOURCE_ONLY'
        and plan['runtime']==base.ev.runtime() and plan['policy_sha256']==POLICY_SHA==old.sha(POLICY)
        and plan['original_report_sha256']==ORIGINAL_SHA==old.sha(old.OUT/'REPORT.json')
        and plan['original_plan_sha256']==ORIGINAL_PLAN==old.sha(old.OUT/'SOURCE_PLAN.json')
        and plan['physical_prefixes']==list(old.PREFIXES) and plan['strategy_trials']==0
        and plan['protected_manifest_sha256']==old.sha(old.PROTECTED),'Source repair plan contract differs')
    base.verify(plan['input_sha256']);base.verify(plan['raw_source_hashes']);accepted_inputs()
    return plan


def prefix_original(original,n):
    out=deepcopy(original)
    out['events']=out['events'][:n]
    out['records']=[r for r in out['records'] if r['query_index']<n]
    return out


def audit(expected):
    plan=gate(expected);base=old.previous()
    old.require(not (OUT/'REPORT.json').exists() and not (OUT/'COMPLETION.json').exists(),'Write-once repaired audit')
    protected=old.read(old.PROTECTED);before=old.CENTRAL.read_bytes();base.verify(protected)
    original=old.read(old.OUT/'REPORT.json');policy=old.read(POLICY);meta=old.read(old.LOCAL/'META.json')
    # Only original price/sector panel loader; do not call load/ranked_forecasts/prepare.
    study=base.local();source,info=study.arch.old.panel_load(meta['etfs'])
    old.require(study.raw_source_hashes(info)==plan['raw_source_hashes'],'Original raw-source map differs')
    adj=source['adj'];calendar=old.read(old.CALENDAR)['sessions'];acceptance=accepted_inputs()
    raw=old.read(ROOT/acceptance['prices_raw_path']);vox=repair.parse_vox(raw,calendar)
    import ownsector238_repair_independent as independent
    raw_check=independent.check_vox(raw,calendar,vox)
    old.require(all(raw_check.get(k) is True for k in ('passed','independent_raw_vox_rebuild','official_sessions_checked','requested_bounds_checked','utc_midnight_checked','unique_dates_checked','adjusted_close_exact')) and raw_check.get('row_count')==len(raw) and raw_check.get('missing_rows_filled') is False and raw_check.get('producer_modules_imported') is False,'Independent raw source check failed')
    report=repair.repair_sources(original,adj,vox,policy)
    check=independent.check_repaired(original,adj,vox,policy,report)
    old.require(all(check.get(k) is True for k in independent.FLAGS) and check.get('counts')==report['counts'] and check.get('repair_counts')==report['repair_counts'] and check.get('cell_count')==len(report['records'])
        and check.get('endpoint_count')==report['counts']['security_benchmark_endpoint_checks']+report['counts']['market_endpoint_checks'], 'Independent repaired scope mismatch')
    prefixes=[]
    for n in old.PREFIXES:
        subset=prefix_original(original,n);cutoff=subset['events'][-1]['date'] if n else None
        aa=adj.loc[:cutoff].copy(deep=True) if n else adj.iloc[:0].copy(deep=True)
        vv=vox.loc[:cutoff].copy(deep=True) if n else vox.iloc[:0].copy(deep=True)
        part=repair.repair_sources(subset,aa,vv,policy)
        ic=independent.check_repaired(subset,aa,vv,policy,part)
        old.require(all(ic.get(k) is True for k in independent.FLAGS) and ic.get('counts')==part['counts'],'Independent prefix mismatch')
        old.require(part==repair.repair_sources(original,adj,vox,policy,limit=n),'Physical/logical prefix differs')
        old.require(part['events']==report['events'][:n] and part['records']==[r for r in report['records'] if r['query_index']<n]
            and part['gaps']==[r for r in report['gaps'] if r['query_index']<n],'Full/prefix source report differs')
        prefixes.append(dict(prefix=n,cutoff=cutoff,counts=part['counts'],status=part['status'],independent=ic))
    gate(expected);base.verify(protected);old.require(old.CENTRAL.read_bytes()==before,'Central changed during source audit')
    result=dict(report,created_utc=old.now(),plan_sha256=expected,independent=check,raw_vox_check=raw_check,
        physical_prefixes=prefixes,input_sha256=plan['input_sha256'],protected_files=21444,
        central_modified=False,observers_modified=False,original_npz_decoded=False,limitations=policy['limitations'])
    old.dump(OUT/'REPORT.json',result)
    old.dump(OUT/'COMPLETION.json',dict(status='COMPLETED_REPAIRED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',created_utc=old.now(),
        admitted=result['admitted'],counts=result['counts'],repair_counts=result['repair_counts'],plan_sha256=expected,
        report_sha256=old.sha(OUT/'REPORT.json'),source_only=True,strategy_trials=0,actual_strategy_math_computed=False,
        original_npz_decoded=False,protected_files=21444,central_modified=False,observers_modified=False,strict_pit=False))
    print(json.dumps(dict(status='COMPLETED_REPAIRED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',admitted=result['admitted'],counts=result['counts'],
        repair_counts=result['repair_counts'],report_sha256=old.sha(OUT/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('test','register','audit'))
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:audit(args.plan_sha256)
