"""Explicit, hash-bound schema repair for unadmitted annual-source rows.

No producer imports at module import or while verifying repair receipts.
Original PLAN, mathematical policy, source rows and code remain unchanged.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_annual_lag158'
PLAN_SHA = '29abc09a1c0a043da286a7d4ac7e1e46f003fcfe9c175d393cf61237b5c64665'
FAILURE = OUT+'/PREPARE_INTERFACE_FAILURE_01.json'
FAILURE_LOG = OUT+'/PREPARE_INTERFACE_FAILURE_01.log'
PREREVIEW = OUT+'/REPAIR_PREREVIEW_01.json'
AMENDMENT = OUT+'/AMENDMENT_01.json'
REPAIR_COMPLETION = OUT+'/REPAIR_COMPLETION_01.json'
PREVIOUS_AMENDMENT = dict(path='outputs/monthly_filing_association157/AMENDMENT_01.json',
    sha256='1e17a0eaccd4669c9706c128e817d5605334efba42e1539d483dc385e43ed73e')
FILES = tuple('work/'+name for name in (
    'annual_lag158_cell_repair.py','test_annual_lag158_cell_repair.py',
    'monthly_annual_lag158_repaired_runner.py','test_monthly_annual_lag158_repaired_runner.py',
    'annual_lag158_repair.py','test_annual_lag158_repair.py',
    'annual_lag158_repair_review.py','test_annual_lag158_repair_review.py'))
SUITES = dict(cell=FILES[1],runner=FILES[3],lifecycle=FILES[5],independent=FILES[7])
REAL = ('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','RANK_AUDIT.json',
    'PRETRAIN_ORDERS.json','PRICE_PARITY.json','CALENDAR.json','CACHE.json',
    'INDEPENDENT_PREPARE.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json',
    'FROZEN.json','DIAGNOSE_STARTED.json','RESULTS.json','VALIDATION.json',
    'INDEPENDENT_OUTCOME.json','COMPLETION.json')
CONTRACT = dict(round=158, kind='absent_amendment_field_on_unadmitted_source_rows',
    original_plan_sha256=PLAN_SHA, same_experiment=True, additional_strategy_trials=0,
    formula_changed=False, parameters_changed=False, original_source_rows_changed=False,
    source_eligibility_changed=False)


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    publish_bytes(path,raw)


def publish_bytes(path,raw):
    path=Path(path);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(str(temp),str(path))
    finally:
        if temp is not None: temp.unlink(missing_ok=True)


def reference(root,path):
    return dict(path=path,sha256=sha(Path(root)/path))


def verify_reference(root,ref,path):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and ref['path']==path,
            'Exact repair reference required')
    require(sha(Path(root)/path)==ref['sha256'],'Repair reference digest changed')
    return read(Path(root)/path)


def implementation_hashes(root):
    return {name:sha(Path(root)/name) for name in FILES}


def unprepared(root):
    found=[name for name in REAL if (Path(root)/OUT/name).exists()]
    require(not found,'Actual preparation artifacts exist: '+repr(found))


def verify_plan(root,plan_sha):
    require(plan_sha==PLAN_SHA and sha(Path(root)/OUT/'PLAN.json')==PLAN_SHA,'Exact original PLAN required')
    plan=read(Path(root)/OUT/'PLAN.json')
    require(plan['round']==158 and plan['status']=='REGISTERED' and plan['dose']==.01,'Original policy changed')
    return plan


def validate_failure(root,doc):
    require(doc.get('original_plan_sha256')==PLAN_SHA and doc.get('exit_code')==1,'Original failed attempt required')
    require(type(doc.get('strategy_returns_calculated')) is int and doc['strategy_returns_calculated']==0 and
        type(doc.get('prepared_artifacts_published')) is int and doc['prepared_artifacts_published']==0 and
        doc.get('lag_feature_build_attempted') is True,'Failed attempt activity must be preserved')
    require(doc.get('log')==reference(root,FAILURE_LOG),'Failure log changed')
    require(b'Explicit source reason list required: known amendments' in (Path(root)/FAILURE_LOG).read_bytes(),
        'Wrong failure log')


def validate_prereview(root,review):
    require(review.get('passed') is True and review.get('synthetic_only') is True and
        review.get('semantics_preserved') is True and review.get('original_plan_sha256')==PLAN_SHA and
        type(review.get('actual_returns_calculated')) is int and review['actual_returns_calculated']==0,
        'Independent synthetic repair review required')
    require(review['implementation_hashes']==implementation_hashes(root),'Repair implementation changed')
    failure=verify_reference(root,review['failure_record'],FAILURE);validate_failure(root,failure)
    require(set(review['tests'])==set(SUITES),'All four repair suites required')
    for name,path in SUITES.items():
        row=review['tests'][name]
        require(row.get('passed') is True and type(row.get('count')) is int and row['count']>0 and
            row.get('source_sha256')==sha(Path(root)/path),'Changed/failed repair test')
        require(re.findall(r'Ran (\d+) tests? in ',row.get('log',''))==[str(row['count'])] and
            row['log'].rstrip().endswith('OK'),'Incomplete repair test log')


def verify_amendment(root,plan_sha,amendment_sha):
    root=Path(root).resolve();plan=verify_plan(root,plan_sha)
    require(type(amendment_sha) is str and re.fullmatch('[0-9a-f]{64}',amendment_sha) and
        sha(root/AMENDMENT)==amendment_sha,'Exact amendment digest required')
    doc=read(root/AMENDMENT)
    require(all(type(doc.get(k)) is type(v) and doc[k]==v for k,v in CONTRACT.items()),'Repair contract changed')
    require(doc['implementation_hashes']==implementation_hashes(root),'Amended code changed')
    failure=verify_reference(root,doc['failure_record'],FAILURE);validate_failure(root,failure)
    review=verify_reference(root,doc['independent_prereview'],PREREVIEW);validate_prereview(root,review)
    require(review['implementation_hashes']==doc['implementation_hashes'],'Unreviewed repair')
    times=[datetime.fromisoformat(x) for x in (plan['registered_utc'],failure['recorded_utc'],
        review['reviewed_utc'],doc['registered_utc'])]
    require(all(x.tzinfo is not None for x in times) and times==sorted(times) and
        times[-1]<=datetime.now(timezone.utc),'Repair chronology changed')
    return doc


def augment_report(root,report,plan_sha,amendment_sha):
    amendment=verify_amendment(root,plan_sha,amendment_sha)
    require(not any(k.startswith('repair_') or k=='amendment_sha256' for k in report),'Repair provenance already present')
    result=dict(report,original_plan_sha256=PLAN_SHA,amendment_sha256=amendment_sha,
        repair_source_hashes=amendment['implementation_hashes'],
        repair_review_sha256=sha(Path(root)/FILES[6]),
        repair_artifact_hashes={str(Path(root).resolve()/p):sha(Path(root)/p)
                               for p in (AMENDMENT,FAILURE,FAILURE_LOG,PREREVIEW)})
    return result


def verify_repair_report(root,report,plan_sha,amendment_sha):
    amendment=verify_amendment(root,plan_sha,amendment_sha)
    require(report.get('original_plan_sha256')==PLAN_SHA and report.get('amendment_sha256')==amendment_sha,
        'Numerical audit lacks amended execution provenance')
    require(report.get('repair_source_hashes')==amendment['implementation_hashes'] and
        report.get('repair_review_sha256')==sha(Path(root)/FILES[6]),'Repair audit implementation changed')
    expected={str(Path(root).resolve()/p):sha(Path(root)/p) for p in (AMENDMENT,FAILURE,FAILURE_LOG,PREREVIEW)}
    require(report.get('repair_artifact_hashes')==expected,'Repair audit closure incomplete')
    return amendment


def record_failure(root,log):
    root=Path(root).resolve();plan=verify_plan(root,PLAN_SHA);unprepared(root)
    import monthly_annual_lag158_lifecycle as lifecycle
    reader=lifecycle.old.Reader(root);reader.verify(plan['hashes'])
    raw=Path(log).read_bytes()
    require(b'Explicit source reason list required: known amendments' in raw,'Unexpected failure')
    if (root/FAILURE_LOG).exists():require((root/FAILURE_LOG).read_bytes()==raw,'Existing log differs')
    else:publish_bytes(root/FAILURE_LOG,raw)
    doc=dict(original_plan_sha256=PLAN_SHA,recorded_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=1,log=reference(root,FAILURE_LOG),exec_session=20364,
        lag_feature_build_attempted=True,raw_price_parity_attempted=True,
        completed_lag_feature_values_not_persisted=True,strategy_returns_calculated=0,
        prepared_artifacts_published=0,
        note='Original preparation stopped at unadmitted ARKK (signal0,candidate2). This later record preserves the actual process log; no earlier timestamp is invented.',
        repair_scope='Only absent fields on the two already-unadmitted early-return branches. No default for explicit null or malformed fields; admitted missing metadata remains failure. Original source bytes, eligibility, formulas, doses, choices and accounting contract unchanged.')
    validate_failure(root,doc);write_once(root/FAILURE,doc)
    return reference(root,FAILURE)


def register(root):
    root=Path(root).resolve();plan=verify_plan(root,PLAN_SHA);unprepared(root)
    import monthly_annual_lag158_lifecycle as lifecycle
    with lifecycle.prior.state_lock(root):
        reader=lifecycle.old.Reader(root);reader.verify(plan['hashes'])
        review=read(root/PREREVIEW);validate_prereview(root,review)
        state,before=lifecycle.prior.state_snapshot(root/lifecycle.STATE);lifecycle.predecessor(state)
        require(state['registered_queue']==[lifecycle.queue_entry(plan,PLAN_SHA)],'Exact original158 queue required')
        bound=state.get('active_execution_amendment')
        predecessor_binding=bound is None or bound==PREVIOUS_AMENDMENT
        if bound==PREVIOUS_AMENDMENT:
            verify_reference(root,bound,PREVIOUS_AMENDMENT['path'])
        if (root/AMENDMENT).exists():
            digest=sha(root/AMENDMENT)
            doc=verify_amendment(root,PLAN_SHA,digest)
            require(predecessor_binding or bound==reference(root,AMENDMENT),'Different active amendment')
            if bound==reference(root,AMENDMENT):
                return dict(registered=True,amendment=bound,original_plan_sha256=PLAN_SHA)
            require(doc.get('continuation_before_sha256')==before,
                    'Unbound amendment continuation changed; preserve it for review')
        else:
            require(predecessor_binding,'Active amendment without its file')
            doc=dict(CONTRACT,registered_utc=datetime.now(timezone.utc).isoformat(),
                implementation_hashes=implementation_hashes(root),failure_record=reference(root,FAILURE),
                independent_prereview=reference(root,PREREVIEW),continuation_before_sha256=before)
            write_once(root/AMENDMENT,doc);digest=sha(root/AMENDMENT)
            verify_amendment(root,PLAN_SHA,digest)
        state.update(current_status='registered_code_repair_before_reprepare',
            active_execution_amendment=reference(root,AMENDMENT),last_goal_turn_classification='progress',
            updated_utc=doc['registered_utc'])
        lifecycle.prior.update_state(root/lifecycle.STATE,before,state)
    return dict(registered=True,amendment=reference(root,AMENDMENT),original_plan_sha256=PLAN_SHA)


def complete(root,amendment_sha):
    root=Path(root).resolve();amendment=verify_amendment(root,PLAN_SHA,amendment_sha)
    for name in ('INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json'):
        verify_repair_report(root,read(root/OUT/name),PLAN_SHA,amendment_sha)
    import monthly_annual_lag158_lifecycle as lifecycle
    with lifecycle.prior.state_lock(root):
        result=lifecycle.complete(root,PLAN_SHA)
        verify_amendment(root,PLAN_SHA,amendment_sha)
        stable=dict(round=158,same_experiment=True,additional_strategy_trials=0,
            original_completion=reference(root,OUT+'/COMPLETION.json'),amendment=reference(root,AMENDMENT),
            failure_record=amendment['failure_record'],independent_prereview=amendment['independent_prereview'],
            independent_prepare=reference(root,OUT+'/INDEPENDENT_PREPARE.json'),
            independent_outcome=reference(root,OUT+'/INDEPENDENT_OUTCOME.json'),
            protected=reference(root,OUT+'/PROTECTED_AFTER.json'))
        if (root/REPAIR_COMPLETION).exists():
            old=read(root/REPAIR_COMPLETION)
            require({k:v for k,v in old.items() if k!='completed_utc'}==stable,'Existing repair completion differs')
        else:write_once(root/REPAIR_COMPLETION,dict(stable,completed_utc=datetime.now(timezone.utc).isoformat()))
        state,before=lifecycle.prior.state_snapshot(root/lifecycle.STATE)
        repair_ref=reference(root,REPAIR_COMPLETION)
        if state.get('latest_execution_repair_receipt')!=repair_ref:
            state['latest_execution_repair_receipt']=repair_ref
            lifecycle.prior.update_state(root/lifecycle.STATE,before,state)
    return dict(result,repair_completion=repair_ref)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('record_failure','register','complete'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--log');parser.add_argument('--amendment-sha')
    args=parser.parse_args()
    result=(record_failure(args.root,args.log) if args.phase=='record_failure' else
            register(args.root) if args.phase=='register' else complete(args.root,args.amendment_sha))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
