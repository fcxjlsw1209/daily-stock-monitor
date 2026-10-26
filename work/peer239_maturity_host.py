"""Registered source-only refinement for officially pre-inception endpoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,html,io,json,re,unittest
import peer239_source_host as old
import peer239_maturity_sources as producer
import monthly_ownsector_gate238_host as common
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_peer239_maturity_audit'
BASE=ROOT/'outputs/monthly_peer239_source_audit'
POLICY=ROOT/'work/peer239_maturity_policy.json'
EVIDENCE=ROOT/'work/peer239_listing_gap_evidence.json'
TESTS=ROOT/'work/peer239_maturity_TESTS.json'
sha,read,dump,require=common.sha,common.read,common.dump,common.require
FILES=[ROOT/'work'/x for x in ('peer239_maturity_host.py','peer239_maturity_sources.py','peer239_maturity_independent.py','test_peer239_maturity_sources.py')]

def now():return datetime.now(timezone.utc).isoformat()

def evidence_check():
    policy=read(POLICY);evidence=read(EVIDENCE)
    require(policy['evidence']['sha256']==sha(EVIDENCE),'Listing review changed')
    primary={s['id']:s for s in evidence['primary_evidence']}
    for b in policy['securities']:
        e=primary[b['evidence_id']]
        require(e['publication_date']<b['available_from'],'Publication available clock')
        require(b['symbol'] in ('SPCX','SKHY'),'New bound requires additional reviewed official proof')
        raw=ROOT/e['raw_path'];require(sha(raw)==e['raw_sha256'],'Official source bytes changed')
        text=html.unescape(re.sub('<[^>]+>',' ',raw.read_text(errors='replace')))
        text=re.sub(r'\s+',' ',text)
        if b['symbol']=='SPCX':
            require(b['kind']=='exchange_new_listing_lower_bound' and b['not_before']=='2026-06-08' and b['available_from']=='2026-06-09','Exchange bound scope')
            require(all(t in text for t in ('Monday, June 8, 2026','84615Q103','upcoming listing','previously used for a different security','should not be linked')),'Exchange source claim absent')
        else:
            require(b['kind']=='issuer_actual_adr_start' and b['not_before']=='2026-07-10' and b['available_from']=='2026-07-11','ADR bound scope')
            require('July 10' in text and 'NASDAQ' in text and 'commencement of trading' in text,'ADR source claim absent')
    return policy,evidence

def pins():
    p=read(BASE/'SOURCE_PLAN.json');c=read(BASE/'COMPLETION.json')
    require(c['status']=='COMPLETED_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED' and c['admitted'] is False and c['plan_sha256']==sha(BASE/'SOURCE_PLAN.json') and c['report_sha256']==sha(BASE/'REPORT.json') and c['independent_review_sha256']==sha(BASE/'INDEPENDENT_REVIEW.json'),'Prior full source audit proof')
    require(read(BASE/'INDEPENDENT_REVIEW.json')['passed'] is True,'Prior oracle incomplete')
    policy,e=evidence_check();out=dict(p['input_sha256'])
    for f in FILES+[POLICY,EVIDENCE,TESTS,ROOT/'work/peer239_maturity_design.json',BASE/'SOURCE_PLAN.json',BASE/'COMPLETION.json',BASE/'REPORT.json',BASE/'INDEPENDENT_REVIEW.json']:
        common.bind(out,f)
    for name,h in e['input_sha256'].items():common.bind(out,ROOT/name,h)
    t=read(TESTS);require(t['passed'] is True and t['synthetic_only'] is True and t['runtime']==common.ev.runtime(),'Tests incomplete')
    require(t['code_sha256']=={str(p):sha(p) for p in FILES},'Source code changed since tests')
    common.verify(out);return out

def tests():
    require(not TESTS.exists(),'Test receipt write-once')
    mod=common.module('test_peer239_maturity',FILES[-1]);stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    print(stream.getvalue(),flush=True);require(r.wasSuccessful(),'Synthetic tests failed')
    dump(TESTS,dict(passed=True,synthetic_only=True,tests=r.testsRun,runtime=common.ev.runtime(),code_sha256={str(p):sha(p) for p in FILES}))

def register():
    require(not OUT.exists(),'Fresh source directory required')
    ps=pins();state=read(common.CENTRAL)
    require(state['current']=='monthly_ownsector_gate238' and not state['registered_queue'],'No concurrent strategy')
    OUT.mkdir();dump(OUT/'SOURCE_PLAN.json',dict(schema='peer239_maturity_source_plan_v1',created_utc=now(),input_sha256=ps,
        policy_sha256=sha(POLICY),original_source_sha256=sha(BASE/'REPORT.json'),runtime=common.ev.runtime(),
        prior_central_sha256=sha(common.CENTRAL),source_only=True,strategy_trials=0,original_failed_audit_preserved=True,
        no_new_actual_projection_before_registration=True,strict_pit=False))
    print(json.dumps(dict(status='REGISTERED_SOURCE_ONLY',plan_sha256=sha(OUT/'SOURCE_PLAN.json'))),flush=True)

def gate(expected):
    require(sha(OUT/'SOURCE_PLAN.json')==expected,'Explicit immutable plan SHA')
    p=read(OUT/'SOURCE_PLAN.json');require(p['input_sha256']==pins() and p['runtime']==common.ev.runtime(),'Source contract/runtime changed')
    require(sha(common.CENTRAL)==p['prior_central_sha256'],'Central state changed during source-only phase')
    return p

def audit(expected):
    gate(expected);require(not (OUT/'REPORT.json').exists(),'Source projection write-once')
    import peer239_maturity_independent as oracle
    source=read(BASE/'REPORT.json');meta=read(old.LOCAL/'META.json');policy=read(POLICY)
    old.assert_source_report(source,meta)
    result=producer.project(source,meta,policy);check=oracle.check_sources(source,meta,policy,result)
    require(check.get('passed') is True and check['counts']==result['counts'],'Independent source projection failed')
    prefixes=[]
    for n in (0,1,53,88,122):
        r=producer.project(source,meta,policy,n);a=oracle.check_sources(source,meta,policy,r,n)
        mm=common.meta_prefix(meta,n);ss=dict(source,events=source['events'][:n],records=[r for r in source['records'] if r['query_index']<n])
        pp=dict(policy,securities=[b for b in policy['securities'] if n and b['available_from']<=mm['dates'][-1]])
        physical=producer.project(ss,mm,pp);ap=oracle.check_sources(ss,mm,pp,physical)
        require(r==physical and r['records']==[x for x in result['records'] if x['query_index']<n] and r['events']==result['events'][:n],'Future history metadata dependency')
        require(a.get('passed') is True and ap.get('passed') is True,'Prefix oracle incomplete')
        prefixes.append(dict(prefix=n,exact=True,history_evidence_visible=len(pp['securities']),counts=r['counts']))
    gate(expected);dump(OUT/'REPORT.json',result)
    dump(OUT/'INDEPENDENT_REVIEW.json',dict(passed=True,full=check,physical_prefixes=prefixes,original_full_inventory_preserved=True))
    dump(OUT/'COMPLETION.json',dict(status='COMPLETED_MATURE_SOURCE_AUDIT_INDEPENDENTLY_VERIFIED',created_utc=now(),admitted=result['admitted'],counts=result['counts'],plan_sha256=expected,
        report_sha256=sha(OUT/'REPORT.json'),independent_review_sha256=sha(OUT/'INDEPENDENT_REVIEW.json'),source_only=True,strategy_trials=0,central_unchanged=True,observers_modified=False,
        old_source_audit_still_not_admitted=True,sector_gaps_deferred_not_filled=True,strict_pit=False))
    print(json.dumps(read(OUT/'COMPLETION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('test','register','audit'));p.add_argument('--plan-sha256');a=p.parse_args()
    if a.phase=='test':tests()
    elif a.phase=='register':register()
    else:require(bool(a.plan_sha256),'Plan SHA required');audit(a.plan_sha256)
