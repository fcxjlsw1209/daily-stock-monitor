"""Offline conditional source admission only; no ratios, signs, ranks or returns.

Freeze a new policy and source/code hashes before mapping the already verified
twenty preflight events. Old preflight and round141 artifacts remain unchanged.
"""
import argparse
import bisect
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/same_report144_admission.py'
TEST = 'work/test_same_report144_admission.py'
OUT = 'outputs/monthly_same_report144_admission'
PRE = 'outputs/monthly_same_report_preflight_v1'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
METHOD = 'same_current_release_annual_comparison'
POLICY_ID = 'same_report144_source_v1'
PINS = {
 PRE+'/PLAN.json':'8dd919fe15302957ab18d153d2e047907d49a21bc6eedf7d25e1d9260f880a4f',
 PRE+'/EVENTS.json':'adcec8a2bfead10f31977e248548061a8b158db57aad4b5e12d673570ebabd00',
 PRE+'/SOURCE_AUDIT.json':'f3d4b79f3a234d80274233e75b54557f7b61048a8b7e1ac81c550dc661850a48',
 PRE+'/MANIFEST.json':'b570352043c667cfb20bdec52d2ead187fc6d4a358ca53458c5d96ac02bc2e72',
 SESSIONS:'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4',
}
YEARS = {'AAPL':list(range(2016,2026)), 'AMD':[2015,2016,2017,2018,2020,2021,2022,2023,2024,2025]}
ELIGIBLE = {'AAPL':list(range(2017,2026)), 'AMD':[2016,2017,2020,2021,2022,2023,2024]}
ISSUERS = {'AAPL':'Apple Inc.','AMD':'Advanced Micro Devices, Inc.'}
REJECTIONS = {('AAPL',2016):['unknown_comparison_duration'],
 ('AMD',2015):['unknown_comparison_duration'],
 ('AMD',2018):['excluded_accounting_comparability'],
 ('AMD',2025):['excluded_accounting_comparability']}
EVENT_KEYS = set('available_from comparison_period current_period eligible_under_listed_research_assumptions event_id fact_admitted financials first_public_version_verified fiscal_year identity model_admitted original_record provenance rejection_codes release_date reviews source source_only strategy_admitted symbol'.split())
AMOUNT_KEYS = set('amount_decimal available_from currency_iso currency_literal date displayed_amount original_cell scale source'.split())
PERIOD_KEYS = set('days_inclusive derivation derived end evidence start start_is_direct_quote'.split())


def require(ok, message):
    if not ok:raise ValueError(message)


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        obj={}
        for key,value in items:
            require(key not in obj,'Duplicate JSON key');obj[key]=value
        return obj
    def bad(value):raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def exact(obj, keys, label):
    require(type(obj)is dict and set(obj)==set(keys),'Unknown or missing '+label+' fields')


def day(value):
    require(type(value)is str,'Date string required')
    parsed=date.fromisoformat(value)
    require(parsed.isoformat()==value,'Canonical ISO date required')
    return parsed


def decimal(value):
    require(type(value)is str and bool(re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',value)),'Plain decimal string required')
    try:result=Decimal(value)
    except InvalidOperation:raise ValueError('Invalid decimal') from None
    require(result.is_finite(),'Finite decimal required')
    return result


def local(root, value):
    require(type(value)is str and bool(value) and not Path(value).is_absolute() and '..'not in Path(value).parts,'Local relative path required')
    require(not any(x in value.lower()for x in ('onedrive','cloudstorage')),'Cloud path forbidden')
    root=Path(root).resolve();path=root/value
    require(path.resolve()==path and path.is_relative_to(root),'Symlink/escaped source forbidden')
    return path


def hash_file(root,value):
    obj=hashlib.sha256()
    with local(root,value).open('rb')as stream:
        for part in iter(lambda:stream.read(1<<20),b''):obj.update(part)
    return obj.hexdigest()


def read(root,value,expected=None):
    raw=local(root,value).read_bytes()
    if expected is not None:require(digest(raw)==expected,'SHA mismatch: '+value)
    return decode(raw)


def verify_hashes(root, mapping):
    require(type(mapping)is dict and bool(mapping),'Nonempty source hash closure')
    for path,expected in mapping.items():
        require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'SHA256 required')
        require(hash_file(root,path)==expected,'SHA mismatch: '+path)


def publish(root,relative,value):
    target=local(root,relative);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.source-admission-',delete=False)as stream:
            temporary=Path(stream.name);stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,target)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def source_refs(value):
    refs={}
    def walk(obj):
        if type(obj)is dict:
            if set(('path','sha256'))<=set(obj):
                path,sha=obj['path'],obj['sha256']
                require(type(path)is str and type(sha)is str and re.fullmatch('[0-9a-f]{64}',sha),'Source reference schema')
                require(path not in refs or refs[path]==sha,'Conflicting source references');refs[path]=sha
            for item in obj.values():walk(item)
        elif type(obj)is list:
            for item in obj:walk(item)
    walk(value)
    return refs


def session_axis(values):
    require(type(values)is list and len(values)>1,'Pinned sessions required')
    parsed=[day(x)for x in values]
    require(parsed==sorted(set(parsed)),'Unique ordered sessions required')
    return parsed


def next_session(release, sessions):
    date_only=day(release);j=bisect.bisect_right(sessions,date_only)
    require(sessions[0]<=date_only and j<len(sessions),'Release outside pinned calendar')
    return sessions[j].isoformat()


def period(obj, fiscal_year, allow_unknown=False):
    require(type(obj)is dict and PERIOD_KEYS<=set(obj)<=PERIOD_KEYS|{'monetary_exclusion_of_boundary_source_unchanged'},'Unknown or missing period fields')
    end=day(obj['end']);require(end.year==fiscal_year,'Fiscal period year mismatch')
    require(type(obj['derived'])is bool and type(obj['start_is_direct_quote'])is bool,'Period evidence flags')
    require(type(obj['evidence'])is list and bool(obj['evidence'])and type(obj['derivation'])is str and bool(obj['derivation']),'Period evidence required')
    if obj['start']is None:
        require(allow_unknown and obj['days_inclusive']is None,'Unknown duration not authorized')
    else:
        start=day(obj['start']);days=(end-start).days+1
        require(type(obj['days_inclusive'])is int and obj['days_inclusive']==days and 330<=days<=400,'Annual duration mismatch')
    return dict(period_start=obj['start'],period_end=obj['end'],days_inclusive=obj['days_inclusive'])


def map_event(event,index,sessions,bindings):
    exact(event,EVENT_KEYS,'preflight event')
    symbol,year=event['symbol'],event['fiscal_year']
    require(symbol in YEARS and type(year)is int and year in YEARS[symbol],'Unexpected symbol/year')
    require(event['event_id']==f'{symbol}:FY{year}:{event["release_date"]}','Event identifier mismatch')
    require(event['first_public_version_verified']is False and event['source_only']is True and all(event[k]is False for k in('fact_admitted','model_admitted','strategy_admitted')),'Upstream source-only/vintage contract')
    expected=REJECTIONS.get((symbol,year),[])
    require(event['rejection_codes']==expected,'Rejection sentinel changed')
    require(type(event['eligible_under_listed_research_assumptions'])is bool and event['eligible_under_listed_research_assumptions']==(not expected),'Conditional eligibility changed')
    available=next_session(event['release_date'],sessions)
    require(event['available_from']==available,'Current-publication next-session availability mismatch')
    identity=event['identity']
    require(identity['current_issuer_name']==ISSUERS[symbol] and identity['certified_daily_security_master']is False,'Issuer identity/master claim mismatch')
    require(identity['valid_from']==available,'Identity must begin at own release availability')
    current=period(event['current_period'],year)
    previous=period(event['comparison_period'],year-1,allow_unknown=(expected==['unknown_comparison_duration']))
    if expected==['unknown_comparison_duration']:require(previous['period_start']is None and previous['days_inclusive']is None,'Preserve unknown prior duration sentinel')
    require(day(current['period_start'])==day(previous['period_end'])+timedelta(days=1),'Noncontiguous annual comparison')
    require(day(current['period_end'])<day(event['release_date']),'Release does not follow current period')
    through=day(current['period_end'])+timedelta(days=550)
    require(identity['valid_through']==through.isoformat()and day(available)<=through,'Identity window or late publication')
    exact(event['financials'],{'current','comparison'},'financial')
    for name,part in (('current',current),('comparison',previous)):
        exact(event['financials'][name],{'gross_profit','total_assets'},'annual financial')
        for input_key,output_key in (('gross_profit','gross_profit'),('total_assets','assets')):
            cell=event['financials'][name][input_key]
            exact(cell,AMOUNT_KEYS,'amount')
            value=decimal(cell['amount_decimal'])
            require(type(cell['scale'])is int and cell['scale']>0,'Positive integer scale required')
            require(cell['currency_literal']=='$'and cell['currency_iso']is None,'Uncertified same-report currency literal required')
            require(cell['available_from']==available and cell['date']==part['period_end'],'Amount period/current availability mismatch')
            require(cell['source']==event['source'],'Four cells must share current source bytes')
            require(output_key!='assets'or value>0,'Nonpositive assets')
            part[output_key]=dict(value=cell['amount_decimal'],scale=str(cell['scale']),currency_literal=cell['currency_literal'])
    refs=source_refs(event)
    require(bool(refs)and all(bindings.get(path)==sha for path,sha in refs.items()),'Unbound event evidence')
    require(type(event['reviews'])is list and bool(event['reviews'])and all(path in bindings for path in event['reviews']),'Independent review binding')
    return dict(event_id=event['event_id'],symbol=symbol,fiscal_year=year,
        issuer_id=symbol+':'+ISSUERS[symbol],security_id=symbol,method=METHOD,
        release_date=event['release_date'],available_from=available,
        identity_valid_from=available,identity_valid_through=through.isoformat(),
        admitted=not expected,rejection_codes=deepcopy(expected),same_report=dict(current=current,comparison=previous),
        currency_key='same-report-dollar:'+event['event_id'],first_public_version_verified=False,
        certified_daily_security_master=False,source=deepcopy(event['source']),
        evidence=dict(preflight_event_index=index,preflight_event_sha256=digest(encoded(event)),
            source_refs=[dict(path=path,sha256=sha)for path,sha in sorted(refs.items())],
            reviews=[dict(path=path,sha256=bindings[path])for path in event['reviews']],
            identity_anchors=deepcopy(identity['anchors']),identity_method=identity['kind'],
            current_period_derivation=deepcopy(event['current_period']),
            comparison_period_derivation=deepcopy(event['comparison_period'])))


def build_dataset(preflight,sessions,bindings,policy):
    exact(preflight,{'schema_version','status','assumptions','counts','events','missing_publications','financial_computations','network_requests'},'preflight root')
    require(type(preflight['schema_version'])is int and preflight['schema_version']==1,'Preflight schema version')
    require(type(preflight['financial_computations'])is int and preflight['financial_computations']==0 and type(preflight['network_requests'])is int and preflight['network_requests']==0,'Preflight scope')
    require(preflight['assumptions']==policy['inherited_preflight_assumptions'],'Frozen inherited assumptions changed')
    rows=preflight['events'];require(type(rows)is list and len(rows)==20,'All twenty events required')
    observed={symbol:[]for symbol in YEARS}
    for row in rows:
        require(type(row)is dict and row.get('symbol')in observed,'Unexpected event symbol')
        observed[row['symbol']].append(row.get('fiscal_year'))
    require(observed==YEARS,'Original event order/years required; no omitted or fabricated events')
    missing=preflight['missing_publications']
    if type(missing)is list and len(missing)==1:exact(missing[0],{'symbol','fiscal_year','event_created','reason'},'missing publication')
    require(type(missing)is list and len(missing)==1 and missing[0].get('symbol')=='AMD'and missing[0].get('fiscal_year')==2019 and missing[0].get('event_created')is False,'Preserve missing AMD FY2019 publication')
    mapped=[map_event(row,i,session_axis(sessions),bindings)for i,row in enumerate(rows)]
    require(len({row['event_id']for row in mapped})==20,'Unique event identifiers')
    eligible={symbol:[r['fiscal_year']for r in mapped if r['symbol']==symbol and r['admitted']]for symbol in YEARS}
    require(eligible==ELIGIBLE,'Exact conditional admission population')
    for row in mapped:
        if not row['admitted']:
            selected=latest_metadata(mapped,row['symbol'],row['available_from'])
            require(selected==dict(event_id=row['event_id'],usable=False,reason='latest_event_rejected'),'Rejected latest event must block fallback')
    return dict(schema_version=1,policy_id=POLICY_ID,method=METHOD,
        admission_scope='Conditional source use under listed research assumptions only',
        first_public_version_verified=False,complete_correction_history_verified=False,
        certified_daily_security_master=False,model_admitted=False,strategy_admitted=False,
        events=mapped,missing_publications=deepcopy(missing),assumptions=deepcopy(policy['assumptions']),
        counts=dict(events=20,conditionally_admitted=16,rejected=4,raw_values=80),
        ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0)


def latest_metadata(events,symbol,signal):
    """Metadata selection only. Select latest before checking its eligibility."""
    cutoff=day(signal);known=[x for x in events if x['symbol']==symbol and day(x['available_from'])<=cutoff]
    if not known:return dict(event_id=None,usable=False,reason='no_known_event')
    latest=max(day(x['available_from'])for x in known)
    selected=[x for x in known if day(x['available_from'])==latest]
    require(len(selected)==1,'Unresolved same-date events')
    row=selected[0]
    reason='conditional_source_available'
    if row['admitted']is not True:reason='latest_event_rejected'
    elif not day(row['identity_valid_from'])<=cutoff<=day(row['identity_valid_through']):reason='identity_expired'
    return dict(event_id=row['event_id'],usable=reason=='conditional_source_available',reason=reason)


def make_policy(preplan):
    assumptions=deepcopy(preplan['assumptions'])
    assumptions['status']='New conditional source admission only; financial calculations and model/strategy admission remain forbidden in this step.'
    return dict(schema_version=1,policy_id=POLICY_ID,method=METHOD,
        eligible_years=ELIGIBLE,expected_years=YEARS,
        inherited_preflight_assumptions=deepcopy(preplan['assumptions']),assumptions=assumptions,
        admission_meaning='admitted=true certifies reviewed source mapping under assumptions, not archival first-public-version or strategy efficacy.',
        latest_event_rule='Select latest available current publication, then reject invalid or expired event; no fallback. Same-date conflicts fail.',
        isolation='AAPL/AMD only. No merge with round141 MSFT/NVDA independently published prior-report definition.',
        comparison_rule='Both periods and all four cells belong to the same CURRENT release and become available together.',
        currency_rule='Per-event literal-dollar key only; do not certify ISO USD or compare currencies across releases.',
        source_scope='Read existing local exact-hash sources only. No source acquisitions, financial ratios/signs, model fitting, ranking or returns.',
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)


def freeze(root):
    require(not local(root,OUT+'/PLAN.json').exists()and not local(root,OUT+'/POLICY.json').exists(),'Policy/plan already frozen')
    verify_hashes(root,PINS)
    preplan=read(root,PRE+'/PLAN.json');manifest=read(root,PRE+'/MANIFEST.json')
    require(manifest['input_hashes']==preplan['input_hashes']and manifest['plan_sha256']==PINS[PRE+'/PLAN.json'],'Preflight source closure mismatch')
    require(manifest['output_hashes']=={name:PINS[PRE+'/'+name]for name in('EVENTS.json','SOURCE_AUDIT.json')},'Preflight output hashes')
    audit=read(root,PRE+'/SOURCE_AUDIT.json')
    require(audit['passed']is True and all(v is True for v in audit['checks'].values())and audit['counts']==dict(conditionally_eligible=16,events=20,fact_admissions=0,model_admissions=0,raw_monetary_values=80,rejection_sentinels=4,strategy_admissions=0),'Completed preflight required')
    sources=dict(preplan['input_hashes'],**PINS)
    verify_hashes(root,sources)
    policy=make_policy(preplan);publish(root,OUT+'/POLICY.json',policy)
    sources[OUT+'/POLICY.json']=hash_file(root,OUT+'/POLICY.json')
    code={path:hash_file(root,path)for path in(SELF,TEST)}
    tests=read(root,OUT+'/TESTS.json')
    require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('failures')==tests.get('errors')==0 and type(tests.get('tests_run'))is int and tests['tests_run']>0 and tests.get('source_code_hashes')==code,'Passing current synthetic tests required')
    sources[OUT+'/TESTS.json']=hash_file(root,OUT+'/TESTS.json')
    plan=dict(schema_version=1,stage='source_admission_plan',policy_id=POLICY_ID,method=METHOD,
        created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,
        source_hashes=sources,source_code_hashes=code,expected=dict(events=20,admitted=16,rejected=4,raw_values=80),
        financial_computations_permitted=False,network_requests_permitted=False,
        outputs=['DATA.json','AUDIT.json','MANIFEST.json'])
    publish(root,OUT+'/PLAN.json',plan)
    return dict(plan_sha256=hash_file(root,OUT+'/PLAN.json'),source_files=len(sources),code_files=len(code))


def load_plan(root,plan_sha):
    require(type(plan_sha)is str and re.fullmatch('[0-9a-f]{64}',plan_sha),'Exact PLAN SHA required')
    plan=read(root,OUT+'/PLAN.json',plan_sha)
    require(plan['stage']=='source_admission_plan'and plan['policy_id']==POLICY_ID and plan['method']==METHOD and plan['runtime']==sys.version,'Frozen plan identity/runtime')
    require(plan['financial_computations_permitted']is False and plan['network_requests_permitted']is False,'Source-only plan')
    verify_hashes(root,plan['source_hashes']);verify_hashes(root,plan['source_code_hashes'])
    require(set(plan['source_code_hashes'])=={SELF,TEST},'Exact mapper/test source closure')
    require(all(plan['source_hashes'].get(k)==v for k,v in PINS.items()),'Frozen upstream sources')
    return plan


def products(root,plan_sha):
    plan=load_plan(root,plan_sha)
    policy=read(root,OUT+'/POLICY.json')
    require(policy==make_policy(read(root,PRE+'/PLAN.json')),'Policy implementation mismatch')
    data=build_dataset(read(root,PRE+'/EVENTS.json'),read(root,SESSIONS)['sessions'],plan['source_hashes'],policy)
    audit=dict(schema_version=1,stage='conditional_source_admission',policy_id=POLICY_ID,passed=True,
        plan_sha256=plan_sha,policy_sha256=hash_file(root,OUT+'/POLICY.json'),data_sha256=digest(encoded(data)),
        counts=data['counts'],eligible_years=ELIGIBLE,source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],
        checks={k:True for k in('all_twenty_events_preserved','all_eighty_values_and_scales_preserved','same_current_release_four_cells','two_annual_periods_checked','strict_next_session_availability','causal_identity_windows','four_rejection_sentinels_retained','missing_AMD2019_not_fabricated','latest_invalid_blocks_fallback','old_measurement_definition_not_merged')},
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,
        limitations=policy['assumptions'])
    verify_hashes(root,plan['source_hashes']);verify_hashes(root,plan['source_code_hashes'])
    return data,audit


def build(root,plan_sha):
    require(not any(local(root,OUT+'/'+name).exists()for name in('DATA.json','AUDIT.json','MANIFEST.json')),'Admission output exists; preserve prior result')
    data,audit=products(root,plan_sha)
    publish(root,OUT+'/DATA.json',data);publish(root,OUT+'/AUDIT.json',audit)
    manifest=dict(schema_version=1,stage='conditional_source_admission_complete',plan_sha256=plan_sha,
        output_hashes={name:hash_file(root,OUT+'/'+name)for name in('DATA.json','AUDIT.json')})
    publish(root,OUT+'/MANIFEST.json',manifest)
    return manifest


def verify(root,plan_sha):
    manifest=read(root,OUT+'/MANIFEST.json');require(manifest['plan_sha256']==plan_sha,'Manifest PLAN binding')
    require(set(manifest['output_hashes'])=={'DATA.json','AUDIT.json'},'Exact admission output closure')
    verify_hashes(root,{OUT+'/'+k:v for k,v in manifest['output_hashes'].items()})
    data,audit=products(root,plan_sha)
    require(read(root,OUT+'/DATA.json')==data and read(root,OUT+'/AUDIT.json')==audit,'Reproduced source mapping mismatch')
    return dict(passed=True,counts=data['counts'],output_hashes=manifest['output_hashes'])


def main():
    cli=argparse.ArgumentParser(description=__doc__);group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-plan',action='store_true');group.add_argument('--build',action='store_true');group.add_argument('--verify',action='store_true')
    cli.add_argument('--plan-sha');args=cli.parse_args()
    result=freeze(ROOT)if args.freeze_plan else build(ROOT,args.plan_sha)if args.build else verify(ROOT,args.plan_sha)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
