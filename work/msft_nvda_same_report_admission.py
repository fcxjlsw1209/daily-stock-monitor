"""Conditional source admission of reviewed same-release MSFT/NVDA facts.

No financial calculation. Freeze plan first, bind the exact completed independent
review second, then map source values. Old preflight and strategies stay frozen.
"""
import argparse
from copy import deepcopy
from datetime import timedelta,datetime,timezone
from pathlib import Path
import re
import sys
import same_report144_admission as base

ROOT=Path(__file__).resolve().parents[1]
SELF='work/msft_nvda_same_report_admission.py';TEST='work/test_msft_nvda_same_report_admission.py'
OUT='outputs/monthly_msft_nvda_same_report_admission_v1'
PRE='outputs/monthly_msft_nvda_same_report_preflight_v1'
REVIEW='work/after144_msft_nvda_source_review.json'
METHOD='same_current_release_annual_comparison';POLICY_ID='msft_nvda_same_report_source_v1'
YEARS={s:list(range(2015,2027))for s in ('MSFT','NVDA')}
ISSUERS={'MSFT':'Microsoft Corporation','NVDA':'NVIDIA Corporation'}
ACCOUNTING={('MSFT',y)for y in (2018,2023,2025,2026)}
PINS={PRE+'/'+n:h for n,h in {
 'PLAN.json':'33024f402deb6dbc0c36ea8b2b1219cf6c69036b417a7865b128345e574882b2',
 'EVENTS.json':'e2c6cada88d94d020d01472641adaace1d9847c5e17b500773224b05b46f2f8a',
 'AUDIT.json':'c720cc0a3d2ee2a58f61a1479d4c984673076a5d7e7c66e38f6a6dabec49c1b4',
 'MANIFEST.json':'eee99b2b6cc369af12b4fbf44780857f6c4983a9d2b7064f38d29cb53b5b253d',
 'SCHEMA.json':'eebdc65dd627bb49a7e803fe7598586fe48f4d7ce0fdfdcb817955b156f9e258',
 'TESTS.json':'3cfe7fe21bc932ab87c723b8915694dece3912ee87a7655e3e1e06c4bbf90ca1'}.items()}
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
BASE_POLICY='outputs/monthly_same_report144_admission/POLICY.json'
BASE_CODE='work/same_report144_admission.py'
PINS.update({SESSIONS:'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4',
 BASE_POLICY:'8b1431ecba8d5c33192c6ded65455f227bb447e97ac92f55798d259228e48507',
 BASE_CODE:'afb62d9286327d9849d9c42ca579c4566cbdc3bfdf7bcdc2274bfd8b30fd41c1'})
EVENT_KEYS=set('event_id symbol fiscal_year release_date available_from source identity current comparison rejection_codes review_flags first_public_version_verified complete_correction_history_verified fact_admitted model_admitted strategy_admitted cell_errors four_cells_extracted'.split())
CELL_KEYS=set('value_decimal displayed_amount scale currency_literal currency_iso period_end available_from source locator header_literals unit_literal statement_context parsed_cell_sha256'.split())
require=base.require;encoded=base.encoded;digest=base.digest;read=base.read;sha=base.hash_file;publish=base.publish;verify=base.verify_hashes;day=base.day


def count(obj,key,value):require(type(obj.get(key))is int and obj[key]==value,'Exact integer counter '+key)
def require_sha(value,label):
    require(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'Explicit SHA256 required: '+label)

def merge(*groups):
    result={}
    for group in groups:
        for k,v in group.items():
            require(k not in result or result[k]==v,'Conflicting evidence SHA');result[k]=v
    return result


def policy(original):
    kept={k:original['assumptions'][k]for k in ('calendar','currency','identity_window','period_boundaries','vintage','status','stop')}
    kept.update(identity='Each event own dated Microsoft/MSFT or NVIDIA/NVDA original release anchor; not a certified daily master.',
        duration='Both annual periods330..400 inclusive days. Both FY2015 comparison starts remain unknown and rejected.',
        accounting='Use current-report comparative cells only. Four Microsoft accounting flags may be resolved only by exact independent source evidence; retain unresolved flags.',
        invalid_events='Retain all24 current releases. Latest available event is selected before admission/identity validity; invalid or expired latest never falls back.',
        source_isolation='MSFT/NVDA same-current-release measurement only. Neither old141 independent-report values nor144 AAPL/AMD events are merged here.')
    return dict(schema_version=1,policy_id=POLICY_ID,method=METHOD,expected_years=YEARS,assumptions=kept,
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        admission_meaning='Reviewed source mapping under explicit research assumptions, not archival version certification or strategy efficacy.',
        review_gate=dict(path=REVIEW,stage='independent_same_report_source_review',events=24,cells=96,all_cells_must_pass=True),
        frozen_review_binding_required_before_mapping=True)


def validate_review(review,events):
    require(type(review)is dict and review.get('stage')=='independent_same_report_source_review' and review.get('passed')is True and review.get('source_only')is True and review.get('method')==METHOD,'Passed independent source review required')
    for k in ('financial_calculations','model_fits','ranks_computed','returns_computed'):count(review,k,0)
    count(review,'events_reviewed',24);count(review,'cells_reviewed',96)
    for k in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):require(review.get(k)is False,'Review cannot certify unavailable history')
    require(type(review.get('source_hashes'))is dict and bool(review['source_hashes']),'Review source closure missing')
    for path,value in review['source_hashes'].items():
        base.local(ROOT,path);require_sha(value,'review source')
    for k,expected in (('preflight_plan_sha256',PINS[PRE+'/PLAN.json']),('events_sha256',PINS[PRE+'/EVENTS.json'])):require(review.get(k)==expected,'Review input binding '+k)
    rows=review.get('events');require(type(rows)is list and len(rows)==24,'All24 reviewed events required')
    require([x.get('event_id')for x in rows]==[e['event_id']for e in events],'Reviewed event order/identity mismatch')
    result={}
    for r,e in zip(rows,events):
        require(r.get('symbol')==e['symbol']and r.get('fiscal_year')==e['fiscal_year'],'Review issuer/year mismatch')
        count(r,'cells_verified',4)
        for k in ('raw_cells_verified','availability_verified','identity_verified_under_assumptions','periods_verified'):require(r.get(k)is True,'Failed independent event check '+k)
        codes=list(e['rejection_codes']);a=r.get('accounting');require(type(a)is dict,'Accounting review missing')
        if (e['symbol'],e['fiscal_year'])in ACCOUNTING:
            require(a.get('status')in ('resolved','unresolved'),'Explicit accounting resolution required')
            if a['status']=='resolved':
                require(type(a.get('resolution_code'))is str and a['resolution_code']and type(a.get('scope'))is str and a['scope'],'Accounting resolution explanation required')
                evidence=a.get('evidence');require(type(evidence)is list and bool(evidence),'Accounting source evidence required')
                for item in evidence:
                    require(type(item)is dict and review['source_hashes'].get(item.get('path'))==item.get('sha256') and item.get('sha256')is not None,'Unbound accounting source evidence')
                require(any(x.get('path')==e['source']['raw_path']and x.get('sha256')==e['source']['raw_sha256']and x.get('quote')and x.get('locator')for x in evidence),'Accounting evidence must bind this current source')
                codes.remove('accounting_context_requires_independent_review')
        else:require(a.get('status')=='no_flag','Unexpected accounting resolution')
        require(r.get('remaining_rejection_codes')==codes,'Review changed unrelated rejection sentinels')
        if e['fiscal_year']==2015:require('unknown_comparison_period_start'in codes,'FY2015 duration rejection must remain')
        result[e['event_id']]=r
    return result


def period(p,year,unknown):
    base.exact(p,{'start','end','days_inclusive','derived_not_direct_quote','evidence'},'period')
    require(day(p['end']).year==year and p['derived_not_direct_quote']is True and type(p['evidence'])is list and p['evidence'],'Period evidence/year required')
    if unknown:require(p['start']is None and p['days_inclusive']is None,'Unknown prior duration must remain')
    else:
        days=(day(p['end'])-day(p['start'])).days+1
        require(type(p['days_inclusive'])is int and p['days_inclusive']==days and 330<=days<=400,'Annual duration mismatch')
    return dict(period_start=p['start'],period_end=p['end'],days_inclusive=p['days_inclusive'])


def map_event(e,r,index,axis,bindings,review_sha):
    base.exact(e,EVENT_KEYS,'preflight event');symbol=e['symbol'];year=e['fiscal_year']
    require(symbol in YEARS and type(year)is int and year in YEARS[symbol],'Source symbol/year')
    require(e['event_id']==f'{symbol}:FY{year}:{e["release_date"]}','Event identifier')
    for k in ('first_public_version_verified','complete_correction_history_verified','fact_admitted','model_admitted','strategy_admitted'):require(e[k]is False,'Preflight cannot claim admission/history')
    require(e['cell_errors']==[]and e['four_cells_extracted']is True,'All four preflight cells required')
    current=period(e['current']['period'],year,False);previous=period(e['comparison']['period'],year-1,year==2015)
    require(day(current['period_start'])==day(previous['period_end'])+timedelta(days=1),'Noncontiguous periods')
    require(day(current['period_end'])<day(e['release_date']),'Release before current year end')
    for proof in e['comparison']['period']['evidence']:
        if 'release_date'in proof:require(day(proof['release_date'])<day(e['release_date']),'Future boundary evidence')
    available=base.next_session(e['release_date'],axis);require(e['available_from']==available,'Current next-session mismatch')
    identity=e['identity'];through=(day(current['period_end'])+timedelta(days=550)).isoformat()
    require(identity['valid_from']==available and identity['valid_through']==through and available<=through,'Causal identity window')
    require(identity['anchor_in_current_source']is True and identity['certified_daily_security_master']is False and identity['anchor'],'Own identity anchor required')
    refs=base.source_refs(e)
    source=dict(path=e['source']['raw_path'],sha256=e['source']['raw_sha256'])
    refs=merge(refs,{source['path']:source['sha256'],e['source']['receipt_path']:e['source']['receipt_sha256']})
    require(all(bindings.get(p)==h for p,h in refs.items()),'Unbound event source')
    for role,p in (('current',current),('comparison',previous)):
        base.exact(e[role],{'period','gross_profit','assets'},'annual facts')
        for k in ('gross_profit','assets'):
            cell=e[role][k];base.exact(cell,CELL_KEYS,'cell')
            value=base.decimal(cell['value_decimal'])
            require(type(cell['scale'])is str and re.fullmatch('[1-9][0-9]*',cell['scale']),'Positive integer scale string')
            require(cell['currency_literal']=='$'and cell['currency_iso']is None,'Same-report literal currency only')
            require(cell['source']==e['source']and cell['period_end']==p['period_end']and cell['available_from']==available,'Cell source/period/availability mismatch')
            require(k!='assets'or value>0,'Nonpositive assets')
            p[k]=dict(value=cell['value_decimal'],scale=cell['scale'],currency_literal='$')
    codes=deepcopy(r['remaining_rejection_codes'])
    return dict(event_id=e['event_id'],symbol=symbol,fiscal_year=year,issuer_id=symbol+':'+ISSUERS[symbol],security_id=symbol,method=METHOD,
        release_date=e['release_date'],available_from=available,identity_valid_from=available,identity_valid_through=through,
        admitted=not codes,rejection_codes=codes,same_report=dict(current=current,comparison=previous),currency_key='same-report-dollar:'+e['event_id'],
        first_public_version_verified=False,certified_daily_security_master=False,source=source,
        evidence=dict(preflight_event_index=index,preflight_event_sha256=digest(encoded(e)),source_refs=[dict(path=p,sha256=h)for p,h in sorted(refs.items())],
            reviews=[dict(path=REVIEW,sha256=review_sha)],identity_anchors=[dict(literal=identity['anchor'],source=source)],identity_method='own_current_release_dated_issuer_ticker_anchor',
            current_period_derivation=deepcopy(e['current']['period']),comparison_period_derivation=deepcopy(e['comparison']['period']),accounting_review=deepcopy(r['accounting']),original_rejection_codes=deepcopy(e['rejection_codes'])))


def build_dataset(preflight,review,sessions,bindings,p,review_sha):
    rows=preflight['events'];require(type(rows)is list and [(e['symbol'],e['fiscal_year'])for e in rows]==[(s,y)for s in YEARS for y in YEARS[s]],'All24 original ordered events required')
    checks=validate_review(review,rows);axis=base.session_axis(sessions)
    mapped=[map_event(e,checks[e['event_id']],i,axis,bindings,review_sha)for i,e in enumerate(rows)]
    for row in mapped:
        if not row['admitted']:require(base.latest_metadata(mapped,row['symbol'],row['available_from'])['usable']is False,'Invalid latest fallback forbidden')
    n=sum(e['admitted']for e in mapped)
    return dict(schema_version=1,policy_id=POLICY_ID,method=METHOD,admission_scope='Conditional source use under listed research assumptions only',
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,model_admitted=False,strategy_admitted=False,
        events=mapped,missing_publications=[],assumptions=deepcopy(p['assumptions']),counts=dict(events=24,conditionally_admitted=n,rejected=24-n,raw_values=96),ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0)


def freeze(root):
    require(not base.local(root,OUT+'/PLAN.json').exists(),'Plan already frozen')
    verify(root,PINS);pre=read(root,PRE+'/PLAN.json');audit=read(root,PRE+'/AUDIT.json')
    require(audit['passed']is True and audit['all_cells_extracted']is True,'Successful complete preflight required')
    source=merge(pre['source_hashes'],pre['source_code_hashes'],PINS);verify(root,source)
    p=policy(read(root,BASE_POLICY));publish(root,OUT+'/POLICY.json',p)
    code={f:sha(root,f)for f in (SELF,TEST)};tests=read(root,OUT+'/TESTS.json')
    require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('source_code_hashes')==code,'Current synthetic tests required')
    for k in ('failures','errors','real_source_mappings'):count(tests,k,0)
    source.update({OUT+'/'+n:sha(root,OUT+'/'+n)for n in ('POLICY.json','TESTS.json')})
    plan=dict(schema_version=1,stage='source_admission_plan',policy_id=POLICY_ID,method=METHOD,created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,
        source_hashes=source,source_code_hashes=code,review_gate=p['review_gate'],review_binding_mode='explicit_exact_sha_then_exclusive_binding_before_mapping',
        financial_calculations_permitted=False,network_requests_permitted=False,outputs=['REVIEW_BINDING.json','DATA.json','AUDIT.json','MANIFEST.json'])
    publish(root,OUT+'/PLAN.json',plan);return dict(plan_sha256=sha(root,OUT+'/PLAN.json'),source_files=len(source))


def load_plan(root,plan_sha):
    require_sha(plan_sha,'plan')
    p=read(root,OUT+'/PLAN.json',plan_sha)
    require(p['runtime']==sys.version and p['stage']=='source_admission_plan'and p['method']==METHOD and p['policy_id']==POLICY_ID,'Plan identity/runtime')
    require(p['financial_calculations_permitted']is False and p['network_requests_permitted']is False,'Source-only plan')
    verify(root,p['source_hashes']);verify(root,p['source_code_hashes'])
    require(p['source_code_hashes']=={f:sha(root,f)for f in (SELF,TEST)},'Exact source implementation')
    return p


def bind_review(root,plan_sha,review_sha):
    require_sha(review_sha,'review')
    p=load_plan(root,plan_sha);review=read(root,REVIEW,review_sha)
    validate_review(review,read(root,PRE+'/EVENTS.json')['events']);verify(root,review['source_hashes'])
    sources=merge(p['source_hashes'],p['source_code_hashes'],review['source_hashes'],{REVIEW:review_sha,OUT+'/PLAN.json':plan_sha})
    verify(root,sources)
    binding=dict(stage='independent_review_binding',plan_sha256=plan_sha,review_path=REVIEW,review_sha256=review_sha,source_hashes=sources)
    publish(root,OUT+'/REVIEW_BINDING.json',binding)
    return dict(binding_sha256=sha(root,OUT+'/REVIEW_BINDING.json'),source_files=len(sources))


def products(root,plan_sha,binding_sha):
    require_sha(binding_sha,'review binding')
    p=load_plan(root,plan_sha);b=read(root,OUT+'/REVIEW_BINDING.json',binding_sha)
    require(b['stage']=='independent_review_binding'and b['plan_sha256']==plan_sha and b['review_path']==REVIEW,'Review binding identity')
    require_sha(b['review_sha256'],'bound review')
    verify(root,b['source_hashes']);review=read(root,REVIEW,b['review_sha256'])
    expected=merge(p['source_hashes'],p['source_code_hashes'],review['source_hashes'],{REVIEW:b['review_sha256'],OUT+'/PLAN.json':plan_sha})
    require(b['source_hashes']==expected,'Review source closure changed')
    pol=read(root,OUT+'/POLICY.json');require(pol==policy(read(root,BASE_POLICY)),'Frozen policy changed')
    data=build_dataset(read(root,PRE+'/EVENTS.json'),review,read(root,SESSIONS)['sessions'],expected,pol,b['review_sha256'])
    audit=dict(schema_version=1,stage='conditional_source_admission',policy_id=POLICY_ID,passed=True,plan_sha256=plan_sha,policy_sha256=sha(root,OUT+'/POLICY.json'),
        data_sha256=digest(encoded(data)),review_binding_sha256=binding_sha,independent_review_sha256=b['review_sha256'],counts=data['counts'],
        source_hashes=merge(expected,{OUT+'/REVIEW_BINDING.json':binding_sha}),source_code_hashes=p['source_code_hashes'],
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        checks={k:True for k in ('all24_events_preserved','all96_cells_independently_verified','both_FY2015_rejections_retained','accounting_flags_resolved_only_by_evidence','same_current_release_four_cells','strict_next_session_availability','causal_identity_windows','latest_invalid_blocks_fallback')},
        financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,limitations=pol['assumptions'])
    verify(root,expected);return data,audit


def build(root,plan_sha,binding_sha):
    require(not any(base.local(root,OUT+'/'+n).exists()for n in ('DATA.json','AUDIT.json','MANIFEST.json')),'Outputs already exist')
    data,audit=products(root,plan_sha,binding_sha);publish(root,OUT+'/DATA.json',data);publish(root,OUT+'/AUDIT.json',audit)
    manifest=dict(stage='conditional_source_admission_complete',plan_sha256=plan_sha,review_binding_sha256=binding_sha,output_hashes={n:sha(root,OUT+'/'+n)for n in ('DATA.json','AUDIT.json')})
    publish(root,OUT+'/MANIFEST.json',manifest);return dict(counts=data['counts'],output_hashes=manifest['output_hashes'])


def verify_outputs(root,plan_sha,binding_sha):
    require_sha(plan_sha,'plan');require_sha(binding_sha,'review binding')
    manifest=read(root,OUT+'/MANIFEST.json');require(manifest['plan_sha256']==plan_sha and manifest['review_binding_sha256']==binding_sha,'Manifest input binding')
    require(set(manifest['output_hashes'])=={'DATA.json','AUDIT.json'},'Exact outputs')
    verify(root,{OUT+'/'+n:h for n,h in manifest['output_hashes'].items()});data,audit=products(root,plan_sha,binding_sha)
    require(read(root,OUT+'/DATA.json')==data and read(root,OUT+'/AUDIT.json')==audit,'Reproduced mapping mismatch')
    return dict(passed=True,counts=data['counts'])


def main():
    cli=argparse.ArgumentParser();g=cli.add_mutually_exclusive_group(required=True)
    for flag in ('freeze-plan','bind-review','build','verify'):g.add_argument('--'+flag,action='store_true')
    cli.add_argument('--plan-sha');cli.add_argument('--review-sha');cli.add_argument('--binding-sha');a=cli.parse_args()
    result=freeze(ROOT)if a.freeze_plan else bind_review(ROOT,a.plan_sha,a.review_sha)if a.bind_review else build(ROOT,a.plan_sha,a.binding_sha)if a.build else verify_outputs(ROOT,a.plan_sha,a.binding_sha)
    print(base.json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
