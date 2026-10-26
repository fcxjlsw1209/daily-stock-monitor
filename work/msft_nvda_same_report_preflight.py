"""Frozen-plan, offline source preflight; no ratios, signs, ranks or returns.

Only original issuer-page cells and their provenance are emitted. Every annual
publication survives errors as an unadmitted event. Independent source review
is required before any financial or strategy use.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import os
from bs4 import BeautifulSoup
import profitability141_issuer_statements as old

ROOT=Path(__file__).resolve().parents[1]
SELF='work/msft_nvda_same_report_preflight.py'
TEST='work/test_msft_nvda_same_report_preflight.py'
OUT='outputs/monthly_msft_nvda_same_report_preflight_v1'
NOTE='work/after144_same_report_coverage_feasibility.json'
OLD_PLAN='outputs/monthly_profitability141_issuer_statements/EXTRACTION_PLAN.json'
NOTE_SHA='237c68ff8442fa39adbc1cfcc630a709dda03ee66d92686d7a2128007ad6455d'
METHOD='same_current_release_annual_comparison'
SCOPE=[(symbol,year) for symbol in ('MSFT','NVDA') for year in range(2015,2027)]
ACCOUNTING_PENDING={('MSFT',year) for year in (2018,2023,2025,2026)}
SPEC_KEYS=set('symbol fiscal_year release_date release_date_literal source_url raw_path raw_sha256 receipt_path receipt_sha256 receipt_entry_key current_period comparison_period comparison_end identity_anchor rows'.split())


def require(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    def bad(v):raise ValueError('Nonfinite JSON token '+v)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)


def path(root,relative):
    require(type(relative)is str and not Path(relative).is_absolute() and '..'not in Path(relative).parts,'Local relative path required')
    require(not any(v in relative.lower() for v in ('onedrive','cloudstorage')),'Cloud path forbidden')
    p=Path(root).resolve()/relative
    require(p.resolve()==p,'Symlink source forbidden');return p


def sha(root,relative):return digest(path(root,relative).read_bytes())
def read(root,relative,expected=None):
    raw=path(root,relative).read_bytes()
    if expected is not None:require(digest(raw)==expected,'Source SHA mismatch: '+relative)
    return decode(raw)


def verify(root,mapping):
    require(type(mapping)is dict and bool(mapping),'Nonempty hash closure required')
    for p,h in mapping.items():
        require(type(h)is str and re.fullmatch('[0-9a-f]{64}',h),'SHA256 required')
        require(sha(root,p)==h,'Source SHA mismatch: '+p)


def publish(root,relative,obj):
    target=path(root,relative);target.parent.mkdir(parents=True,exist_ok=True);tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.preflight-',delete=False)as f:
            tmp=Path(f.name);f.write(encoded(obj));f.flush();os.fsync(f.fileno())
        os.link(tmp,target)
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)


def day(value):
    require(type(value)is str,'Date string required');d=date.fromisoformat(value)
    require(d.isoformat()==value,'Canonical date required');return d


def next_session(release,sessions):
    require(type(sessions)is list and len(sessions)>1 and sessions==sorted(set(sessions)),'Ordered unique sessions required')
    for x in sessions:day(x)
    day(release);i=bisect_right(sessions,release)
    require(sessions[0]<=release and i<len(sessions),'Release outside pinned calendar')
    return sessions[i]


def schema():
    return dict(schema_version=1,method=METHOD,status='SOURCE_PREFLIGHT_ONLY',
        event_fields=['event_id','symbol','fiscal_year','release_date','available_from','identity','current','comparison','source','rejection_codes','review_flags','fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','cell_errors','four_cells_extracted'],
        period_fields=['start','end','days_inclusive','derived_not_direct_quote','evidence'],
        cell_fields=['value_decimal','displayed_amount','scale','currency_literal','currency_iso','period_end','available_from','source','locator','header_literals','unit_literal','statement_context','parsed_cell_sha256'],
        amount_rule='Plain exact decimal strings and separate positive integer scale; no normalization, ratios or signs.',
        time_rule='All four cells available only on first pinned session strictly after CURRENT release date.',
        period_rule='Both periods330..400 inclusive days; previous boundary evidence must already be public. Two FY2015 unknown starts remain null.',
        identity_rule='Own dated issuer/ticker anchor; validity from own next session through current fiscal end+550calendar days. No future event used.',
        vintage='Current-retrieval original-dated pages; first-public version and complete correction history are unverified.',
        currency='Same-current-report literal dollar unit; no inferred ISO or cross-report currency equivalence.',
        error_rule='Preserve all24 events and unparsed cells as null with reasons; hash/provenance mismatch aborts entire extraction.',
        accounting_rule='MSFT2018 restatement and2023/2025/2026 recast scope require independent table-level review. No restated comparative is available before its CURRENT release.',
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        financial_calculations_permitted=False,network_requests_permitted=False)


def validate_periods(spec):
    require(set(spec)==SPEC_KEYS,'Unexpected source specification fields')
    symbol,year=spec['symbol'],spec['fiscal_year']
    require(type(year)is int and (symbol,year)in SCOPE,'Out-of-scope issuer/year')
    current=deepcopy(spec['current_period']);previous=deepcopy(spec['comparison_period'])
    for period in (current,previous):
        require(type(period)is dict and set(period)=={'start','end','days_inclusive','derived_not_direct_quote','evidence'},'Period schema')
        require(period['derived_not_direct_quote']is True and type(period['evidence'])is list and period['evidence'],'Period evidence required')
        day(period['end'])
    require(previous['end']==spec['comparison_end'],'Comparison end mismatch')
    require(day(current['end']).year==year and day(previous['end']).year==year-1,'Period fiscal year mismatch')
    require(day(current['start'])==day(previous['end'])+timedelta(days=1),'Noncontiguous current annual period')
    errors=[]
    for which,period in (('current',current),('comparison',previous)):
        if period['start']is None:
            require(which=='comparison' and year==2015 and period['days_inclusive']is None,'Unexpected unknown duration')
            errors.append('unknown_comparison_period_start');continue
        n=(day(period['end'])-day(period['start'])).days+1
        require(type(period['days_inclusive'])is int and period['days_inclusive']==n and 330<=n<=400,'Annual duration mismatch')
    if year==2015:require(previous['start']is None,'FY2015 unknown boundary must be retained')
    for p in previous['evidence']:
        if 'release_date'in p:require(day(p['release_date'])<day(spec['release_date']),'Future comparison boundary evidence')
    require(day(current['end'])<day(spec['release_date']),'Publication precedes current fiscal end')
    return current,previous,errors


def select_cell(soup,row_spec,symbol,end,available,source,current=False):
    tables=soup.find_all('table');table=tables[row_spec['table_index_zero_based']]
    rows,grid,physical=old.table_grid(table);ri=row_spec['row_index_zero_based']
    kind=row_spec['kind'];expected='Gross margin'if symbol=='MSFT' else 'Gross profit'
    if kind=='year_end_assets':expected='Total assets'
    require(kind in ('annual_gross_amount','year_end_assets') and row_spec['row_label']==expected,'Exact financial row label required')
    labels=[i for i,r in enumerate(rows)if any(old.norm(c.get_text(' ',strip=True)).lower()==expected.lower()for c in r.find_all(['td','th'],recursive=False))]
    require(labels==[ri],'Ambiguous or shifted statement label')
    context=old.statement_context(table);text=old.norm(table.get_text(' ',strip=True))
    old.validate_statement(context,text,kind,symbol)
    unit=old.norm(row_spec['units']['literal']);require(unit in old.norm(context+' '+text),'Planned scale literal missing')
    scale=old.scale_from(unit);require('$'in text,'Dollar unit missing')
    stop=old.header_stop(rows);matches=[]
    for (r,pi),item in physical.items():
        if r!=ri:continue
        headers=old.headers_for(grid,ri,item['columns'][0],stop)
        if old.header_dates(headers)!=[end]:continue
        if kind=='annual_gross_amount' and not any('Twelve Months Ended'in h for h in headers):continue
        if old.norm(item['text'])in ('','$'):continue
        value=old.exact_amount(item['text'])
        require(re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',value),'Plain exact decimal required')
        matches.append((pi,item,headers,value))
    require(len(matches)==1,'Exactly one annual date-matched monetary cell required')
    pi,item,headers,value=matches[0]
    if current:
        require(len(row_spec['matching_cells'])==1 and pi==row_spec['matching_cells'][0]['source_cell_index'] and item['columns']==row_spec['matching_cells'][0]['grid_columns'],'Current cell differs from reviewed original locator')
    cell=rows[ri].find_all(['td','th'],recursive=False)[pi]
    return dict(value_decimal=value,displayed_amount=item['text'],scale=str(scale),currency_literal='$',currency_iso=None,
        period_end=end,available_from=available,source=deepcopy(source),
        locator=dict(table_index_zero_based=row_spec['table_index_zero_based'],row_index_zero_based=ri,physical_cell_index_zero_based=pi,logical_columns=item['columns']),
        header_literals=headers,unit_literal=unit,statement_context=context,parsed_cell_sha256=digest(str(cell).encode()))


def extract_event(spec,raw,sessions):
    current,previous,rejections=validate_periods(spec)
    available=next_session(spec['release_date'],sessions)
    through=day(current['end'])+timedelta(days=550)
    require(day(available)<=through,'Publication after causal identity expiry')
    soup=BeautifulSoup(raw,'html.parser');text=old.norm(soup.get_text(' ',strip=True))
    identity_ok=old.norm(spec['identity_anchor'])in text
    if not identity_ok:rejections.append('own_issuer_ticker_anchor_missing')
    if old.norm(spec['release_date_literal'])not in text:rejections.append('current_publication_date_literal_missing')
    if (spec['symbol'],spec['fiscal_year'])in ACCOUNTING_PENDING:rejections.append('accounting_context_requires_independent_review')
    source={k:spec[k]for k in ('raw_path','raw_sha256','source_url','receipt_path','receipt_sha256','receipt_entry_key')}
    event=dict(event_id=f'{spec["symbol"]}:FY{spec["fiscal_year"]}:{spec["release_date"]}',symbol=spec['symbol'],fiscal_year=spec['fiscal_year'],
        release_date=spec['release_date'],available_from=available,source=source,
        identity=dict(anchor=spec['identity_anchor'],anchor_in_current_source=identity_ok,valid_from=available,valid_through=through.isoformat(),certified_daily_security_master=False),
        current=dict(period=current,gross_profit=None,assets=None),comparison=dict(period=previous,gross_profit=None,assets=None),
        rejection_codes=rejections,review_flags=['independent_comparative_cell_review_required','historical_first_public_version_not_certified'],
        first_public_version_verified=False,complete_correction_history_verified=False,fact_admitted=False,model_admitted=False,strategy_admitted=False)
    errors=[]
    for role,period in (('current',current),('comparison',previous)):
        for rs in spec['rows']:
            name='gross_profit'if rs['kind']=='annual_gross_amount'else'assets'
            try:event[role][name]=select_cell(soup,rs,spec['symbol'],period['end'],available,source,current=(role=='current'))
            except (ValueError,KeyError,IndexError,TypeError)as exc:
                errors.append(dict(role=role,measure=name,reason=str(exc)))
                event['rejection_codes'].append('cell_unresolved:'+role+':'+name)
    event['cell_errors']=errors
    event['four_cells_extracted']=not errors
    return event


def make_specs(note,prior_plan):
    result=[]
    for n in note['rows']:
        s=next(x for x in prior_plan['source_scope']['sources']if x['symbol']==n['symbol']and x['fiscal_year']==n['fiscal_year'])
        source=dict(path=s['raw_path'],sha256=s['raw_sha256'],release_date=s['release_date'])
        current={k:n['current_period'][k]for k in ('start','end','days_inclusive')};current.update(derived_not_direct_quote=True,evidence=[source])
        p=n['comparison_period_candidate']
        previous=dict(start=None,end=n['comparison_end_from_same_report_headers'],days_inclusive=None,derived_not_direct_quote=True,evidence=[source])
        if p:
            previous.update({k:p[k]for k in ('start','end','days_inclusive')})
            b=p['boundary_evidence'];previous['evidence']=[dict(path=b['raw_path'],sha256=b['raw_sha256'],release_date=p['boundary_source_release_date'])]
        spec={k:s[k]for k in ('symbol','fiscal_year','release_date','release_date_literal','source_url','raw_path','raw_sha256','receipt_path','receipt_sha256','receipt_entry_key','rows')}
        spec.update(current_period=current,comparison_period=previous,comparison_end=previous['end'],identity_anchor=n['causal_identity']['own_release_anchor'])
        # Unknown FY2015 boundary evidence refers to the CURRENT page solely for its known end.
        if not p:previous['evidence']=[dict(path=s['raw_path'],sha256=s['raw_sha256'])]
        validate_periods(spec);result.append(spec)
    require([(s['symbol'],s['fiscal_year'])for s in result]==SCOPE,'Exact24-event source order required')
    return result


def freeze(root):
    require(not path(root,OUT+'/PLAN.json').exists(),'Plan already frozen')
    note=read(root,NOTE,NOTE_SHA);prior=read(root,OLD_PLAN)
    bindings=dict(note['input_sha256']);bindings[NOTE]=NOTE_SHA
    cal=read(root,prior['inputs']['calendar_plan']['path']);bindings.update(cal['bindings'])
    for p in ('work/profitability141_issuer_statements.py','run_local.sh','local_run.py','local.sandbox.sb'):bindings[p]=sha(root,p)
    code={p:sha(root,p)for p in (SELF,TEST)}
    tests=read(root,OUT+'/TESTS.json')
    require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('source_code_hashes')==code,'Current synthetic tests required')
    for key in ('errors','failures','real_source_cells_extracted'):require(type(tests.get(key))is int and tests[key]==0,'Clean synthetic test scope')
    verify(root,bindings);specs=make_specs(note,prior)
    publish(root,OUT+'/SCHEMA.json',schema());bindings[OUT+'/SCHEMA.json']=sha(root,OUT+'/SCHEMA.json');bindings[OUT+'/TESTS.json']=sha(root,OUT+'/TESTS.json')
    plan=dict(schema_version=1,stage='frozen_source_only_extraction_plan',created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,
        method=METHOD,source_hashes=bindings,source_code_hashes=code,sources=specs,sessions=prior['inputs']['calendar_sessions'],
        expected=dict(events=24,monetary_cells=96,unknown_comparison_starts=2),accounting_pending=[list(x)for x in sorted(ACCOUNTING_PENDING)],
        financial_calculations_permitted=False,network_requests_permitted=False,fact_admission_permitted=False,outputs=['EVENTS.json','AUDIT.json','MANIFEST.json'])
    publish(root,OUT+'/PLAN.json',plan)
    return dict(plan_sha256=sha(root,OUT+'/PLAN.json'),schema_sha256=sha(root,OUT+'/SCHEMA.json'),source_files=len(bindings),events=len(specs))


def products(root,plan_sha):
    plan=read(root,OUT+'/PLAN.json',plan_sha)
    require(plan['runtime']==sys.version and plan['stage']=='frozen_source_only_extraction_plan' and plan['method']==METHOD,'Frozen source plan/runtime')
    require(all(plan[k]is False for k in ('financial_calculations_permitted','network_requests_permitted','fact_admission_permitted')),'Source-only plan flags')
    verify(root,plan['source_hashes']);verify(root,plan['source_code_hashes'])
    require(plan['source_code_hashes']=={p:sha(root,p)for p in (SELF,TEST)}and read(root,OUT+'/SCHEMA.json')==schema(),'Source implementation/schema changed')
    require([(s['symbol'],s['fiscal_year'])for s in plan['sources']]==SCOPE,'Exact24-event source scope')
    sessions=read(root,plan['sessions']['path'],plan['sessions']['sha256'])['sessions'];events=[]
    for spec in plan['sources']:
        raw=path(root,spec['raw_path']).read_bytes();require(digest(raw)==spec['raw_sha256'],'Raw source changed')
        receipt=read(root,spec['receipt_path'],spec['receipt_sha256'])
        if spec['receipt_entry_key']is not None:receipt=receipt[spec['receipt_entry_key']]
        old.receipt_check(raw,receipt,spec)
        events.append(extract_event(spec,raw,sessions))
    count=sum(e[r][k]is not None for e in events for r in ('current','comparison')for k in ('gross_profit','assets'))
    counts=dict(events=len(events),extracted_monetary_cells=count,complete_four_cell_events=sum(e['four_cells_extracted']for e in events),
        events_without_rejection_codes=sum(not e['rejection_codes']for e in events),events_with_rejection_codes=sum(bool(e['rejection_codes'])for e in events),fact_admissions=0,model_admissions=0,strategy_admissions=0)
    data=dict(schema_version=1,method=METHOD,status='SOURCE_PREFLIGHT_REQUIRES_INDEPENDENT_REVIEW',events=events,counts=counts,
        first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
        ratios_computed=0,signs_computed=0,ranks_computed=0,model_fits=0,returns_computed=0)
    audit=dict(schema_version=1,stage='source_preflight',passed=True,all_cells_extracted=(count==96),plan_sha256=plan_sha,counts=counts,
        data_sha256=digest(encoded(data)),source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],
        independent_source_review_passed=False,financial_calculations=0,network_requests=0,
        checks=dict(all24_events_preserved=len(events)==24,all_cells_use_current_availability=all(e[r][k]is None or e[r][k]['available_from']==e['available_from']for e in events for r in ('current','comparison')for k in ('gross_profit','assets')),all_admissions_false=True),
        rejection_summary=[dict(event_id=e['event_id'],codes=e['rejection_codes'],cell_errors=e['cell_errors'])for e in events if e['rejection_codes']])
    verify(root,plan['source_hashes']);verify(root,plan['source_code_hashes'])
    return data,audit


def build(root,plan_sha):
    require(not any(path(root,OUT+'/'+n).exists()for n in ('EVENTS.json','AUDIT.json','MANIFEST.json')),'Outputs already exist')
    data,audit=products(root,plan_sha)
    publish(root,OUT+'/EVENTS.json',data);publish(root,OUT+'/AUDIT.json',audit)
    manifest=dict(plan_sha256=plan_sha,output_hashes={n:sha(root,OUT+'/'+n)for n in ('EVENTS.json','AUDIT.json')})
    publish(root,OUT+'/MANIFEST.json',manifest)
    return dict(counts=data['counts'],all_cells_extracted=audit['all_cells_extracted'],output_hashes=manifest['output_hashes'])


def verify_outputs(root,plan_sha):
    manifest=read(root,OUT+'/MANIFEST.json');require(manifest['plan_sha256']==plan_sha and set(manifest['output_hashes'])=={'EVENTS.json','AUDIT.json'},'Manifest scope')
    verify(root,{OUT+'/'+n:h for n,h in manifest['output_hashes'].items()})
    data,audit=products(root,plan_sha)
    require(read(root,OUT+'/EVENTS.json')==data and read(root,OUT+'/AUDIT.json')==audit,'Source reproduction mismatch')
    return dict(passed=True,counts=data['counts'])


def main():
    cli=argparse.ArgumentParser();group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-plan',action='store_true');group.add_argument('--build',action='store_true');group.add_argument('--verify',action='store_true');cli.add_argument('--plan-sha')
    a=cli.parse_args();print(json.dumps(freeze(ROOT)if a.freeze_plan else build(ROOT,a.plan_sha)if a.build else verify_outputs(ROOT,a.plan_sha),sort_keys=True))
if __name__=='__main__':main()
