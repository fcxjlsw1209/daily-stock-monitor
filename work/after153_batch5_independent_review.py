"""Independent PLTR/PYPL/TMO source review; no producer imports or ratios."""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from bs4 import BeautifulSoup, NavigableString
from after149_batch1_independent_review import Reader, grid_from_html, parsed_dates, signed_text, norm, sha, require
from after150_batch2_independent_review import verify_published_cell
from after151_batch3_independent_review import receipt_evidence, verify_accounting, position_counts

ROOT=Path(__file__).resolve().parents[1]
BASE='outputs/monthly_financial_coverage_batch5_v1'
PLAN,EXTRACTIONS=BASE+'/PLAN.json',BASE+'/EXTRACTIONS.json'
CONTEXT='work/after153_next_source_context.json'
POLICY='work/after153_batch5_admission_policy.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SELF='work/after153_batch5_independent_review.py'
TEST='work/test_after153_batch5_independent_review.py'
OUTPUT='work/after153_batch5_independent_review.json'
ACCOUNTING='work/after153_batch5_accounting_review.json'
FIELDS=('annual_net_income','current_total_assets','prior_total_assets')
STATUS='standard_HTML_annual_header_context_located_not_admitted'
CATALOG_KEYS=('event_id','symbol','fiscal_year','publication_date','available_session','available_session_verified','official_actual_period_end','source','source_kind','original_report_bytes_verified','context_status','context_holds')
ANNUAL=re.compile(r'(?:years?|(?:twelve|12) months?)\s+ended(?:\s+[A-Za-z]+\s+\d{1,2},?(?:\s+\d{4})?)?',re.I)
SHORT=re.compile(r'(?:quarters?|(?:three|six|nine|3|6|9) months?)\s+ended',re.I)
BAD=re.compile(r'non[\s\-\u2010-\u2015]*gaap|pro\s+forma|reconciliation|as\s+adjusted',re.I)
OTHER=re.compile(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b',re.I)
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)'
DATE_CELL=re.compile(r'(?:'+MONTH+r'\s+\d{1,2}\s*,?\s*\d{4}|\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))',re.I)
PUNCT=re.compile(r'[\s$()\-\u2212]+')
BODY=re.compile(r'(?:assets|current assets|(?:net |total )?revenues?|(?:net )?sales|cash(?: and cash equivalents)?|net (?:income|loss))\s*:?',re.I)
ISSUERS={'PLTR':'Palantir Technologies Inc.','PYPL':'PayPal Holdings, Inc.','TMO':'Thermo Fisher Scientific Inc.'}
HOSTS={'PLTR':{'d18rn0p25nwr6d.cloudfront.net'},'PYPL':{'newsroom.paypal-corp.com'},'TMO':{'india.newsroom.thermofisher.com'}}
SCALES={'PLTR':'1000','PYPL':'1000000','TMO':'1000000'}
FULL=[('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2026)]+[('TMO',y) for y in range(2012,2023)]
SELECTED=[('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2020)]+[('TMO',y) for y in range(2012,2022)]
CENSUS='outputs/monthly_financial_coverage_census_v1/CENSUS.json'
MONTH_DAY=re.compile(r'(?:(?:as of|(?:years?|(?:three|six|nine|twelve|3|6|9|12) months?) ended)\s+)?('+MONTH+r')\s+(\d{1,2}),?',re.I)
TITLE=re.compile(r'\bconsolidated\s+(?:statements?\s+of\s+(?:income|operations|earnings)|balance\s+sheets?)\b',re.I)


def source_files():
    return [SELF,TEST,'work/after153_batch5_accounting_review.py','work/after149_batch1_independent_review.py','work/after150_batch2_independent_review.py','work/after151_batch3_independent_review.py']


def relative(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(not p.is_symlink(),'Symlink input')
    return str(p.resolve().relative_to(ROOT))


def dates(text):
    result=parsed_dates(re.sub(r'\s+,',',',text))
    for m,d,y in re.findall(r'(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?!\d)',text):
        try:result.add(date(int(y) if len(y)==4 else 2000+int(y),int(m),int(d)).isoformat())
        except ValueError:pass
    return result


def date_groups(headers,row,annual,header_stop=None):
    """Resolve full dates or layered month/day+year within one annual domain."""
    parents=[c for c in headers if ANNUAL.fullmatch(c['text'])]
    if annual:
        require(len(parents)==1,'Unique full annual parent required')
        domain=set(parents[0]['columns'])
        require(not any(SHORT.search(c['text']) and set(c['columns']) & domain for c in headers),'Annual and interim domains overlap')
    else:
        label=min(row,key=lambda c:min(c['columns']))
        domain=set(range(max(label['columns'])+1,max(max(c['columns']) for c in row)+1))
    def header_span_ok(c):
        return (c['rowspan']==1 or
                (type(header_stop) is int and type(c.get('row')) is int and
                 c['rowspan']>0 and c['row']+c['rowspan']<=header_stop))
    for c in headers:
        if c['text'].startswith('%') or c['text'].lower()=='revenues':
            domain.difference_update(c['columns'])
    groups={}
    for col in sorted(domain):
        full=[];years=[];monthdays=[]
        for c in headers:
            if col not in c['columns']:continue
            text=c['text']
            if DATE_CELL.fullmatch(text):
                found=dates(text);require(len(found)==1,'Invalid full date cell')
                full.append(next(iter(found)))
                require(header_span_ok(c) and set(c['columns'])<=domain,'Date crosses header/domain')
            elif re.fullmatch(r'(?:19|20)\d{2}',text):
                years.append(text)
                require(header_span_ok(c) and set(c['columns'])<=domain,'Year crosses header/domain')
            else:
                md=MONTH_DAY.fullmatch(text)
                if md:monthdays.append(md.group(1)+' '+md.group(2))
        require(len(full)<=1 and len(years)<=1,'Duplicate full date/year anchor')
        require(len(set(monthdays))<=1,'Conflicting month/day headers')
        resolved=set(full)
        if years and monthdays:
            composite=dates(monthdays[0]+', '+years[0])
            require(len(composite)==1,'Invalid compound annual date')
            resolved.update(composite)
        require(len(resolved)<=1,'Conflicting layered dates')
        if resolved:groups.setdefault(next(iter(resolved)),[]).append(col)
    require(len(groups)==2,'Wrong annual/instant date count')
    owners={}
    for col in domain:
        dist={day:min(abs(col-n) for n in cols) for day,cols in groups.items()}
        nearest=[d for d,n in dist.items() if n==min(dist.values())]
        owners[col]=nearest[0] if len(nearest)==1 else None
    return groups,owners


def external_header(table):
    """Bounded nearby raw text, stopping at the first prior table/title."""
    parts=[]
    for node in table.previous_elements:
        if getattr(node,'name',None)=='table':break
        if not isinstance(node,NavigableString):continue
        if node.find_parent('table') is not None:break
        text=norm(str(node))
        if not text:continue
        parts.insert(0,text)
        combined=' '.join(parts)
        if TITLE.search(combined):return parts
        if len(parts)>25 or len(combined)>1200:break
    return []


def statement(soup,context,field):
    spec=context['fields'][field];ti,ri=spec['table'],spec['row'];tables=soup.find_all('table')
    require(type(ti) is int and 0<=ti<len(tables) and type(ri) is int and ri>=0,'Invalid table/row')
    table=tables[ti];require(table.find('table') is None,'Nested statement')
    cells,grid=grid_from_html(table)
    fixed=[s for s in context['statement_contexts'] if s['table']==ti]
    require(len(fixed)==1,'Unique frozen statement required');fixed=fixed[0];stop=fixed['header_stop_row']
    require(type(stop) is int and 0<stop<=ri,'Invalid header stop')
    headers=[c for c in cells.values() if c['row']<stop]
    for c in headers:
        before=[x for x in cells.values() if x['row']==c['row'] and x['physical_cell']<c['physical_cell']]
        require(not (BODY.fullmatch(c['text']) and all(not x['text'] for x in before)),'Body inside header')
    for h in fixed['headers']+spec['header_evidence']:
        c=cells.get((h['row'],h['physical_cell']))
        require(c is not None and c['row']<stop and all(c[k]==h[k] for k in ('columns','rowspan','text')),'Frozen header mismatch')
    annual=field==FIELDS[0]
    core=r'\bconsolidated\s+'+(r'statements?\s+of\s+(?:income|operations|earnings)' if annual else r'balance\s+sheets?')
    title_re=re.compile(core,re.I)
    titles=[c['text'] for c in headers if title_re.search(c['text'])]
    if not titles:
        for c in headers:
            if not re.fullmatch(r'(?:unaudited\s+)?(?:condensed\s+)?consolidated',c['text'],re.I):continue
            for nxt in headers:
                if nxt['row']==c['row']+1 and min(nxt['columns'])==min(c['columns']):
                    combined=c['text']+' '+nxt['text']
                    if title_re.search(combined):titles.append(combined)
    outside=external_header(table) if fixed.get('external_metadata') else []
    for text in fixed.get('external_metadata',[]):
        require(norm(text) in norm(' '.join(outside)),'External metadata not found before statement')
    if not titles:titles=[s for s in outside if title_re.search(s)]
    require(len(titles)==1 and not any(BAD.search(s) for s in [c['text'] for c in headers]+outside),'Formal statement required')
    require(fixed['kind']==('income' if annual else 'balance'),'Wrong statement kind')
    units=[c for c in headers if re.search(r'\bin\s+(?:thousands|millions|billions)\b',c['text'],re.I)]
    units += [dict(row=-1,physical_cell=i,text=s,external_source_text=True,markup_sha256=sha(s.encode()))
              for i,s in enumerate(outside) if re.search(r'\bin\s+(?:thousands|millions|billions)\b',s,re.I)]
    names={n.lower() for c in units for n in re.findall(r'\bin\s+(thousands|millions|billions)\b',c['text'],re.I)}
    require(len(names)==1,'Missing/conflicting scale')
    scale={'thousands':'1000','millions':'1000000','billions':'1000000000'}[next(iter(names))]
    require(spec['scale']==fixed['scale']==scale==SCALES[context['symbol']] and spec['currency_literal']=='$' and spec['currency_iso'] is None,'Scale/currency context mismatch')
    text=table.get_text(' ',strip=True);require('$' in text and not OTHER.search(text),'Dollar literal absent or competing currency')
    label=re.compile(r'(?:consolidated\s+)?net (?:income|earnings|loss)(?:\s*\((?:income|loss)\))?' if annual else r'total assets',re.I)
    labels=[]
    for c in cells.values():
        if not label.fullmatch(c['text'].rstrip(':')):continue
        require(all(not x['text'] for x in cells.values() if x['row']==c['row'] and x['physical_cell']<c['physical_cell']),'Target follows another row label')
        labels.append(c)
    require(len(labels)==1 and labels[0]['row']==ri and labels[0]['physical_cell']==spec['label_cell'] and labels[0]['rowspan']==1,'Exact consolidated total row missing or duplicated')
    require(labels[0]['text'].rstrip(':')==spec['label'].rstrip(':'),'Label context mismatch')
    row=sorted([c for c in cells.values() if c['row']==ri],key=lambda c:c['physical_cell'])
    require([{k:c[k] for k in ('row','physical_cell','columns','rowspan')} for c in row]==spec['row_geometry_without_values'],'Frozen row geometry mismatch')
    return spec,cells,grid,headers,units,titles[0],row


def source_field(soup,context,field):
    spec,cells,grid,headers,units,title,row=statement(soup,context,field)
    stop=next(s['header_stop_row'] for s in context['statement_contexts'] if s['table']==spec['table'])
    groups,owners=date_groups(headers,row,field==FIELDS[0],stop)
    current,prior=context['periods']['current_end'],context['periods']['prior_end']
    require(current==context['official_actual_period_end'] and current in groups and prior in groups,'Fiscal endpoints not both printed')
    require(all(prior<=d<=current for d in groups),'Unexpected instant outside annual endpoints')
    day=prior if field==FIELDS[2] else current;cols=groups[day];ri=spec['row']
    require(cols==spec['logical_columns'] and day==spec['as_of'],'Independent date columns differ')
    require(all((ri,x) in grid and grid[ri,x]['row']==ri for x in cols),'Missing/cross-row cells')
    chosen=[c for c in row if set(c['columns']) & set(cols)];require(chosen,'Missing physical value')
    for ix in (row.index(chosen[0])-1,row.index(chosen[-1])+1):
        if 0<=ix<len(row) and row[ix]['text'] and PUNCT.fullmatch(row[ix]['text']) and any(x in owners for x in row[ix]['columns']) and not any(d!=day and set(row[ix]['columns'])<=set(cs) for d,cs in groups.items()):
            require(all(owners.get(x)==day for x in row[ix]['columns']),'Sign crosses/ambiguously belongs to date');chosen.append(row[ix])
    chosen.sort(key=lambda c:c['physical_cell'])
    for c in chosen:require(c['rowspan']==1 and all(owners.get(x)==day for x in c['columns']),'Cross-date/cross-row amount')
    for c in row:
        require(not (c['text'] and any(x in owners and owners[x] is None for x in c['columns'])),'Nonempty cell has ambiguous annual ownership')
        if c not in chosen and c['text'] and any(owners.get(x)==day for x in c['columns']):raise ValueError('Unconsumed nonempty owned-period cell')
    display=' '.join(c['text'] for c in chosen if c['text']);value=signed_text(display)
    require(Decimal(value).is_finite() and (field==FIELDS[0] or Decimal(value)>0),'Invalid monetary amount/assets')
    return dict(table=spec['table'],row=ri,value_decimal=value,displayed_amount=display,scale=spec['scale'],as_of=day,columns=cols,physical_cells=chosen,header_cells=[c for c in headers if set(c['columns']) & set(cols)],scale_cells=units,statement_title=title,all_date_groups=groups)


def review_position(soup,context,field,event,period):
    cell=event['fields'][field]
    if cell is not None:
        reading=source_field(soup,context,field)
        return verify_published_cell(cell,reading,field,event,period),reading
    gaps=[g for g in event['gaps'] if g.get('field')==field]
    require(len(gaps)==1 and type(gaps[0].get('reason')) is str and gaps[0]['reason'],'Null requires one explicit frozen gap')
    # A readable source does not authorize filling the frozen null.
    reading=source_field(soup,context,field)
    return dict(is_null=True,null_reason_verified=True,source_value_present=True,frozen_null_preserved=True,null_classification='producer_gap_despite_independently_readable_source',frozen_gap=gaps[0],independent_source_reading=reading),reading


def periods(context,event,policy):
    p=context['periods'];end,prior=p['current_end'],p['prior_end'];start=(date.fromisoformat(prior)+timedelta(days=1)).isoformat();length=(date.fromisoformat(end)-date.fromisoformat(start)).days+1
    expected=dict(period_start=start,period_end=end,days_inclusive=length)
    require(p['proposed_current_start']==start and p['start_is_source_quoted'] is p['start_assumption_approved'] is False and p['start_assumption']=='same_current_report_prior_annual_end_plus_one_calendar_day','Original proposed boundary changed')
    require(event['periods']['current']==expected and event['periods']['comparison']==dict(period_end=prior),'Published annual periods mismatch')
    aid=context['symbol']+'_consecutive_annual_boundaries_v1';setting=policy['periods']['boundary_assumptions'][context['symbol']]
    require(setting['not_a_quoted_fact'] is True and setting['requires_independent_header_verification'] is True and setting['assumption_id']==aid and policy['periods']['allowed_days']==[365,366] and policy['periods']['required_year_end']=='12-31','Explicit annual assumption missing')
    require(event['periods']['start_basis']=='derived_same_report_prior_annual_end_plus_one_not_directly_quoted' and event['periods']['start_assumption_certified'] is False and event['periods']['assumption_id']==aid,'Derived start upgraded or mislabelled')
    require(length in (365,366),'Not a calendar-year annual interval')
    require(end==str(context['fiscal_year'])+'-12-31' and prior==str(context['fiscal_year']-1)+'-12-31','Issuer annual endpoints disagree')
    require(p.get('days_inclusive')==length and p.get('annual_date_binding')=='same_report_full_date_headers','Context annual date binding differs')
    return expected


def review_event(reader,row,event,context,sessions,policy,accounting=None):
    require(all(event[k]==context[k]==row[k] for k in ('event_id','symbol','fiscal_year')),'Event identity mismatch')
    require(event['event_id']==event['symbol']+'_'+context['publication_date'],'ID/date mismatch')
    require(context['source_kind']=='HTML_original' and context['original_report_bytes_verified'] is True,'Original HTML required')
    require(event['source']==context['source'],'Source record changed');source=event['source'];raw=reader.read(row['source_path'],row['source_sha256'])
    require(relative(source['path'])==relative(row['source_path'])==relative(context['evidence_path']) and sha(raw)==source['sha256']==context['evidence_sha256'],'Raw/context mismatch')
    require(source['url']==context['source_url'],'Conflicting source URL')
    if accounting is not None:verify_accounting(accounting,event,raw)
    receipt=receipt_evidence(reader,context,raw);soup=BeautifulSoup(raw,'html.parser')
    for n in soup(['script','style']):n.decompose()
    text=norm(soup.get_text(' ',strip=True));symbol=event['symbol'];issuer=ISSUERS[symbol]
    require(issuer.lower() in text.lower() and re.search(r'\b'+symbol+r'\b',text),'Issuer/security anchor missing')
    require(context['identity_anchors'] and all(norm(a).lower() in text.lower() for a in context['identity_anchors']),'Frozen identity anchors changed')
    require(event['identity']['issuer']==issuer and event['identity']['symbol']==symbol and event['identity']['source_anchors']==context['identity_anchors'] and event['identity']['certified_daily_security_master'] is False,'Published source identity mismatch')
    url=urlsplit(source['url']);require(url.scheme=='https' and url.hostname in HOSTS[symbol] and not url.username and not url.password and url.port in (None,443),'Source publisher URL mismatch')
    release,available=context['publication_date'],context['available_session']
    require(release in dates(text) and event['publication_date']==release and event['available_from']==available and context['official_actual_period_end']<release<available,'Release/availability mismatch')
    require(sessions and sessions==sorted(set(sessions)),'Invalid sessions');clock=sessions[0]<=release<sessions[-1]
    if clock:require(sessions[bisect_right(sessions,release)]==available,'Not first next pinned session')
    require(context['available_session_verified'] is clock,'Context session status differs')
    require(set(event['fields'])==set(FIELDS) and all(event[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted','first_public_time_verified','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master')),'Fields/admission flags')
    require(type(event.get('financial_calculations')) is int and event['financial_calculations']==0,'Source-only calculation counter')
    require(event.get('complete_required_fields') is all(event['fields'][f] is not None for f in FIELDS),'Completeness flag differs from physical positions')
    require(event['context_holds']==context['context_holds'] and event['accounting_review_flags']==context['accounting_terms'] and event['accounting_context_numbers_redacted']==context['accounting_context_numbers_redacted'],'Inherited context lost')
    expected=periods(context,event,policy);readings={};fields={}
    for f in FIELDS:fields[f],readings[f]=review_position(soup,context,f,event,expected)
    require({g['field'] for g in event['gaps']}=={f for f in FIELDS if event['fields'][f] is None},'Gaps differ from null positions')
    header=' '.join(c['text'] for r in readings.values() for c in r['header_cells'])
    require(not re.search(r'transition period|stub period|change.*fiscal year',header,re.I),'Contradictory annual header')
    return dict(event_id=event['event_id'],symbol=symbol,fiscal_year=event['fiscal_year'],passed=True,metadata_verified=True,identity_verified=True,issuer_id=issuer,security_id=symbol,periods_verified=True,currency_verified=True,clock_verified=clock,clock_scope='first_pinned_next_session' if clock else 'outside_pinned_calendar_not_certified',release_date=release,available_from=available,source=source,receipt=receipt,fields=fields,retained_context_holds=event['context_holds'],accounting_review_flags=event['accounting_review_flags'],accounting_context_numbers_redacted=event['accounting_context_numbers_redacted'],accounting_comparability_admitted=False,start_is_source_quoted=False,first_public_time_verified=False,first_public_version_verified=False,certified_daily_security_master=False,source_admitted=False,boundary_assumption_verified=True,assumption_id=symbol+'_consecutive_annual_boundaries_v1',derived_start=expected['period_start'],boundary_evidence={f:dict(table=readings[f]['table'],date=readings[f]['as_of'],headers=readings[f]['header_cells']) for f in FIELDS})


def verify_population(plan,data,context,policy):
    docs=context['events'];ids=[e['event_id'] for e in docs];chosen=[e for e in docs if e['context_status']==STATUS]
    require(len(ids)==len(set(ids))==28 and [(e['symbol'],e['fiscal_year']) for e in docs]==FULL,'Full28 catalog required')
    require([(e['symbol'],e['fiscal_year']) for e in chosen]==SELECTED,'Fixed21 report scope')
    require([e['event_id'] for e in plan['records']]==[e['event_id'] for e in data['events']]==[e['event_id'] for e in chosen],'Extraction event order')
    require(plan['catalog_ids']==data['catalog_ids']==ids and plan['catalog_events']==28,'Catalog missing/reordered')
    projection=[{k:e[k] for k in CATALOG_KEYS} for e in docs]
    require(policy['fixed_catalog_metadata']==plan['catalog_metadata']==data['catalog_metadata']==projection,'Policy/plan full28 catalog metadata changed')
    require(policy['required_fields']==list(FIELDS),'Source policy fields')
    require(data['counts']==dict(events=21,by_symbol={'PLTR':6,'PYPL':5,'TMO':10},nonnull_fields=sum(e['fields'][f] is not None for e in data['events'] for f in FIELDS),unprocessed_catalog_events=7),'Extraction counts')
    require(sum(e['source_kind']=='HTML_original' and e['original_report_bytes_verified'] is True for e in docs)==27 and sum(e['source_kind']=='index_or_unresolved_not_original_report' and e['original_report_bytes_verified'] is False for e in docs)==1,'Original/index distinction changed')
    return {e['event_id']:e for e in chosen}


def verify_original_catalog(census,plan,data,context):
    rows=[e for e in census['annual_documents'] if e['symbol'] in ISSUERS]
    require(len(rows)==28 and rows==plan['original_catalog_metadata']==data['original_catalog_metadata'],'Original census records changed')
    for raw,ctx in zip(rows,context['events']):
        require(all(ctx.get(k)==v for k,v in raw.items() if k!='source_kind')
                and ctx.get('census_source_kind')==raw['source_kind'],'Original/context catalog mismatch')


def audit(plan_sha,extractions_sha,publish=False):
    require(all(re.fullmatch(r'[0-9a-f]{64}',x or '') for x in (plan_sha,extractions_sha)),'Explicit frozen hashes required')
    reader=Reader();plan,data=reader.load(PLAN,plan_sha),reader.load(EXTRACTIONS,extractions_sha)
    require(data['plan_sha256']==plan_sha and plan['context']==CONTEXT and plan['policy']==POLICY
            and plan['stage']=='frozen_source_only_plan'
            and data['status']=='source_values_extracted_pending_independent_review','Frozen plan linkage')
    for p,d in {**plan['inputs'],**plan['code']}.items():reader.read(p,d)
    for p in source_files():reader.read(p)
    context,policy=reader.load(CONTEXT),reader.load(POLICY)
    require(data['catalog_ref']==dict(path=CONTEXT,sha256=reader.hashes[CONTEXT])
            and plan['inputs'][CONTEXT]==policy['input_hashes'][CONTEXT]==reader.hashes[CONTEXT],'Context hash binding')
    require(policy['policy_id']=='pltr_pypl_tmo_annual_ni_same_report_assets_batch5_v1','Wrong source policy')
    for p,d in policy.get('input_hashes',{}).items():reader.read(p,d)
    for key in ('new_financial_admissions','new_strategy_trials','ratios_computed','returns_computed'):
        require(type(data.get(key)) is int and data[key]==0,'No prior admission or financial calculations')
    contexts=verify_population(plan,data,context,policy)
    verify_original_catalog(reader.load(CENSUS),plan,data,context)
    sessions=reader.load(SESSIONS)['sessions'];events=[];failures=[]
    accounting=reader.load(ACCOUNTING)
    require(accounting['schema_version']==1 and accounting['stage']=='independent_field_specific_accounting_review' and accounting['passed'] is True and accounting['plan_sha256']==plan_sha and accounting['extractions_sha256']==extractions_sha,'Accounting review linkage')
    for p,d in accounting['input_hashes'].items():reader.read(p,d)
    decisions={e['event_id']:e for e in accounting['events']};require(len(decisions)==len(accounting['events'])==21 and set(decisions)==set(contexts),'All21 accounting decisions')
    for row,event in zip(plan['records'],data['events']):
        try:
            reviewed=review_event(reader,row,event,contexts[row['event_id']],sessions,policy,decisions[event['event_id']])
            reviewed['accounting_review']=verify_accounting(decisions[event['event_id']],event,reader.read(row['source_path'],row['source_sha256']));events.append(reviewed)
        except Exception as exc:failures.append(dict(event_id=row['event_id'],error_type=type(exc).__name__,error=str(exc)))
    nonnull,nulls=position_counts(events);passed=not failures and len(events)==21 and nonnull+nulls==63
    for p,d in list(reader.hashes.items()):reader.read(p,d)
    result=dict(schema_version=1,stage='independent_source_value_and_null_classification_review',passed=passed,created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_sha,extractions_sha256=extractions_sha,event_reviews=events,failures=failures,counters=dict(events=len(events),published_nonnull_values_verified=nonnull,null_fields_reviewed=nulls,catalog_events=28,unprocessed_catalog_events=7,original_HTML_catalog_events=27,index_catalog_events=1,clock_verified_events=sum(e['clock_verified'] for e in events),clock_held_events=sum(not e['clock_verified'] for e in events),by_symbol=dict(Counter(e['symbol'] for e in events)),input_hashes_verified=len(reader.hashes),boundary_assumptions_verified=sum(e['boundary_assumption_verified'] for e in events)),input_hashes=reader.hashes,source_hashes={p:reader.hashes[p] for p in source_files()},runtime=sys.version,scope='21 declared raw HTML reports,63 positions; all28 catalog events retained; source correctness only',limitations=['Two early TMO publications remain outside the pinned calendar.','Calendar-year starts are explicit prior-end-plus-one assumptions, not directly quoted.','Accounting and context holds remain recorded; numerical verification is not source or strategy admission.','No first-public version, correction-history, ISO currency or continuous-security-master certification.','One index entry is not original report evidence; seven catalog entries remain unprocessed.'],financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        target=OUTPUT if passed else OUTPUT.replace('.json','_FAILURE.json')
        with (ROOT/target).open('x') as f:json.dump(result,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=passed,counters=result['counters'],failures=failures,receipt_sha256=sha((ROOT/target).read_bytes()) if publish else None)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);p.add_argument('--extractions-sha',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();raise SystemExit(0 if audit(a.plan_sha,a.extractions_sha,a.publish)['passed'] else 1)
