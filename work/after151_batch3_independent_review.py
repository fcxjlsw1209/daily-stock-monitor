"""Independent KO/LRCX raw-HTML source review; never imports a producer.

Raw headers establish date groups before published amount coordinates are read.
This is source correctness only: accounting and calendar holds are preserved.
"""
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
from bs4 import BeautifulSoup
from after149_batch1_independent_review import Reader, grid_from_html, parsed_dates, signed_text, norm, sha, require
from after150_batch2_independent_review import external_title, verify_published_cell

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/monthly_financial_coverage_batch3_v1'
PLAN, EXTRACTIONS = BASE+'/PLAN.json', BASE+'/EXTRACTIONS.json'
CONTEXT = 'work/after151_next_source_context.json'
POLICY = 'work/after151_batch3_admission_policy.json'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SELF, TEST = 'work/after151_batch3_independent_review.py', 'work/test_after151_batch3_independent_review.py'
OUTPUT = 'work/after151_batch3_independent_review.json'
ACCOUNTING = 'work/after151_batch3_accounting_review.json'
FIELDS = ('annual_net_income','current_total_assets','prior_total_assets')
STATUS = 'standard_HTML_annual_header_context_located_not_admitted'
ASSUMPTION = 'LRCX_consecutive_annual_boundaries_v1'
ANNUAL = re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b|\byears?\s+ended\b',re.I)
SHORT = re.compile(r'\b(?:(?:three|six|nine|3|6|9)\s+months?|quarters?)\s+ended\b',re.I)
INCOME = re.compile(r'(?:condensed\s+)?consolidated\s+statements?\s+of\s+(?:income|operations|earnings)',re.I)
BALANCE = re.compile(r'(?:condensed\s+)?consolidated\s+balance\s+sheets?',re.I)
BAD = re.compile(r'non[\s\-\u2010-\u2015]*gaap|pro\s+forma|reconciliation|as\s+adjusted',re.I)
OTHER = re.compile(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b',re.I)
BODY = re.compile(r'(?:assets|current assets|(?:net |total |net operating )?revenues?|(?:net )?sales|cash(?: and cash equivalents)?|(?:consolidated )?net (?:income|loss))\s*:?',re.I)
PUNCT = re.compile(r'[\s$()\-\u2212]+')


def source_files():
    return [SELF,TEST,'work/after149_batch1_independent_review.py','work/after150_batch2_independent_review.py']


def relative(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(not p.is_symlink(),'Symlink input')
    return str(p.resolve().relative_to(ROOT))


def dates(text):
    return parsed_dates(re.sub(r'\s+,',',',text))


def statement(soup,context,field,accounting=None):
    spec=context['fields'][field];ti,row=spec['table'],spec['row']
    tables=soup.find_all('table')
    require(type(ti) is int and 0<=ti<len(tables) and type(row) is int and row>=0,'Invalid table/row')
    table=tables[ti];require(table.find('table') is None,'Nested statement')
    cells,grid=grid_from_html(table)
    statements=[s for s in context['statement_contexts'] if s['table']==ti]
    require(len(statements)==1,'Unique frozen statement required')
    fixed=statements[0];stop=fixed['header_stop_row']
    require(type(stop) is int and 0<stop<=row,'Invalid header stop')
    headers=[c for c in cells.values() if c['row']<stop]
    for c in headers:
        before=[x for x in cells.values() if x['row']==c['row'] and x['physical_cell']<c['physical_cell']]
        require(not (BODY.fullmatch(c['text']) and all(not x['text'] for x in before)),'Body inside header')
    for h in fixed['headers']:
        c=cells.get((h['row'],h['physical_cell']))
        require(c is not None and c['row']<stop and all(c[k]==h[k] for k in ('columns','rowspan','text')),'Frozen header mismatch')
    for h in spec['header_evidence']:
        c=cells.get((h['row'],h['physical_cell']))
        require(c is not None and c['row']<stop and all(c[k]==h[k] for k in ('columns','rowspan','text')),'Field header mismatch')
    annual=field==FIELDS[0];pattern=INCOME if annual else BALANCE
    titles=[c['text'] for c in headers if pattern.fullmatch(c['text'])]
    if not titles:
        title=external_title(table,pattern);titles=[title] if title else []
    require(len(titles)==1 and not BAD.search(titles[0]),'Formal GAAP statement required')
    for c in headers:
        if not BAD.search(c['text']):continue
        # A separately source-verified manual decision can explain only the
        # exact retrospective header. Non-GAAP/pro-forma remains rejected.
        allow=(accounting is not None and accounting.get('status')=='resolved'
               and accounting.get('gaap_retrospective_adjustment_verified') is True
               and re.fullmatch(r'as\s+adjusted(?:\s+\d+)?',c['text'],re.I))
        require(allow,'Formal GAAP statement required')
    require(fixed['kind']==('income' if annual else 'balance'),'Wrong statement kind')
    units=[c for c in headers if re.search(r'\bin\s+(?:thousands|millions|billions)\b',c['text'],re.I)]
    names={x.lower() for c in units for x in re.findall(r'\bin\s+(thousands|millions|billions)\b',c['text'],re.I)}
    wanted='millions' if context['symbol']=='KO' else 'thousands'
    require(names=={wanted},'Missing/conflicting scale')
    scale='1000000' if wanted=='millions' else '1000'
    require(spec['scale']==fixed['scale']==scale and spec['currency_literal']=='$' and spec['currency_iso'] is None,'Scale/currency context mismatch')
    text=table.get_text(' ',strip=True)
    require('$' in text and not OTHER.search(text),'Dollar literal absent or competing currency')
    label=re.compile(r'consolidated net income' if context['symbol']=='KO' else r'net (?:income|loss)(?:\s*\((?:income|loss)\))?',re.I) if annual else re.compile(r'total assets',re.I)
    labels=[]
    for c in cells.values():
        if not label.fullmatch(c['text'].rstrip(':')):continue
        require(all(not x['text'] for x in cells.values() if x['row']==c['row'] and x['physical_cell']<c['physical_cell']),'Target follows another row label')
        labels.append(c)
    require(len(labels)==1 and labels[0]['row']==row and labels[0]['physical_cell']==spec['label_cell'] and labels[0]['rowspan']==1,'Exact consolidated total row missing or duplicated')
    require(label.fullmatch(spec['label'].rstrip(':')) and labels[0]['text'].rstrip(':')==spec['label'].rstrip(':'),'Label context mismatch')
    ordered=sorted([c for c in cells.values() if c['row']==row],key=lambda c:c['physical_cell'])
    geometry=[{k:c[k] for k in ('row','physical_cell','columns','rowspan')} for c in ordered]
    require(geometry==spec['row_geometry_without_values'],'Frozen row geometry mismatch')
    return spec,cells,grid,headers,units,titles[0],ordered,scale


def date_groups(headers,row,annual):
    """Independent per-column stacked date reconstruction, including BS quarter."""
    groups={}
    for col in sorted({x for c in headers for x in c['columns']}):
        hs=[c for c in headers if col in c['columns']];text=' '.join(c['text'] for c in hs)
        if annual and (not ANNUAL.search(text) or SHORT.search(text)):continue
        found=dates(text)
        if not found:continue
        require(len(found)==1,'Ambiguous raw date column')
        d=next(iter(found))
        anchors=[c for c in hs if re.search(r'\b(?:19|20)\d{2}\b',c['text'])]
        require(len(anchors)==1 and set(re.findall(r'\b(?:19|20)\d{2}\b',text))=={d[:4]},'Conflicting/duplicated year anchor')
        groups.setdefault(d,[]).append(col)
    require(len(groups)==2 if annual else 2<=len(groups)<=3,'Wrong annual/instant date count')
    for cols in groups.values():require(cols==list(range(min(cols),max(cols)+1)),'Disjoint date group')
    if annual:
        parents=[c for c in headers if ANNUAL.search(c['text']) and not SHORT.search(c['text'])]
        require(len(parents)==1 and all(set(v)<=set(parents[0]['columns']) for v in groups.values()),'Ambiguous annual parent')
        domain=set(parents[0]['columns'])
    else:domain=set(range(1,max(max(c['columns']) for c in row)+1))
    owners={}
    for col in domain:
        distances={d:min(abs(col-v) for v in cols) for d,cols in groups.items()}
        closest=[d for d,v in distances.items() if v==min(distances.values())]
        owners[col]=closest[0] if len(closest)==1 else None
    return groups,owners


def source_field(soup,context,field,accounting=None):
    spec,cells,grid,headers,units,title,row,scale=statement(soup,context,field,accounting)
    groups,owners=date_groups(headers,row,field==FIELDS[0])
    current,prior=context['periods']['current_end'],context['periods']['prior_end']
    require(current==context['official_actual_period_end'] and current in groups and prior in groups,'Fiscal endpoints not both printed')
    require(all(prior<=d<=current for d in groups),'Unexpected instant outside annual endpoints')
    endpoint=prior if field==FIELDS[2] else current;cols=groups[endpoint]
    require(cols==spec['logical_columns'] and endpoint==spec['as_of'],'Independent date columns differ')
    ri=spec['row'];require(all((ri,col) in grid and grid[ri,col]['row']==ri for col in cols),'Missing/cross-row cells')
    selected=[c for c in row if set(c['columns'])&set(cols)]
    require(selected,'Missing physical value')
    lo,hi=row.index(selected[0]),row.index(selected[-1])
    for i in (lo-1,hi+1):
        if 0<=i<len(row) and row[i]['text'] and PUNCT.fullmatch(row[i]['text']):
            require(all(owners.get(x)==endpoint for x in row[i]['columns']),'Sign crosses/ambiguously belongs to date')
            selected.append(row[i])
    selected.sort(key=lambda c:c['physical_cell'])
    for c in selected:require(c['rowspan']==1 and all(owners.get(x)==endpoint for x in c['columns']),'Cross-date/cross-row amount')
    displayed=' '.join(c['text'] for c in selected if c['text']);value=signed_text(displayed)
    require(Decimal(value).is_finite() and (field==FIELDS[0] or Decimal(value)>0),'Invalid monetary amount/assets')
    return dict(table=spec['table'],row=ri,value_decimal=value,displayed_amount=displayed,scale=scale,as_of=endpoint,columns=cols,physical_cells=selected,header_cells=[c for c in headers if set(c['columns'])&set(cols)],scale_cells=units,statement_title=title,all_date_groups=groups)


def review_position(soup,context,field,event,period,accounting=None):
    cell=event['fields'][field]
    if cell is not None:
        reading=source_field(soup,context,field,accounting)
        return verify_published_cell(cell,reading,field,event,period),reading
    gaps=[g for g in event['gaps'] if g.get('field')==field]
    require(len(gaps)==1 and type(gaps[0].get('reason')) is str and gaps[0]['reason'],'Null requires one explicit frozen gap')
    try:
        reading=source_field(soup,context,field,accounting)
        result=dict(is_null=True,null_reason_verified=True,source_value_present=True,frozen_null_preserved=True,null_classification='producer_gap_despite_independently_readable_source',frozen_gap=gaps[0],independent_source_reading=reading)
    except ValueError as exc:
        allowed={'Ambiguous monetary syntax','Unbalanced sign','Double negative','Multiple currency signs','Misplaced currency sign','Invalid monetary amount/assets','Cross-date/cross-row amount','Sign crosses/ambiguously belongs to date','Missing/cross-row cells'}
        require(str(exc) in allowed,'Null cannot bypass source metadata failure: '+str(exc))
        spec,cells,grid,headers,units,title,row,scale=statement(soup,context,field,accounting)
        groups,_=date_groups(headers,row,field==FIELDS[0]);endpoint=spec['as_of'];columns=groups[endpoint]
        reading=dict(table=spec['table'],row=spec['row'],as_of=endpoint,columns=columns,header_cells=[c for c in headers if set(c['columns'])&set(columns)],scale_cells=units,statement_title=title,all_date_groups=groups,raw_target_row=row)
        result=dict(is_null=True,null_reason_verified=True,source_value_present=None,frozen_null_preserved=True,null_classification='source_present_value_or_sign_geometry_unresolved',frozen_gap=gaps[0],independent_error=str(exc),independent_source_reading=reading)
    return result,reading


def periods(context,event,policy):
    p=context['periods'];end,prior=p['current_end'],p['prior_end']
    start=(date.fromisoformat(prior)+timedelta(days=1)).isoformat();length=(date.fromisoformat(end)-date.fromisoformat(start)).days+1
    expected=dict(period_start=start,period_end=end,days_inclusive=length)
    require(p['proposed_current_start']==start and p['start_is_source_quoted'] is p['start_assumption_approved'] is False and p['start_assumption']=='same_current_report_prior_annual_end_plus_one_calendar_day','Original proposed boundary changed')
    require(event['periods']['current']==expected and event['periods']['comparison']==dict(period_end=prior),'Published annual periods mismatch')
    require(event['periods']['start_basis']=='derived_same_report_prior_annual_end_plus_one_not_directly_quoted' and event['periods']['start_assumption_certified'] is False,'Derived start upgraded to source quotation')
    if context['symbol']=='KO':
        year=context['fiscal_year'];require(end==f'{year}-12-31' and prior==f'{year-1}-12-31' and start==f'{year}-01-01' and length in (365,366),'KO calendar year mismatch')
    else:
        setting=policy['periods']['LRCX_52_53_week_assumption']
        require(setting['enabled'] is True and setting['not_a_quoted_fact'] is True and setting['assumption_id']==ASSUMPTION,'LRCX explicit assumption missing')
        require(length in (364,371),'LRCX not a 52/53-week annual interval')
    return expected


def receipt_evidence(reader,context,raw):
    ref=context['receipt'];source=context['source'];require(relative(ref['path'])==relative(source['receipt_path']) and ref['sha256']==source['receipt_sha256'],'Receipt reference conflict')
    doc=reader.load(ref['path'],ref['sha256']);items=doc if isinstance(doc,list) else [doc] if 'sha256' in doc else list(doc.values())
    found=[x for x in items if isinstance(x,dict) and x.get('sha256')==sha(raw)]
    require(len(found)==1,'Raw receipt missing/ambiguous');item=found[0]
    require(item.get('status',item.get('http_status'))==200 and item['url']==source['url'],'Receipt HTTP/URL')
    if 'bytes' in item:require(item['bytes']==len(raw),'Receipt length')
    stamp=datetime.fromisoformat(item['retrieved_at'].replace('Z','+00:00'));require(stamp.tzinfo is not None and item['retrieved_at']==ref['retrieved_at'],'Receipt timestamp')
    return dict(path=relative(ref['path']),sha256=ref['sha256'],url=item['url'],http_status=200,retrieved_at=item['retrieved_at'])


def review_event(reader,row,event,context,sessions,policy,accounting=None):
    require(all(event[k]==context[k]==row[k] for k in ('event_id','symbol','fiscal_year')),'Event identity mismatch')
    require(event['event_id']==event['symbol']+'_'+context['publication_date'],'ID/date mismatch')
    require(event['source']==context['source'],'Source record changed')
    source=event['source'];raw=reader.read(row['source_path'],row['source_sha256'])
    require(relative(source['path'])==relative(row['source_path'])==relative(context['evidence_path']) and sha(raw)==source['sha256']==context['evidence_sha256'],'Raw/context mismatch')
    require(source['url']==context['source_url'],'Conflicting source URL')
    if accounting is not None:verify_accounting(accounting,event,raw)
    receipt=receipt_evidence(reader,context,raw);soup=BeautifulSoup(raw,'html.parser')
    for n in soup(['script','style']):n.decompose()
    text=norm(soup.get_text(' ',strip=True));symbol=event['symbol']
    issuer,host,exchange=('Coca-Cola Company','investors.coca-colacompany.com','NYSE') if symbol=='KO' else ('Lam Research Corporation','newsroom.lamresearch.com','NASDAQ')
    require(issuer.lower() in text.lower() and re.search(exchange+r'\s*:\s*'+symbol+r'\b',text,re.I),'Issuer/security anchor missing')
    require(context['identity_anchors'] and all(norm(a).lower() in text.lower() for a in context['identity_anchors']),'Frozen identity anchors changed')
    url=urlsplit(source['url']);require(url.scheme=='https' and url.hostname==host and not url.username and not url.password and url.port in (None,443),'Official issuer URL mismatch')
    release,available=context['publication_date'],context['available_session']
    require(release in dates(text) and event['publication_date']==release and event['available_from']==available and context['official_actual_period_end']<release<available,'Release/availability mismatch')
    require(sessions and sessions==sorted(set(sessions)),'Invalid sessions');clock=sessions[0]<=release<sessions[-1]
    if clock:require(sessions[bisect_right(sessions,release)]==available,'Not first next pinned session')
    require(set(event['fields'])==set(FIELDS) and all(event[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')),'Fields/admission flags')
    require(event['context_holds']==context['context_holds'] and event['accounting_review_flags']==context['accounting_terms'] and event['accounting_context_numbers_redacted']==context['accounting_context_numbers_redacted'],'Inherited context lost')
    expected=periods(context,event,policy);readings={};fields={}
    for f in FIELDS:fields[f],readings[f]=review_position(soup,context,f,event,expected,accounting)
    require({g['field'] for g in event['gaps']}=={f for f in FIELDS if event['fields'][f] is None},'Gaps differ from null positions')
    header=' '.join(c['text'] for r in readings.values() for c in r['header_cells'])
    require(not re.search(r'transition period|stub period|change.*fiscal year|from .+? to .+?',header,re.I),'Contradictory annual header')
    result=dict(event_id=event['event_id'],symbol=symbol,fiscal_year=event['fiscal_year'],passed=True,metadata_verified=True,identity_verified=True,issuer_id=('The Coca-Cola Company' if symbol=='KO' else 'Lam Research Corporation'),security_id=symbol,periods_verified=True,currency_verified=True,clock_verified=clock,clock_scope='first_pinned_next_session' if clock else 'outside_pinned_calendar_not_certified',release_date=release,available_from=available,source=source,receipt=receipt,fields=fields,retained_context_holds=event['context_holds'],accounting_review_flags=event['accounting_review_flags'],accounting_context_numbers_redacted=event['accounting_context_numbers_redacted'],accounting_comparability_admitted=False,start_is_source_quoted=False,first_public_version_verified=False,certified_daily_security_master=False,source_admitted=False)
    if symbol=='LRCX':result.update(boundary_assumption_verified=True,assumption_id=ASSUMPTION,derived_start=expected['period_start'],boundary_evidence={f:dict(table=readings[f]['table'],date=readings[f]['as_of'],headers=readings[f]['header_cells']) for f in FIELDS})
    else:result.update(boundary_assumption_verified=True,assumption_id='KO_calendar_annual_boundaries_v1',annual_start_interpretation='January1 from explicit calendar year ended December31',derived_start=expected['period_start'],boundary_evidence={f:dict(table=readings[f]['table'],date=readings[f]['as_of'],headers=readings[f]['header_cells']) for f in FIELDS})
    return result


def verify_population(plan,data,context,policy):
    docs=context['events'];ids=[e['event_id'] for e in docs]
    require(len(ids)==len(set(ids))==32 and Counter(e['symbol'] for e in docs)=={'KO':16,'LRCX':16},'Full32 catalog required')
    chosen=[e for e in docs if e['context_status']==STATUS]
    require(len(chosen)==10 and Counter(e['symbol'] for e in chosen)=={'KO':6,'LRCX':4},'Fixed10 report scope')
    require([e['fiscal_year'] for e in chosen if e['symbol']=='KO']==list(range(2011,2017)) and [e['fiscal_year'] for e in chosen if e['symbol']=='LRCX']==list(range(2013,2017)),'Fixed report years')
    require([e['event_id'] for e in plan['records']]==[e['event_id'] for e in data['events']]==[e['event_id'] for e in chosen],'Extraction event order')
    require(plan['catalog_ids']==data['catalog_ids']==ids and plan['catalog_events']==32,'Catalog missing/reordered')
    require(policy['required_fields']==list(FIELDS),'Source policy fields')
    projection=[{k:e[k] for k in ('event_id','symbol','fiscal_year','publication_date','available_session','official_actual_period_end','source','context_status')} for e in docs]
    require(policy['fixed_catalog_metadata']==projection,'Policy full32 catalog metadata changed')
    require(data['counts']==dict(events=10,by_symbol={'KO':6,'LRCX':4},nonnull_fields=sum(e['fields'][f] is not None for e in data['events'] for f in FIELDS),unprocessed_catalog_events=22),'Extraction counts')
    return {e['event_id']:e for e in chosen}


def verify_accounting(item,event,raw):
    """Validate a separately authored field-specific assessment, never infer it."""
    require(item['event_id']==event['event_id'] and item['source']==event['source'],'Accounting source identity')
    require(item['status'] in ('resolved','unresolved') and item['scope']=='current_annual_net_income_same_report_assets','Accounting decision scope')
    require(type(item['reasons']) is list and all(type(x) is str and x for x in item['reasons']),'Accounting reasons')
    require(type(item['basis']) is str and item['basis'].strip(),'Accounting explanation')
    require(set(item['field_checks'])==set(FIELDS) and all(type(v) is bool for v in item['field_checks'].values()) and type(item['current_prior_common_basis']) is bool,'Explicit field-specific checks')
    soup=BeautifulSoup(raw,'html.parser')
    for n in soup(['script','style']):n.decompose()
    text=norm(soup.get_text(' ',strip=True))
    require(type(item['evidence']) is list,'Accounting evidence list')
    for e in item['evidence']:
        require(relative(e['path'])==relative(event['source']['path']) and e['sha256']==sha(raw)==event['source']['sha256'],'Accounting raw source SHA')
        loc=e['locator'];require(set(loc)=={'kind','start','end'} and loc['kind']=='document_text','Exact accounting text locator')
        start,end=loc['start'],loc['end']
        require(type(start) is int and type(end) is int and 0<=start<end<=len(text),'Accounting text range')
        require(type(e['quote']) is str and e['quote']==text[start:end],'Accounting quote mismatch')
    if item.get('gaap_retrospective_adjustment_verified') is True:
        quotes=' '.join(e['quote'] for e in item['evidence']).lower()
        require(item['status']=='resolved' and
                'change in accounting methodology has been applied retrospectively' in quotes and
                'adjusted all prior period financial information presented herein as required' in quotes and
                'pension plans' in quotes,'Retrospective GAAP exception lacks explicit source proof')
    if item['status']=='resolved':
        require(not item['reasons'] and item['evidence'] and all(item['field_checks'].values()) and item['current_prior_common_basis'],'Resolved accounting lacks proof')
    else:require(item['reasons'],'Unresolved accounting needs explicit cause')
    return item


def position_counts(events):
    fields=[f for e in events for f in e['fields'].values()]
    require(all(type(f['is_null']) is bool for f in fields),'Exact null flags required')
    return sum(not f['is_null'] for f in fields),sum(f['is_null'] for f in fields)


def audit(plan_sha,extractions_sha,publish=False):
    require(all(re.fullmatch(r'[0-9a-f]{64}',x or '') for x in (plan_sha,extractions_sha)),'Explicit frozen hashes required')
    reader=Reader();plan,data=reader.load(PLAN,plan_sha),reader.load(EXTRACTIONS,extractions_sha)
    require(data['plan_sha256']==plan_sha and plan['context']==CONTEXT and plan['policy']==POLICY,'Frozen plan linkage')
    for p,d in {**plan['inputs'],**plan['code']}.items():reader.read(p,d)
    for p in source_files():reader.read(p)
    context,policy=reader.load(CONTEXT),reader.load(POLICY)
    require(policy['source_scope']['source_context']==plan['catalog_ref']==data['catalog_ref']==dict(path=CONTEXT,sha256=reader.hashes[CONTEXT]),'Context hash binding')
    require(policy['policy_id']=='ko_lrcx_annual_ni_same_report_assets_batch3_v1','Wrong source policy')
    for p,d in policy.get('input_hashes',{}).items():reader.read(p,d)
    contexts=verify_population(plan,data,context,policy);sessions=reader.load(SESSIONS)['sessions'];events=[];failures=[]
    accounting=reader.load(ACCOUNTING)
    require(accounting['schema_version']==1 and accounting['stage']=='independent_field_specific_accounting_review' and accounting['passed'] is True and accounting['plan_sha256']==plan_sha and accounting['extractions_sha256']==extractions_sha,'Accounting review linkage')
    for p,d in accounting['input_hashes'].items():reader.read(p,d)
    decisions={e['event_id']:e for e in accounting['events']}
    require(len(decisions)==len(accounting['events'])==10 and set(decisions)==set(contexts),'All10 accounting decisions')
    for row,event in zip(plan['records'],data['events']):
        try:
            reviewed=review_event(reader,row,event,contexts[row['event_id']],sessions,policy,decisions[event['event_id']])
            reviewed['accounting_review']=verify_accounting(decisions[event['event_id']],event,reader.read(row['source_path'],row['source_sha256']))
            events.append(reviewed)
        except Exception as exc:failures.append(dict(event_id=row['event_id'],error_type=type(exc).__name__,error=str(exc)))
    nonnull,nulls=position_counts(events)
    passed=not failures and len(events)==10 and nonnull+nulls==30
    for p,d in list(reader.hashes.items()):reader.read(p,d)
    result=dict(schema_version=1,stage='independent_source_value_and_null_classification_review',passed=passed,created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_sha,extractions_sha256=extractions_sha,event_reviews=events,failures=failures,counters=dict(events=len(events),published_nonnull_values_verified=nonnull,null_fields_reviewed=nulls,catalog_events=32,unprocessed_catalog_events=22,by_symbol=dict(Counter(e['symbol'] for e in events)),input_hashes_verified=len(reader.hashes),boundary_assumptions_verified=sum(e.get('boundary_assumption_verified') is True for e in events)),input_hashes=reader.hashes,source_hashes={p:reader.hashes[p] for p in source_files()},runtime=sys.version,scope='10 declared raw HTML reports, 30 positions; all32 catalog events retained; source correctness only',limitations=['Calendar-exterior publication remains unverified.','Annual starts are derived, not directly quoted; LRCX requires explicit52/53-week boundary policy.','Accounting holds and redacted context are preserved; numerical verification is not accounting or strategy admission.','No first-public version, correction-history, ISO currency or continuous-security-master certification.'],financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        require(passed,'Review failed; no success receipt published')
        with (ROOT/OUTPUT).open('x') as f:json.dump(result,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=passed,counters=result['counters'],failures=failures,receipt_sha256=sha((ROOT/OUTPUT).read_bytes()) if publish else None)))
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--plan-sha',required=True);cli.add_argument('--extractions-sha',required=True);cli.add_argument('--publish',action='store_true');a=cli.parse_args()
    raise SystemExit(0 if audit(a.plan_sha,a.extractions_sha,a.publish)['passed'] else 1)
