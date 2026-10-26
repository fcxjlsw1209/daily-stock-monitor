"""Offline, plan-bound extraction of displayed annual issuer-statement amounts.

This module never calculates a financial ratio, ranking, trade or return.
"""
import argparse
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = 'outputs/monthly_profitability141_issuer_statements'
PLAN = OUT + '/EXTRACTION_PLAN.json'
PLAN_SHA = 'a47fab1805b8578b80b958ae67784b67f1a66cc198060a55ac3b6bb8b379c4b6'
SELF = 'work/profitability141_issuer_statements.py'
TEST = 'work/profitability141_issuer_statements_tests.py'
MONTHS = {m:i for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
DATE_RE = re.compile(r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2}),?\s+(\d{4})\b')


def norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(relative, root=ROOT):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Non-relative source path')
    p = root / p
    if p.resolve() != p:
        raise ValueError('Symlink source path')
    return p


def read_pinned(binding, root=ROOT):
    raw = safe_path(binding['path'], root).read_bytes()
    if sha(raw) != binding['sha256']:
        raise ValueError('Source digest mismatch: ' + binding['path'])
    return raw


def exact_amount(text):
    text = norm(text).replace('\u2212', '-')
    if '%' in text or not text:
        raise ValueError('Percentage or missing amount')
    text = text.replace('$', '').strip()
    negative = text.startswith('(') and text.endswith(')')
    if negative:
        text = text[1:-1].strip()
    if not re.fullmatch(r'-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', text):
        raise ValueError('Ambiguous displayed amount: ' + text)
    value = Decimal(text.replace(',', ''))
    if negative:
        value = -value
    return format(value, 'f')


def scale_from(text):
    units = re.findall(r'\bin\s+(millions|thousands)\b', norm(text), re.I)
    if len(set(u.lower() for u in units)) != 1:
        raise ValueError('Missing or ambiguous statement scale')
    return {'millions':1000000, 'thousands':1000}[units[0].lower()]


def table_grid(table):
    """Retain physical cell and logical columns, including rowspan/colspan."""
    rows = [r for r in table.find_all('tr') if r.find_parent('table') is table]
    grid, physical = {}, {}
    for ri, row in enumerate(rows):
        ci = 0
        for pi, cell in enumerate(row.find_all(['td','th'], recursive=False)):
            while (ri,ci) in grid:
                ci += 1
            rs, cs = int(cell.get('rowspan',1)), int(cell.get('colspan',1))
            if not (1 <= rs <= 100 and 1 <= cs <= 100):
                raise ValueError('Invalid table span')
            item = {'text':norm(cell.get_text(' ',strip=True)), 'row':ri,
                    'physical_cell':pi, 'columns':list(range(ci,ci+cs))}
            physical[(ri,pi)] = item
            for r in range(ri,ri+rs):
                for c in range(ci,ci+cs):
                    if (r,c) in grid:
                        raise ValueError('Overlapping table spans')
                    grid[(r,c)] = item
            ci += cs
    return rows, grid, physical


def headers_for(grid, ri, col, stop):
    result = []
    for r in range(min(ri,stop)):
        item = grid.get((r,col))
        if item and item['text'] and item['text'] not in result:
            result.append(item['text'])
    return result


def header_stop(rows):
    for i,row in enumerate(rows):
        cells = [norm(c.get_text(' ',strip=True)).lower() for c in row.find_all(['td','th'],recursive=False)]
        if any(re.fullmatch(r'(?:total )?revenues?:?|assets|cash(?: and cash equivalents)?',c) for c in cells):
            return i
    raise ValueError('Could not establish end of statement headers')


def header_dates(parts):
    return sorted({date(int(y),MONTHS[m],int(d)).isoformat()
                   for m,d,y in DATE_RE.findall(' '.join(parts))})


def statement_context(table):
    caption = table.find('caption')
    if caption:
        return norm(caption.get_text(' ',strip=True))
    # Microsoft older pages place the statement heading immediately before c-table.
    if table.parent.name == 'div' and 'c-table' in table.parent.get('class',[]):
        snippets=[]
        for s in table.parent.previous_siblings:
            if getattr(s,'name',None):
                if s.name == 'div' or s.find('table'):
                    break
                snippets.append(norm(s.get_text(' ',strip=True)))
                if len(snippets)>=5:
                    break
        return ' '.join(reversed(snippets))
    # NVIDIA embeds the formal heading and unit in the initial table rows.
    return ' '.join(norm(r.get_text(' ',strip=True)) for r in table.find_all('tr')[:6])


def validate_statement(context, table_text, kind, symbol):
    needed = r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS)' if kind=='annual_gross_amount' else r'BALANCE SHEETS'
    if not re.search(needed, context, re.I):
        raise ValueError('Formal statement title missing')
    if re.search(r'non[ -]?GAAP|reconciliation|as adjusted',context,re.I):
        raise ValueError('Non-GAAP or reconciliation table')
    if re.search(r'non[ -]?GAAP|as adjusted',table_text,re.I):
        raise ValueError('Adjusted measures inside selected table')
    if kind=='annual_gross_amount' and symbol=='MSFT' and not re.search(r'cost of (?:revenue|goods)',table_text,re.I):
        raise ValueError('MSFT monetary income-statement context missing')


def extract_cell(soup, spec, symbol, end):
    tables=soup.find_all('table'); table=tables[spec['table_index_zero_based']]
    rows,grid,physical=table_grid(table); ri=spec['row_index_zero_based']
    label=norm(spec['row_label']); matching=[]
    for r,row in enumerate(rows):
        if any(norm(c.get_text(' ',strip=True)).lower()==label.lower() for c in row.find_all(['td','th'],recursive=False)):
            matching.append(r)
    if matching != [ri]:
        raise ValueError('Ambiguous or shifted statement row')
    if len(spec['matching_cells']) != 1:
        raise ValueError('Ambiguous current-year cell')
    target=spec['matching_cells'][0]; item=physical[(ri,target['source_cell_index'])]
    if item['columns'] != target['grid_columns']:
        raise ValueError('Cell logical-column mismatch')
    stop=header_stop(rows); headers=headers_for(grid,ri,item['columns'][0],stop)
    for required in target['header_literals']:
        if norm(required) not in ' '.join(headers):
            raise ValueError('Header literal missing: '+required)
    dates=header_dates(headers)
    if dates != [end]:
        raise ValueError('Current-year statement date mismatch: '+str(dates))
    if spec['kind']=='annual_gross_amount' and 'Twelve Months Ended' not in ' '.join(headers):
        raise ValueError('Not the annual duration column')
    context=statement_context(table); text=norm(table.get_text(' ',strip=True))
    validate_statement(context,text,spec['kind'],symbol)
    expected_unit=norm(spec['units']['literal'])
    if expected_unit not in norm(context+' '+text):
        raise ValueError('Planned monetary unit is absent')
    multiplier=scale_from(expected_unit)
    if '$' not in text:
        raise ValueError('Statement dollar unit not evidenced')
    all_header_dates=sorted({d for c in {c for r,c in grid} for d in header_dates(headers_for(grid,ri,c,stop))})
    return {'measure_label':label,'displayed_amount':item['text'],
            'amount_decimal':exact_amount(item['text']),'scale':multiplier,
            'unit_literal':expected_unit,'currency_label':'$',
            'date':end,'statement_title_and_unit_context':context,
            'header_literals':headers,'all_statement_header_dates':all_header_dates,
            'locator':{'table_index_zero_based':spec['table_index_zero_based'],
                       'row_index_zero_based':ri,'physical_cell_index_zero_based':item['physical_cell'],
                       'logical_column_zero_based':item['columns'][0], 'covered_logical_columns':item['columns']},
            'raw_cell_sha256':sha(str(rows[ri].find_all(['td','th'],recursive=False)[item['physical_cell']]).encode())}


def derive_period(end, prior_end, symbol):
    e,p=date.fromisoformat(end),date.fromisoformat(prior_end)
    start=p+timedelta(days=1);days=(e-start).days+1
    if not 330 <= days <= 400:
        raise ValueError('Nonannual fiscal boundary interval')
    if symbol=='MSFT' and (e.month,e.day,p.month,p.day)!=(6,30,6,30):
        raise ValueError('MSFT calendar year boundary changed')
    return {'start':start.isoformat(),'end':end,'days_inclusive':days,
            'start_is_direct_quote':False,
            'derivation':'day_after_explicit_previous_fiscal_year_end_in_same_balance_sheet',
            'prior_year_end_literal_date':prior_end,
            'duration_literal':'Twelve Months Ended',
            'derivation_requires_independent_review':True}


def first_session_after(day,sessions,start=None):
    date.fromisoformat(day)
    if not sessions or sessions!=sorted(set(sessions)):
        raise ValueError('Invalid official session sequence')
    if day < (start or sessions[0]):
        return None
    i=bisect_right(sessions,day)
    return sessions[i] if i<len(sessions) else None


def receipt_check(raw,receipt,spec):
    if receipt.get('http_status') != 200 or receipt.get('sha256') != sha(raw) or receipt.get('url') != spec['source_url']:
        raise ValueError('Original receipt mismatch')
    stamp=datetime.fromisoformat(receipt['retrieved_at'].replace('Z','+00:00'))
    if stamp.utcoffset() is None:
        raise ValueError('Naive retrieval time')
    if 'bytes' in receipt and receipt['bytes']!=len(raw):
        raise ValueError('Original byte-count mismatch')
    if urlparse(spec['source_url']).hostname not in ['www.microsoft.com','nvidianews.nvidia.com']:
        raise ValueError('Unregistered issuer host')


def extract_source(spec,events,sessions,calendar_start,root=ROOT):
    raw=read_pinned({'path':spec['raw_path'],'sha256':spec['raw_sha256']},root)
    receipt_raw=read_pinned({'path':spec['receipt_path'],'sha256':spec['receipt_sha256']},root)
    receipt=json.loads(receipt_raw)
    if spec['receipt_entry_key'] is not None:
        receipt=receipt[spec['receipt_entry_key']]
    receipt_check(raw,receipt,spec)
    event=[e for e in events if e['ticker']==spec['symbol'] and e['fiscal_year']==spec['fiscal_year'] and e['fiscal_quarter']==4]
    if len(event)!=1:raise ValueError('Ambiguous release event')
    event=event[0]
    if event.get('sha256',event.get('evidence_sha256'))!=spec['raw_sha256'] or event['source_url']!=spec['source_url'] or event['release_date']!=spec['release_date']:
        raise ValueError('Original event and raw binding mismatch')
    if event['fiscal_period_end_actual']!=spec['fiscal_year_end']:raise ValueError('Period-event mismatch')
    soup=BeautifulSoup(raw,'html.parser');text=norm(soup.get_text(' ',strip=True))
    if norm(spec['release_date_literal']) not in text:raise ValueError('Original publication date missing')
    policy=re.search(r'(?:To supplement.{0,350}?presented in accordance with GAAP|financial results.{0,100}?reported in accordance with.{0,100}?GAAP|financial results in accordance with.{0,100}?GAAP|measures of financial performance prepared in accordance with GAAP)',text,re.I)
    if policy is None:raise ValueError('Issuer GAAP presentation policy missing')
    amounts={s['kind']:extract_cell(soup,s,spec['symbol'],spec['fiscal_year_end']) for s in spec['rows']}
    gp,assets=amounts['annual_gross_amount'],amounts['year_end_assets']
    if (gp['currency_label'],gp['scale'],gp['date'])!=(assets['currency_label'],assets['scale'],assets['date']):
        raise ValueError('Mismatched currency/scale/end in pair')
    prior=[d for d in assets['all_statement_header_dates'] if d<assets['date']]
    if len(prior)!=1:raise ValueError('Previous annual boundary ambiguous')
    period=derive_period(assets['date'],prior[0],spec['symbol'])
    usd=re.search(r'.{0,70}converted into United States dollars.{0,100}',text,re.I)
    usable=first_session_after(spec['release_date'],sessions,calendar_start)
    if usable is None:raise ValueError('Official-session coverage missing')
    return {'symbol':spec['symbol'],'cik':{'MSFT':'0000789019','NVDA':'0001045810'}[spec['symbol']],
            'fiscal_year_label':spec['fiscal_year'],'schema':'issuer_statement_raw_annual_pair_v1',
            'gross_amount':gp,'total_assets':assets,'annual_period':period,
            'currency':{'original_label':'$','both_statement_currency_labels_match':True,
                        'both_scales_match':True,'iso_currency':'USD' if usd else None,
                        'USD_verified':bool(usd),'reporting_currency_context':usd.group(0) if usd else None,
                        'note':'ISO interpretation is separate from matching the original dollar reporting label; no FX or ratio computed.'},
            'gaap':{'selected_formal_consolidated_statements':True,'non_gaap_tables_excluded':True,
                    'issuer_policy_literal':policy.group(0),'interpretation':'Unadjusted consolidated statement amounts; MSFT monetary Gross margin is distinct from percentage margin summaries.'},
            'availability':{'release_date':event['release_date'],'publication_timestamp':event.get('publication_timestamp'),
                            'timestamp_precision':event['precision'],'publication_timezone':event.get('publication_timezone'),
                            'first_conservative_session_after_release_date':usable,
                            'first_public_version_verified':False,'current_webpage_retrieved_at':receipt['retrieved_at'],
                            'rule':'Strictly later official session; no same-day use, fabricated midnight, or conference-call-time substitution.'},
            'source':{'raw_path':spec['raw_path'],'raw_sha256':spec['raw_sha256'],
                      'receipt_path':spec['receipt_path'],'receipt_sha256':spec['receipt_sha256'],
                      'receipt_entry_key':spec['receipt_entry_key'],'source_url':spec['source_url'],
                      'receipt_and_event_verified':True},
            'review_flags':['derived_annual_start_not_verbatim','historical_first_webpage_version_not_certified']+([] if usd else ['ISO_currency_unresolved_original_dollar_label_preserved']),
            'raw_fact_pair_extracted':True,'factor_or_strategy_admitted':False}


def execute(root=ROOT):
    raw=read_pinned({'path':PLAN,'sha256':PLAN_SHA},root);plan=json.loads(raw)
    read_pinned(plan['separate_from_unchanged_SEC_DATA_PLAN'],root)
    inputs={k:json.loads(read_pinned(b,root)) for k,b in plan['inputs'].items()}
    cal=inputs['calendar_sessions'];calendar=inputs['calendar_plan']
    if cal['plan_sha256'] != plan['inputs']['calendar_plan']['sha256']:
        raise ValueError('Calendar-plan binding mismatch')
    for path,digest in calendar['bindings'].items():read_pinned({'path':path,'sha256':digest},root)
    events=inputs['MSFT_events']+inputs['NVDA_events'];pairs=[];gaps=[]
    for spec in plan['source_scope']['sources']:
        try:pairs.append(extract_source(spec,events,cal['sessions'],calendar['start'],root))
        except (ValueError,IndexError,KeyError,TypeError) as e:
            gaps.append({'symbol':spec['symbol'],'fiscal_year_label':spec['fiscal_year'],
                         'raw_path':spec['raw_path'],'error':str(e)})
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':PLAN_SHA,
            'status':'RAW_FACTS_FOR_INDEPENDENT_REVIEW_NOT_STRATEGY_ADMISSION',
            'source_count':len(plan['source_scope']['sources']),'pair_count':len(pairs),'gap_count':len(gaps),
            'USD_resolved_pairs':sum(p['currency']['USD_verified'] for p in pairs),
            'matched_original_currency_pairs':sum(p['currency']['both_statement_currency_labels_match'] for p in pairs),
            'ratios_ranks_returns_computed':0,'pairs':pairs,'gaps':gaps,
            'inputs':plan['inputs'],'extractor_sha256':sha(safe_path(SELF,root).read_bytes()),
            'tests_sha256':sha(safe_path(TEST,root).read_bytes()),
            'runtime_guard_sha256':{n:sha(safe_path(n,root).read_bytes()) for n in ['local_run.py','run_local.sh','local.sandbox.sb']}}
    target=safe_path(OUT+'/RAW_ANNUAL_FACTS.json',root)
    with target.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['pairs','gaps','inputs','runtime_guard_sha256']}))
    if gaps:print(json.dumps(gaps))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');args=parser.parse_args()
    if args.extract:execute()
