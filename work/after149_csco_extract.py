"""Pure, unadmitted CSCO annual NI and same-report asset extraction.

The caller must pin the full context record and raw bytes in a source-only PLAN
before using real reports. This module performs no I/O, calendar construction,
financial ratios, features, fitting or replay. extract_document never admits a
fact. Source/identity/period corruption raises; field ambiguity remains a gap.
"""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import re
from bs4 import BeautifulSoup
import after141_amd_source_extract as gridlib

FIELDS=('annual_net_income','current_total_assets','prior_total_assets')
CF=re.compile(r'(?:condensed\s+)?consolidated\s+statements?\s+of\s+cash\s+flows?',re.I)
BS=re.compile(r'(?:condensed\s+)?consolidated\s+balance\s+sheets?',re.I)
NI=re.compile(r'^net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?$',re.I)
ASSETS=re.compile(r'^total\s+assets$',re.I)
BODY=re.compile(r'^(?:net\s+(?:income|loss).*|assets:?|current\s+assets:?|cash(?:\s+and\s+cash\s+equivalents)?|total\s+assets)$',re.I)
ANNUAL=re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b|\byears?\s+ended\b',re.I)
QUARTER=re.compile(r'\b(?:three|six|nine|3|6|9)\s+months?\s+ended\b',re.I)
BAD=re.compile(r'non[\s\-\u2010-\u2015]*gaap|pro\s+forma|as\s+adjusted',re.I)
ADJUST=re.compile(r'adjustments?\s+to\s+reconcile\s+(?:net\s+)?(?:income|loss)',re.I)
CFO=re.compile(r'^(?:net\s+)?cash\s+(?:provided\s+by|from|generated\s+by)\s+(?:operating\s+activities|operations)$',re.I)
MONEY=re.compile(r'^\$?\s*(\()?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?$')


def require(test, message):
    if not test:raise ValueError(message)


def digest(value):return hashlib.sha256(value).hexdigest()


def schema():
    return {'schema_version':1,'stage':'unadmitted_csco_annual_source_candidates',
            'fields':list(FIELDS),'scope':'CSCO FY2011..2026 original annual releases',
            'annual_net_income':'Exact consolidated NI/loss starting line in the annual indirect cash-flow statement; no parent/common/continuing-operations or non-GAAP substitution.',
            'assets':'Exact total-assets line at current and prior annual endpoints in the SAME current report; no predecessor-value splice.',
            'units':'Preserve signed displayed amount and decimal string in reported units; separate scale string and literal dollar, no certified ISO currency.',
            'admission':'Always false; preserve FY2011 duration, pre2015 calendar and FY2019 accounting-transition holds.'}


def source_files():
    return ['work/after149_csco_extract.py','work/test_after149_csco_extract.py',
            'work/after141_amd_source_extract.py']


def _day(text):
    require(isinstance(text,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',text),'invalid_date')
    return date.fromisoformat(text)


def _validate(raw,e):
    require(type(raw)is bytes and raw and not raw.startswith(b'%PDF-'),'nonempty_HTML_bytes_required')
    require(isinstance(e,dict),'context_event_required')
    year=e.get('fiscal_year');require(type(year)is int and 2011<=year<=2026,'outside_fixed_CSCO_year_scope')
    release=_day(e['release_date']);available=_day(e['available_session'])
    require(e['event_id']=='CSCO_'+release.isoformat(),'event_identity_mismatch')
    require(available>release,'availability_must_be_after_release_date')
    source=e['source'];require(isinstance(source,dict)and source.get('sha256')==digest(raw),'source_SHA_mismatch')
    require(isinstance(source.get('path'),str)and source['path'] and source.get('url','').startswith('https://newsroom.cisco.com/'),'original_CSCO_source_required')
    require(e.get('source_issuer_name_found')is True and isinstance(e.get('source_listing_reference'),str),'missing_original_identity_context')
    metadata=e['release_metadata']
    for key,value in [('ticker','CSCO'),('release_date',release.isoformat()),('fiscal_year',year),('fiscal_quarter',4),('evidence_path',source['path']),('evidence_sha256',source['sha256'])]:
        require(metadata.get(key)==value,'release_context_mismatch:'+key)
    require(metadata.get('first_public_time_verified')is False,'unexpected_first_public_certification')
    periods=e['periods'];require(set(periods)=={'current','comparison'},'both_same_report_periods_required')
    for side,p in periods.items():
        end=_day(p['period_end']);require(end<release,'period_not_complete_at_release')
        if p['period_start']is None:
            require(side=='comparison'and year==2011 and p['days_inclusive']is None,'unexpected_missing_period_boundary')
        else:
            days=(end-_day(p['period_start'])).days+1
            require(type(p['days_inclusive'])is int and 330<=days<=400 and days==p['days_inclusive'],'annual_duration_mismatch')
    require(_day(periods['current']['period_start'])==_day(periods['comparison']['period_end'])+timedelta(days=1),'annual_endpoints_not_contiguous')
    require(metadata.get('fiscal_period_end_actual')==periods['current']['period_end'],'release_period_end_mismatch')
    require(isinstance(e.get('tables'),list)and e['tables'],'frozen_statement_context_required')
    return year


def _amount(texts):
    displayed=' '.join(s for s in texts if s.strip()).strip()
    # Only whitespace next to a thousands comma and Unicode minus are normalized.
    normalized=re.sub(r'\s*,\s*',',',gridlib.norm(displayed)).replace('\u2212','-')
    match=MONEY.fullmatch(normalized)
    require(match is not None and bool(match[1])==bool(match[3]),'invalid_signed_monetary_literal')
    require(not(match[1]and match[2].startswith('-')),'conflicting_monetary_signs')
    try:value=Decimal(match[2].replace(',',''))
    except InvalidOperation:raise ValueError('invalid_decimal') from None
    if match[1]:value=-value
    require(value.is_finite(),'nonfinite_decimal')
    return displayed,normalized,format(value,'f')


def _expected(e,field,wanted):
    pattern=CF if field=='annual_net_income'else BS
    label=NI if field=='annual_net_income'else ASSETS
    found=[]
    for t in e['tables']:
        if not any(pattern.search(title)for title in t['title']):continue
        cols=[c['column']for c in t['columns']if c['dates']==[wanted]and(field!='annual_net_income'or(c['annual']and not c['quarter']))]
        for row in t['labels']:
            if label.fullmatch(row['label']):found.append((t['table'],row['row'],cols))
    require(len(found)==1 and found[0][2],'frozen_context_locator_missing_or_ambiguous')
    return found[0]


def _candidates(soup,field,wanted):
    found=[];rejected=[]
    for ti,table in enumerate(soup.find_all('table')):
        try:rows,grid,physical=gridlib.table_grid(table)
        except ValueError:continue
        starts=[ri for(ri,_),v in physical.items()if BODY.fullmatch(v['text'])]
        if not starts:continue
        stop=min(starts)
        heads=[v['text']for(ri,_),v in physical.items()if ri<stop and v['text']]
        if table.caption:heads.insert(0,gridlib.norm(table.caption.get_text(' ',strip=True)))
        ctx=' '.join(heads+gridlib.external_context(table))
        cf=bool(CF.search(ctx));bs=bool(BS.search(ctx))
        if not(cf if field=='annual_net_income'else bs)or BAD.search(ctx):continue
        try:
            require(cf!=bs,'ambiguous_statement_type')
            require(not table.find('table'),'nested_financial_table')
            scale=set(x.lower()for x in re.findall(r'\b(millions|thousands)\b',ctx,re.I))
            require(len(scale)==1,'missing_or_ambiguous_statement_scale')
            scale=str({'millions':1000000,'thousands':1000}[next(iter(scale))])
            require(not re.search(r'[\u20ac\u00a3\u00a5]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b',table.get_text(' ',strip=True),re.I),'conflicting_currency')
            labels=[(ri,v['text'])for(ri,_),v in physical.items()if(NI if field=='annual_net_income'else ASSETS).fullmatch(v['text'])]
            require(len(labels)==1,'missing_or_duplicate_exact_row')
            ri,label=labels[0]
            if field=='annual_net_income':
                require(not any(re.match(r'^net\s+(?:income|loss)\b',v['text'],re.I)for(r,_),v in physical.items()if r<ri),'NI_not_first_consolidated_start')
                require(not any(re.match(r'^(?:income|profit|earnings)(?:\s+\(loss\))?\s+(?:before|from\s+continuing)',v['text'],re.I)for(r,_),v in physical.items()if r<ri),'NI_follows_different_earnings_scope')
                adjustments=[r for(r,_),v in physical.items()if r>ri and ADJUST.search(v['text'])]
                totals=[r for(r,_),v in physical.items()if r>ri and CFO.fullmatch(v['text'])]
                require(adjustments and totals and min(adjustments)<min(totals),'NI_missing_indirect_reconciliation_context')
            selected=[];evidence=[];annual_dates=set()
            for col in sorted({c for r,c in grid}):
                hv=gridlib.header_evidence(grid,stop,col);texts=[v['text']for v in hv]
                dates=gridlib.dates(texts)
                valid=(field!='annual_net_income'or(ANNUAL.search(' '.join(texts))and not QUARTER.search(' '.join(texts))))
                if valid:annual_dates.update(dates)
                if valid and dates==[wanted]:selected.append(col);evidence.extend(hv)
            require(selected and selected==list(range(min(selected),max(selected)+1)),'annual_or_point_in_time_column_missing_or_ambiguous')
            if field!='prior_total_assets':require(annual_dates and max(annual_dates)==wanted,'current_period_not_latest_printed_column')
            cells=[]
            for pi,v in sorted((pi,v)for(r,pi),v in physical.items()if r==ri and set(v['columns'])&set(selected)):
                require(v['rowspan']==1 and set(v['columns'])<=set(selected),'financial_cell_crosses_period_or_row')
                require(not re.search('[A-Za-z]',v['text']),'nonmonetary_text_in_financial_column')
                node=rows[ri].find_all(['td','th'],recursive=False)[pi]
                cells.append(dict(physical_cell_index_zero_based=pi,logical_columns=v['columns'],raw_text=node.get_text(' ',strip=False),markup_sha256=digest(str(node).encode())))
            require(cells,'missing_physical_cells')
            currency=any('$'in v['text']for(r,_),v in physical.items()if r>=stop and set(v['columns'])&set(selected))
            require(currency,'missing_selected_column_dollar_evidence')
            unique_headers=[];seen=set()
            for v in evidence:
                key=(v['row'],v['physical_cell'])
                if key in seen:continue
                seen.add(key);unique_headers.append(dict(row_index_zero_based=v['row'],physical_cell_index_zero_based=v['physical_cell'],logical_columns=v['columns'],text=v['text'],markup_sha256=v['parsed_cell_markup_sha256']))
            found.append(dict(table_index_zero_based=ti,row_index_zero_based=ri,logical_columns=selected,row_label=label,
                statement_title=(CF if cf else BS).search(ctx).group(0),statement_context=ctx,
                header_cells=unique_headers,physical_cells=cells,scale=scale,currency_literal='$',period_end=wanted))
        except ValueError as exc:rejected.append(dict(table_index_zero_based=ti,reason=str(exc)))
    return found,rejected


def extract_document(raw_bytes,context_event):
    year=_validate(raw_bytes,context_event);e=context_event
    soup=BeautifulSoup(raw_bytes,'html.parser')
    for node in soup(['script','style']):node.decompose()
    text=soup.get_text(' ',strip=True)
    require(re.search(r'CISCO\s+SYSTEMS[,. ]+INC',text,re.I)and re.search(r'NASDAQ\s*[:(]\s*CSCO',text,re.I),'original_report_identity_not_found')
    result={k:deepcopy(e[k])for k in ('event_id','fiscal_year','source','periods','release_date')}
    result.update(schema_version=1,stage=schema()['stage'],symbol='CSCO',available_from=e['available_session'],fields={f:None for f in FIELDS},gaps=[],original_context=deepcopy(e),
                  accounting_review_flags=deepcopy(e.get('accounting_context',[])),context_holds=[],fact_admitted=False,model_admitted=False,strategy_admitted=False,
                  first_public_version_verified=False,currency_iso_verified=False,financial_ratios_computed=0,factors_computed=0,returns_computed=0)
    if year==2011:result['context_holds'].append('comparison_FY2010_period_start_unverified')
    if e.get('calendar_verification',{}).get('status')!='first_strictly_later_pinned_session_verified':result['context_holds'].append('strict_next_session_calendar_not_independently_verified')
    if year==2019:result['context_holds'].extend(['ASC606_modified_retrospective_NI_asset_comparability_unresolved','restricted_cash_prior_flow_presentation_flag_retained'])
    if e.get('accounting_context'):result['context_holds'].append('report_accounting_notes_require_field_specific_review')
    for field in FIELDS:
        side='comparison'if field=='prior_total_assets'else'current';wanted=e['periods'][side]['period_end']
        candidates,rejected=_candidates(soup,field,wanted)
        try:
            require(len(candidates)==1,'exactly_one_formal_candidate_required')
            loc=candidates[0];expected=_expected(e,field,wanted)
            require((loc['table_index_zero_based'],loc['row_index_zero_based'],loc['logical_columns'])==expected,'frozen_context_locator_changed')
            displayed,normalized,value=_amount([c['raw_text']for c in loc['physical_cells']])
            result['fields'][field]=dict(measure='net_income'if field=='annual_net_income'else'total_assets',
                basis='current_annual_indirect_CF_start'if field=='annual_net_income'else'same_report_year_end_balance',
                displayed_amount=displayed,normalized_monetary_literal=normalized,value_decimal=value,scale=loc['scale'],currency_literal='$',currency_iso=None,
                period=deepcopy(e['periods']['current'])if field=='annual_net_income'else None,as_of=wanted,
                available_from=e['available_session'],source=deepcopy(e['source']),locator=loc,fact_admitted=False)
        except ValueError as exc:result['gaps'].append(dict(field=field,reason=str(exc),candidate_count=len(candidates),rejected_contexts=rejected))
    return result
