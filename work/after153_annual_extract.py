"""Pure current-report annual NI/asset extraction for fixed PLTR/PYPL/TMO pages.

No I/O, ratios, admission or inferred currency ISO. Caller freezes all bytes.
"""
import copy
import datetime as dt
import re

import after149_amzn_extract as value_parser
import after152_mu_extract as reference
import after153_next_source_context as context_parser
from after141_amd_source_extract import table_grid

FIELDS=reference.FIELDS
SourceError,FieldGap=reference.SourceError,reference.FieldGap
PROFILES={
 'PLTR':dict(symbol='PLTR',issuer='Palantir Technologies Inc.',years=tuple(range(2020,2026)),hosts=('d18rn0p25nwr6d.cloudfront.net',),issuer_pattern=r'\bPalantir Technologies\s+Inc\.',ticker_pattern=r'\bPLTR\b'),
 'PYPL':dict(symbol='PYPL',issuer='PayPal Holdings, Inc.',years=tuple(range(2015,2020)),hosts=('newsroom.paypal-corp.com',),issuer_pattern=r'\bPayPal Holdings,?\s+Inc\.',ticker_pattern=r'\bPYPL\b'),
 'TMO':dict(symbol='TMO',issuer='Thermo Fisher Scientific Inc.',years=tuple(range(2012,2022)),hosts=('india.newsroom.thermofisher.com',),issuer_pattern=r'\bThermo Fisher Scientific(?:,?\s+Inc\.)?',ticker_pattern=r'\bTMO\b'),
}


def source_files():
 return ('work/after153_annual_extract.py','work/test_after153_annual_extract.py',
         'work/after153_next_source_context.py','work/test_after153_next_source_context.py',
         'work/after152_mu_extract.py','work/after149_amzn_extract.py','work/after141_amd_source_extract.py')


def periods(c):
 p=c.get('periods')
 if not isinstance(p,dict):raise FieldGap('missing_period_metadata')
 end,prior,start=(value_parser._date(p.get(k)) for k in ('current_end','prior_end','proposed_current_start'))
 if (end!=dt.date(c['fiscal_year'],12,31) or prior!=dt.date(c['fiscal_year']-1,12,31)
     or start!=prior+dt.timedelta(days=1) or (end-start).days+1 not in (365,366)
     or c.get('official_actual_period_end')!=str(end)):
  raise FieldGap('calendar_year_boundary_mismatch')
 if (p.get('start_is_source_quoted') is not False or p.get('start_assumption_approved') is not False
     or p.get('start_assumption')!='same_current_report_prior_annual_end_plus_one_calendar_day'
     or p.get('annual_date_binding')!='same_report_full_date_headers'
     or p.get('days_inclusive')!=(end-start).days+1):raise FieldGap('changed_boundary_assumption')
 if not end<value_parser._date(c['publication_date'])<value_parser._date(c['available_session']):raise FieldGap('invalid_publication_availability_order')
 return dict(current=dict(period_start=str(start),period_end=str(end),days_inclusive=(end-start).days+1),
             comparison=dict(period_end=str(prior)),start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',
             start_assumption_certified=False,assumption_id=c['symbol']+'_consecutive_annual_boundaries_v1')


def _physical(table):
 if table.find('table') is not None:raise FieldGap('nested_financial_table')
 rows,grid,physical=table_grid(table)
 for (r,pi),cell in physical.items():
  cell['original_text']=rows[r].find_all(['td','th'],recursive=False)[pi].get_text()
  cell['markup_sha256']=cell['parsed_cell_markup_sha256']
 return rows,grid,physical


def _owned(table_context,physical,grid,spec,annual):
 row=spec['row'];columns=spec['logical_columns']
 if (not isinstance(columns,list) or not columns or columns!=sorted(set(columns)) or
     any(type(x) is not int or x<0 for x in columns)):raise FieldGap('invalid_fixed_columns')
 headers=[x for x in physical.values() if x['row']<table_context['header_stop_row']]
 groups=sorted((dict(date=d,columns=cols) for d,cols in table_context['date_columns'].items()),key=lambda g:min(g['columns']))
 if len(groups)!=2:raise FieldGap('two_annual_date_groups_required')
 if annual:
  parents=table_context['annual_header_locators']
  if len(parents)!=1:raise FieldGap('unique_annual_parent_required')
  domain=set(parents[0]['columns'])
  short=[x for x in headers if reference.SHORT.search(x['text'])]
  if any(set(x['columns']) & domain for x in short):raise FieldGap('annual_quarter_parent_overlap')
 else:
  label=physical[(row,spec['label_cell'])]
  domain=set(range(max(label['columns'])+1,max(max(x['columns']) for x in value_parser._cells(physical,row))+1))
 percentage=set()
 for h in headers:
  text=context_parser.norm(h['text'])
  if text.startswith('%') or (text.lower()=='revenues' and not context_parser.left_label(h,list(physical.values()))):percentage.update(h['columns'])
 domain-=percentage
 if any(not set(g['columns'])<=domain for g in groups):raise FieldGap('date_group_outside_annual_numeric_domain')
 selected=[i for i,g in enumerate(groups) if g['date']==spec['as_of'] and g['columns']==columns]
 if len(selected)!=1:raise FieldGap('fixed_date_columns_differ')
 i=selected[0];lo=min(domain) if i==0 else (max(groups[i-1]['columns'])+min(groups[i]['columns']))//2+1
 hi=max(domain) if i==len(groups)-1 else (max(groups[i]['columns'])+min(groups[i+1]['columns']))//2
 owned={x for x in domain if lo<=x<=hi}
 if any((row,col) not in grid for col in columns):raise FieldGap('incomplete_amount_geometry')
 return owned,groups,headers


def extract_document(raw_bytes,context):
 c=copy.deepcopy(context)
 profile=PROFILES.get(c.get('symbol'))
 if profile is None:raise SourceError('outside_fixed_issuer_scope')
 source,soup=reference._reference(raw_bytes,c,profile)
 result=dict(schema_version=1,stage='source_only_annual_NI_assets_candidates',symbol=c['symbol'],event_id=c['event_id'],fiscal_year=c['fiscal_year'],
  identity=dict(issuer=profile['issuer'],symbol=c['symbol'],source_anchors=copy.deepcopy(c['identity_anchors']),certified_daily_security_master=False),
  source=source,publication_date=c['publication_date'],available_from=c['available_session'],periods={},fields=dict.fromkeys(FIELDS),gaps=[],
  context_holds=copy.deepcopy(c['context_holds']),accounting_review_flags=copy.deepcopy(c.get('accounting_terms',[])),
  accounting_context_numbers_redacted=copy.deepcopy(c.get('accounting_context_numbers_redacted',[])),
  fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_version_verified=False,complete_correction_history_verified=False,
  first_public_time_verified=False,certified_daily_security_master=False,financial_calculations=0)
 try:
  result['periods']=periods(c)
  tables=soup.find_all('table');contexts=[]
  for ti,table in enumerate(tables):
   if table.find('table'):continue
   try:item=context_parser.table_context(table,ti)
   except (KeyError,ValueError):item=None
   if item is not None:contexts.append(item)
  selected,gaps=context_parser.choose_pair(contexts,c['fiscal_year'],c['official_actual_period_end'])
  if selected is None:raise FieldGap('formal_annual_pair_not_unique:'+','.join(gaps))
  if selected['periods']!=c['periods']:raise FieldGap('reconstructed_period_metadata_differs')
 except (FieldGap,KeyError,TypeError,ValueError) as exc:
  result['gaps']=[dict(field=name,reason=str(exc)) for name in FIELDS];result['complete_required_fields']=False;return result
 for name in FIELDS:
  try:
   spec=c.get('fields',{}).get(name);expected=copy.deepcopy(selected['fields'][name])
   expected.update(source=copy.deepcopy(source),available_from=c['available_session'],availability_verified=c['available_session_verified'])
   if spec!=expected:raise FieldGap('fixed_field_context_differs')
   table=tables[spec['table']];tc=next(t for t in contexts if t['table']==spec['table'])
   if tc['structural_review_flags']:raise FieldGap('adjusted_or_restated_header_requires_review')
   if value_parser.OTHER_CURRENCY.search(table.get_text(' ',strip=True)):raise FieldGap('conflicting_currency_literal')
   rows,grid,physical=_physical(table)
   owned,groups,headers=_owned(tc,physical,grid,spec,name=='annual_net_income')
   val,display,cells=value_parser._value(physical,grid,spec['row'],spec['logical_columns'],owned)
   selected_ids={(x['row'],x['physical_cell']) for x in cells}
   if any(value_parser._text(x['text']) and set(x['columns'])&owned and (x['row'],x['physical_cell']) not in selected_ids for x in value_parser._cells(physical,spec['row'])):raise FieldGap('unconsumed_content_within_period_group')
   label=physical[(spec['row'],spec['label_cell'])]
   result['fields'][name]=dict(value_decimal=val,displayed_amount=display,scale=spec['scale'],currency_literal='$',currency_iso=None,
    currency_basis='same_statement_dollar_symbol_not_ISO_certification',source=copy.deepcopy(source),available_from=c['available_session'],
    as_of=spec['as_of'],period=copy.deepcopy(result['periods']['current']) if name=='annual_net_income' else None,fact_admitted=False,
    locator=dict(table=spec['table'],row=spec['row'],label=copy.deepcopy(label),header_columns=copy.deepcopy(spec['logical_columns']),
     logical_columns=copy.deepcopy(spec['logical_columns']),owned_logical_columns=sorted(owned),date_groups=groups,physical_cells=copy.deepcopy(cells),
     header_evidence=copy.deepcopy(spec['header_evidence']),external_metadata=copy.deepcopy(tc['external_metadata']),split_title_evidence=copy.deepcopy(tc['split_title_evidence'])))
  except (FieldGap,KeyError,TypeError,ValueError) as exc:result['gaps'].append(dict(field=name,reason=str(exc)))
 result['complete_required_fields']=not result['gaps']
 return result
