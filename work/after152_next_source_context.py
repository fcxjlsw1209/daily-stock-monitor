"""MU/PEP pre-extraction structure context; never parse financial amounts."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import re

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid, external_context
import after151_next_source_context as prior
import after152_next_source_feasibility as f

ROOT=f.ROOT
PREVIOUS=f.OUT
OUTPUT='work/after152_next_source_context.json'
FIELDS=('annual_net_income','current_total_assets','prior_total_assets')
INCOME=re.compile(r'consolidated\s+(?:(?:condensed|summary)\s+)*statements?\s+of\s+(?:income|operations|earnings)\b',re.I)
BALANCE=re.compile(r'consolidated\s+(?:(?:condensed|summary)\s+)*balance\s+sheets?\b',re.I)
NI=re.compile(r'^(?:consolidated\s+)?net\s+(?:income|earnings|loss)(?:\s*\((?:loss|income)\))?\s*$',re.I)
PARENT=re.compile(r'^(?:net\s+(?:income|earnings|loss)(?:\s*\((?:loss|income)\))?).*attributable\s+to',re.I)
ANNUAL=re.compile(r'^(?:for\s+the\s+)?(?:(?:fiscal\s+)?years?|twelve\s+months?|12\s+months?|52\s+weeks?|53\s+weeks?)\s+ended$',re.I)
UNIT=re.compile(r'\b(millions|thousands|billions)\b',re.I)


SLASH_DATE=re.compile(r'(?<![\d/])(\d{1,2})/(\d{1,2})/((?:19|20)\d{2}|\d{2})(?![\d/])')


def parsed_dates(parts):
    result=set(prior.parsed_dates(parts))
    for text in parts:
        for match in SLASH_DATE.finditer(text):
            month,day,year=map(int,match.groups())
            if year<100:year+=2000
            try:result.add(date(year,month,day).isoformat())
            except ValueError:pass
    return sorted(result)


def units(text):
    text=f.norm(text)
    without_dates=SLASH_DATE.sub('[date]',prior.DATE.sub('[date]',text))
    return bool(UNIT.search(text) and len(text)<350 and not re.search(r'[\d%]',without_dates))


def table_context(table,ti):
    rows,grid,physical=table_grid(table);cells=list(physical.values())
    ni=[c for c in cells if NI.fullmatch(f.norm(c['text']))]
    assets=[c for c in cells if f.norm(c['text']).lower()=='total assets']
    if not ni and not assets:return None
    first=min(c['row'] for c in ni+assets)
    body=[c['row'] for c in cells if c['row']<first and f.BODY.fullmatch(f.norm(c['text']))]
    if not body:return None
    stop=min(body);before=[c for c in cells if c['row']<stop]
    outside=[f.norm(t) for t in external_context(table) if f.is_metadata(t) or units(t)]
    texts=[f.norm(c['text']) for c in before]+outside
    titles=[t for t in texts if INCOME.search(t) or BALANCE.search(t)]
    kinds={('income' if INCOME.search(t) else 'balance') for t in titles}
    if len(kinds)!=1:return None
    kind=next(iter(kinds));labels=ni if kind=='income' else assets
    if len(labels)!=1:return None
    # No title/label-based claim of GAAP certification; reconcile/adjusted
    # headings remain a structural caution for independent field review.
    flags=sorted({m.group(0).lower() for t in texts for m in re.finditer(r'non.gaap|reconciliation|as adjusted|restated',t,re.I)})
    annual=[c for c in before if ANNUAL.fullmatch(f.norm(c['text']))]
    allowed={j for c in annual for j in c['columns']} if kind=='income' else {j for _,j in grid}
    groups={};year_columns={}
    for col in sorted(allowed):
        parts=[f.norm(c['text']) for c in before if col in c['columns']]
        date_parts=[text for text in parts if f.HEADER.fullmatch(text) or SLASH_DATE.fullmatch(text)]
        ds=parsed_dates(date_parts)
        if len(ds)==1:groups.setdefault(ds[0],[]).append(col)
        ys={int(t) for t in parts if re.fullmatch(r'(?:19|20)\d{2}',t)}
        if len(ys)==1:year_columns.setdefault(str(next(iter(ys))),[]).append(col)
    headers=[f.cell_metadata(c) for c in before if f.is_metadata(c['text']) or SLASH_DATE.fullmatch(f.norm(c['text'])) or ANNUAL.fullmatch(f.norm(c['text'])) or units(c['text'])]
    unit_evidence=[f.cell_metadata(c) for c in before if units(c['text'])]
    unit_outside=[t for t in outside if units(t)]
    scales={'1000000' if x.group(1).lower()=='millions' else '1000' if x.group(1).lower()=='thousands' else '1000000000'
            for t in [c['text'] for c in unit_evidence]+unit_outside for x in UNIT.finditer(t)}
    label=labels[0]
    return dict(table=ti,kind=kind,header_stop_row=stop,headers=headers,external_metadata=outside,
        annual_header_locators=[f.cell_metadata(c) for c in annual],date_columns=groups,year_columns=year_columns,
        label=f.cell_metadata(label),parent_attributable_labels_excluded=[f.cell_metadata(c) for c in cells if PARENT.match(f.norm(c['text'])) and f.LABEL.fullmatch(f.norm(c['text']))],
        unit_evidence=unit_evidence,external_unit_evidence=unit_outside,scale=next(iter(scales)) if len(scales)==1 else None,
        currency_literal='$' if '$' in table.get_text() else None,currency_iso=None,
        row_geometry_without_values=[{k:c[k] for k in ('row','physical_cell','columns','rowspan')} for c in cells if c['row']==label['row']],
        unclassified_header_shapes_numbers_redacted=[f.masked(c['text']) for c in before if f.norm(c['text']) and not (f.is_metadata(c['text']) or SLASH_DATE.fullmatch(f.norm(c['text'])) or units(c['text']))],
        structural_review_flags=flags,monetary_cell_geometry_verified=False)


def choose_pair(tables,fy,end):
    pairs=[]
    for inc in (t for t in tables if t['kind']=='income' and t['annual_header_locators']):
        for bs in (t for t in tables if t['kind']=='balance'):
            previous=[d for d in bs['date_columns'] if int(d[:4])==fy-1]
            if end not in bs['date_columns'] or len(previous)!=1:continue
            prev=previous[0]
            if end in inc['date_columns'] and prev in inc['date_columns']:
                cols=inc['date_columns'][end];basis='same_report_full_date_headers'
            elif str(fy) in inc['year_columns'] and str(fy-1) in inc['year_columns']:
                cols=inc['year_columns'][str(fy)];basis='annual_year_header_to_same_report_balance_date_pending_review'
            else:continue
            pairs.append((inc,bs,prev,cols,basis))
    if len(pairs)!=1:return None,['unique_same_report_annual_NI_assets_date_pair_not_established']
    inc,bs,prev,cols,basis=pairs[0]
    if any(t['scale'] is None or t['currency_literal'] is None for t in (inc,bs)):
        return None,['unit_or_currency_context_unresolved']
    def field(t,day,columns):
        return dict(table=t['table'],row=t['label']['row'],label_cell=t['label']['physical_cell'],label=t['label']['text'],
                    logical_columns=columns,as_of=day,scale=t['scale'],currency_literal=t['currency_literal'],currency_iso=None,
                    header_evidence=t['headers'],external_metadata=t['external_metadata'],row_geometry_without_values=t['row_geometry_without_values'],
                    monetary_cell_geometry_verified=False)
    start=(date.fromisoformat(prev)+timedelta(days=1)).isoformat()
    return dict(fields=dict(annual_net_income=field(inc,end,cols),current_total_assets=field(bs,end,bs['date_columns'][end]),
                            prior_total_assets=field(bs,prev,bs['date_columns'][prev])),
                periods=dict(current_end=end,prior_end=prev,proposed_current_start=start,start_is_source_quoted=False,
                             start_assumption='same_current_report_prior_annual_end_plus_one_calendar_day',start_assumption_approved=False,
                             annual_date_binding=basis),excluded_parent_NI_labels=inc['parent_attributable_labels_excluded']),[]


def build():
    pins={};raw=f.read_local(PREVIOUS,pins);previous=json.loads(raw)
    for p,h in previous['input_hashes'].items():f.read_local(p,pins,h)
    docs=previous['documents']
    if len(docs)!=len({d['event_id'] for d in docs}) or Counter(d['symbol'] for d in docs)!={'MU':7,'PEP':16}:raise ValueError('Full23 catalog required')
    events=[]
    for doc in docs:
        item={k:v for k,v in doc.items() if k!='context'};item['inherited_feasibility_context']=doc['context']
        receipt=doc['receipt'];item['source']=dict(path=doc['evidence_path'],sha256=doc['evidence_sha256'],url=doc['source_url'],
            receipt_path=receipt['path'] if receipt else None,receipt_sha256=receipt['sha256'] if receipt else None)
        holds=['monetary_cells_and_sign_geometry_not_yet_independently_verified','currency_ISO_and_first_public_version_not_certified',
               'accounting_notes_require_field_specific_independent_review']+doc['gaps']
        if not doc['available_session_verified']:holds.append('publication_outside_pinned_calendar_range')
        item.update(fields=dict.fromkeys(FIELDS),periods=None,statement_contexts=[],fact_admitted=False,model_admitted=False,strategy_admitted=False)
        if doc['source_kind']!='HTML_original':status='index_only_original_report_not_available_in_entry'
        else:
            raw=f.read_local(doc['evidence_path'],pins,doc['evidence_sha256']);soup=BeautifulSoup(raw,'html.parser');tables=[]
            for ti,table in enumerate(soup.find_all('table')):
                if table.find('table'):continue
                try:context=table_context(table,ti)
                except (KeyError,ValueError):context=None
                if context is not None:tables.append(context)
            item['statement_contexts']=tables;selected,gaps=choose_pair(tables,doc['fiscal_year'],doc['official_actual_period_end'])
            if selected:
                item.update(selected);status='standard_HTML_annual_header_context_located_not_admitted'
                holds.append('proposed_annual_start_boundary_assumption_not_yet_approved')
                if item['periods']['annual_date_binding']!='same_report_full_date_headers':holds.append('annual_year_to_balance_date_link_requires_independent_review')
                for field in item['fields'].values():field.update(source=dict(item['source']),available_from=doc['available_session'],availability_verified=doc['available_session_verified'])
            else:
                status='current_page_formal_pair_unresolved';holds+=gaps
                if not tables:holds.append('linked_statement_attachment_or_unhandled_layout_not_reviewed')
            if any(t['structural_review_flags'] for t in tables):holds.append('adjusted_or_reconciliation_heading_requires_field_scope_review')
        item.update(context_status=status,context_holds=sorted(set(holds)),page_without_table_does_not_prove_report_missing=True)
        events.append(item)
    for path in ('work/after152_next_source_context.py','work/after151_next_source_context.py','work/test_after152_next_source_context.py'):
        f.read_local(path,pins)
    return dict(schema_version=1,stage='pre_extraction_source_structure_context_not_admission',created_utc=datetime.now(timezone.utc).isoformat(),
        source_scope=list(f.SYMBOLS),parent_feasibility=dict(path=PREVIOUS,sha256=pins[PREVIOUS]),
        events=events,input_hashes=pins,counts=dict(catalog_events=len(events),by_symbol=dict(Counter(d['symbol'] for d in events)),
        by_context_status=dict(Counter(d['context_status'] for d in events)),clock_verified=sum(d['available_session_verified'] for d in events)),
        coordinate_convention='HTML table,row,physical_cell,logical_column zero based; no monetary values in output',
        limitations=['Full23 frozen catalog retained; seven index entries are not original PDFs.',
            'Calendar years on an annual income header may require independent linkage to balance-sheet fiscal dates. No start is quoted or certified.',
            'No actual monetary cell parsing; physical values/signs and field-specific accounting must be independently reviewed before admission.',
            'Total consolidated NI is distinct from parent-attributable NI. A page without the table does not prove the report is absent.',
            'No first-public-version, ISO currency, continuous-security-master or complete event-history certification.'],
        financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--details',action='store_true');args=p.parse_args();result=build()
    if args.write:
        with (ROOT/OUTPUT).open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(counts=result['counts'],output=OUTPUT if args.write else None,
        events=[dict(event_id=e['event_id'],status=e['context_status'],tables=e['statement_contexts'] if args.details else len(e['statement_contexts']),periods=e['periods']) for e in result['events']]),sort_keys=True))


if __name__=='__main__':main()
