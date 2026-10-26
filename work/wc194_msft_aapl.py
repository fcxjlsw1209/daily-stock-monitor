"""Frozen-scope source-only extraction of three signed trade cash-flow rows.

All twelve MSFT and ten AAPL reports retained, including original holds. No
component sum, financial ratio, factor, model, orders, returns or new admission.
Only source parsing/geometry helpers and isolated PDF runtime are reused.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date,datetime,timezone
from decimal import Decimal
import io,json,re,sys
from pathlib import Path
from bs4 import BeautifulSoup
import capex193_msft_aapl as base

ROOT=Path(__file__).resolve().parents[1]
SCOPE='work/working_capital194_source_scope.json'
SCOPE_SHA='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
CONCEPT='work/working_capital194_experiment_concept.json'
CONCEPT_SHA='380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
OUT=ROOT/'outputs/monthly_working_capital194_source/msft_aapl'
PINS={**base.PINS,'work/capex193_msft_aapl.py':'01bf8a7de433422ddc9981705b39d9723dab9f9a6ac4c0bbd04c2d445d635208',CONCEPT:CONCEPT_SHA}
COMPONENTS=('accounts_receivable','inventories','accounts_payable')
LABELS={'accounts_receivable':re.compile(r'Accounts receivable(?:,?\s+net)?',re.I),
        'inventories':re.compile(r'Inventories',re.I),'accounts_payable':re.compile(r'Accounts payable',re.I)}
require,sha,norm,local,verify,publish=base.require,base.sha,base.norm,base.local,base.verify,base.publish


def signed_cell(literal,scale,confirmed_cell=True):
    """No sign inversion. An explicit dash is zero only inside a confirmed cell."""
    require(type(literal) is str and literal.strip(),'Explicit cash-flow literal required')
    normalized=re.sub(r'[ \t]+(?=\)$)','',literal.strip().replace('/dollarsign','$'))
    normalized=re.sub(r'^\$\s*','',normalized).strip()
    dash=normalized in ('-','—','–')
    if dash:
        require(confirmed_cell,'Dash requires confirmed current annual cell');value='0'
    else:
        require(bool(re.fullmatch(r'(?:\((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)|-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)',normalized)),'Ambiguous monetary literal')
        value=normalized.replace(',','')
        if value.startswith('('):value='-'+value[1:-1]
        value=format(Decimal(value),'f')
    require(str(scale) in ('1000','1000000'),'Explicit statement monetary scale required')
    return dict(signed_literal=literal,normalized_monetary_literal=normalized,signed_value_decimal=value,
                scale=str(scale),explicit_dash_zero=dash)


def monetary_columns(tail,count):
    tail=tail.replace('/dollarsign','$').replace('$','')
    number=r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    token=re.compile(r'\('+number+r'[ \t]*\)|-?'+number+r'|[—–-]')
    matches=list(token.finditer(tail));require(len(matches)==count,'Exact annual column count required')
    position=0;raw=[];clean=[]
    for m in matches:
        require(not tail[position:m.start()].strip(),'Unmapped monetary tail')
        parsed=signed_cell(m[0],'1000000');raw.append(m[0]);clean.append(parsed['normalized_monetary_literal']);position=m.end()
    require(not tail[position:].strip(),'Unmapped monetary tail')
    return raw,clean


def html_fields(raw,event):
    loc=base.verify_cfo_locator(base.html2.inventory(raw,event['periods']),event)
    soup=BeautifulSoup(raw,'html.parser');table=soup.find_all('table')[loc['table_index_zero_based']]
    rows,grid,physical=base.v1.html.table_grid(table)
    segment=loc.get('statement_segment',{});start=segment.get('start_row',0);end=segment.get('end_row_exclusive',len(rows))
    labels={ri:[x['text'] for (r,_),x in physical.items() if r==ri and re.search('[A-Za-z]',x['text'])] for ri in range(start,end)}
    cfo=loc['row_index_zero_based']
    changes=[ri for ri,items in labels.items() if ri<cfo and any(re.search(r'changes in.*(?:operating assets|assets and liabilities)',norm(s),re.I) for s in items)]
    require(len(changes)==1,'Unique operating working-capital reconciliation heading required')
    columns=loc['logical_columns'];result={};evidence={}
    top=next((r for r in range(start,cfo) if any(base.v1.NI.fullmatch(norm(s)) for s in labels[r])),None)
    require(top is not None,'Cash-flow indirect-method starting row required')
    headers=[]
    for col in columns:
        cells=[x for x in base.v1.html.header_evidence(grid,top,col) if x['row']>=start]
        parts=[x['text'] for x in cells]
        require(base.v1.html.dates(parts)==[loc['period_end']] and base.v1.ANNUAL.search(' '.join(parts))
                and not base.v1.QUARTER.search(' '.join(parts)),'Exact current annual cash-flow column required')
        headers.append(dict(logical_column=col,cells=deepcopy(cells)))
    context=base.v1.html.external_context(table)
    if not base.v1.CF_TITLE.search(' '.join(context)):context.append(base.v1.legacy.statement_context(table))
    units=[x['text'] for (r,_),x in physical.items() if start<=r<top and re.search(r'\bmillions|\bthousands',x['text'],re.I)]
    units += [x for x in context if re.search(r'\bmillions|\bthousands',x,re.I)]
    require(units and base.v1.units(units)==loc['scale'],'Statement units missing')
    require('$' in table.get_text(),'Statement dollar literal absent')
    for key,pattern in LABELS.items():
        matches=[ri for ri in range(changes[0]+1,cfo) if any(pattern.fullmatch(norm(s)) for s in labels[ri])]
        clues=[dict(row_index_zero_based=ri,row_text=norm(rows[ri].get_text(' ',strip=True)))
               for ri in range(changes[0]+1,cfo) if any(re.search({'accounts_receivable':'receivabl','inventories':'inventor','accounts_payable':'payabl'}[key],s,re.I) for s in labels[ri])]
        evidence[key]=dict(candidates=clues,hold_reasons=[])
        try:
            require(len(matches)==1,'Unique distinct pure '+key+' row required')
            ri=matches[0];label=next(norm(s) for s in labels[ri] if pattern.fullmatch(norm(s)))
            chosen=sorted({grid[(ri,c)]['physical_cell'] for c in columns if (ri,c) in grid})
            require(chosen and all(physical[(ri,p)]['rowspan']==1 and set(physical[(ri,p)]['columns']).issubset(columns) for p in chosen),'Financial cell crosses period columns')
            display=' '.join(physical[(ri,p)]['text'] for p in chosen if physical[(ri,p)]['text']).strip()
            parsed=signed_cell(display,loc['scale']);row=rows[ri];row_text=norm(row.get_text(' ',strip=True));cells=row.find_all(['td','th'],recursive=False)
            result[key]=dict(parsed,component=key,measure='current_annual_operating_cashflow_contribution',row_label=label,
                row_text=row_text,row_text_sha256=sha(row_text.encode()),row_html=str(row),row_html_sha256=sha(str(row).encode()),
                cell_html=[str(cells[p]) for p in chosen],cell_html_sha256=[sha(str(cells[p]).encode()) for p in chosen],
                currency_literal='$' if '$' in display else None,currency_iso=None,statement_currency_literal='$',
                currency_scope='same formal cash-flow table; ISO unverified',
                statement_currency_evidence=dict(original_cfo_locator=deepcopy(loc),format='html',table_index_zero_based=loc['table_index_zero_based'],statement_segment=segment or None),
                source=deepcopy(event['source']),period=deepcopy(event['periods']['current']),available_from=event['available_from'],
                locator=dict(format='html',table_index_zero_based=loc['table_index_zero_based'],row_index_zero_based=ri,
                    physical_cell_indices_zero_based=chosen,logical_columns=columns,statement_segment=segment or None,
                    statement_title=loc['statement_title'],table_html_sha256=sha(str(table).encode()),header_literals=loc['header_literals'],
                    header_cells=headers,unit_evidence=units,period_end=loc['period_end'],
                    operating_changes_heading_row_zero_based=changes[0],operating_changes_heading=labels[changes[0]],
                    original_cfo_locator=deepcopy(loc)),extraction_verified=True,fact_admitted=False)
        except (ValueError,KeyError,IndexError) as error:
            result[key]=None;evidence[key]['hold_reasons'].append(str(error))
    return result,evidence


def pdf_layout(texts,event):
    loc=base.verify_cfo_locator(base.pdf4.inventory(texts,event['periods']),event)
    lines=texts[loc['page_1based']-1].splitlines();cfo=loc['line_1based']-1
    changes=[i for i,line in enumerate(lines[:cfo]) if re.search(r'changes in.*(?:operating assets|assets and liabilities)',line,re.I)]
    require(len(changes)==1,'Unique operating working-capital reconciliation heading required')
    require(any('$' in s or '/dollarsign' in s for s in lines),'Statement currency literal absent')
    result={};evidence={}
    for key,pattern in LABELS.items():
        matches=[]
        for ri in range(changes[0]+1,cfo):
            for label in ({'accounts_receivable':['Accounts receivable, net','Accounts receivable'],
                           'inventories':['Inventories'],'accounts_payable':['Accounts payable']}[key]):
                hit=base.v1.pdf_row_match(lines[ri],label)
                if hit:matches.append((ri,label,hit[1]))
        clues=[dict(line_1based=ri+1,row_text=lines[ri]) for ri in range(changes[0]+1,cfo)
               if re.search({'accounts_receivable':'receivabl','inventories':'inventor','accounts_payable':'payabl'}[key],lines[ri],re.I)]
        evidence[key]=dict(candidates=clues,hold_reasons=[])
        pending=None
        try:
            require(len(matches)==1,'Unique distinct pure '+key+' PDF row required')
            ri,label,tail=matches[0]
            pending=dict(locator=dict(loc,row_label=label,line_1based=ri+1,original_cfo_locator=deepcopy(loc),
                operating_changes_heading_line_1based=changes[0]+1,operating_changes_heading=lines[changes[0]]),
                row_text=lines[ri],monetary_tail=tail)
            raw,clean=monetary_columns(tail,len(loc['header_dates']))
            selected=raw[loc['financial_column_index_0based']];parsed=signed_cell(selected,loc['scale'])
            component_loc=dict(loc,row_label=label,line_1based=ri+1,original_cfo_locator=deepcopy(loc),
                operating_changes_heading_line_1based=changes[0]+1,operating_changes_heading=lines[changes[0]])
            result[key]=dict(parsed,component=key,measure='current_annual_operating_cashflow_contribution',row_label=label,
                row_text=lines[ri],row_text_sha256=sha(lines[ri].encode()),all_row_cells=clean,all_row_signed_literals=raw,
                source=deepcopy(event['source']),period=deepcopy(event['periods']['current']),available_from=event['available_from'],
                currency_literal='$' if '$' in lines[ri] else '/dollarsign' if '/dollarsign' in lines[ri] else None,currency_iso=None,
                statement_currency_literal='$',currency_scope='same formal cash-flow page; ISO unverified',
                statement_currency_evidence=dict(format='pdf',page_1based=loc['page_1based'],original_cfo_locator=deepcopy(loc)),
                locator=component_loc,fact_admitted=False)
        except (ValueError,KeyError,IndexError) as error:
            result[key]=None;evidence[key]['hold_reasons'].append(str(error))
            if pending is not None:evidence[key]['pending_geometry']=pending
    return result,evidence


def geometry_reconcile(rows,loc,layout_tail):
    """Recover no digits: compare all actual PDFium glyphs with the whole layout tail.

    PDFium source-character continuity defines financial tokens and year-header
    geometry defines columns. Layout's only ignored characters are whitespace.
    This is not a rule to join arbitrary text numbers or infer a missing cell.
    """
    label=base.pdf4.compact(loc['row_label']);matches=[]
    for row in rows:
        if base.pdf4.row_text(row).lower().startswith(label):
            cells,_=base.pdf4.numeric_tail(row[len(label):]);matches.append([c['text'] for c in cells])
    require(len(matches)==1,'Unique glyph row required for layout reconciliation')
    cells=matches[0];tail=layout_tail.replace('/dollarsign','$').replace('$','')
    require(''.join(tail.split())==''.join(cells),'PDF layout/glyph character disagreement')
    geometry=base.pdf4.geometry_rows(rows,loc,cells)
    for value in cells:signed_cell(value,loc['scale'])
    return cells,geometry


def pdf_fields(raw,event,runtime,render_dir=None):
    reader,pdfium=base.configure_pdf193(runtime)
    document=reader(io.BytesIO(raw));require(1<=len(document.pages)<=12,'Bounded annual PDF required')
    texts=[page.extract_text(extraction_mode='layout') for page in document.pages]
    fields,evidence=pdf_layout(texts,event)
    page_number=event['fields']['current']['operating_cash_flow']['locator']['page_1based']
    doc=pdfium.PdfDocument(raw);page=doc[page_number-1];tp=page.get_textpage();render=None
    try:
        rows=base.pdf4.physical_rows(tp)
        for key,field in list(fields.items()):
            if field is None:
                pending=evidence[key].get('pending_geometry')
                if pending is None:continue
                try:
                    loc=pending['locator'];cells,geometry=geometry_reconcile(rows,loc,pending['monetary_tail'])
                    literal=cells[loc['financial_column_index_0based']]
                    field=dict(signed_cell(literal,loc['scale']),component=key,
                        measure='current_annual_operating_cashflow_contribution',row_label=loc['row_label'],
                        row_text=pending['row_text'],row_text_sha256=sha(pending['row_text'].encode()),
                        all_row_cells=cells,all_row_signed_literals=cells,
                        literal_origin='PDFium original glyphs inside independently verified annual column',
                        layout_whitespace_reconciled=True,original_layout_monetary_tail=pending['monetary_tail'],
                        source=deepcopy(event['source']),period=deepcopy(event['periods']['current']),available_from=event['available_from'],
                        currency_literal='$' if '$' in pending['row_text'] else None,currency_iso=None,
                        statement_currency_literal='$',currency_scope='same formal cash-flow page; ISO unverified',
                        statement_currency_evidence=dict(format='pdf',page_1based=loc['page_1based'],original_cfo_locator=deepcopy(loc['original_cfo_locator'])),
                        locator=loc,pdfium_geometry=geometry,pdfium_text_agrees=True,extraction_verified=True,fact_admitted=False)
                    evidence[key]['resolved_layout_notes']=evidence[key]['hold_reasons'];evidence[key]['hold_reasons']=[]
                    fields[key]=field
                except (ValueError,IndexError,KeyError) as error:
                    evidence[key]['hold_reasons'].append('geometry_reconciliation:'+str(error));continue
            try:
                field['pdfium_geometry']=base.pdf4.geometry_rows(rows,field['locator'],field['all_row_cells'])
                field['pdfium_text_agrees']=True;field['extraction_verified']=True
            except (ValueError,IndexError,KeyError) as error:
                evidence[key]['parsed_unverified_field']=field;fields[key]=None
                evidence[key]['hold_reasons'].append('geometry:'+str(error))
        if render_dir is not None:
            path=Path(render_dir)/('AAPL_FY'+str(event['fiscal_year'])+'_cashflow.png')
            require(not path.exists(),'Write-once image exists');bitmap=page.render(scale=1.6)
            try:
                with path.open('xb') as stream:bitmap.to_pil().save(stream,format='PNG')
            finally:bitmap.close()
            render=dict(path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes()),page_1based=page_number,scale=1.6,visual_review_pending=True)
            for field in fields.values():
                if field is not None:field['render']=render
    finally:tp.close();page.close();doc.close()
    return fields,evidence,render


def record(event,fields,evidence,render=None):
    reasons=['inherited:'+x for x in event['cash_quality_admission']['reasons']]
    if not event['cash_quality_admission']['admitted'] and not reasons:reasons.append('inherited:admission_false')
    for key in COMPONENTS:
        if fields[key] is None:reasons += [key+':'+x for x in evidence[key]['hold_reasons']]
    complete=all(fields[key] is not None and fields[key].get('extraction_verified') is True for key in COMPONENTS)
    return dict(event_id=event['event_id'],symbol=event['symbol'],fiscal_year=event['fiscal_year'],
        release_date=event['release_date'],available_from=event['available_from'],period=deepcopy(event['periods']['current']),
        inherited_admitted=event['cash_quality_admission']['admitted'],inherited_holds=deepcopy(event['cash_quality_admission']['reasons']),
        inherited_rejection_codes=deepcopy(event['inherited_rejection_codes']),inherited_accounting_review_flags=deepcopy(event['accounting_review_flags']),
        inherited_identity=deepcopy(event['identity']),source=deepcopy(event['source']),fields=fields,component_evidence=evidence,render=render,
        extraction_verified=complete,status='verified' if complete and not reasons else 'held',hold_reasons=reasons,
        fact_admitted=False,strategy_admitted=False,first_public_version_verified=False,complete_correction_history_verified=False,
        certified_daily_security_master=False)


def gate(expected):
    require(expected==SCOPE_SHA,'Exact194 scope SHA required');verify({SCOPE:expected,**PINS})
    scope=json.loads(local(SCOPE).read_text());verify(scope['input_sha256'])
    require(scope['expected_event_count']==44 and scope['inherited_admitted_count']==31 and scope['component_sum_permitted'] is False,'Source-only full scope differs')
    events=json.loads(local(base.DATA).read_text())['events']
    require(len(events)==44 and sum(e['cash_quality_admission']['admitted'] for e in events)==31,'Original44/31 census changed')
    events=[e for e in events if e['symbol'] in ('MSFT','AAPL')]
    targets=[t for t in scope['prospective_targets'] if t['symbol'] in ('MSFT','AAPL')]
    require(len(events)==len(targets)==22 and len({e['event_id'] for e in events})==22,'All22 distinct reports required')
    pins={SCOPE:expected,**PINS,**scope['input_sha256']};calendar=json.loads(local(base.CALENDAR).read_text())['sessions']
    for e,t in zip(events,targets):
        require(e['event_id']==t['event_id'] and e['source']==t['source'] and e['available_from']==t['inherited_available_from']
                and e['cash_quality_admission']['admitted']==t['inherited_admitted']
                and e['fields']['current']['operating_cash_flow']['locator']==t['current_cfo_locator'],'Scope identity/source/clock/locator changed')
        require(calendar[bisect_right(calendar,e['release_date'])]==e['available_from'],'First subsequent session differs')
        current=e['periods']['current'];days=(date.fromisoformat(current['period_end'])-date.fromisoformat(current['period_start'])).days+1
        require(days==current['days_inclusive'] and 330<=days<=400,'Annual period mismatch')
        for item in (e['source'],e['source']['receipt']):pins[item['path']]=item['sha256']
    verify(pins);runtime=json.loads(local(base.PDF_RUNTIME).read_text());base.configure_pdf193(runtime)
    return scope,events,pins,runtime


def run(expected):
    scope,events,pins,runtime=gate(expected)
    require(not (OUT/'EXTRACTION.json').exists(),'Write-once extraction exists')
    code={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__).read_bytes()),
          'work/wc194_msft_aapl_test.py':sha(local('work/wc194_msft_aapl_test.py').read_bytes())}
    OUT.mkdir(parents=True,exist_ok=True);renders=OUT/'renders';renders.mkdir(exist_ok=True);records=[]
    for event in events:
        raw=local(event['source']['path']).read_bytes();require(sha(raw)==event['source']['sha256'],'Source changed')
        try:
            if event['symbol']=='AAPL':fields,evidence,render=pdf_fields(raw,event,runtime,renders)
            else:fields,evidence=html_fields(raw,event);render=None
            records.append(record(event,fields,evidence,render))
        except (ValueError,KeyError,IndexError) as error:
            records.append(record(event,{k:None for k in COMPONENTS},{k:dict(candidates=[],hold_reasons=[str(error)]) for k in COMPONENTS}))
    verify(pins);verify(code);base.configure_pdf193(runtime)
    result=dict(schema='working_capital194_msft_aapl_source_extraction_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        scope_sha256=expected,concept_sha256=CONCEPT_SHA,records=records,report_count=22,
        extraction_verified_count=sum(r['extraction_verified'] for r in records),verified_count=sum(r['status']=='verified' for r in records),
        held_count=sum(r['status']=='held' for r in records),inherited_admitted_count=sum(r['inherited_admitted'] for r in records),
        source_hashes=pins,code_hashes=code,pdf_runtime_hashes=runtime['pdf_runtime_hashes'],runtime=sys.version,
        component_sums=0,financial_ratios=0,factors=0,model_fits=0,orders=0,returns=0,fact_admissions=0,network_requests=0,
        pdf_visual_review_pending=True,limits=scope['source_limitations'])
    publish(OUT/'EXTRACTION.json',result)
    return dict(path=str((OUT/'EXTRACTION.json').relative_to(ROOT)),sha256=sha((OUT/'EXTRACTION.json').read_bytes()),
        reports=22,extraction_verified=result['extraction_verified_count'],verified=result['verified_count'],held=result['held_count'],
        holds=[dict(event_id=r['event_id'],reasons=r['hold_reasons']) for r in records if r['status']=='held'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope-sha256',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:
        _,events,pins,runtime=gate(a.scope_sha256);inspection=[]
        for event in events:
            try:
                raw=local(event['source']['path']).read_bytes()
                if event['symbol']=='AAPL':fields,evidence,_=pdf_fields(raw,event,runtime)
                else:fields,evidence=html_fields(raw,event)
                inspection.append(dict(event_id=event['event_id'],fields={k:None if f is None else
                    dict(label=f['row_label'],signed=f['signed_value_decimal'],row=f['row_text']) for k,f in fields.items()},
                    holds={k:v for k,v in evidence.items() if v['hold_reasons']}))
            except (ValueError,IndexError,KeyError) as error:inspection.append(dict(event_id=event['event_id'],error=str(error)))
        verify(pins);print(json.dumps(inspection,indent=2))
    else:print(json.dumps(run(a.scope_sha256),indent=2))
