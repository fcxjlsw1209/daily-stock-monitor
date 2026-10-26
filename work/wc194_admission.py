"""Source-only194 gate: preserve original44 events; never aggregate components."""
from copy import deepcopy
from datetime import date
from bisect import bisect_right
from decimal import Decimal
import hashlib
import re

DEFINITION='current_annual_trade_working_capital_signed_contributions_v1'
METHOD='current_annual_cash_before_trade_working_capital_v1'
KEYS=('accounts_receivable','inventories','accounts_payable')
META=('event_id','symbol','release_date','available_from')
LABELS={'accounts_receivable':r'(?:trade\s+)?accounts? receivable(?:,?\s+net)?',
        'inventories':r'inventor(?:y|ies)',
        'accounts_payable':r'(?:trade\s+)?accounts? payable'}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def indexed(rows,ids):
    require(type(rows) is list and all(type(r) is dict and type(r.get('event_id')) is str for r in rows),'Event records required')
    result={r['event_id']:r for r in rows}
    require(len(result)==len(rows) and set(result)==set(ids),'Exact unique source census required')
    return result

def amount(x):
    require(type(x) is str and re.fullmatch(r'-?\d+(?:\.\d+)?',x),'Explicit plain decimal required')
    value=Decimal(x);require(value.is_finite(),'Finite source value required');return value

def literal_value(x):
    require(type(x) is str and x.strip(),'Signed literal required')
    text=re.sub(r'\s+|\$|/dollarsign','',x).replace('−','-')
    if text in ('-','—','–'):return Decimal(0)
    require(bool(re.fullmatch(r'\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?',text)),'Malformed signed literal')
    require(('(' in text)==(')' in text),'Unpaired sign parentheses')
    if text.startswith('('):
        require('-' not in text,'Conflicting source sign');text='-'+text[1:-1]
    return amount(text.replace(',',''))

def check_field(field,key,original,target,independent):
    require(type(field) is dict and field.get('component')==key,'Component identity differs')
    require(re.fullmatch(LABELS[key],re.sub(r'\s+',' ',field.get('row_label','')).strip(),re.I),'Mixed or wrong component label')
    require(amount(field['signed_value_decimal'])==literal_value(field['signed_literal']),'Signed literal/value mismatch')
    require(amount(field['scale'])>0,'Positive source scale required')
    row=field.get('row_text')
    require(type(row) is str and row and hashlib.sha256(row.encode()).hexdigest()==field.get('row_text_sha256'),'Row text hash differs')
    require(independent.get('verified') is True,'Independent component verification missing')
    for k in ('signed_value_decimal','scale'):
        require(independent.get(k)==field[k],'Independent component '+k+' differs')
    loc=field.get('locator');cfo=original['fields']['current'].get('operating_cash_flow')
    require(type(loc) is dict and type(cfo) is dict,'Current CFO and locator required for existing component')
    cfo_loc=cfo.get('locator')
    require(type(cfo_loc) is dict and loc.get('original_cfo_locator')==cfo_loc,'Original CFO locator differs')
    require(loc.get('period_end')==original['periods']['current']['period_end']==independent.get('annual_period_end'),'Current annual period differs')
    require(cfo['source']==target['source'] and amount(cfo['scale'])==amount(field['scale']) and cfo['currency_literal']=='$','Paired CFO source/unit differs')
    require(field.get('currency_iso') is None and field.get('currency_literal') in (None,'$','/dollarsign'),'Currency conflict')
    require(field.get('statement_currency_literal') in (None,'$'),'Conflicting statement currency')
    evidence=field.get('statement_currency_evidence',{})
    require(type(evidence) is dict and evidence.get('original_cfo_locator')==cfo_loc,'Statement currency evidence absent')
    fmt=loc.get('format')
    require(fmt==cfo_loc.get('format')==evidence.get('format'),'Cashflow format differs')
    if fmt=='html':
        number=loc.get('table_index_zero_based')
        require(type(number) is int and number>=0 and number==cfo_loc.get('table_index_zero_based')==evidence.get('table_index_zero_based'),'Cashflow table differs')
        require(loc.get('statement_segment')==cfo_loc.get('statement_segment'),'Cashflow statement segment differs')
        if 'statement_segment' in evidence:require(evidence['statement_segment']==cfo_loc.get('statement_segment'),'Currency evidence statement segment differs')
        if field.get('statement_currency_literal')=='$':
            require(field.get('currency_scope')=='same formal cash-flow table; ISO unverified','Statement currency scope absent')
        else:
            require(evidence.get('literal')=='$' and type(evidence.get('rows')) is list and evidence['rows'],'Explicit same-table dollar evidence absent')
            require(all(type(r.get('row_text')) is str and '$' in r['row_text'] and re.fullmatch('[0-9a-f]{64}',r.get('row_html_sha256','')) for r in evidence['rows']),'Dollar evidence rows incomplete')
        ri=loc.get('row_index_zero_based');end=cfo_loc.get('row_index_zero_based')
        heads=loc.get('operating_reconciliation_heading_rows')
        if heads is None:heads=[loc.get('operating_changes_heading_row_zero_based')]
        require(type(ri) is int and type(end) is int and type(heads) is list and heads and all(type(n) is int for n in heads) and any(h<ri<end for h in heads),'Outside operating reconciliation')
        position=(fmt,number,loc.get('statement_segment',{}).__repr__(),ri)
    elif fmt=='pdf':
        number=loc.get('page_1based')
        require(type(number) is int and number>0 and number==cfo_loc.get('page_1based')==evidence.get('page_1based'),'Cashflow page differs')
        require(field.get('statement_currency_literal')=='$' and field.get('currency_scope')=='same formal cash-flow page; ISO unverified','PDF currency scope absent')
        ri=loc.get('line_1based');head=loc.get('operating_changes_heading_line_1based');end=cfo_loc.get('line_1based')
        require(all(type(n) is int for n in (head,ri,end)) and head<ri<end,'Outside operating reconciliation')
        require(field.get('pdfium_text_agrees') is True and type(field.get('pdfium_geometry')) is dict and field['pdfium_geometry'],'PDF column geometry missing')
        position=(fmt,number,ri)
    else:raise ValueError('Unknown source format')
    require(field.get('fact_admitted',False) is False,'Raw source admission flag changed')
    result=deepcopy(field)
    for k,v in [('source',target['source']),('period',original['periods']['current']),('available_from',original['available_from']),('fact_admitted',False)]:
        result.setdefault(k,deepcopy(v));require(result[k]==v,'Normalized evidence differs: '+k)
    return result,position

def check_record(original,target,producer,independent,sessions):
    for k in META:require(producer.get(k)==original.get(k),'Producer metadata differs: '+k)
    require(target['event_id']==independent.get('event_id')==original['event_id'],'Event binding differs')
    inherited=original['cash_quality_admission']['admitted']
    require(type(inherited) is bool and producer.get('inherited_admitted') is inherited and target.get('inherited_admitted') is inherited,'Inherited gate changed')
    require(producer.get('source')==target['source']==original['source'],'Source reference changed')
    require(target['inherited_available_from']==original['available_from'],'Source clock changed')
    require(independent.get('source_sha256')==target['source']['sha256'] and independent.get('review_passed') is True,'Completed independent review required')
    require(type(sessions) is list and sessions==sorted(set(sessions)),'Sorted unique sessions required')
    for d in (original['release_date'],original['available_from']):require(date.fromisoformat(d).isoformat()==d,'Canonical date required')
    ix=bisect_right(sessions,original['release_date'])
    require(ix<len(sessions) and sessions[ix]==original['available_from'],'First session after release required')
    require(type(producer.get('fields')) is dict and set(producer['fields'])==set(KEYS),'Exactly three raw component keys required')
    require(type(independent.get('components')) is dict and set(independent['components'])==set(KEYS),'Exactly three independent component keys required')
    require(type(producer.get('hold_reasons')) is list and type(producer.get('extraction_verified')) is bool,'Explicit hold/extraction result required')
    result_fields={};positions=[];reasons=[]
    if not inherited:reasons.append('preserved_original_cash_quality_hold')
    for key in KEYS:
        field=producer['fields'][key];review=independent['components'][key]
        if field is None:
            require(review.get('verified') is False and all(review.get(k) is None for k in ('signed_value_decimal','scale','annual_period_end')),'Missing independent component disagreement')
            require(producer['hold_reasons'],'Missing component needs explanation')
            result_fields[key]=None;reasons.append(key+':missing_or_ambiguous')
        else:
            normalized,pos=check_field(field,key,original,target,review)
            require(independent.get('paired_cfo_verified') is True,'Paired CFO review required')
            if inherited:require(independent.get('paired_assets_verified') is True,'Paired assets review required')
            result_fields[key]=normalized;positions.append(pos)
    require(len(set(positions))==len(positions),'Duplicate component rows')
    complete=all(result_fields[k] is not None for k in KEYS)
    require(producer['extraction_verified'] is complete,'Producer extraction completeness differs')
    reasons=list(dict.fromkeys(reasons+producer['hold_reasons']))
    require(producer.get('status')==('held' if reasons else 'verified'),'Producer status/reasons differ')
    require(complete or reasons,'Incomplete extraction cannot pass')
    result=deepcopy(original)
    result.update(trade_working_capital=result_fields,trade_working_capital_source_evidence=deepcopy(producer),trade_working_capital_admission=dict(admitted=not reasons,reasons=reasons,definition=DEFINITION))
    require({k:v for k,v in result.items() if k not in ('trade_working_capital','trade_working_capital_source_evidence','trade_working_capital_admission')}==original,'Original event changed')
    return result
