"""Independent 21-report accounting scope only; never parse monetary cells."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup
import after153_batch5_accounting_resolution as helper

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after153_batch5_accounting_review.py'
TEST='work/test_after153_batch5_accounting_review.py'
OUT='work/after153_batch5_accounting_review.json'
PLAN='outputs/monthly_financial_coverage_batch5_v1/PLAN.json'
PLAN_SHA='77eae28f976174c657b85d4160c47ab056fb5b3abdaf31d9c8c45cb410d49c6e'
EXTRACTIONS='outputs/monthly_financial_coverage_batch5_v1/EXTRACTIONS.json'
EXTRACTIONS_SHA='938c2528953d73340139e63a80bb0f4afed703020f8be7fbc73e1e1124d78fa0'
CONTEXT='work/after153_next_source_context.json'
CONTEXT_SHA='1bd968ca401a313af2ee6c1e92328ae8b270e7c78e607b14edb61060ecc6f241'
FEASIBILITY='work/after153_batch5_accounting_feasibility_review.json'
FEASIBILITY_SHA='6e8cc03a6502b8194792b6d17e65f3c5fec993c7bd638ee44c4fee4a4bb08b94'
RESOLUTION='work/after153_batch5_accounting_resolution.json'
RESOLUTION_SHA='2091d65f39e8e0947128294230ad43951caa6298d59ccb9c0121203ad0fed896'
FIELDS=('annual_net_income','current_total_assets','prior_total_assets')
EXPECTED=[('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2020)]+[('TMO',y) for y in range(2012,2022)]
check=helper.check
norm=helper.norm


def load(name,pins,sha=None):return json.loads(helper.read(name,pins,sha))


def visible_document(soup):
    for tag in soup(['script','style']):tag.decompose()
    strings=[(node,norm(str(node))) for node in soup.strings if norm(str(node))]
    positions={};parts=[];offset=0
    for node,value in strings:
        positions[id(node)]=offset;parts.append(value);offset+=len(value)+1
    text=' '.join(parts)
    check(text==norm(soup.get_text(' ',strip=True)),'Visible text convention')
    return text,positions


def quote_cell(cell,source,text,positions,ti,row,physical):
    quote=norm(cell.get_text(' ',strip=True))
    node=next((n for n in cell.strings if norm(str(n))),None)
    check(node is not None and quote,'Empty evidence cell')
    start=positions[id(node)]
    check(text[start:start+len(quote)]==quote,'Exact selected-cell text location')
    return dict(path=source['path'],sha256=source['sha256'],
      locator=dict(kind='document_text',start=start,end=start+len(quote)),quote=quote,
      physical_cell=dict(table=ti,row=row,physical_cell=physical))


def quote_literal(quote,source,text):
    start=text.find(quote);check(bool(quote) and start>=0,'Missing explicit accounting quote')
    return dict(path=source['path'],sha256=source['sha256'],locator=dict(kind='document_text',start=start,end=start+len(quote)),quote=quote)


def table_rows(table):return [r for r in table.find_all('tr') if r.find_parent('table') is table]


def metadata_evidence(context,soup,text,positions):
    proof=[];seen=set();tables=soup.find_all('table')
    for field in FIELDS:
        spec=context['fields'][field];ti=spec['table'];rows=table_rows(tables[ti]);row=spec['row'];physical=spec['label_cell']
        cells=rows[row].find_all(['td','th'],recursive=False);cell=cells[physical]
        check(norm(cell.get_text(' ',strip=True))==spec['label'],'Selected field label changed')
        if field=='annual_net_income':
            check(re.fullmatch(r'(?:consolidated\s+)?net\s+(?:income|earnings|loss)(?:\s*\((?:loss|income)\))?',spec['label'],re.I),'Total NI label required')
        else:check(spec['label'].lower()=='total assets','Total assets required')
        candidates=[(row,physical,cell)]
        for h in spec['header_evidence']:
            head=rows[h['row']].find_all(['td','th'],recursive=False)[h['physical_cell']]
            check(norm(head.get_text(' ',strip=True))==h['text'],'Raw header differs from context')
            # Context only: no monetary body or amounts are selected here.
            candidates.append((h['row'],h['physical_cell'],head))
        for rr,cc,element in candidates:
            if (ti,rr,cc) in seen:continue
            seen.add((ti,rr,cc));proof.append(quote_cell(element,context['source'],text,positions,ti,rr,cc))
    check(bool(proof),'No metadata evidence')
    return proof


def assert_scope(context,event):
    check((context['symbol'],context['fiscal_year']) in EXPECTED,'Unexpected source scope')
    for key in ('event_id','symbol','fiscal_year','source','context_holds','publication_date'):
        check(event[key]==context[key],'Event metadata changed '+key)
    check(set(event['fields'])==set(FIELDS),'Three fields required')
    check(all(event[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')),'Premature admission')
    p=context['periods'];e=event['periods']
    check(p['days_inclusive'] in (365,366),'Calendar year length')
    check(p['current_end'][5:]=='12-31' and p['prior_end'][5:]=='12-31','Annual endpoints')
    check(e['current']['period_start']==p['proposed_current_start'] and e['current']['period_end']==p['current_end'] and e['current']['days_inclusive']==p['days_inclusive'],'Published current period')
    check(e['comparison']['period_end']==p['prior_end'],'Published prior endpoint')
    check(event['available_from']==context['available_session'],'Clock metadata changed')
    expected_scale='1000' if context['symbol']=='PLTR' else '1000000'
    for name in FIELDS:
        spec=context['fields'][name]
        check(spec['scale']==expected_scale and spec['currency_literal']=='$' and spec['currency_iso'] is None,'Common report-unit scope')
        published=event['fields'][name]
        if published is None:
            check(any(g['field']==name and g['reason'] for g in event['gaps']),'Unexplained null')
            continue
        check(published['source']==context['source'] and published['available_from']==context['available_session'],'Field source/time changed')
        check(published['scale']==spec['scale'] and published['currency_literal']=='$' and published['currency_iso'] is None,'Field units changed')
        check(published['as_of']==spec['as_of'],'Field date changed')
        check(published['period']==(e['current'] if name=='annual_net_income' else None),'Field period mismatch')
        check(published['fact_admitted'] is False,'Field premature admission')


def scoped_decision(previous,resolutions):
    eid=previous['event_id']
    basis=' '.join(previous['findings'])
    limitations=['Assessment is limited to current annual total NI and both same-report reported total-asset endpoints; no constant economic perimeter or counterfactual standalone adjustment is asserted.']
    if eid in resolutions:
        item=resolutions[eid]
        check(item['status']=='resolved' and item['scope']=='current_annual_net_income_same_report_assets','Resolution scope')
        check(item['field_scope_checks']=={f:True for f in FIELDS},'Resolution field scope')
        basis=' '.join(item['findings']);limitations+=item['limitations']
    else:
        # The original independently inspected report-specific narrative is
        # retained. Only two identified accounting questions require overrides.
        check(not previous['unresolved_questions'] or all('Publication lies outside' in q for q in previous['unresolved_questions']),'Unresolved accounting question lacks explicit resolution')
    return dict(status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis=basis,
      field_checks={f:True for f in FIELDS},current_prior_common_basis=True,limitations=limitations)


def verify_item(item,source,text):
    check(item['source']==source and item['status'] in ('resolved','unresolved'),'Item identity/status')
    check(item['scope']=='current_annual_net_income_same_report_assets','Item purpose')
    check(set(item['field_checks'])==set(FIELDS),'Item fields')
    check(item['basis'] and item['evidence'],'Missing supporting rationale')
    if item['status']=='resolved':check(all(item['field_checks'].values()) and item['current_prior_common_basis'] is True,'Resolved claim inconsistent')
    for ev in item['evidence']:
        check(ev['path']==source['path'] and ev['sha256']==source['sha256'],'Evidence source changed')
        loc=ev['locator'];check(loc['kind']=='document_text' and text[loc['start']:loc['end']]==ev['quote'] and ev['quote'],'Evidence location changed')


def build(plan_sha,extractions_sha,publish=False):
    check(plan_sha==PLAN_SHA and extractions_sha==EXTRACTIONS_SHA,'Exact authorized source run required')
    pins={};p=load(PLAN,pins,plan_sha);data=load(EXTRACTIONS,pins,extractions_sha)
    context=load(CONTEXT,pins,CONTEXT_SHA);old=load(FEASIBILITY,pins,FEASIBILITY_SHA);res=load(RESOLUTION,pins,RESOLUTION_SHA)
    check(data['plan_sha256']==plan_sha,'Extraction plan linkage')
    check(p['inputs'][CONTEXT]==CONTEXT_SHA,'Plan context pin')
    for group in (p['inputs'],p['code'],old['input_hashes'],res['input_hashes']):
        for path,digest in group.items():helper.read(path,pins,digest)
    contexts={e['event_id']:e for e in context['events']};prior={e['event_id']:e for e in old['event_reviews']};resolved={e['event_id']:e for e in res['events']}
    check([(e['symbol'],e['fiscal_year']) for e in data['events']]==EXPECTED,'Exact21 event population')
    check([r['event_id'] for r in p['records']]==[e['event_id'] for e in data['events']],'Plan event order')
    check(len(context['events'])==p['catalog_events']==28,'Full28 catalog')
    check(data['catalog_ids']==p['catalog_ids']==[e['event_id'] for e in context['events']],'Catalog IDs changed')
    check(data['catalog_metadata']==p['catalog_metadata'],'Catalog metadata changed')
    check(data['original_catalog_metadata']==p['original_catalog_metadata'],'Original catalog changed')
    results=[];nulls=0
    for e in data['events']:
        eid=e['event_id'];c=contexts[eid];previous=prior[eid];assert_scope(c,e)
        check(previous['source']==e['source'],'Earlier independent source review mismatch')
        raw=helper.read(e['source']['path'],pins,e['source']['sha256'])
        receipt=load(e['source']['receipt_path'],pins,e['source']['receipt_sha256'])
        check(receipt['sha256']==e['source']['sha256'] and receipt['url']==e['source']['url'] and receipt.get('status',receipt.get('http_status'))==200,'Source receipt')
        soup=BeautifulSoup(raw,'html.parser');text,positions=visible_document(soup)
        proof=metadata_evidence(c,soup,text,positions)
        # Revalidate every previously inspected redacted accounting span.
        earlier=[]
        for ev in previous['evidence']:
            loc=ev['locator'];check(ev['path']==e['source']['path'] and ev['sha256']==e['source']['sha256'],'Context evidence source')
            check(helper.redact(text[loc['start']:loc['end']])==ev['quote_numbers_redacted'],'Context quote changed')
            earlier.append(deepcopy(ev))
        if eid=='TMO_2019-01-30':proof.append(quote_literal('The results for 2017 have been restated for the immaterial impacts of adopting new guidance on pension accounting.',e['source'],text))
        if eid=='PYPL_2016-01-27':proof.append(quote_literal("To supplement the company's condensed consolidated financial statements presented in accordance with generally accepted accounting principles, or GAAP, the company uses non-GAAP measures of certain components of financial performance.",e['source'],text))
        decision=scoped_decision(previous,resolved)
        item=dict(event_id=eid,source=deepcopy(e['source']),**decision,evidence=proof,
          previously_reviewed_accounting_context_evidence=earlier,
          retained_context_holds=deepcopy(c['context_holds']),periods=deepcopy(e['periods']),
          available_from=e['available_from'],clock_verified_from_context=c['available_session_verified'],
          selected_values_verified=False,field_checks_are_accounting_scope_only=True,
          common_basis_definition='Same-publication reported GAAP consolidated/combined-and-consolidated NI and asset scope; not invariant legal perimeter or unchanged values relative to old publications.',
          financial_fields_present={f:e['fields'][f] is not None for f in FIELDS},
          fact_admitted=False,model_admitted=False,strategy_admitted=False)
        verify_item(item,e['source'],text);results.append(item);nulls+=sum(e['fields'][f] is None for f in FIELDS)
    helper.read(SELF,pins);helper.read(TEST,pins);helper.read('work/after153_batch5_accounting_resolution.py',pins)
    for path,digest in list(pins.items()):helper.read(path,pins,digest)
    result=dict(schema_version=1,stage='independent_field_specific_accounting_review',passed=True,
      created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,plan_sha256=plan_sha,extractions_sha256=extractions_sha,
      input_hashes=pins,events=results,counters=dict(events=len(results),resolved_for_stated_three_field_purpose=sum(e['status']=='resolved' for e in results),unresolved_for_stated_three_field_purpose=sum(e['status']=='unresolved' for e in results),source_nulls_preserved=nulls),
      scope='Current annual reported GAAP total NI and same-current-report total assets only. Monetary-cell correctness is a separate independent review.',
      limitations=['Two early TMO clocks remain held independently of accounting scope.','Seven unprocessed entries remain in the complete 28-event catalog and are not replaced by older valid events.','Annual Jan1 starts are explicit 365/366-day same-report endpoint assumptions, not source-quoted start dates.','Dollar notation is not ISO-currency certification; first-public version and complete security-master history remain uncertified.','Reported gains, losses, discontinued operations, acquisition effects, separation costs and historical allocations remain in GAAP totals.','The two scoped resolutions do not establish counterfactual standalone costs, invariant legal entities, pension classification-only effects or zero asset impact.'],
      catalog_events=28,unprocessed_catalog_events=7,numeric_fields_verified=0,
      financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        with (ROOT/OUT).open('x') as output:json.dump(result,output,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);output.write('\n')
        check(json.loads((ROOT/OUT).read_bytes())==result,'Exact published JSON')
    return result


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--plan-sha',required=True);cli.add_argument('--extractions-sha',required=True);cli.add_argument('--publish',action='store_true');a=cli.parse_args()
    result=build(a.plan_sha,a.extractions_sha,a.publish)
    print(json.dumps(dict(passed=result['passed'],counters=result['counters'],output=OUT if a.publish else None,sha256=hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest() if a.publish else None)))
if __name__=='__main__':main()
