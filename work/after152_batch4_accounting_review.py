"""Explicit source-bound accounting decisions; no ratios, models or returns."""
import argparse
from datetime import datetime,timezone
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import after152_batch4_independent_review as audit
PLAN_SHA='9b1016e51ad22eae4faf13629afeae7e06a8f7efe3b2f0e8d7e77dce0d232643'
EXTRACTIONS_SHA='6855162969dd271aab13a8e62608684cbf9d285703fe8897913dfa3360afbe8c'
ROOT=Path(__file__).resolve().parents[1]
SELF='work/after152_batch4_accounting_review.py'

def inputs():
    r=audit.Reader();p=r.load(audit.PLAN,PLAN_SHA);d=r.load(audit.EXTRACTIONS,EXTRACTIONS_SHA);c=r.load(audit.CONTEXT,p['inputs'][audit.CONTEXT]);audit.require(d['plan_sha256']==PLAN_SHA,'Plan linkage')
    contexts={x['event_id']:x for x in c['events']};events={x['event_id']:x for x in d['events']};items=[]
    for row in p['records']:
        raw=r.read(row['source_path'],row['source_sha256']);s=BeautifulSoup(raw,'html.parser')
        for n in s(['script','style']):n.decompose()
        items.append((contexts[row['event_id']],events[row['event_id']],s,audit.norm(s.get_text(' ',strip=True))))
    return r,items

def inspect(only=None,details=False):
    r,items=inputs()
    for c,e,s,t in items:
        if only and only!=e['event_id']:continue
        rx=re.compile(r'generally accepted|accounting principles|reclassif|restat|accounting change|retrospect|comparability|deconsolid|fiscal year.{0,80}weeks|53.{0,20}week|52.{0,20}week|derived from|audited|recast|revis(?:ed|ion)',re.I)
        spans=[]
        for x in rx.finditer(t):
            lo,hi=max(0,x.start()-150),min(len(t),x.end()+500)
            if spans and lo<=spans[-1][1]:spans[-1]=(spans[-1][0],max(hi,spans[-1][1]))
            else:spans.append((lo,hi))
        tables=[]
        for st in c['statement_contexts']:
            if st['table'] not in {x['table'] for x in c['fields'].values()}:continue
            tab=s.find_all('table')[st['table']];cells,_=audit.grid_from_html(tab)
            tables.append(dict(table=st['table'],headers=[dict(row=x['row'],text=x['text']) for x in cells.values() if x['row']<st['header_stop_row']],body=audit.norm(tab.get_text(' ',strip=True)) if details else None))
        values={}
        for f in audit.FIELDS:
            try:
                value=audit.source_field(s,c,f);result=audit.verify_published_cell(e['fields'][f],value,f,e,e['periods']['current']);values[f]=dict(verified=result['verified'],source_reading=value)
            except Exception as exc:values[f]=dict(error_type=type(exc).__name__,error=str(exc))
        print(json.dumps(dict(event_id=e['event_id'],context_notes=c['accounting_context_numbers_redacted'],tables=tables,values=values,quotes=[dict(start=lo,end=hi,quote=t[lo:hi]) for lo,hi in spans]),ensure_ascii=False))

# Filled only after the explicit source inspection above; no keyword-based clearance.
def decision(basis,quotes=(),resolved=True):
    return dict(status='resolved' if resolved else 'unresolved',reasons=[] if resolved else ['Inventory accounting method change is disclosed without sufficient opening-balance or retrospective-treatment detail to verify a common basis.'],basis=basis,quotes=list(quotes),field_checks={f:resolved for f in audit.FIELDS},current_prior_common_basis=resolved)

DECISIONS={
 'MU_2024-09-25':decision('The consolidated operations statement explicitly separates quarter columns from the Year Ended columns. Current annual net income is the total after tax and equity-method earnings, before any per-share calculation. The balance sheet presents current August, interim May and prior August dates; only both August annual endpoints are used. The inventory NRV write-down note describes charges within reported GAAP cost and inventory, not an alternative accounting basis or a restated adjusted total. Both same-report total assets remain consolidated reported amounts; no non-GAAP exclusion is made.', ['GAAP represents U.S. Generally Accepted Accounting Principles.']),
 'MU_2025-09-23':decision('Current full-year consolidated net income and both August consolidated balance-sheet totals belong to the same reported financial statements. The middle May asset column is not the prior annual endpoint. The separate non-GAAP presentation is expressly an exclusion-based supplement and supplies none of the selected fields. The printed consolidated rows, common million-dollar scale and annual date groups establish the stated current-report basis.', ['GAAP represents U.S. Generally Accepted Accounting Principles.']),
 'PEP_2011-02-10':decision('The selected Year Ended Net income is the consolidated total before the separate noncontrolling-interest deduction and Net Income Attributable to PepsiCo row. The two annual total-assets figures are from the same condensed consolidated balance sheet. Bottler acquisition, integration and fair-value effects remain in the reported totals; no core, organic or constant-currency adjustment is substituted. The prior annual income column is labelled as the audited comparative; this is a current-report presentation assessment, not a claim of constant economic business perimeter.'),
 'PEP_2012-02-09':decision('The values and annual consolidated identities are readable, but the release specifically attributes part of QFNA performance to an inventory accounting change in the first quarter. It does not identify the old and new methods or establish whether the opening annual assets were restated to the new method. Consequently current-year NI and average-assets accounting comparability cannot be certified from this release alone. This unresolved purpose assessment does not deny source correctness and does not replace the separate early-calendar hold.', ['an inventory accounting change in the first quarter which contributed 2 percentage points.'],False),
 'PEP_2013-02-14':decision('The annual consolidated Net income row precedes noncontrolling interests and parent-only net income. Both annual total-assets endpoints occur in the same condensed consolidated balance sheet. The tax-court benefit and deferred-tax effects discussed in the release remain in the reported annual income rather than being replaced by core EPS or another adjusted measure. The selected data use the same report and consolidated basis, without requiring the unused prior-year income to enter a new calculation.'),
 'PEP_2014-02-13':decision('The same release supplies the annual consolidated income statement and both annual asset endpoints. Net income is the total before attribution to noncontrolling interests and PepsiCo shareholders. The prior annual income column is explicitly marked as the audited comparative. Reported restructuring and impairment effects are retained; the separate core results are not used. No adjustment to total assets or replacement with a segment or parent-only balance is made.'),
 'PEP_2015-02-11':decision('The Year Ended consolidated Net income row and current/prior total assets are taken from the same reported condensed consolidated statements. The selected income includes noncontrolling interests, matching the consolidated asset scope. Comparative asset subcategories are accepted exactly as presented in this current report; no older report is spliced in. Non-GAAP core and constant-currency results are separate from these totals.'),
 'PEP_2016-02-11':decision('The annual consolidated Net income includes the reported Venezuela impairment charge and translation-loss reclassification. The release explicitly explains loss of control, deconsolidation at the end of the third quarter, and subsequent cost-method investments. That reported change in business perimeter is retained in current NI and consolidated total assets; this purpose does not require a constant set of subsidiaries through the year. Both asset endpoints use this same report, with no substitution of organic/core results or a continuing-business pro forma balance.', ['we deconsolidated our wholly-owned Venezuelan subsidiaries effective as of the end of the third quarter of 2015.']),
 'PEP_2017-02-15':decision('The income-statement footnote explicitly identifies the prior-year cost reclassification and states that it has no impact on consolidated net income. The two selected assets are both reported on the same consolidated balance sheet at the two annual dates; the footnote is not treated as proof about assets by itself. The 2015 Venezuela deconsolidation and impairment remain in the printed prior results and assets; no constant-perimeter or core adjustment is applied. The selected current NI is the consolidated total before noncontrolling-interest attribution. The 53-week year is retained under the declared annual-boundary assumption.', ["These reclassifications had no impact on our consolidated net revenue, operating profit, net interest expense, provision for income taxes, net income or earnings per share.","PepsiCo's 2016 fiscal year includes 53 weeks of results."])
}


def quote_span(text,quote,source):
    start=text.find(quote);audit.require(start>=0 and quote,'Exact source quote absent')
    return dict(path=audit.relative(source['path']),sha256=source['sha256'],locator=dict(kind='document_text',start=start,end=start+len(quote)),quote=quote)

def build(publish=False):
    r,items=inputs();audit.require(set(DECISIONS)=={e['event_id'] for c,e,s,t in items},'Explicit decision population mismatch')
    p=r.load(audit.PLAN,PLAN_SHA)
    for path,h in {**p['inputs'],**p['code']}.items():r.read(path,h)
    for path in [SELF,*audit.source_files()]:r.read(path)
    results=[]
    for c,e,s,t in items:
        d=DECISIONS[e['event_id']];proof=[]
        for ti in sorted({x['table'] for x in c['fields'].values()}):proof.append(quote_span(t,audit.norm(s.find_all('table')[ti].get_text(' ',strip=True)),e['source']))
        for q in d['quotes']:proof.append(quote_span(t,q,e['source']))
        item=dict(event_id=e['event_id'],source=e['source'],status=d['status'],scope='current_annual_net_income_same_report_assets',reasons=d['reasons'],basis=d['basis'],field_checks=d['field_checks'],current_prior_common_basis=d['current_prior_common_basis'],evidence=proof)
        audit.verify_accounting(item,e,r.read(e['source']['path'],e['source']['sha256']));results.append(item)
    for path,h in list(r.hashes.items()):r.read(path,h)
    result=dict(schema_version=1,stage='independent_field_specific_accounting_review',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,extractions_sha256=EXTRACTIONS_SHA,input_hashes=r.hashes,events=results,counters=dict(events=len(results),resolved_for_stated_three_field_purpose=sum(e['status']=='resolved' for e in results),unresolved_for_stated_three_field_purpose=sum(e['status']=='unresolved' for e in results),source_nulls_preserved=sum(v is None for c,e,s,t in items for v in e['fields'].values())),scope='Current annual consolidated net income and same-report current/prior consolidated total assets only.',limitations=['Presentation assessment is limited to the pinned releases and does not certify all later corrections.','Four early publication clocks remain outside the pinned calendar.','Annual starts are explicit 52/53-week assumptions, not source-quoted beginning dates.','Dollar symbols do not certify ISO currency; first-public timing and security-master history remain uncertified.','No parent-only, adjusted, per-share or segment measure replaces the consolidated reported totals.'],financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        with (ROOT/audit.ACCOUNTING).open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,counters=result['counters'],accounting_sha256=audit.sha((ROOT/audit.ACCOUNTING).read_bytes()) if publish else None)))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--inspect',action='store_true');g.add_argument('--publish',action='store_true');p.add_argument('--event');p.add_argument('--details',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.event,a.details)
    else:build(a.publish)
