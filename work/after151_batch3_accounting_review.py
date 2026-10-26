"""Bound raw-text accounting review support; no extraction, ratios or strategy."""
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import after151_batch3_independent_review as audit
PLAN_SHA='6dd69ae328c0ddf0c995eda11962716b53ec85caff10e9493454e9adbd01c352'
EXTRACTIONS_SHA='f023c218e88679b45bfb28b1ebc4c213220fe905b2f02e5fb0c730c51857abef'
ROOT=Path(__file__).resolve().parents[1]
SELF='work/after151_batch3_accounting_review.py'

def inputs():
    r=audit.Reader();plan=r.load(audit.PLAN,PLAN_SHA);data=r.load(audit.EXTRACTIONS,EXTRACTIONS_SHA);context=r.load(audit.CONTEXT,plan['inputs'][audit.CONTEXT])
    events={e['event_id']:e for e in data['events']};docs={e['event_id']:e for e in context['events']};items=[]
    for row in plan['records']:
        c=docs[row['event_id']];raw=r.read(row['source_path'],row['source_sha256']);soup=BeautifulSoup(raw,'html.parser')
        for n in soup(['script','style']):n.decompose()
        text=audit.norm(soup.get_text(' ',strip=True));items.append((c,events[c['event_id']],soup,text))
    return r,items

def inspect(details=False):
    r,items=inputs()
    pattern=re.compile(r'accounting methodology|retrospectively|reclassified historical|classified as held for sale|fiscal year consisted|fiscal year[^.]{0,100}weeks|change in[^.]{0,100}accounting|reclassifications|not restated|different[^.]{0,100}basis|revised[^.]{0,100}financial',re.I)
    if details:pattern=re.compile(r'generally accepted accounting|accounting principles generally|derived from|audited financial|reclassif|restat|52.week|53.week|fifty.two|fifty.three|weeks|accounting change',re.I)
    result=[]
    for c,e,soup,text in items:
        spans=[]
        for match in pattern.finditer(text):
            lo=max(0,match.start()-100);hi=min(len(text),match.end()+350)
            if spans and lo<spans[-1][1]:spans[-1]=(spans[-1][0],max(hi,spans[-1][1]))
            else:spans.append((lo,hi))
        headers=[]
        for st in c['statement_contexts']:
            if st['table'] not in {x['table'] for x in c['fields'].values()}:continue
            table=soup.find_all('table')[st['table']];cells,_=audit.grid_from_html(table)
            headers.append(dict(table=st['table'],text=' | '.join(x['text'] for x in cells.values() if x['row']<st['header_stop_row']),footer=audit.norm(table.get_text(' ',strip=True))[-900:] if details else ''))
        result.append(dict(event_id=c['event_id'],context_notes=c['accounting_context_numbers_redacted'] if details else [],headers=headers,quotes=[dict(start=lo,end=hi,quote=text[lo:hi]) for lo,hi in spans]))
    print(json.dumps(result,ensure_ascii=False,indent=2))

# Decisions authored after reading each raw statement, footnotes and retained
# context. These explicit dispositions are not generated from keyword absence.
DECISIONS={
 'KO_2012-02-07':'The release identifies reported financial statements as U.S. GAAP. The selected full-year consolidated net-income row precedes the noncontrolling-interest deduction, and both assets endpoints belong to the same consolidated balance sheet. Non-GAAP comparable measures occur in separately titled reconciliations. Their generic comparability caveats do not reclassify the selected totals.',
 'KO_2013-02-12':'The annual income statement and balance-sheet footnote 1 explicitly apply the pension market-related asset-value accounting change retrospectively to all prior-period information. Thus As Adjusted 1 identifies GAAP retrospective presentation, not a non-GAAP measure. The three original nulls remain null: the source-only parser rejected this header conservatively. Held-for-sale assets remain included in consolidated total assets. The share split note affects share/per-share data, not these monetary totals.',
 'KO_2014-02-18':'The source describes reclassified historical operating-segment data after organizational changes; the selected measures are consolidated totals, not segment measures. The Philippine held-for-sale classification and disposal are disclosed within reported GAAP results. Both balance-sheet dates and annual consolidated net income are presented together; no selected total is substituted from an adjusted reconciliation or another report.',
 'KO_2015-02-10':'The selected annual consolidated net income is before the noncontrolling-interest deduction, with same-report consolidated assets for both year ends. Refranchising impairment and held-for-sale charges are part of the reported GAAP amounts; they are not removed. Hedge adjustments recognized in earnings are likewise retained. Generic non-GAAP comparability adjustments do not alter the selected reported basis.',
 'KO_2016-02-09':'The formal annual reported GAAP income and balance sheets present full consolidated totals at both year ends. Held-for-sale items are included within assets; no continuing-operations or parent-only numerator is used. Separate reconciliation measures and general accounting-risk language are outside the selected three-field scope.',
 'KO_2017-02-09':'The release explicitly distinguishes the consolidated financial statements reported on a GAAP basis from supplementary non-GAAP measures. The selected annual consolidated net income and both total-assets dates use that current-report basis. Held-for-sale presentation remains part of total assets. This review does not assert equality with prior releases or certify all later corrections; only same-report presentation is assessed.',
 'LRCX_2013-07-31':'The condensed consolidated operations statement contains separately grouped three-month and twelve-month columns; only the twelve-month current-year net-income column is used. The same release balance sheet contains current June, March and prior June dates; the two June endpoints are selected. The prior annual column is marked derived from audited financial statements. Separate non-GAAP summaries are excluded. Consolidation/acquisition economics remain in reported totals without adjustment.',
 'LRCX_2014-07-30':'The formal consolidated operations statement supplies current twelve-month net income, and the balance sheet supplies the current/prior June total assets; March is not the prior annual endpoint. Prior annual information is identified as derived from audited financial statements. Reported gain on real-estate sale remains in consolidated net income, while separate non-GAAP summaries do not supply any selected value.',
 'LRCX_2015-07-29':'The selected twelve-month net income and two June balance-sheet endpoints are formal consolidated reported measures; the March interim column is excluded. The prior annual footnote identifies audited-source information. Goodwill impairment remains within reported net income and reported assets, rather than being removed using the separately presented non-GAAP summary.',
 'LRCX_2016-07-27':'The formal operations and balance-sheet tables jointly present the current twelve-month net income and current/prior June consolidated assets. Prior annual information is derived from audited financial statements; March and the separate non-GAAP summary are excluded. Financing transactions change the reported amounts economically without creating a different selected accounting basis.'
}

def quote_span(text,quote,source,start=0):
    offset=text.find(quote,start);audit.require(offset>=0,'Exact source quote absent')
    return dict(path=audit.relative(source['path']),sha256=source['sha256'],locator=dict(kind='document_text',start=offset,end=offset+len(quote)),quote=quote)

def build(publish=False):
    r,items=inputs();audit.require(set(DECISIONS)=={e['event_id'] for _,e,_,_ in items},'Explicit decision population mismatch')
    plan=r.load(audit.PLAN,PLAN_SHA)
    for p,d in plan['inputs'].items():r.read(p,d)
    for p in [SELF,*audit.source_files()]:r.read(p)
    results=[]
    for c,e,soup,text in items:
        evidence=[];tables=soup.find_all('table')
        for ti in sorted({v['table'] for v in c['fields'].values()}):
            q=audit.norm(tables[ti].get_text(' ',strip=True));evidence.append(quote_span(text,q,e['source']))
        if e['symbol']=='KO':
            match=re.search(r'The Company reports its financial results in accordance with accounting principles generally accepted in the United States[^.]*\.',text)
            audit.require(match is not None,'Reviewed GAAP declaration missing');evidence.append(quote_span(text,match.group(),e['source']))
        else:
            match=re.search(r'\(1\) Derived from audited financial statements\.?',text)
            audit.require(match is not None,'Reviewed annual audited-source footnote missing');evidence.append(quote_span(text,match.group(),e['source']))
        if e['event_id']=='KO_2014-02-18':
            match=re.search(r'The reclassified historical operating segment data[^.]*\.',text);audit.require(match is not None,'Reviewed segment note missing');evidence.append(quote_span(text,match.group(),e['source']))
        item=dict(event_id=e['event_id'],source=e['source'],status='resolved',scope='current_annual_net_income_same_report_assets',reasons=[],basis=DECISIONS[e['event_id']],field_checks={f:True for f in audit.FIELDS},current_prior_common_basis=True,gaap_retrospective_adjustment_verified=e['event_id']=='KO_2013-02-12',evidence=evidence)
        audit.verify_accounting(item,e,r.read(e['source']['path'],e['source']['sha256']));results.append(item)
    for p,d in list(r.hashes.items()):r.read(p,d)
    result=dict(schema_version=1,stage='independent_field_specific_accounting_review',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,extractions_sha256=EXTRACTIONS_SHA,input_hashes=r.hashes,events=results,counters=dict(events=10,resolved_for_stated_three_field_purpose=10,source_nulls_preserved=3),scope='Current annual consolidated net income and same-report current/prior consolidated total assets only.',limitations=['Assessment is limited to the pinned releases and their printed presentation, not an audit of the issuers or all later restatements.','Resolved accounting presentation does not admit a missing value, certify first publication or override an unverified trading calendar.','Annual-start derivation remains an explicit source-use assumption; no beginning date is certified as directly quoted.','No parent-only, per-share, segment or non-GAAP adjustment is substituted into the selected measures.'],financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        with (ROOT/audit.ACCOUNTING).open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,events=len(results),accounting_sha256=audit.sha((ROOT/audit.ACCOUNTING).read_bytes()) if publish else None)))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--inspect',action='store_true');g.add_argument('--publish',action='store_true');p.add_argument('--details',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.details)
    else:build(a.publish)
