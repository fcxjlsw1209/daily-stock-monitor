"""Fixed MU/PEP metadata census. Never parse or emit monetary cell values."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid, external_context
import after150_next_source_context as old
from after149_batch1_admission import first_session, normalize_sessions

ROOT=Path(__file__).resolve().parents[1]
OUT='work/after152_next_source_feasibility.json'
CENSUS,CONCEPT=old.CENSUS,old.CONCEPT
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SYMBOLS=('MU','PEP')
LABEL=re.compile(r'^(?:(?:consolidated\s+)?net (?:income|loss|earnings)(?:\s*\((?:loss|income)\))?(?:\s+attributable\s+to\b[^\d$]*)?|total assets)\s*(?:\([a-z0-9]+\))?\s*:?$',re.I)
PREFIX=re.compile(r'^(?:(?:consolidated\s+)?net (?:income|loss|earnings)(?:\s*\((?:loss|income)\))?(?:\s+attributable\s+to\b[^\d$]*)?|total assets)\b',re.I)
TITLE=re.compile(r'(?:condensed\s+)?consolidated\s+(?:(?:condensed|unaudited|summary)\s+)*(?:balance\s+sheets?|statements?\s+of\s+(?:income|operations|earnings|cash\s+flows?))',re.I)
BODY=re.compile(r'^(?:assets|current assets|cash(?: and cash equivalents)?|cash and short.term investments|net operating revenues|net revenues?|revenues?|net sales|sales|operating activities)\s*:?$',re.I)
HEADER=re.compile(r'^(?:\(?unaudited\)?|(?:fiscal\s+)?(?:years?|quarters?)(?:\s+ended)?|(?:three|six|nine|twelve|3|6|9|12)\s+months?(?:\s+ended)?|(?:52|53)\s+weeks?(?:\s+ended)?|(?:fiscal\s+)?(?:19|20)\d{2}|'+old.MONTH+r'(?:\s+\d{1,2})?\s*,?(?:\s+(?:19|20)\d{2})?|(?:\d{1,2},?\s+)?(?:19|20)\d{2}|(?:as of|for the|for|ended|and|at))$',re.I)
IDENTITY={'MU':re.compile(r'Micron Technology,?\s+Inc\.?|(?:NASDAQ|Nasdaq)\s*[:(]\s*MU\)?',re.I),
          'PEP':re.compile(r'PepsiCo,?\s+Inc\.?|(?:NYSE|NASDAQ|Nasdaq)\s*[:(]\s*PEP\)?',re.I)}


def norm(text):return old.norm(text)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def masked(text):return re.sub(r'\d[\d,.\s]*','[number]',norm(text))


def read_local(path,pins,expected=None):
    p=ROOT/path
    if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or 'CloudStorage' in p.parts:raise ValueError('Nonlocal input: '+path)
    raw=p.read_bytes();sha=digest(raw)
    if expected is not None and sha!=expected:raise ValueError('Hash mismatch: '+path)
    if path in pins and pins[path]!=sha:raise ValueError('Input changed: '+path)
    pins[path]=sha;return raw


def is_metadata(text):
    text=norm(text)
    return bool(old.UNIT.fullmatch(text) or HEADER.fullmatch(text) or
                (TITLE.search(text) and len(text)<260 and not re.search(r'[\d$%]',text)))


def cell_metadata(cell):
    return {key:cell[key] for key in ('row','physical_cell','columns','rowspan')}|{'text':norm(cell['text'])}


def html_info(raw):
    soup=BeautifulSoup(raw,'html.parser')
    for node in soup(['script','style']):node.decompose()
    text=norm(soup.get_text(' ',strip=True));tables=[]
    for ti,table in enumerate(soup.find_all('table')):
        if table.find('table'):continue
        try:rows,grid,physical=table_grid(table)
        except ValueError:continue
        targets=[c for c in physical.values() if LABEL.fullmatch(norm(c['text']))]
        if not targets:continue
        first=min(c['row'] for c in targets)
        starts=[c['row'] for c in physical.values() if c['row']<=first and BODY.fullmatch(norm(c['text']))]
        stop=min(starts) if starts else first
        tables.append(dict(table=ti,total_physical_rows=len(rows),header_stop_row=stop,
            headers=[cell_metadata(c) for c in physical.values() if c['row']<stop and is_metadata(c['text'])],
            external_metadata=[norm(s) for s in external_context(table) if is_metadata(s)],
            labels=[dict(row=c['row'],physical_cell=c['physical_cell'],columns=c['columns'],label=masked(c['text'])) for c in targets],
            dollar_symbol_present='$' in table.get_text()))
    pres=[]
    for pi,pre in enumerate(soup.find_all('pre')):
        lines=[]
        for li,line in enumerate(pre.get_text().splitlines()):
            if is_metadata(line):lines.append(dict(line_1based=li+1,kind='header',text=norm(line)))
            elif PREFIX.match(norm(line)):lines.append(dict(line_1based=li+1,kind='label_only',text=masked(PREFIX.match(norm(line)).group(0))))
        if lines:pres.append(dict(pre_block=pi,lines=lines))
    links=[dict(label=masked(a.get_text(' ',strip=True)),url=a['href']) for a in soup.find_all('a',href=True)
           if re.search(r'\.pdf(?:$|\?)',a['href'],re.I) or re.search(r'financial statements|financial tables',a.get_text(),re.I)]
    return dict(table_count=len(soup.find_all('table')),statement_candidates=tables,pre_candidates=pres,attachment_links=links),text


def inspect(row,pins):
    raw=read_local(row['evidence_path'],pins,row['evidence_sha256']);path=Path(row['evidence_path'])
    gaps=[];receipt_info=None;index_metadata=None
    if path.name=='INDEXED_PDF_OBSERVATIONS.json':
        matches=[x for x in json.loads(raw) if x.get('source_url')==row['source_url']]
        if len(matches)!=1 or matches[0].get('original_response_available') is not False:raise ValueError('Indexed event mismatch')
        item=matches[0];index_metadata={k:item.get(k) for k in ('evidence_kind','original_response_available','retrieved_at','observed_period_end_text','observed_publication_date_text')}
        info,text={},'';kind='index_observation_not_original_report';gaps.append('original_report_bytes_and_receipt_not_in_this_catalog_entry')
    elif path.name=='indexed_recent_results.json':
        data=json.loads(raw)
        if data.get('raw_original_bytes_available') is not False:raise ValueError('Unexpected index original status')
        rp=str(path)+'.meta.json';receipt=json.loads(read_local(rp,pins))
        if receipt.get('sha256')!=row['evidence_sha256']:raise ValueError('Index receipt mismatch')
        receipt_info=dict(path=rp,sha256=pins[rp],kind='index_retrieval_metadata_not_original_receipt',retrieved_at=receipt.get('retrieved_at'))
        index_metadata={k:data.get(k) for k in ('raw_original_bytes_available','evidence_class','captured_at')}
        info,text={},'';kind='index_observation_not_original_report';gaps.append('original_report_bytes_and_receipt_not_in_this_catalog_entry')
    else:
        rp=str(path.with_suffix('.meta.json' if row['symbol']=='MU' else '.json'))
        receipt=json.loads(read_local(rp,pins));status=receipt.get('status',receipt.get('http_status'))
        if receipt.get('sha256')!=row['evidence_sha256'] or receipt.get('url')!=row['source_url'] or status!=200:raise ValueError('Original receipt mismatch: '+row['event_id'])
        receipt_info=dict(path=rp,sha256=pins[rp],kind='original_response_receipt',status=status,retrieved_at=receipt.get('retrieved_at'),url_and_sha_match=True)
        if raw.startswith(b'%PDF'):
            kind='PDF_original_geometry_pending';info,text={},'';gaps.append('PDF_metadata_and_visual_geometry_not_reviewed')
        else:kind='HTML_original';info,text=html_info(raw)
    terms=sorted(set(m.group(0).lower() for m in old.ACCOUNTING.finditer(text)));snippets=[]
    for match in old.ACCOUNTING.finditer(text):
        quote=masked(text[max(0,match.start()-65):match.end()+175])
        if quote not in snippets:snippets.append(quote)
    identity=sorted(set(m.group(0) for m in IDENTITY[row['symbol']].finditer(text)))
    if not identity:gaps.append('report_local_identity_not_verified')
    return dict(row,source_kind=kind,original_report_bytes_verified=kind=='HTML_original',receipt=receipt_info,index_metadata=index_metadata,
                context=info,identity_anchors=identity,accounting_terms=terms,accounting_context_numbers_redacted=snippets,gaps=gaps,
                financial_values_extracted=0,fact_admitted=False,model_admitted=False,strategy_admitted=False)


def build():
    pins={};census=json.loads(read_local(CENSUS,pins,old.EXPECTED_CENSUS));concept=json.loads(read_local(CONCEPT,pins))
    order=concept['source_preparation_order']['remaining_text_lead_issuers']
    if order[order.index('MU'):order.index('MU')+2]!=list(SYMBOLS):raise ValueError('Fixed source-order mismatch')
    sessions=normalize_sessions(json.loads(read_local(SESSIONS,pins)))
    rows=[r for r in census['annual_documents'] if r['symbol'] in SYMBOLS]
    if len(rows)!=len({r['event_id'] for r in rows}) or Counter(r['symbol'] for r in rows)!={'MU':7,'PEP':16}:raise ValueError('Expected complete MU7/PEP16 frozen census')
    docs=[]
    for row in rows:
        doc=inspect(row,pins);session=first_session(row['publication_date'],sessions)
        doc.update(pinned_next_session=session,available_session_verified=session is not None and session==row['available_session']);docs.append(doc)
    for p in ('work/after152_next_source_feasibility.py','work/after150_next_source_context.py','work/after141_amd_source_extract.py','work/after149_batch1_admission.py'):read_local(p,pins)
    return dict(schema_version=1,stage='bounded_source_metadata_feasibility_not_admission',created_utc=datetime.now(timezone.utc).isoformat(),
        source_scope=list(SYMBOLS),documents=docs,input_hashes=pins,counts=dict(catalog_events=len(docs),by_symbol=dict(Counter(d['symbol'] for d in docs)),
        source_kinds=dict(Counter(d['source_kind'] for d in docs)),clock_verified=sum(d['available_session_verified'] for d in docs)),
        limitations=['Full frozen MU7/PEP16 catalog retained; absence of earlier MU events is not proof of complete issuer history.',
          'Index observations and linked PDFs are not original response bytes. No attachment download or cross-tree source search performed.',
          'No monetary cell parsing or financial calculations. Titles and dollar symbols do not certify GAAP scope or ISO currency.',
          'Parent-attributable NI is not substituted for consolidated NI; current publication governs all comparative columns.',
          'First-public version, correction history, continuous security identity and accounting comparability remain unverified.'],
        financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=build()
    if args.write:
        with (ROOT/OUT).open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(counts=result['counts'],output=OUT if args.write else None,
        documents=[dict(event_id=d['event_id'],kind=d['source_kind'],tables=len(d['context'].get('statement_candidates',[])),pre=len(d['context'].get('pre_candidates',[]))) for d in result['documents']]),sort_keys=True))


if __name__=='__main__':main()
