"""Prospective, quantity-free inventory of basic weighted-average share locators."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after158_basic_shares_inventory.py'
TEST='work/test_after158_basic_shares_inventory.py'
PLAN='work/after158_basic_shares_source_plan.json'
OUT='outputs/after158_basic_shares_inventory_v1'
UNION='outputs/monthly_cashflow_source_union_v1'
PINS={UNION+'/RAW_FACTS.json':'fb26792299d6e6b75ab8ae321e1bcca65fc978876ddf4f1e0a0e983b25c41c06',UNION+'/PLAN.json':'61abd53de8f2c99c2e979d77ff7712ddd9f3536d8fdedb370949240eb5ee75d3',UNION+'/MANIFEST.json':'40915dc0e562da20163864d9da2a46e6ac47bcc30f30ffb7561ad7a7b86b56b2'}
ANNUAL=re.compile(r'\b(?:years?\s+ended|(?:twelve|12)\s+months?\s+ended|fiscal\s+years?|(?:52|53|fifty.two|fifty.three)\s+weeks?\s+ended)\b',re.I)
INTERIM=re.compile(r'\b(?:(?:three|six|nine|3|6|9)\s+months?\s+ended|quarters?\s+ended)\b',re.I)
WEIGHTED=re.compile(r'weighted[\s\-]*average',re.I)
SHARES=re.compile(r'\bshares?\b',re.I)
BASIC=re.compile(r'\bbasic\b',re.I)
DILUTED=re.compile(r'\bdilut\w*\b',re.I)
TITLE=re.compile(r'(?:consolidated\s+)?(?:statements?\s+of\s+(?:income|operations|earnings)|income\s+statements?)',re.I)
BAD=re.compile(r'non[\s\-]*gaap|pro\s*forma|reconciliation|as\s+adjusted',re.I)
YEAR=re.compile(r'(?<![\d,.])(?:19|20)\d{2}(?![\d,.])')
MONTH=re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b',re.I)
FLAGS={'split_context':r'\b(?:stock[\s-]+split|share[\s-]+split|split[\s-]+adjust\w*|reverse[\s-]+split)\b','retrospective_context':r'\b(?:retrospect\w*|retroactiv\w*|recast\w*|restat\w*)\b','accounting_change_context':r'\b(?:reclassif\w*|accounting\s+standards?|accounting\s+polic\w*|adopt\w*)\b'}


def norm(text):return re.sub(r'\s+',' ',text).strip()
def redact(text):return norm(re.sub(r'\d+(?:[,.]\d+)*','[number]',text))
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def encoded(obj):return (json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):return json.loads((ROOT/path).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write_once(path,obj):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb')as f:f.write(encoded(obj))
def verify(pins):
    for name,h in pins.items():require(sha(ROOT/name)==h,'Hash mismatch: '+name)
def code_pins():
    result={p:sha(ROOT/p)for p in(SELF,TEST,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb')}
    import bs4,soupsieve
    for module in(bs4,soupsieve):
        for p in sorted(Path(module.__file__).parent.rglob('*.py')):
            result[str(p.resolve())]=sha(p)
    return result


def rules():
    return dict(field='Current and same-report comparative annual BASIC weighted-average shares',
        interpretation='Smoothed ownership-base proxy only; not actual net issuance, not fiscal-year-end split-adjusted shares and not the published net-issuance factor.',
        scope='Exactly the 44 original union events in original order; no source substitution, new issuer or outcome-based selection.',
        stages='Write this plan and hashes before reading any original financial document or derived PDF text for this field; inventory requires its exact digest.',
        allowed='Presence booleans, source/DOM/line locators, annual/date headers, units and split/accounting word flags only.',
        prohibited='No share quantities, numeric row tails, ratios, signs, factors, ranks, orders, prices, returns or model/strategy registration.',
        semantics='Explicit weighted-average basic-share labels are distinct from basic EPS-denominator labels with implicit weighting. Diluted counts, basic EPS values, end-balance outstanding shares and quarterly-only values never satisfy the field.',
        html='Parse only owned table rows with bounded 100-column spans; associate Basic with nearest preceding share-denominator heading (six rows), not an EPS-value heading. Report every candidate, not a convenient one.',
        annual='Annual current/prior year columns must be present in the same report. For mixed interim/annual tables require per-column annual evidence and no interim evidence. Global textual annual evidence alone cannot disambiguate mixed columns.',
        pdf='Use existing source-adjacent .pages.txt only, bind exact text hash before scanning. No new quantity extraction or OCR. Text-only mixed-period candidates remain ambiguous pending geometric header review.',
        limits=dict(max_source_bytes=33554432,max_tables=2000,max_rows_per_table=4000,max_span=100,max_context_preceding_rows=6),
        flags='Bounded lexical split/retrospective/accounting flags are review leads, not proof of comparability or absence of other changes.',
        prior_holds='Copy old identity/provenance status and all inherited rejection codes separately. Old CFO/NI/assets completeness does not gate this independent field. No old or new admission is promoted.')


def freeze_plan():
    verify(PINS);data=read(UNION+'/RAW_FACTS.json');events=data['events'];require(len(events)==44,'44 events required')
    require(len({e['event_id']for e in events})==44,'Duplicate event')
    source_pins=dict(PINS);targets=[]
    for index,e in enumerate(events):
        source=deepcopy(e['source'])
        for p,h in((source['path'],source['sha256']),(source['receipt']['path'],source['receipt']['sha256'])):
            require(p not in source_pins or source_pins[p]==h,'Conflicting source hashes');source_pins[p]=h
        item={k:deepcopy(e[k])for k in('event_id','symbol','fiscal_year','release_date','available_from','identity','periods','inherited_rejection_codes','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master','fact_admitted','prior_field_admitted','model_admitted','strategy_admitted')}
        item.update(original_index=index,source=source,format=e['union_provenance']['producer'],original_event_sha256=digest(encoded(e)))
        require(item['format']in('pdf','html'),'Unexpected format')
        item['existing_text']=None
        if item['format']=='pdf':
            p=Path(source['path']).with_suffix('.pages.txt')
            if(ROOT/p).is_file():
                source_pins[str(p)]=sha(ROOT/p);item['existing_text']={'path':str(p),'sha256':source_pins[str(p)],'source_raw_sha256':source['sha256'],'binding_scope':'Existing source-adjacent derived text; hash/path association retained, no fresh geometric certification.'}
        targets.append(item)
    require(Counter(e['symbol']for e in targets)=={'AAPL':10,'AMD':10,'MSFT':12,'NVDA':12},'Unexpected population')
    verify(source_pins)
    plan=dict(schema='after158_basic_shares_source_plan_v1',created_utc=datetime.now(timezone.utc).isoformat(),rules=rules(),events=targets,source_hashes=source_pins,implementation_hashes=code_pins(),financial_source_contents_scanned_for_new_field=False,source_admission_changed=False,activity={'new_features':0,'returns':0,'orders':0,'share_quantities_extracted':0})
    write_once(PLAN,plan)
    print(json.dumps({'plan':PLAN,'sha256':sha(ROOT/PLAN),'events':len(targets),'source_pins':len(source_pins)}))


def label_type(label,previous):
    """Reject EPS values and diluted counts before considering a share heading."""
    label=norm(label)
    if not BASIC.search(label)or DILUTED.search(label):return None
    direct=SHARES.search(label)and WEIGHTED.search(label)
    bare=bool(re.fullmatch(r'basic(?:\s+(?:shares?|weighted[\s-]*average|average|outstanding))*\s*[:*†‡()]*',label,re.I))
    if not(direct or bare or re.fullmatch(r'(?:weighted[\s-]*average\s+)?(?:basic\s+shares?|shares?\s+basic)(?:\s+outstanding)?\s*[:*()]*',label,re.I)):return None
    if direct:return 'explicit_basic_weighted_average_shares'
    for text in reversed(previous[-6:]):
        if SHARES.search(text)and(re.search(r'\b(?:used|comput\w*|calcula\w*|outstanding|weighted|average)\b',text,re.I)):
            return 'explicit_basic_weighted_average_shares'if WEIGHTED.search(text)else'basic_eps_denominator_weighting_implicit'
        if re.search(r'(?:earnings|income|loss)\s+per\s+(?:common\s+)?share',text,re.I):return None
    return None


def word_flags(text):
    return {name:sorted({m.group(0).lower()for m in re.finditer(pattern,text,re.I)})for name,pattern in FLAGS.items()}

def unit_evidence(parts):
    result=[]
    for text in parts:
        if re.search(r'\b(?:thousands|millions|billions)\b|\bshares?\s+in\b',text,re.I):
            result.append(redact(text)[:500])
    return sorted(set(result))


def table_grid(table):
    rows=[r for r in table.find_all('tr')if r.find_parent('table')is table]
    require(len(rows)<=4000,'table_row_bound');grid={};physical=[]
    for ri,row in enumerate(rows):
        line=[];col=0
        for pi,cell in enumerate(row.find_all(['td','th'],recursive=False)):
            require(cell.find('table')is None,'nested_table_container')
            while(ri,col)in grid:col+=1
            rs=int(cell.get('rowspan',1));cs=int(cell.get('colspan',1));require(1<=rs<=100 and 1<=cs<=100 and col+cs<=100,'span_bound')
            item={'text':norm(cell.get_text(' ',strip=True)),'row':ri,'physical':pi,'columns':list(range(col,col+cs))}
            for rr in range(ri,ri+rs):
                for cc in item['columns']:
                    require((rr,cc)not in grid,'overlapping_span');grid[rr,cc]=item
            line.append(item);col+=cs
        physical.append(line)
    return rows,grid,physical


def external_context(table):
    result=[]
    for step,node in enumerate(table.previous_elements):
        if step>=150:break
        if not getattr(node,'name',None):continue
        if node.name=='table'or node.find_parent('table')is not None:break
        if node.name not in('p','div','h1','h2','h3','h4','strong','b')or node.find('table'):continue
        t=norm(node.get_text(' ',strip=True))
        if 0<len(t)<=300 and t not in result:result.append(t)
        if len(result)>=8:break
    return list(reversed(result))


def header_like(text,years):
    return bool(ANNUAL.search(text)or INTERIM.search(text)or MONTH.search(text)or(re.search(r'\d',text)and not re.search('[A-Za-z]',text)and set(YEAR.findall(text)).intersection(years)))


def html_inventory(raw,event):
    soup=BeautifulSoup(raw,'html.parser')
    for node in soup(['script','style']):node.decompose()
    tables=soup.find_all('table');require(len(tables)<=2000,'table_count_bound');candidates=[];structural=[]
    years={str(event['fiscal_year']),str(event['fiscal_year']-1)}
    wanted={s:event['periods'][s]['period_end'][:4]for s in('current','comparison')}
    for ti,table in enumerate(tables):
        try:_,grid,rows=table_grid(table)
        except(ValueError,TypeError):structural.append({'table_index':ti,'reason':'nested_or_unsupported_table_structure'});continue
        external=external_context(table);labels=[];block_start=0
        for ri,cells in enumerate(rows):
            texts=[c['text']for c in cells];alpha=[t for t in texts if re.search('[A-Za-z]',t)]
            label=norm(' '.join(alpha));rowtext=norm(' '.join(texts))
            if TITLE.search(label):block_start=ri
            kind=label_type(label,labels)
            if not kind:labels.append(label);continue
            heads=[c['text']for rr in rows[block_start:ri]for c in rr if header_like(c['text'],years)]
            context=external+[c['text']for rr in rows[block_start:min(block_start+8,ri)]for c in rr]
            annual=bool(ANNUAL.search(' '.join(heads)));interim=bool(INTERIM.search(' '.join(heads)))
            mapped={s:[]for s in wanted}
            for c in cells:
                if not re.search(r'\d',c['text'])or re.search('[A-Za-z]',c['text']):continue
                for col in c['columns']:
                    pieces=[];seen=set()
                    for rr in range(block_start,ri):
                        h=grid.get((rr,col))
                        if h and(h['row'],h['physical'])not in seen and header_like(h['text'],years):pieces.append(h['text']);seen.add((h['row'],h['physical']))
                    joined=' '.join(pieces);column_years=set(YEAR.findall(joined))
                    colannual=bool(ANNUAL.search(joined));colinterim=bool(INTERIM.search(joined))
                    for side,year in wanted.items():
                        if column_years=={year}and colannual and not colinterim:
                            mapped[side].append({'logical_column':col,'physical_cell':c['physical'],'header_literals_redacted':[redact(p)for p in pieces],'period_end_from_frozen_metadata':event['periods'][side]['period_end'],'year_header_matches':True})
            unique={s:sorted({x['physical_cell']for x in locs})for s,locs in mapped.items()}
            pair=all(len(unique[s])==1 for s in unique)and unique['current']!=unique['comparison']
            blockers=[]
            if not pair:blockers.append('current_and_prior_annual_columns_not_uniquely_located')
            if not annual:blockers.append('annual_header_not_found')
            if kind.endswith('implicit'):blockers.append('weighted_average_semantics_implicit_needs_review')
            if BAD.search(' '.join(context)):blockers.append('adjusted_or_non_gaap_context')
            units=unit_evidence(context+labels[max(0,len(labels)-6):]+[label])
            if not units:blockers.append('share_unit_scale_not_located')
            candidates.append(dict(format='html',table_index=ti,row_index=ri,label=redact(label),semantic_kind=kind,ancestor_labels_redacted=[redact(t)for t in labels[-6:]if t],annual_header_present=annual,interim_header_present=interim,header_literals_redacted=sorted(set(redact(h)for h in heads)),units=units,side_locators=mapped,same_report_annual_pair_locator_present=pair,blockers=blockers,quantities_retained=False))
            labels.append(label)
    return dict(candidates=candidates,structure_notes=structural,lexical_review_flags=word_flags(soup.get_text(' ',strip=True)),text_source='original_html')


def pdf_inventory(text,event):
    # Source-adjacent text may have form feeds or explicit PAGE markers. Never infer a geometric column.
    chunks=re.split(r'\f|(?im:^\s*(?:={2,}\s*)?PAGE\s+\d+\s*(?:={2,})?\s*$)',text)
    candidates=[];line_offset=0;wanted={str(event['fiscal_year']),str(event['fiscal_year']-1)}
    for page,chunk in enumerate(chunks,1):
        lines=chunk.splitlines();labels=[]
        for i,line in enumerate(lines):
            label=norm(re.split(r'\s{2,}(?=\(?[\d$])',line.strip(),maxsplit=1)[0])
            # Also handles plain-text extraction without aligned numeric columns.
            label=norm(re.sub(r'\s+\(?[\d$].*$','',label))
            kind=label_type(label,labels)
            if kind:
                head=lines[:min(i,45)];annual=bool(ANNUAL.search(' '.join(head)));interim=bool(INTERIM.search(' '.join(head)))
                headers=[h for h in head if header_like(h,wanted)]
                found=set(YEAR.findall(' '.join(headers)));pair=annual and not interim and wanted.issubset(found)
                blockers=['text_only_share_row_requires_column_geometry_review']
                if not pair:blockers.append('current_and_prior_annual_columns_not_uniquely_located')
                if kind.endswith('implicit'):blockers.append('weighted_average_semantics_implicit_needs_review')
                units=unit_evidence(head[:12]+labels[-6:]+[label])
                if not units:blockers.append('share_unit_scale_not_located')
                candidates.append(dict(format='pdf_existing_text',text_chunk_index_1based=page,text_line_1based=line_offset+i+1,label=redact(label),semantic_kind=kind,ancestor_labels_redacted=[redact(t)for t in labels[-6:]if t],annual_header_present=annual,interim_header_present=interim,header_literals_redacted=sorted(set(redact(h)for h in headers)),units=units,same_report_annual_pair_locator_present=False,annual_pair_text_context_present=pair,blockers=blockers,quantities_retained=False))
            labels.append(label)
        line_offset+=len(lines)
    return dict(candidates=candidates,structure_notes=[],lexical_review_flags=word_flags(text),text_source='frozen_source_adjacent_existing_pdf_text')


def summarize(result):
    cs=result['candidates'];explicit=[c for c in cs if c['semantic_kind']=='explicit_basic_weighted_average_shares']
    located=[c for c in explicit if c['same_report_annual_pair_locator_present']and not c['blockers']]
    status='no_basic_share_candidate_located'if not cs else'explicit_annual_pair_locator_present'if len(located)==1 else'candidate_present_needs_source_review'
    if len(located)>1:status='multiple_annual_pair_candidates_ambiguous'
    return dict(status=status,candidate_count=len(cs),explicit_weighted_candidate_count=len(explicit),unambiguous_complete_locator_count=len(located))


def inventory(plan_sha):
    require(re.fullmatch('[a-f0-9]{64}',plan_sha or'')is not None,'Explicit plan SHA required')
    require(sha(ROOT/PLAN)==plan_sha,'Plan SHA mismatch');plan=read(PLAN)
    verify(plan['source_hashes']);verify(plan['implementation_hashes']);require(plan['rules']==rules(),'Rules mismatch')
    rows=[]
    for e in plan['events']:
        raw=(ROOT/e['source']['path']).read_bytes();require(len(raw)<=33554432,'Source size bound')
        if e['format']=='html':result=html_inventory(raw,e)
        elif e['existing_text']is not None:result=pdf_inventory((ROOT/e['existing_text']['path']).read_text(),e)
        else:result=dict(candidates=[],structure_notes=[{'reason':'existing_pdf_text_unavailable'}],lexical_review_flags={},text_source=None)
        rows.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source=deepcopy(e['source']),existing_text=deepcopy(e['existing_text']),release_date=e['release_date'],available_from=e['available_from'],periods=deepcopy(e['periods']),identity=deepcopy(e['identity']),inherited_holds={k:deepcopy(e[k])for k in('inherited_rejection_codes','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master','fact_admitted','prior_field_admitted','model_admitted','strategy_admitted')},summary=summarize(result),inventory=result,source_admission_changed=False,new_field_admitted=False))
    counts=Counter(r['summary']['status']for r in rows)
    report=dict(schema='after158_basic_shares_inventory_v1',plan_sha256=plan_sha,created_utc=datetime.now(timezone.utc).isoformat(),events=rows,counts=dict(events=len(rows),by_status=dict(counts),by_symbol={s:dict(Counter(r['summary']['status']for r in rows if r['symbol']==s))for s in sorted({r['symbol']for r in rows})}),source_admission_changed=False,share_quantities_retained=False,activity={'features':0,'ratios':0,'ranks':0,'orders':0,'returns':0,'model_fits':0,'new_trials':0})
    verify(plan['source_hashes']);verify(plan['implementation_hashes']);require(sha(ROOT/PLAN)==plan_sha,'Plan changed')
    write_once(OUT+'/INVENTORY.json',report)
    manifest=dict(schema='after158_basic_shares_inventory_manifest_v1',plan=PLAN,plan_sha256=plan_sha,input_hashes={**plan['source_hashes'],**plan['implementation_hashes'],PLAN:plan_sha},output_hashes={'INVENTORY.json':sha(ROOT/OUT/'INVENTORY.json')},source_admission_changed=False)
    write_once(OUT+'/MANIFEST.json',manifest)
    print(json.dumps({'counts':report['counts'],'inventory_sha256':manifest['output_hashes']['INVENTORY.json'],'manifest_sha256':sha(ROOT/OUT/'MANIFEST.json')},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-plan',action='store_true');p.add_argument('--inventory',action='store_true');p.add_argument('--plan-sha');a=p.parse_args();require(a.freeze_plan!=a.inventory,'Choose one stage')
    freeze_plan()if a.freeze_plan else inventory(a.plan_sha)
if __name__=='__main__':main()
