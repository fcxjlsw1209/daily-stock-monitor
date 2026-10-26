"""Independent original-HTML review of the fixed remaining 27 share pairs.

No producer parser/locator imports. Rebuild physical table occupancy, choose
the Basic row by its immediately preceding nonblank share-denominator heading,
and resolve its annual dates from original header cells. No financial factor.
"""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_basic_shares_independent_remaining.py'
EVIDENCE = 'work/after158_basic_shares_independent_remaining_evidence.json'
PLAN = 'work/after158_basic_shares_source_plan.json'
MANUAL = 'work/after158_basic_shares_independent_manual.json'
PREVIOUS = 'work/after158_basic_shares_independent_review.json'
PINS = {PLAN:'3ff000c9a1b9c17d04d2a2edf2f79f21b18811f4e32263b38b768ace805ca60b',
        MANUAL:'e0c53b3cde86c38646042e7e57ac235263b7e313a9a0f6efb2a2da239e271bad',
        PREVIOUS:'d0a108bbd11dbe79715d9ad63abcb3be05dbd04c597b7f78ffd491a27269a1f3'}
MONTHS = {name.lower():i+1 for i,name in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'])}
MONTH = '('+'|'.join(MONTHS)+')'
TITLE = re.compile(r'(?:statements? of (?:income|operations)|income statements?)', re.I)
UNITS = re.compile(r'\b(?:in\s+)?(thousands|millions|billions)\b',re.I)
ANNUAL = re.compile(r'\b(?:year(?:s)?|twelve months|12 months) ended\b',re.I)
INTERIM = re.compile(r'\b(?:three|six|nine|3|6|9) months ended\b|quarter',re.I)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def norm(value):
    return re.sub(r'\s+',' ',value).strip()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local(name):
    p=Path(name)
    need(not p.is_absolute() and '..' not in p.parts, 'Nonlocal evidence path')
    out=ROOT/p
    need(out.resolve().is_relative_to(ROOT), 'Escaping evidence path')
    return out


def raw(name, expected=None):
    value=local(name).read_bytes()
    need(expected is None or digest(value)==expected,'Hash mismatch: '+name)
    return value


def source_rows(table):
    trs=[r for r in table.find_all('tr') if r.find_parent('table') is table]
    rows=[]; occupancy={}
    for y,tr in enumerate(trs):
        cells=[c for c in tr.find_all(['td','th']) if c.find_parent('tr') is tr and c.find_parent('table') is table]
        row=[];x=0
        for physical,node in enumerate(cells):
            while (y,x) in occupancy:
                x+=1
            width=int(node.get('colspan',1));height=int(node.get('rowspan',1))
            need(1<=width<=100 and 1<=height<=100,'Invalid table span')
            cell={'row':y,'physical':physical,'columns':list(range(x,x+width)),
                  'text':norm(node.get_text(' ',strip=True)),'line':node.sourceline}
            for yy in range(y,y+height):
                for xx in range(x,x+width):
                    need((yy,xx) not in occupancy,'Overlapping cells')
                    occupancy[yy,xx]=cell
            row.append(cell);x+=width
        rows.append(row)
    return rows,occupancy


def date_header(text):
    md=re.findall(r'\b'+MONTH+r'\s+(\d{1,2})\b',text,re.I)
    years=re.findall(r'\b(?:19|20)\d{2}\b',text)
    need(len(set(md))==1 and len(set(years))==1,'Ambiguous original date header: '+text)
    month,day=md[0]
    return date(int(years[0]),MONTHS[month.lower()],int(day)).isoformat()


def previous_context(table):
    result=[]
    for n,node in enumerate(table.previous_elements):
        if n>300:
            break
        if getattr(node,'name',None)=='table':
            break
        if getattr(node,'name',None) not in ('font','span','p','div','b','strong','h2','h3'):
            continue
        if node.find_parent('table') is not None:
            break
        text=norm(node.get_text(' ',strip=True))
        if not text or len(text)>250 or text in result:
            continue
        result.append(text)
        if TITLE.search(text):
            break
    return result


def parse_original(raw_html, event):
    soup=BeautifulSoup(raw_html,'html.parser')
    for node in soup(['script','style']):
        node.decompose()
    matches=[]
    for ti,table in enumerate(soup.find_all('table')):
        rows,grid=source_rows(table)
        texts=[norm(' '.join(c['text'] for c in row)) for row in rows]
        for ri,row in enumerate(rows):
            nonblank=[c for c in row if c['text']]
            if not nonblank or nonblank[0]['text'].lower()!='basic':
                continue
            prior=next((texts[k] for k in range(ri-1,-1,-1) if texts[k]),'')
            if not(re.search(r'\bshares\b',prior,re.I) and re.search(r'weighted|used',prior,re.I)) or re.search('diluted',prior,re.I):
                continue
            body=next((i for i,t in enumerate(texts[:ri]) if re.match(r'^(?:net |total )?(?:revenue|sales)\b',t,re.I)),None)
            need(body is not None,'No original statement body boundary')
            caption=table.find('caption',recursive=False)
            internal=([norm(caption.get_text(' ',strip=True))] if caption else [])+texts[:body]
            external=previous_context(table)
            titles=[x for x in internal if TITLE.search(x)] or [x for x in external if TITLE.search(x)]
            need(titles and not any(re.search(r'non[ -]?gaap|reconciliation|pro forma',x,re.I) for x in titles),'Nonformal source table')
            unit_text=[x for x in internal if UNITS.search(x)] or [x for x in external if UNITS.search(x)]
            unit_names={u.lower() for x in unit_text for u in UNITS.findall(x)}
            need(len(unit_names)==1,'Ambiguous local table units')
            need(not any(re.search(r'except.*(?:number of shares|share counts)',x,re.I) for x in unit_text),'Unparsed share-specific exception')
            scale={'thousands':1000,'millions':1000000,'billions':1000000000}[next(iter(unit_names))]
            cells={}
            for c in nonblank[1:]:
                if not re.fullmatch(r'\d[\d,]*(?:\.\d+)?',c['text']):
                    continue
                choices=[]
                for x in c['columns']:
                    headers=[];seen=set()
                    for y in range(body):
                        h=grid.get((y,x))
                        if h is not None and (h['row'],h['physical']) not in seen:
                            headers.append(h);seen.add((h['row'],h['physical']))
                    joined=' '.join(h['text'] for h in headers)
                    if not ANNUAL.search(joined) or INTERIM.search(joined):
                        continue
                    when=date_header(joined)
                    choices.append((when,headers,x))
                if not choices:
                    continue
                need(len({x[0] for x in choices})==1,'One physical value spans conflicting dates')
                when=choices[0][0]
                side=next((s for s in ('current','comparison') if event['periods'][s]['period_end']==when),None)
                if side is None:
                    continue
                need(side not in cells,'Duplicate annual value for one period')
                cells[side]={'value_decimal':str(Decimal(c['text'].replace(',',''))),
                    'displayed_value':c['text'],'share_scale':scale,'period_end':when,
                    'physical_cell':c['physical'],'source_line_1based':c['line'],
                    'headers':[h for h in choices[0][1] if h['text']],
                    'annual_logical_columns':[x[2] for x in choices]}
            need(set(cells)=={'current','comparison'},'Two exact annual values required')
            matches.append({'event_id':event['event_id'],'source':event['source'],'table_index':ti,
                'row_index':ri,'row_text':texts[ri],'denominator_heading':prior,
                'formal_title':titles,'unit_literals':unit_text,'cells':cells,
                'header_rows':texts[:body],
                'semantic_basis':'explicit_weighted_average' if re.search('weighted',prior,re.I) else 'implicit_basic_eps_denominator'})
    need(len(matches)==1,'Expected exactly one original annual Basic pair, got '+str(len(matches)))
    return matches[0]


def write_once(name,value):
    data=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with local(name).open('xb') as handle:
        handle.write(data)
    return digest(data)


def inspect():
    bound={k:json.loads(raw(k,v)) for k,v in PINS.items()}
    events=bound[PLAN]['events'];old_ids={r['event_id'] for r in bound[MANUAL]['manual_source_pairs']}
    targets=[e for e in events if e['event_id'] not in old_ids]
    need(len(events)==44 and len(old_ids)==17 and len(targets)==27,'Fixed remaining denominator changed')
    need(all(e['format']=='html' for e in targets),'Remaining sources must be HTML')
    pins=dict(PINS);pins[SELF]=digest(raw(SELF));records=[]
    for event in targets:
        source=event['source']
        value=raw(source['path'],source['sha256'])
        raw(source['receipt']['path'],source['receipt']['sha256'])
        pins[source['path']]=source['sha256'];pins[source['receipt']['path']]=source['receipt']['sha256']
        try:
            record=parse_original(value,event)
            record['failure']=None
        except ValueError as ex:
            record={'event_id':event['event_id'],'source':source,'failure':str(ex)}
        records.append(record)
    for name,h in pins.items():
        raw(name,h)
    result={'schema':'after158_basic_shares_independent_remaining_evidence_v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'input_hashes':pins,'records':records,
        'scope':'Exactly the 27 original HTML events not in the prior 17-pair manual oracle; no producer numeric output or locators used by this program.',
        'source_admission_changed':False,'factors_computed':0,'returns_computed':0,'orders_computed':0}
    h=write_once(EVIDENCE,result)
    print(json.dumps({'path':EVIDENCE,'sha256':h,'records':len(records),'failures':[r for r in records if r['failure']]},indent=2))


class TestOriginalTable(unittest.TestCase):
    def fixture(self, annual='Year Ended', scale='millions', value='1,250', prior='1,400', repeat=False):
        span=' colspan="2"' if repeat else ''
        width='4' if repeat else '2'
        html='''<p>Non-GAAP narrative unrelated to next table</p><table><caption>CONSOLIDATED STATEMENTS OF INCOME (In %s, except per-share data)</caption>
        <tr><td></td><td colspan="2">Three Months Ended</td><td colspan="%s">%s</td></tr>
        <tr><td></td><td>December 31, 2024</td><td>December 31, 2023</td><td%s>December 31, 2024</td><td%s>December 31, 2023</td></tr>
        <tr><td>Revenue</td><td>99</td><td>88</td><td%s>1</td><td%s>2</td></tr>
        <tr><td>Net income per share</td></tr><tr><td>Basic</td><td>1</td><td>2</td><td%s>3</td><td%s>4</td></tr>
        <tr><td>Weighted average shares used in per share computation:</td></tr>
        <tr><td>Basic</td><td>77</td><td>66</td><td%s>%s</td><td%s>%s</td></tr>
        <tr><td>Diluted</td><td>9</td><td>8</td><td%s>7</td><td%s>6</td></tr></table>''' % (scale,width,annual,span,span,span,span,span,span,span,value,span,prior,span,span)
        event={'event_id':'TEST','source':{},'periods':{'current':{'period_end':'2024-12-31'},'comparison':{'period_end':'2023-12-31'}}}
        return html.encode(),event

    def test_exact_annual_basic_not_quarter_or_eps(self):
        r=parse_original(*self.fixture())
        self.assertEqual(r['cells']['current']['value_decimal'],'1250')
        self.assertEqual(r['cells']['comparison']['value_decimal'],'1400')

    def test_physical_colspan_not_duplicate(self):
        r=parse_original(*self.fixture(repeat=True))
        self.assertEqual(len(r['cells']['current']['annual_logical_columns']),2)

    def test_reject_quarter_only(self):
        with self.assertRaises(ValueError):parse_original(*self.fixture(annual='Three Months Ended'))

    def test_reject_wrong_source_date(self):
        html,event=self.fixture();event['periods']['comparison']['period_end']='2023-12-30'
        with self.assertRaises(ValueError):parse_original(html,event)

    def test_reject_missing_annual_number(self):
        with self.assertRaises(ValueError):parse_original(*self.fixture(value='—'))

    def test_reject_own_nongaap_caption(self):
        html,event=self.fixture();html=html.replace(b'CONSOLIDATED STATEMENTS',b'NON-GAAP CONSOLIDATED STATEMENTS')
        with self.assertRaises(ValueError):parse_original(html,event)

    def test_unit_and_zero_source(self):
        r=parse_original(*self.fixture(scale='thousands',value='0'))
        self.assertEqual(r['cells']['current']['share_scale'],1000)
        self.assertEqual(r['cells']['current']['value_decimal'],'0')

    def test_duplicate_tables_fail(self):
        html,event=self.fixture()
        with self.assertRaises(ValueError):parse_original(html+html,event)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inspect',action='store_true');parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args();need(args.inspect != args.test_only,'Choose one phase')
    if args.test_only:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestOriginalTable))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    inspect()
