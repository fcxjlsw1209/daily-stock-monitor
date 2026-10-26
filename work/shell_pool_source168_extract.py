"""Registered Shell share-capital source evidence only; offline, no market values."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import sys
from datetime import datetime
from calendar import monthrange

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_shell_pool_source168'
PLAN_SHA = '587fa9546ef99a84a947a8f51bb6e66b768454fdb9474faeefac9a2cf41687eb'
PDF_PINS = ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    assert sha(OUT/'PLAN.json') == PLAN_SHA
    plan=json.loads((OUT/'PLAN.json').read_text())
    assert len(plan['records'])==20
    for rec in plan['records']:
        for p,h in rec['inputs'].items():assert sha(ROOT/p)==h,p
    for p,h in plan['anchors'].items():assert sha(ROOT/p)==h,p
    assert sha(PDF_PINS)=='cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
    pins=json.loads(PDF_PINS.read_text())
    for p,h in pins['pdf_runtime_hashes'].items():assert sha(Path(p))==h,p
    sys.path.append(pins['pdf_runtime']['bundle_site_packages'])
    return plan,pins


def publish(path, content):
    if path.exists():
        assert path.read_bytes()==content, 'Refuse to replace derived bytes: '+str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)


def derive(plan):
    from pypdf import PdfReader
    found=[]
    for rec in plan['records']:
        identity=f"{rec['year']}Q{rec['quarter']}"
        pdf=PdfReader(io.BytesIO((ROOT/rec['binary']).read_bytes()))
        selected=[]
        for page_id,page in enumerate(pdf.pages,1):
            text=page.extract_text(extraction_mode='layout')
            publish(OUT/'derived'/identity/f'page{page_id:02d}.txt',text.encode())
            if re.search(r'(?:issued and fully paid|share capital|Shares held in trust|date of this announcement)',text,re.I):
                selected.append(page_id)
        found.append(dict(period=identity,pages=len(pdf.pages),relevant_pages=selected))
    publish(OUT/'derived'/'INDEX.json',(json.dumps(found,indent=2)+'\n').encode())
    print(json.dumps(found,indent=2))


def render(plan):
    import pypdfium2 as pdfium
    index=json.loads((OUT/'derived/INDEX.json').read_text())
    for rec,info in zip(plan['records'],index):
        ident=f"{rec['year']}Q{rec['quarter']}"
        candidates=[]
        for p in sorted((OUT/'derived'/ident).glob('page*.txt')):
            if re.search('ISSUED AND FULLY PAID|Issued and fully paid',p.read_text()):
                candidates.append(int(p.stem[4:]))
        assert len(candidates)==1
        with pdfium.PdfDocument((ROOT/rec['binary']).read_bytes()) as pdf:
            for num in (candidates[0],info['pages']):
                target=OUT/'derived'/ident/f'page{num:02d}.png'
                if not target.exists():pdf[num-1].render(scale=1.5).to_pil().save(target)
        print(ident,candidates[0],info['pages'])


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def amount(token):
    if token in ('-', '—'):return 0
    assert re.fullmatch(r'\(?\d{1,3}(?:,\d{3})*\)?|\(?\d+\)?',token),token
    return int(token.replace(',','').replace('(','-').replace(')',''))


def row_values(line,n):
    # Only fixed current-year issued-share table rows reach this parser.
    # Exactly n cells are A/B(/deferred) or A/B and three nominal amounts.
    if line.startswith('At '):
        label=re.match(r'^At [A-Z][a-z]+ \d{1,2}, \d{4}\s+',line)
    else:
        label=re.match(r'^(?:Scrip dividends|Repurchases of shares|Shares issued\d?(?: for the(?: acquisition)?)?)\s+',line)
    assert label, ('unrecognized row label',line)
    cells=line[label.end():].split()
    assert len(cells)==n, ('exact numeric column count required',line)
    return [amount(x) for x in cells]


def parse_table(text,year,quarter):
    lines=[norm(x) for x in text.splitlines()]
    start=next(i for i,x in enumerate(lines) if re.match(r'ISSUED AND FULLY PAID|Issued and fully paid',x))
    n=3 if (year,quarter)<(2016,4) else 5
    month=quarter*3;day=monthrange(year,month)[1]
    end=f'{datetime(year,month,day):%B} {day}, {year}'
    body=lines[start:]
    if (year,quarter)==(2017,3):
        # The source PDF paints bold cells four times. Preserve the layout
        # defect and use a visually checked transcription, never a rollforward
        # inference. Verify all four literal instances of both printed values.
        opening=next(x for x in body if x.startswith('At January 1, 2017777'))
        ending_start=next(i for i,x in enumerate(body) if x.startswith('At At At At September'))
        ending=' '.join(body[ending_start:ending_start+2])
        checks=[(opening,'4,428,903,813'),(opening,'3,745,486,731'),
                (ending,'4,544,414,617'),(ending,'3,745,486,731')]
        for line,value in checks:assert line.count(value)==4,(line,value)
        av,bv=4428903813,3745486731
        ae,be=4544414617,3745486731
        movement=next(x for x in body if x.startswith('Scrip dividends'))
        vals=row_values(movement,n)
        movements=[dict(label='Scrip dividends',A=vals[0],B=vals[1],evidence=movement)]
        method='manual visual transcription of repeated PDF text; four literal copies per A/B cell independently verified, not inferred from sum'
    else:
        opening_i=next(i for i,x in enumerate(body) if x.startswith(f'At January 1, {year} '))
        ending_i=next(i for i,x in enumerate(body) if x.startswith('At '+end+' '))
        assert ending_i>opening_i
        opening,ending=body[opening_i],body[ending_i]
        av,bv=row_values(opening,n)[:2];ae,be=row_values(ending,n)[:2]
        movements=[]
        for line in body[opening_i+1:ending_i]:
            if not line:continue
            if line.startswith(('Scrip dividends','Repurchases of shares','Shares issued')):
                vals=row_values(line,n)
                label=re.split(r'\(?\d[\d,]*\)?|[-—]',line,maxsplit=1)[0].strip()
                movements.append(dict(label=label,A=vals[0],B=vals[1],evidence=line))
            else:
                assert line in ('acquisition of BG Group plc1','of BG Group plc1'), ('unrecognized movement row',line)
        method='fixed issued-share table current-year rows; first A/B cells, separate from nominal/deferred columns'
    calculated_a=av+sum(x['A'] for x in movements);calculated_b=bv+sum(x['B'] for x in movements)
    assert (calculated_a,calculated_b)==(ae,be), 'Rollforward disagreement'
    return dict(period_end=f'{year}-{month:02d}-{day:02d}',A=ae,B=be,total_derived=ae+be,
                ending_evidence=ending,method=method,
                rollforward=dict(opening_date=f'{year}-01-01',A_open=av,B_open=bv,opening_evidence=opening,
                                 movements=movements,computed_A_end=calculated_a,computed_B_end=calculated_b,agrees=True))


def extract(plan,pins):
    from pypdf import PdfReader
    records=[]
    for rec in plan['records']:
        ident=f"{rec['year']}Q{rec['quarter']}"
        files=sorted((OUT/'derived'/ident).glob('page*.txt'))
        pages=[(int(p.stem[4:]),p.read_text()) for p in files]
        table_pages=[(num,t) for num,t in pages if re.search('ISSUED AND FULLY PAID|Issued and fully paid',t)]
        assert len(table_pages)==1
        num,text=table_pages[0]
        # Re-extract selected original pages to bind derived excerpts to source.
        reader=PdfReader(io.BytesIO((ROOT/rec['binary']).read_bytes()))
        assert reader.pages[num-1].extract_text(extraction_mode='layout')==text
        facts=parse_table(text,rec['year'],rec['quarter'])
        releases=[]
        for p,t in pages:
            for m in re.finditer(r'date of this announcement,\s*([A-Z][a-z]+)\s+(\d{1,2})\s*,\s*(\d{4})',norm(t)):
                date=datetime.strptime(' '.join(m.groups()),'%B %d %Y').date().isoformat()
                releases.append((p,date,m.group()))
        assert len(releases)==1, (ident,releases)
        release_page,release_date,release_evidence=releases[0]
        assert reader.pages[release_page-1].extract_text(extraction_mode='layout')==dict(pages)[release_page]
        metadata_path=next(p for p in rec['inputs'] if p.endswith('.meta.json'))
        metadata=json.loads((ROOT/metadata_path).read_text())
        assert metadata['sha256']==rec['inputs'][rec['binary']]
        existing_text=next((p for p in rec['inputs'] if p.endswith('.text.json')),None)
        old_provenance=None
        if existing_text:
            prov=json.loads((ROOT/existing_text).read_text())
            txt=existing_text.replace('.text.json','.txt')
            assert prov['source_sha256']==rec['inputs'][rec['binary']] and prov['text_sha256']==rec['inputs'][txt]
            old_provenance=dict(path=existing_text,parser=prov['parser'],source_and_text_digest_verified=True)
        trust_pages=[p for p,t in pages if re.search('Shares held in trust',norm(t),re.I)]
        eps_pages=[p for p,t in pages if re.search('Weighted average number of shares',norm(t),re.I)]
        treasury_pages=[p for p,t in pages if re.search(r'\btreasury\b',t,re.I)]
        assert trust_pages and eps_pages
        deferred=[norm(x) for x in text.splitlines() if '50,000' in x and 'deferred' in x.lower()]
        deferred_evidence=deferred if deferred else [norm(x) for x in text.splitlines() if 'Sterling deferred shares' in x or 'Number of shares' in x]+[facts['ending_evidence']]
        printed_footer=norm(text.splitlines()[-1])
        item=dict(period=ident,period_end=facts.pop('period_end'),
                  source=dict(binary=rec['binary'],url=rec['url'],sha256=rec['inputs'][rec['binary']],metadata=metadata_path,
                              retrieved_at=metadata['retrieved_at'],existing_text_provenance=old_provenance),
                  issued_ordinary_shares=dict(A=facts.pop('A'),B=facts.pop('B'),total_derived=facts.pop('total_derived'),
                                              units='ordinary shares; not millions',nominal_per_ordinary_share_eur='0.07',
                                              source_page_1based=num,printed_footer=printed_footer,
                                              evidence=facts.pop('ending_evidence'),method=facts.pop('method')),
                  release=dict(stated_date=release_date,source_page_1based=release_page,evidence=release_evidence,
                               intraday_time=None,historical_availability_proven=False,
                               note='Issuer-stated announcement date in later retrieved PDF; not a vintage or contemporaneous archive certificate'),
                  rollforward=facts.pop('rollforward'),
                  caveats=dict(deferred=dict(count=50000,units='sterling deferred shares of £1; excluded from A+B',source_page_1based=num,evidence=deferred_evidence),
                               employee_trust=dict(source_pages_1based=trust_pages,evidence='Shares held in trust',
                                                   A_period_end_count=None,B_period_end_count=None,
                                                   note='Monetary statement presentation is not class-specific trust share counts; issued A/B is not independently verified free-float or net outstanding'),
                               weighted_EPS=dict(source_pages_1based=eps_pages,evidence='Weighted average number of shares',used_as_period_end_count=False),
                               treasury=dict(word_match_pages_1based=treasury_pages,A_period_end_count=None,B_period_end_count=None,
                                             note='No separate class-specific treasury counts admitted; absence of a text match is not proof of zero treasury shares'),
                               nominal_value=dict(used_as_share_count=False,note='Dollar-million A/B/Total columns are nominal capital, not share-count totals'),
                               market_cap=dict(derived=False,note='Class-specific historical quotes, ADS conversion and vendor-series scope remain outside this source audit')),
                  missing_or_ambiguous=['Exact announcement intraday timestamp and original historical version not proven',
                                        'Class-specific net outstanding/treasury/employee-trust counts not established by the issued-share table'],
                  strategy_admitted=False,pool_modified=False)
        if ident=='2017Q3':item['resolved_representation_issue']='Four overprinted copies of bold rows; visually transcribed and literal-copy checked'
        records.append(item)
    continuity=[]
    for previous,current in zip(records,records[1:]):
        if current['period'].endswith('Q1'):
            ok=tuple(previous['issued_ordinary_shares'][k] for k in ('A','B'))==tuple(current['rollforward'][k] for k in ('A_open','B_open'))
            assert ok
            continuity.append(dict(previous_period=previous['period'],next_period=current['period'],previous_end_equals_open=True))
    payload=dict(schema='shell_pool_source168_extraction_v1',plan_sha256=PLAN_SHA,
                 extractor_sha256=sha(Path(__file__)),pdf_runtime_pins_source=str(PDF_PINS.relative_to(ROOT)),pdf_runtime_pins_sha256=sha(PDF_PINS),
                 input_digest_checks=sum(len(r['inputs']) for r in plan['records'])+len(plan['anchors']),
                 records=records,summary=dict(documents=20,exact_A_B_issued_rows=20,release_dates=20,rollforwards_passed=20,
                                             quoted_A_plus_B_share_count_total=0,derived_A_plus_B_totals=20,year_opening_continuity=continuity),
                 visual_review=dict(producer_pages=['2015Q4/16','2016Q1/18','2016Q4/20','2017Q3/12','2020Q2/15'],
                                    remaining_pages='independent reviewer verification requested; no claim of producer visual inspection for all20'),
                 limitations=['Legal issued share capital is not automatically a usable issuer market-cap denominator.',
                              'No class-price equality, fixed share multiplier or complete daily share series assumed.',
                              'Quarter-end facts are not made available before stated publication dates.',
                              'No price/cap/rank/factor/holding/return computation or frozen-input amendment.'],
                 network_requests=0,strategy_trials_added=0,frozen_files_modified=0)
    target=OUT/'EXTRACTION.json';publish(target,(json.dumps(payload,indent=2)+'\n').encode())
    print('period | A issued | B issued | announcement | PDF page')
    for r in records:
        print(r['period'],r['issued_ordinary_shares']['A'],r['issued_ordinary_shares']['B'],r['release']['stated_date'],r['issued_ordinary_shares']['source_page_1based'],sep=' | ')
    print('EXTRACTION_SHA256',sha(target))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--derive',action='store_true');parser.add_argument('--render',action='store_true');parser.add_argument('--extract',action='store_true')
    args=parser.parse_args();plan,pins=verify()
    if args.derive:derive(plan)
    if args.render:render(plan)
    if args.extract:extract(plan,pins)


if __name__=='__main__':main()
