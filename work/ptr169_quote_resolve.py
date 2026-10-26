"""Verify the one open-market missing PTR A-share quotation offline."""
import hashlib
import json
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from ptr169_quote_audit import EXPECTED, normalize_chart

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169/missing_quote'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def parse(body):
    assert body['code']==0 and body['msg']=='', 'Provider error'
    assert set(body['data'])=={'sh601857'}, 'Wrong instrument'
    stock=body['data']['sh601857']
    assert 'day' in stock and 'qfqday' not in stock and 'hfqday' not in stock, 'Not unadjusted bars'
    rows={}
    for raw in stock['day']:
        assert len(raw)==6, 'Unexpected bar schema'
        day=raw[0]
        assert datetime.strptime(day,'%Y-%m-%d').strftime('%Y-%m-%d')==day
        assert day not in rows and '2019-04-22'<=day<='2019-04-30'
        op,cl,hi,lo,vol=map(float,raw[1:])
        assert all(math.isfinite(x) and x>0 for x in (op,cl,hi,lo))
        assert lo<=min(op,cl)<=max(op,cl)<=hi
        assert math.isfinite(vol) and vol>=0
        rows[day]=dict(open=op,close=cl,high=hi,low=lo)
    assert list(rows)==sorted(rows)
    assert '2019-04-29' in rows, 'Missing target'
    return rows

def main():
    source=OUT/'tencent_201904.json';receipt=OUT/'tencent_201904.receipt.json'
    plan=OUT/'TENCENT_PLAN.json';r=read(receipt)
    assert r['status']==200 and r['body_sha256']==sha(source)
    assert r['url']==read(plan)['url']==r['final_url'] and r['plan_sha256']==sha(plan)
    assert read(plan)['code_sha256']==sha(ROOT/'work/ptr169_tencent_quote.py')
    rows=parse(read(source))
    yahoo=ROOT/'outputs/monthly_ptr_pool_source169/601857.SS.json'
    yr=read(yahoo.with_name(yahoo.name+'.receipt.json'))
    assert yr['body_sha256']==sha(yahoo)
    _,other,_=normalize_chart(read(yahoo),EXPECTED[yahoo.name])
    overlap=[]
    for day,row in rows.items():
        if day not in other:continue
        assert other[day]['ohlc_range_consistent']
        diff={k:abs(v-other[day][k]) for k,v in row.items()}
        assert max(diff.values())<0.00001,(day,diff)
        overlap.append(dict(date=day,absolute_differences=diff))
    assert [r['date'] for r in overlap]==['2019-04-22','2019-04-23','2019-04-24','2019-04-25','2019-04-26']
    assert '2019-04-29' not in other and '2019-04-30' not in other
    pdf=OUT/'cjsc_20190430.pdf'
    assert sha(pdf)=='1d373dd32143e854cecd6fa1d38cf2e81d1e7e7824ce8f10ad48c5b9e47fea31'
    assert '中国石油股份 0857.HK 601857.SH 5.06 7.40' in (OUT/'cjsc_page08.txt').read_text()
    runtime=ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
    assert sha(runtime)=='cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
    cfg=read(runtime)
    for p,h in cfg['pdf_runtime_hashes'].items():assert sha(Path(p))==h
    sys.path.append(cfg['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    img=OUT/'cjsc_page07.png'
    assert not img.exists()
    doc=pdfium.PdfDocument(pdf.read_bytes());doc[6].render(scale=1.75).to_pil().save(img);doc.close()
    paths=[source,receipt,plan,yahoo,yahoo.with_name(yahoo.name+'.receipt.json'),pdf,
           OUT/'EXTRACTION.json',OUT/'cjsc_page01.txt',OUT/'cjsc_page07.txt',OUT/'cjsc_page08.txt',img]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='QUOTE_CORROBORATED_PENDING_HEADER_VISUAL_REVIEW',
        code_sha256=sha(Path(__file__)),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        instrument='601857.SH',share_class='A',currency='CNY',date='2019-04-29',ohlc=rows['2019-04-29'],
        current_qt_section_excluded=True,volume_excluded='Tencent volume convention not needed or admitted',
        five_nearby_OHLC_rows_match_yahoo=overlap,
        yahoo_missing_in_probe=['2019-04-29','2019-04-30'],
        development_note='Initial expected six overlap rows failed: Yahoo lacks April30 as well as target April29. Five available prior rows match all OHLC; April30 is not admitted or used.',
        broker_corroboration=dict(report_date='2019-04-30',prior_session_discussed='2019-04-29',
            A_quote=7.40,H_quote=5.06,table='Table6 A+H relative performance, header p7 and row p8',
            row_date_inferred_from_report_context=True,pdf_p1_p8_visually_checked_by_root=True),
        limits=['Historical prices retrieved in2026; not an archived2019 API vintage.',
                'Broker report appeared after target US close; it corroborates a historical quote, not source availability before that close.',
                'Tencent qt current metadata, current shares and market state are excluded.',
                'No existing cache is filled or overwritten; downstream use requires separate registered valuation policy.'],
        existing_inputs_modified=False,market_caps_computed=False,returns_computed=False)
    with (OUT/'RESOLUTION.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],ohlc=result['ohlc'],overlap=len(overlap)),ensure_ascii=False))

class Tests(unittest.TestCase):
    def good(self):return {'code':0,'msg':'','data':{'sh601857':{'day':[['2019-04-29','7.410','7.400','7.460','7.370','746527']]}}}
    def test_valid(self):self.assertEqual(parse(self.good())['2019-04-29']['close'],7.4)
    def test_adjusted_rejected(self):
        b=self.good();b['data']['sh601857']['qfqday']=b['data']['sh601857'].pop('day')
        with self.assertRaises(AssertionError):parse(b)
    def test_wrong_instrument_rejected(self):
        b=self.good();b['data']['sh600000']=b['data'].pop('sh601857')
        with self.assertRaises(AssertionError):parse(b)
    def test_range_rejected(self):
        b=self.good();b['data']['sh601857']['day'][0][3]='7.3'
        with self.assertRaises(AssertionError):parse(b)
    def test_current_quote_cannot_supply_missing_target(self):
        b=self.good();b['data']['sh601857'].update(day=[],qt={'close':'7.4'})
        with self.assertRaises(AssertionError):parse(b)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='test':
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        sys.exit(0 if r.wasSuccessful() else 1)
    main()
