"""Separate acquisition and offline extraction of PTR share-class evidence."""
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169/annual'
URLS = {
    '2016': 'https://www.hkexnews.hk/listedco/listconews/sehk/2017/0421/ltn20170421377.pdf',
    '2017': 'https://www.hkexnews.hk/listedco/listconews/sehk/2018/0416/ltn20180416392.pdf',
    '2018': 'https://www1.hkexnews.hk/listedco/listconews/sehk/2019/0416/ltn20190416287.pdf',
    '2019': 'https://www.hkexnews.hk/listedco/listconews/sehk/2020/0421/2020042100545.pdf',
}

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def write(p, v):
    with p.open('x') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def register():
    OUT.mkdir(exist_ok=True)
    inputs = ['outputs/monthly_ptr_pool_source169/SOURCE_PLAN.json',
              'outputs/monthly_ptr_pool_source169/DOCUMENT_AUDIT.json',
              'outputs/dynamic_top100/priority_sec/PTR_submissions.json']
    write(OUT/'PLAN.json', dict(created_utc=now(), scope='Four annual share-class tables, 2016 through 2019. Source acquisition and transcription only, no valuation or strategy results.',
        sources=URLS, source_discovery='Public web search: title PetroChina annual report and exact share-class strings; HKEX hosts.',
        code_sha256=sha(Path(__file__)), input_sha256={p:sha(ROOT/p) for p in inputs},
        rules=['One request per resource; reuse hash-verified receipt.',
               'Stop all HKEX requests after 403 or 429; retain Retry-After; no retries.',
               'Later documents cannot prove earlier knowability. URL dates are hints only.',
               'Keep period-end counts separate from report approval, release and retrieval dates.',
               'No cap, pool, feature, return or observer changes.']))
    print('Registered four annual sources.')

def validated_plan():
    plan=read(OUT/'PLAN.json')
    assert plan['code_sha256']==sha(Path(__file__))
    for p,h in plan['input_sha256'].items():
        assert sha(ROOT/p)==h, p
    return plan

def collect():
    p=validated_plan()
    # This program is run outside the offline wrapper only for acquisition.
    for receipt in OUT.glob('*.receipt.json'):
        old=read(receipt)
        if old['status'] in (403,429) and old['requested_utc'][:10]==now()[:10]:
            print('HKEX stopped today; no request.');return
    for year,url in p['sources'].items():
        target=OUT/(year+'.pdf'); receipt=OUT/(year+'.receipt.json')
        if receipt.exists():
            old=read(receipt)
            if old.get('body_sha256'):
                assert sha(target)==old['body_sha256']
            continue
        record=dict(url=url,requested_utc=now(),plan_sha256=sha(OUT/'PLAN.json'),provider='HKEX')
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 public historical research'})
        try:
            with urllib.request.urlopen(request,timeout=40) as response:
                body=response.read()
                record.update(status=response.status, final_url=response.url,content_type=response.headers.get('Content-Type'),retry_after=response.headers.get('Retry-After'))
        except urllib.error.HTTPError as e:
            body=e.read()
            record.update(status=e.code,final_url=e.url,content_type=e.headers.get('Content-Type'),retry_after=e.headers.get('Retry-After'))
        except Exception as e:
            body=None
            record.update(status='transport_error',error=type(e).__name__,message=str(e))
        if body is not None:
            with target.open('xb') as f:f.write(body)
            record.update(body_sha256=sha(target),bytes=len(body),pdf_magic=body.startswith(b'%PDF-'))
        record['finished_utc']=now();write(receipt,record)
        print(json.dumps(dict(year=year,status=record['status'],bytes=record.get('bytes'))),flush=True)
        if record['status'] in (403,429):return

def extract():
    plan=validated_plan()
    runtime=ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
    assert sha(runtime)=='cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
    cfg=read(runtime)
    for p,h in cfg['pdf_runtime_hashes'].items():assert sha(Path(p))==h
    sys.path.append(cfg['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    result=dict(created_utc=now(),plan_sha256=sha(OUT/'PLAN.json'),code_sha256=sha(Path(__file__)),
                runtime_sha256=sha(runtime),scope='Text locations and page renderings; visual review and publication-time evidence remain separate.',reports={})
    for year in plan['sources']:
        source=OUT/(year+'.pdf');receipt=read(OUT/(year+'.receipt.json'))
        assert receipt['status']==200 and receipt['pdf_magic'] and receipt['body_sha256']==sha(source)
        assert receipt['plan_sha256']==sha(OUT/'PLAN.json')
        doc=pdfium.PdfDocument(source.read_bytes()); matches=[]
        for index in range(min(30,len(doc))):
            page=doc[index];text=page.get_textpage().get_text_range()
            if all(x in text for x in ('161,922,077,818','21,098,900,000','Pre-movement','Post-movement')):
                base=OUT/(year+'_page%03d'%(index+1))
                txt=base.with_suffix('.txt');png=base.with_suffix('.png')
                assert not txt.exists() and not png.exists()
                txt.write_text(text);page.render(scale=1.5).to_pil().save(png)
                matches.append(dict(pdf_page=index+1,text=str(txt.relative_to(ROOT)),text_sha256=sha(txt),image=str(png.relative_to(ROOT)),image_sha256=sha(png)))
        doc.close()
        assert len(matches)==1,(year,matches)
        result['reports'][year]=dict(source_sha256=sha(source),receipt_sha256=sha(OUT/(year+'.receipt.json')),pages=matches)
    write(OUT/'EXTRACTION.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    {'register':register,'collect':collect,'extract':extract}[sys.argv[1]]()
