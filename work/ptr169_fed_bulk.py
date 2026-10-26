"""Registered Fed H.10 archive acquisition and offline rate extraction only."""
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ptr169_fed_sources import Inspector

ROOT=Path(__file__).resolve().parents[1]
FED=ROOT/'outputs/monthly_ptr_pool_source169/fed'
OUT=FED/'bulk'
METHOD_SHA='3bb87095d74a616654bcc8412a911bf39c395cdf6c0b3d93c7d5fb691f8173be'
SOURCE_CODE_SHA='49326c421954785cc23c11e399871e55a361d9941b616e9bb4af0a8bcc3bd0be'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def write(name,value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')


def register():
    OUT.mkdir(exist_ok=True)
    assert sha(FED/'RELEASE_METHOD.json')==METHOD_SHA
    assert sha(ROOT/'work/ptr169_fed_sources.py')==SOURCE_CODE_SHA
    method=load(FED/'RELEASE_METHOD.json')
    for name,digest in method['source_pins'].items(): assert sha(FED/name)==digest,name
    requests=[]
    for item in method['target_release_candidates']:
        requests.append(dict(url=item['release_url'],date=item['release_date'],
                             name=item['release_date'].replace('-','')+'.html',
                             cached=item['release_date']=='2016-08-15'))
    assert len(requests)==len({r['url'] for r in requests})==53
    write('PLAN.json',dict(schema='ptr169_fed_bulk_v1',registered_utc=now(),
        purpose='Collect and parse already-published official H.10 CNY/HKD rates; no valuation or strategy calculation.',
        method_sha256=METHOD_SHA,source_code_sha256=SOURCE_CODE_SHA,code_sha256=sha(Path(__file__)),
        requests=requests,source_pins=method['source_pins'],target_file=method['target_file'],target_sha256=method['target_sha256'],
        cached_body='../sample_20160815.html',cached_receipt='../sample_20160815.html.receipt.json',
        release_rule='Official page release date must be strictly before asof; same-day publication is excluded.',
        quote_rule='Within that release choose the latest observation date where BOTH CNY and HKD are numeric and non-ND. Preserve all original cells; no cross-page or cross-currency filling.',
        timing='Record observation-to-asof and release-to-asof calendar-day staleness. No assumption of contemporaneous fixing.',
        acquisition='One attempt per URL, serial, 0.5 seconds between requests; 403/429 stops entire provider for UTC day; no automatic retries.',
        limitations='Retrieval in 2026 of official unrevised release archive; historical per-release delivery timestamps and outage history not certified.',
        strategy_trials_added=0,cap_computed=False,pool_computed=False,returns_computed=False))
    print('PLAN registered',sha(OUT/'PLAN.json'))


def verify_plan():
    plan=load(OUT/'PLAN.json')
    assert sha(FED/'RELEASE_METHOD.json')==plan['method_sha256']==METHOD_SHA
    assert sha(ROOT/'work/ptr169_fed_sources.py')==plan['source_code_sha256']==SOURCE_CODE_SHA
    assert sha(Path(__file__))==plan['code_sha256'],'bulk code drift'
    for name,digest in plan['source_pins'].items(): assert sha(FED/name)==digest,name
    assert sha(ROOT/plan['target_file'])==plan['target_sha256']
    return plan


def cached_or_new(item):
    if item['cached']:
        return FED/'sample_20160815.html',FED/'sample_20160815.html.receipt.json'
    return OUT/item['name'],OUT/(item['name']+'.receipt.json')


def verify_receipt(item,bodyfile,receiptfile):
    receipt=load(receiptfile)
    assert receipt['url']==item['url']
    if not item['cached']: assert receipt['plan_sha256']==sha(OUT/'PLAN.json')
    if receipt.get('body_sha256'):
        assert sha(bodyfile)==receipt['body_sha256']
        assert bodyfile.stat().st_size==receipt['bytes']
    return receipt


def collect():
    plan=verify_plan(); previous_request=False
    stopped=any(z.get('status') in (403,429) and z['requested_utc'][:10]==now()[:10]
                for z in (load(p) for folder in (FED,OUT) for p in folder.glob('*.receipt.json')))
    for item in plan['requests']:
        bodyfile,receiptfile=cached_or_new(item)
        if receiptfile.exists():
            receipt=verify_receipt(item,bodyfile,receiptfile)
            print(json.dumps(dict(date=item['date'],reused=True,status=receipt['status'])),flush=True)
            continue
        assert not item['cached'],'Missing registered cached sample'
        if stopped:
            print('Fed provider stopped for UTC day',flush=True); break
        if previous_request: time.sleep(.5)
        assert urllib.parse.urlparse(item['url']).hostname=='www.federalreserve.gov'
        receipt=dict(url=item['url'],requested_utc=now(),plan_sha256=sha(OUT/'PLAN.json'))
        req=urllib.request.Request(item['url'],headers={'User-Agent':'Mozilla/5.0 public source research'})
        try:
            with urllib.request.urlopen(req,timeout=40) as response:
                body=response.read(); receipt.update(status=response.status,final_url=response.url,
                    content_type=response.headers.get('Content-Type'),retry_after=response.headers.get('Retry-After'),
                    last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
        except urllib.error.HTTPError as error:
            body=error.read();receipt.update(status=error.code,final_url=error.url,
                    content_type=error.headers.get('Content-Type'),retry_after=error.headers.get('Retry-After'))
        except Exception as error:
            body=None;receipt.update(status='transport_error',error=type(error).__name__,message=str(error))
        if body is not None:
            with bodyfile.open('xb') as stream: stream.write(body)
            receipt.update(body_sha256=sha(bodyfile),bytes=len(body))
        receipt['finished_utc']=now();write(receiptfile.name,receipt)
        previous_request=True;stopped=receipt['status'] in (403,429)
        print(json.dumps(dict(date=item['date'],status=receipt['status'],bytes=receipt.get('bytes'),stopped=stopped)),flush=True)


def observation_date(text,release):
    cleaned=' '.join(text.replace('.','').split())
    possible=[]
    for year in (release.year-1,release.year):
        for fmt in ('%b %d %Y','%B %d %Y'):
            try: day=datetime.strptime(cleaned+' '+str(year),fmt).date()
            except ValueError: continue
            if 0<(release-day).days<=14: possible.append(day)
    possible=sorted(set(possible))
    assert len(possible)==1,('Ambiguous or non-prior observation date',text,release.isoformat())
    return possible[0].isoformat()


def parse_page(html,expected_date,url):
    parser=Inspector(url);parser.feed(html)
    text=' '.join(' '.join(parser.text).split())
    matches=re.findall(r'Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})',text)
    assert len(matches)==1,('Missing/ambiguous release date',matches)
    release=datetime.strptime(matches[0],'%B %d, %Y').date()
    assert release.isoformat()==expected_date,('URL/index/page release disagreement',expected_date,release.isoformat())
    assert 'Rates in currency units per U.S. dollar except as noted' in text,'Unverified rate unit'
    tables=[t for t in parser.tables if t and t[0][:2]==['COUNTRY','CURRENCY']]
    assert len(tables)==1,'Missing/ambiguous COUNTRY/CURRENCY table'
    table=tables[0];dates=[observation_date(s,release) for s in table[0][2:]]
    assert dates==sorted(set(dates)) and len(dates)==5,('Unexpected observation headers',dates)
    currencies={}
    for label,unit,key in [('CHINA, P.R.','YUAN','CNY'),('HONG KONG','DOLLAR','HKD')]:
        rows=[r for r in table if r[0]==label]
        assert len(rows)==1 and rows[0][1]==unit and len(rows[0])==7,('Missing/unexpected currency row',key)
        currencies[key]=[]
        for cell in rows[0][2:]:
            if cell in ('ND','ND.'):
                currencies[key].append(None)
            else:
                assert re.fullmatch(r'\d+(?:\.\d+)?',cell),('Unrecognized rate cell',cell)
                assert Decimal(cell)>0
                currencies[key].append(cell)
    observations=[dict(observation_date=d,cny_per_usd=currencies['CNY'][i],hkd_per_usd=currencies['HKD'][i]) for i,d in enumerate(dates)]
    return dict(release_date=release.isoformat(),unit='local currency per USD',header_cells=table[0],
                observations=observations,ND_cells=sum(v is None for z in currencies.values() for v in z),
                raw_currency_rows={k:next(r for r in table if r[0]==label) for label,k in [('CHINA, P.R.','CNY'),('HONG KONG','HKD')]})


def choose(page,asof):
    assert page['release_date']<asof,'Same-day/future release not eligible'
    valid=[o for o in page['observations'] if o['cny_per_usd'] is not None and o['hkd_per_usd'] is not None]
    assert valid,'No same-date non-ND CNY/HKD observation in this release'
    selected=max(valid,key=lambda o:o['observation_date'])
    assert selected['observation_date']<page['release_date']<asof
    return dict(**selected,release_date=page['release_date'],
        quote_age_calendar_days=(datetime.fromisoformat(asof)-datetime.fromisoformat(selected['observation_date'])).days,
        release_age_calendar_days=(datetime.fromisoformat(asof)-datetime.fromisoformat(page['release_date'])).days,
        ND_observations_skipped=sum(o['observation_date']>selected['observation_date'] for o in page['observations']),
        cross_release_fill=False,cross_currency_date_mix=False)


def parse():
    plan=verify_plan();method=load(FED/'RELEASE_METHOD.json');pages={};errors=[];pins={}
    for item in plan['requests']:
        bodyfile,receiptfile=cached_or_new(item)
        if not receiptfile.exists():
            errors.append(dict(url=item['url'],stage='collection',reason='not_collected'));continue
        receipt=verify_receipt(item,bodyfile,receiptfile)
        pins[str(receiptfile.relative_to(ROOT))]=sha(receiptfile)
        if bodyfile.exists():pins[str(bodyfile.relative_to(ROOT))]=sha(bodyfile)
        if receipt['status']!=200:
            errors.append(dict(url=item['url'],stage='collection',reason=receipt['status']));continue
        try: page=parse_page(bodyfile.read_text(encoding='utf-8-sig'),item['date'],item['url'])
        except (AssertionError,ValueError) as error:
            errors.append(dict(url=item['url'],stage='parse',reason=str(error)));continue
        page.update(url=item['url'],body_sha256=receipt['body_sha256'],retrieved_utc=receipt['requested_utc'])
        pages[item['date']]=page
    selected=[]
    for row in method['target_release_candidates']:
        try:
            assert row['release_date'] in pages,'Selected release unavailable/unverified'
            value=choose(pages[row['release_date']],row['asof'])
            selected.append(dict(signal_date=row['signal_date'],asof=row['asof'],url=row['release_url'],**value))
        except AssertionError as error:
            errors.append(dict(signal_date=row['signal_date'],asof=row['asof'],stage='selection',reason=str(error)))
    for path,digest in pins.items():assert sha(ROOT/path)==digest
    result=dict(schema='ptr169_fed_bulk_rates_v1',created_utc=now(),plan_sha256=sha(OUT/'PLAN.json'),
        code_sha256=sha(Path(__file__)),method_sha256=METHOD_SHA,inputs=pins,parsed_releases=list(pages.values()),
        selected_rates=selected,errors=errors,complete=len(selected)==53 and not errors,
        counts=dict(expected=53,parsed_releases=len(pages),selected=len(selected),errors=len(errors),
                    ND_cells=sum(p['ND_cells'] for p in pages.values())),
        limitations=plan['limitations'],rate_policy=plan['quote_rule'],
        cap_computed=False,pool_computed=False,returns_computed=False)
    write('RATES.json',result)
    print(json.dumps(dict(sha256=sha(OUT/'RATES.json'),complete=result['complete'],**result['counts'])))


class Tests(unittest.TestCase):
    def test_cross_year(self): self.assertEqual(observation_date('Dec. 30',datetime(2017,1,3).date()),'2016-12-30')
    def test_same_year(self): self.assertEqual(observation_date('Aug. 12',datetime(2016,8,15).date()),'2016-08-12')
    def test_future_date_rejected(self):
        with self.assertRaises(AssertionError): observation_date('Aug. 16',datetime(2016,8,15).date())
    def page(self): return dict(release_date='2016-08-15',observations=[dict(observation_date='2016-08-11',cny_per_usd='6.6',hkd_per_usd='7.7'),dict(observation_date='2016-08-12',cny_per_usd=None,hkd_per_usd='7.8')])
    def test_same_day_cutoff(self):
        with self.assertRaises(AssertionError): choose(self.page(),'2016-08-15')
    def test_ND_same_currency_date(self):
        result=choose(self.page(),'2016-08-16');self.assertEqual(result['observation_date'],'2016-08-11');self.assertEqual(result['quote_age_calendar_days'],5)
    def html(self,unit):
        return '<div>Release Date: August 15, 2016</div><div>'+unit+'</div><table><tr>'+''.join('<th>'+x+'</th>' for x in ['COUNTRY','CURRENCY','Aug. 8','Aug. 9','Aug. 10','Aug. 11','Aug. 12'])+'</tr>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in [['CHINA, P.R.','YUAN','6.6','6.6','6.6','ND','6.6'],['HONG KONG','DOLLAR','7.7','7.7','7.7','7.7','7.7']])+'</table>'
    def test_parse_ND_and_unit(self):
        p=parse_page(self.html('Rates in currency units per U.S. dollar except as noted'),'2016-08-15','https://example.test/');self.assertEqual(p['ND_cells'],1)
    def test_wrong_unit_rejected(self):
        with self.assertRaises(AssertionError): parse_page(self.html('US dollars per foreign unit'),'2016-08-15','https://example.test/')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['test','register','collect','parse']);args=parser.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not result.wasSuccessful(): raise SystemExit(1)
    else: globals()[args.phase]()
