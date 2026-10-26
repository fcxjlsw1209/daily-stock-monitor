"""Bounded H.10 source acquisition and offline HTML inspection, no valuations."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169/fed'
HOST = 'www.federalreserve.gov'
BASE = 'https://' + HOST


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def register():
    OUT.mkdir(exist_ok=True)
    target = ROOT / 'outputs/monthly_2016_extension/EARLY_RANKS.json'
    pairs = load(target)
    assert len(pairs) == 53
    write('SOURCE_PLAN.json', dict(
        schema='ptr169_fed_source_plan_v1', registered_utc=now(),
        purpose='Identify official release dates, units and extractability for already-published H.10 rates; no cap, pool, strategy or returns.',
        target_file=str(target.relative_to(ROOT)), target_sha256=sha(target.read_bytes()),
        target_count=53, first_asof=pairs[0]['asof'], last_asof=pairs[-1]['asof'],
        requests=[dict(name='index.html', url=BASE+'/releases/h10/default.htm'),
                  dict(name='sample_20160815.html', url=BASE+'/releases/h10/20160815/')],
        followups='At most three separately registered official source URLs observed in saved source HTML/JS/JSON. No 53-page bulk collection in this task.',
        collector_sha256=sha(Path(__file__).read_bytes()),
        rules=['One attempt per URL, verify and reuse saved responses.',
               '403 or 429 stops this provider for the UTC day; retain Retry-After.',
               'Raw public bodies and request/response metadata are immutable.',
               'Parse offline via run_local.sh; no network in the parser.',
               'A same-day 16:15 ET release cannot be known before same-day US close.',
               'Do not assume a current historical page is the original unrevised release vintage.'],
        strategy_trials_added=0, returns_computed=False))
    print('SOURCE_PLAN registered', sha((OUT/'SOURCE_PLAN.json').read_bytes()))


class Inspector(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base=base; self.links=[]; self.scripts=[]; self.text=[]; self.tables=[]
        self.table=None; self.row=None; self.cell=None

    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag == 'a' and 'href' in a:
            self.links.append(urllib.parse.urljoin(self.base, a['href']))
        if tag == 'script' and 'src' in a:
            self.scripts.append(urllib.parse.urljoin(self.base, a['src']))
        if tag == 'table':
            assert self.table is None, 'Nested table requires explicit parser update'
            self.table=[]
        elif tag == 'tr' and self.table is not None:
            self.row=[]
        elif tag in ('td','th') and self.row is not None:
            self.cell=[]

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split())); self.cell=None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row); self.row=None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table); self.table=None


def verify_plan(planfile):
    plan=load(planfile)
    assert plan['collector_sha256'] == sha(Path(__file__).read_bytes()), 'Collector drift'
    source=load(OUT/'SOURCE_PLAN.json')
    assert sha((ROOT/source['target_file']).read_bytes()) == source['target_sha256']
    if planfile.name != 'SOURCE_PLAN.json':
        assert plan['source_plan_sha256'] == sha((OUT/'SOURCE_PLAN.json').read_bytes())
    return plan


def verify_response(item, planfile):
    receiptfile=OUT/(item['name']+'.receipt.json'); receipt=load(receiptfile)
    assert receipt['url'] == item['url'] and receipt['plan_sha256'] == sha(planfile.read_bytes())
    if receipt.get('body_sha256'):
        body=(OUT/item['name']).read_bytes()
        assert sha(body) == receipt['body_sha256'] and len(body) == receipt['bytes']
    return receipt


def collect(plan_name):
    planfile=OUT/plan_name; plan=verify_plan(planfile)
    assert planfile.parent == OUT and planfile.suffix == '.json'
    stopped=any(r.get('status') in (403,429) and r['requested_utc'][:10] == now()[:10]
                for r in (load(f) for f in OUT.glob('*.receipt.json')))
    for item in plan['requests']:
        if (OUT/(item['name']+'.receipt.json')).exists():
            old=verify_response(item,planfile)
            print(json.dumps(dict(name=item['name'],reused=True,status=old['status'])),flush=True)
            continue
        assert urllib.parse.urlparse(item['url']).hostname == HOST
        assert Path(item['name']).name == item['name']
        if stopped:
            print('Provider stopped for UTC day',flush=True); break
        oldurls={load(f)['url'] for f in OUT.glob('*.receipt.json')}
        assert item['url'] not in oldurls, 'URL already requested under another name'
        receipt=dict(url=item['url'],requested_utc=now(),plan_sha256=sha(planfile.read_bytes()))
        req=urllib.request.Request(item['url'],headers={'User-Agent':'Mozilla/5.0 public source research'})
        try:
            with urllib.request.urlopen(req,timeout=40) as response:
                body=response.read()
                receipt.update(status=response.status,final_url=response.url,
                    content_type=response.headers.get('Content-Type'),retry_after=response.headers.get('Retry-After'),
                    last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
        except urllib.error.HTTPError as error:
            body=error.read(); receipt.update(status=error.code,final_url=error.url,
                    content_type=error.headers.get('Content-Type'),retry_after=error.headers.get('Retry-After'))
        except Exception as error:
            body=None; receipt.update(status='transport_error',error=type(error).__name__,message=str(error))
        if body is not None:
            with (OUT/item['name']).open('xb') as stream: stream.write(body)
            receipt.update(body_sha256=sha(body),bytes=len(body))
        receipt['finished_utc']=now(); write(item['name']+'.receipt.json',receipt)
        stopped=receipt['status'] in (403,429)
        print(json.dumps(dict(name=item['name'],status=receipt['status'],bytes=receipt.get('bytes'),stopped=stopped)),flush=True)


def inspect(output_name):
    assert Path(output_name).name == output_name and output_name.endswith('.json')
    inspected=[]; pins={}
    for planfile in [OUT/'SOURCE_PLAN.json']+sorted(OUT.glob('FOLLOWUP_PLAN_*.json')):
        plan=verify_plan(planfile); pins[planfile.name]=sha(planfile.read_bytes())
        for item in plan['requests']:
            if not (OUT/(item['name']+'.receipt.json')).exists(): continue
            receipt=verify_response(item,planfile)
            pins[item['name']+'.receipt.json']=sha((OUT/(item['name']+'.receipt.json')).read_bytes())
            if receipt['status'] != 200:
                inspected.append(dict(name=item['name'],status=receipt['status'])); continue
            body=(OUT/item['name']).read_bytes(); pins[item['name']]=sha(body)
            text=body.decode('utf-8-sig'); parser=Inspector(receipt.get('final_url',item['url']))
            parser.feed(text)
            normalized=' '.join(' '.join(parser.text).split())
            terms=('4:15','16:15','Monday','previous','noon','revised','revision','CHINA','HONG KONG','Currency','per U.S.')
            snippets=[]
            for term in terms:
                for match in list(re.finditer(re.escape(term),normalized,re.I))[:6]:
                    snippets.append(dict(term=term,text=normalized[max(0,match.start()-120):match.end()+240]))
            literals=[]
            for match in re.finditer(r'''["']([^"'\s]+(?:h10|H10|calendar|Calendar|\.json)[^"'\s]*)["']''',text):
                literal=match.group(1)
                if not literal.startswith(('<','{')):
                    literals.append(urllib.parse.urljoin(receipt.get('final_url',item['url']),literal))
            inspected.append(dict(name=item['name'],url=item['url'],status=200,
                links=sorted(set(parser.links)),scripts=sorted(set(parser.scripts)),
                observed_source_url_literals=sorted(set(literals)),tables=parser.tables,snippets=snippets))
    write(output_name,dict(created_utc=now(),source_pins=pins,sources=inspected,
        caveat='Current HTTP retrieval; historical page date is not proof of an unmodified original vintage.',
        cap_computed=False,pool_computed=False,returns_computed=False))
    print(output_name,sha((OUT/output_name).read_bytes()))


def followup(discovery_name,url,name):
    assert Path(discovery_name).name == discovery_name
    discovery=load(OUT/discovery_name)
    observed={u for item in discovery['sources'] for k in ('links','scripts','observed_source_url_literals') for u in item.get(k,[])}
    assert url in observed, 'Unobserved URL'
    assert urllib.parse.urlparse(url).hostname == HOST and url.startswith('https://')
    assert Path(name).name == name and not (OUT/name).exists()
    plans=list(OUT.glob('FOLLOWUP_PLAN_*.json')); assert len(plans)<3, 'Bounded discovery budget exhausted'
    for file,digest in discovery['source_pins'].items(): assert sha((OUT/file).read_bytes()) == digest
    number=len(plans)+1
    write(f'FOLLOWUP_PLAN_{number}.json',dict(registered_utc=now(),
        source_plan_sha256=sha((OUT/'SOURCE_PLAN.json').read_bytes()),
        collector_sha256=sha(Path(__file__).read_bytes()),
        observed_in=discovery_name,observation_sha256=sha((OUT/discovery_name).read_bytes()),
        requests=[dict(name=name,url=url)],purpose='Official history/release-calendar index discovery only, not bulk rate collection.'))
    print(f'FOLLOWUP_PLAN_{number}.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='phase',required=True)
    sub.add_parser('register')
    c=sub.add_parser('collect'); c.add_argument('--plan',default='SOURCE_PLAN.json')
    i=sub.add_parser('inspect'); i.add_argument('output_name')
    f=sub.add_parser('followup'); f.add_argument('discovery_name'); f.add_argument('url'); f.add_argument('name')
    a=p.parse_args()
    if a.phase == 'register': register()
    elif a.phase == 'collect': collect(a.plan)
    elif a.phase == 'inspect': inspect(a.output_name)
    else: followup(a.discovery_name,a.url,a.name)
