"""Independent metadata intersection. No producer imports or price-value access.

Print --rules-only, then bind both returned SHA values for --write. Outputs
are write-once. Opaque hashes do not decode market values; NPZ access is
restricted to dates/symbols. No strategy or numerical factor is run here.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after156_association_source_review.py'
OUT = 'work/after156_association_source_review.json'
D = 'work/after156_delayed_partition.json'
E = 'work/after156_exception_preflight.json'
I = 'work/after156_issuer_gate.json'
W = 'work/after156_filing_warmup.json'
PROTOCOL = 'work/after156_association_protocol.json'
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
SOURCE_ROWS = 'outputs/monthly_revenue_pilot_source156/ROWS.json'
TZ = 'work/venv/lib/python3.9/site-packages/pytz/zoneinfo/America/New_York'
PINS = {
    D:'11fb070a64723752ffedfb81b6fbbb00aacb0ce1ec5e3269e0d560c6e30e9a92',
    E:'ec4d4a2335133de2d3463b20211714942d6999301c5249c9decc8144cb847331',
    I:'5cf48494c55eb48dd37a846ebf2708ba83a0198bfe5287b622e835df37344648',
    W:'535eaba07cff687330be47c998cbd614985f83b771db4de6bcb8d857fb065837',
    PROTOCOL:'9906d41f2aad9f5981030b4264540cdca0f40f8fb5c723e1060ddc32b00fa88e',
    'work/after156_delayed_partition.py':'59e8e55083644bce888c0a8ac5134c9039a0d67b36209e2da529f7dd73d083a4',
    'work/after156_exception_preflight.py':'89452c953ebf76c508182b3f1aabc2ad5fb408ec2c9b09baec3ed7630392d3f3',
    'work/after156_issuer_gate.py':'6023c848fd72c43c9d43b9844a85aa656b784bc1409ba3130875276f1a98ae4f',
    'work/after156_filing_warmup.py':'8353fa41dc5984e7ea0336bae78f873ae0befcf6a6fbc25c10b5591f79a386d5',
    'work/after156_joint_source_review.json':'8ec105d773801393ea8574e282cda7bb5a18d01b59ff713513d1a2d25895dc5c',
    'outputs/index_union/NEW_PRICE_VALIDATION.json':'a2a74ced299329bd29b3a175519bb9b20b71608a97976570ff83a7baa4fb7d56',
}
FIELDS = ('accessionNumber','form','filingDate','reportDate','raw_acceptanceDateTime','primaryDocument')
CONFLICT = ('form','filingDate','reportDate','primaryDocument')
ERAS = [('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122)]
RULES = {
    'version':1,
    'definition':'research_eligible means the metadata-only intersection under the separate F-only association protocol and explicit issuer continuity assumptions. It is not exact acceptance, historical-vintage certification, first earnings news, positive-finite prices, a computed feature, or registration.',
    'scope':'Preserve the original 122x49 coordinates, all 3904 stock rows and 2074 neutral ETF rows; original stock denominators never shrink.',
    'join':'Join four frozen reports by exact (signal_index,candidate_index,symbol,date). Require identical selected snapshot, actual new delayed event IDs, context IDs and original decision cutoff. Never use old issuer event membership.',
    'source':'Rehash the complete declared four-report input closure and independently check raw occurrence fields/receipts, advertised page closure, retrieval lower bound and lexicographic source choice. No source substitution after quality failure.',
    'partition':'Reconstruct all 210 E/B/N labels by interval overwrite in anchor order; anchors are second XNYS session strictly after filingDate. Amendments override a same-anchor bundle. Count only 19 owned E days as a full original bundle. At least two bundles, 38 E days and 80 N days.',
    'exceptions':'Reconstruct receipted CIK/accession variants from all consumed snapshot sources, including other-form variants. Compare every observed form/date/report/document variant and raw-clock alternative to the original signal decision cutoff; conflicting or unparseable metadata only excludes, never rewrites or reanchors.',
    'identity':'Adopt only the declared reviewable same-issuer/provider continuity screen. Independently enforce unique current CIK, exact frozen route, actual event CIK equality; AVGO/DIS/XOM/CHL/SKHY stay excluded, GOOGL full price window starts >=2015-10-22, SHEL >=2022-01-31, Meta uses the explicit META route and bridge, BRK.B remains class B. Current observations are not historical known-at evidence.',
    'warmup':'Recheck only NPZ date/symbol axes for P[t-504:t-42], 463 dates including SPY. Also require metadata-reported bounds of frozen raw routes to cover every required date. Missing reported bounds fail closed; requested dates are not observation bounds. No close/open arrays or raw price values are decoded.',
    'raw_bounds':'Use frozen normalizer first/last bounds for exact base/alias routes, the fixed manifest replacement order, then published44/top35 only where no base route exists, and 2016 left-fill extension bounds only before its frozen join date. A recorded range is only a metadata envelope; no claim all interior values are finite. CHL remains excluded and no Tiingo bounds are fabricated.',
    'floors':'Apply unchanged first88 >=50% and >=12 distinct CIKs; each early53/recent35 >=40% and >=8 distinct CIKs; >=40/26 respective months with >=8 distinct usable issuers. Later34 is diagnostic only.',
    'result':'Write row-level reasons and exact hash-bound evidence references. Independent arithmetic and metadata eligibility do not authorize actual prices or strategy activity; actual common finite mask and unchanged floors must be checked after a registered PLAN.',
}


def require(ok, text):
    if not ok: raise ValueError(text)


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class Inputs:
    def __init__(self):self.hashes={};self.cache={}
    def name(self,name):
        p=Path(name)
        if p.is_absolute():p=p.relative_to(ROOT)
        require('..' not in p.parts and not p.is_absolute(),'Out-of-root path')
        path=ROOT/p
        require(not path.is_symlink() and path.resolve().is_relative_to(ROOT),'Unsafe input path')
        return str(p)
    def bind(self,name,expected=None):
        name=self.name(name);value=sha(ROOT/name)
        require(expected is None or value==expected,'Hash changed: '+name)
        require(name not in self.hashes or self.hashes[name]==value,'Conflicting input: '+name)
        self.hashes[name]=value
        return name
    def obj(self,name,expected=None):
        name=self.bind(name,expected)
        if name not in self.cache:self.cache[name]=json.loads((ROOT/name).read_text())
        return self.cache[name]
    def closure(self,report):
        for name,value in report['input_hashes'].items():self.bind(name,value)
    def recheck(self):
        for name,value in self.hashes.items():require(sha(ROOT/name)==value,'Input mutated: '+name)


def keyed(rows,meta,date_key='signal_date'):
    result={}
    for index,row in enumerate(rows):
        i,j=row['signal_index'],row['candidate_index']
        require(type(i) is int and type(j) is int and 0<=i<122 and 0<=j<49,'Bad coordinate')
        require((i,j) not in result,'Duplicate coordinate')
        require(row['symbol']==meta['symbols'][i][j],'Symbol mismatch')
        if date_key:require(row[date_key]==meta['dates'][i],'Date mismatch')
        result[i,j]=(index,row)
    return result


def partition_oracle(t,events,sessions):
    left,right=t-251,t-42
    grouped=defaultdict(list)
    for event in events:
        a=bisect_right(sessions,event['filingDate'])+1
        require(t-271<=a<=right,'Unexpected anchor in envelope')
        grouped[a].append(event)
    cells={d:('N',None) for d in range(left,right+1)}
    for a,group in sorted(grouped.items()):
        amended=any(x['form'].endswith('/A') for x in group)
        for k in range(21):
            if a+k in cells:cells[a+k]=('B' if amended or k<2 else 'E',a)
    full=sum(not any(e['form'].endswith('/A') for e in group) and
        all(cells.get(a+k)==('E',a) for k in range(2,21)) for a,group in grouped.items())
    labels=''.join(cells[d][0] for d in cells)
    return labels,[cells[d][1] for d in cells],full


def clock_ns(raw,eastern):
    match=re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z',raw) if isinstance(raw,str) else None
    if not match:return None
    try:dt=datetime.strptime(match[1],'%Y-%m-%dT%H:%M:%S')
    except ValueError:return None
    fraction=int((match[2] or '').ljust(9,'0'));epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    def number(v):
        d=v.astimezone(timezone.utc)-epoch
        return (d.days*86400+d.seconds)*10**9+fraction
    values={number(dt.replace(tzinfo=timezone.utc))};valid=False
    for fold in (0,1):
        v=dt.replace(tzinfo=eastern,fold=fold)
        if v.astimezone(timezone.utc).astimezone(eastern).replace(tzinfo=None)==dt:
            values.add(number(v));valid=True
    return values if valid else None


def exception_ok(event,variants,day,cutoff,eastern):
    try:
        require(event['form'] in ('10-Q','10-K','10-Q/A','10-K/A'),'form')
        require(re.fullmatch(r'\d{10}-\d{2}-\d{6}',event['accessionNumber']) is not None,'accession')
        for key in ('filingDate','reportDate'):require(date.fromisoformat(event[key]).isoformat()==event[key],key)
        require(event['reportDate']<=event['filingDate']<=day,'dates')
        require(isinstance(event['primaryDocument'],str) and event['primaryDocument'].strip(),'document')
        require(len({tuple(v[k] for k in CONFLICT) for v in variants})==1,'variants')
        for v in variants:
            require(v['filingDate']<=day,'variant date')
            times=clock_ns(v['raw_acceptanceDateTime'],eastern)
            require(times is not None and max(times)<=cutoff,'clock')
        return bool(variants)
    except (ValueError,TypeError,KeyError):return False


def covered(dates,intervals):
    return bool(intervals) and all(any(a<=d<=b for a,b in intervals) for d in dates)


def issuer_ok(symbol,column,start,event_ciks):
    route=column['direct_frozen_route'] or column['explicit_alias_route'] or column['supplemental_route']
    if symbol in ('AVGO','DIS','XOM','CHL','SKHY') or not route:return False
    ciks=column['expected_current_ciks']
    if len(ciks)!=1 or any(c!=ciks[0] for c in event_ciks):return False
    boundary={'GOOGL':'2015-10-22','SHEL':'2022-01-31'}.get(symbol)
    return boundary is None or start>=boundary


class Tests(unittest.TestCase):
    def test_partition_overlap(self):
        sessions=[(date(2000,1,1)+timedelta(days=k)).isoformat() for k in range(700)]
        event=lambda a,f:dict(filingDate=sessions[a-2],form=f)
        a,b,full=partition_oracle(600,[event(349,'10-Q'),event(359,'10-Q/A'),event(420,'10-K')],sessions)
        self.assertEqual(full,1);self.assertEqual(a[10:31],'B'*21)
    def test_same_anchor_amendment(self):
        ss=[(date(2000,1,1)+timedelta(days=k)).isoformat() for k in range(700)]
        a,_,full=partition_oracle(600,[dict(filingDate=ss[347],form=f) for f in ['10-Q','10-Q/A']],ss)
        self.assertEqual(full,0);self.assertEqual(a[:21],'B'*21)
    def test_bounds_missing_middle(self):self.assertFalse(covered(['a','b','c'],[('a','a'),('c','c')]))
    def test_bounds_union(self):self.assertTrue(covered(['a','b','c'],[('a','b'),('c','d')]))
    def test_required_left_boundary(self):self.assertFalse(covered(['a','b'],[('b','c')]))
    def test_column_successor(self):
        c=dict(direct_frozen_route=[1],explicit_alias_route=[],supplemental_route=None,expected_current_ciks=['1'])
        self.assertFalse(issuer_ok('XOM',c,'2020-01-01',['1']))
        self.assertFalse(issuer_ok('GOOGL',c,'2015-10-21',['1']))
        self.assertTrue(issuer_ok('GOOGL',c,'2015-10-22',['1']))
        self.assertFalse(issuer_ok('AAPL',c,'2020-01-01',['2']))
    def test_no_later34_selection(self):
        def n(rows):return sum(r['research_eligible'] for r in rows if r['signal_index']<88)
        self.assertEqual(n([dict(signal_index=0,research_eligible=True),dict(signal_index=88,research_eligible=False)]),
                         n([dict(signal_index=0,research_eligible=True),dict(signal_index=88,research_eligible=True)]))
    def test_clock_utc_and_wall_bound(self):
        tz=ZoneInfo('America/New_York');times=clock_ns('2020-06-01T17:00:00Z',tz)
        self.assertEqual(max(times)-min(times),4*3600*10**9)
        self.assertIsNone(clock_ns('2020-03-08T02:30:00Z',tz))
        self.assertEqual(len(clock_ns('2020-11-01T01:30:00Z',tz)),3)


def bounds_metadata(src,issuer):
    """Follow fixed loader provenance using metadata envelopes, never bars."""
    base='outputs/index_union/'
    norm=src.obj(base+'NEW_PRICE_VALIDATION.json',PINS[base+'NEW_PRICE_VALIDATION.json'])['files']
    lookup={(r['symbol'],r['provider']):r for r in norm}
    norm_by_raw={src.name(r['source']):r for r in norm}
    manifest=src.obj(base+'input_diagnostics/MANIFEST.json')
    alias=src.obj(base+'alias_resolved_inputs/MANIFEST.json')
    source_hashes=src.obj('outputs/monthly_2016_extension/SOURCE_HASHES.json')
    routes={};intervals=defaultdict(list)
    def add(symbol,r,path,raw,rawsha,low=None,high=None):
        low=low or r.get('first');high=high or r.get('last')
        if not low or not high:return
        require(date.fromisoformat(low).isoformat()==low and date.fromisoformat(high).isoformat()==high,'Bounds format')
        raw=src.name(raw);src.bind(raw,rawsha)
        record=dict(metadata_path=path,metadata_sha256=src.hashes[path],raw_source_path=raw,
                    raw_source_sha256=rawsha,first=low,last=high,claim='reported_bounds_only_not_interior_finiteness')
        intervals[symbol].append((low,high));routes.setdefault(symbol,[]).append(record)
    for symbol,column in issuer['columns'].items():
        provider_symbol='META' if symbol=='research_equity_FB' else symbol
        if symbol=='research_equity_FB':
            choices=[x for x in alias['alias_decisions'] if x['security_id']==symbol]
            require(len(choices)==1 and choices[0]['price_provider_symbol']=='META','Meta must not use FB ETF')
        for route in manifest['price_sources']:
            if route['symbol']!=provider_symbol:continue
            n=lookup.get((provider_symbol,route['provider']))
            if n:
                require(route['input_sha256']==n['normalized_sha256'] and route['input_file']==n['normalized_file'],'Normalizer route mismatch')
                src.bind(base+route['input_file'],route['input_sha256'])
                add(symbol,n,base+'NEW_PRICE_VALIDATION.json',n['source'],n['sha256'])
    # Original loader replaces each named column in this exact fixed order.
    for path in ['outputs/four_factor_rebound/ETF_SOURCES.json','outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']:
        for r in src.obj(path):
            s=r['symbol']
            if s not in issuer['columns'] and s!='SPY':continue
            intervals[s]=[];routes[s]=[]
            raw=src.name(r['file']);n=r if r.get('first') else norm_by_raw.get(raw,{})
            if n:
                require(n.get('sha256',r['sha256'])==r['sha256'],'Manifest raw hash disagreement')
                add(s,n,path if r.get('first') else base+'NEW_PRICE_VALIDATION.json',raw,r['sha256'])
    for folder in ['monthly_published44_reconstruction','monthly_top35_extension']:
        for p in sorted((ROOT/'outputs'/folder/'prices').glob('*.source.json')):
            s=p.name[:-len('.source.json')]
            if s not in issuer['columns'] or intervals[s]:continue
            path=src.name(p);r=src.obj(path);raw=path.replace('.source.json','.csv')
            require(source_hashes.get(str(ROOT/raw))==r['sha256'],'Loader source hash mismatch')
            add(s,r,path,raw,r['sha256'])
    joins={r['symbol']:r['join'] for r in src.obj('outputs/monthly_2016_extension/PRICE_JOINS.json')}
    for p in sorted((ROOT/'outputs/monthly_2016_extension/prices').glob('*.source.json')):
        s=p.name[:-len('.source.json')]
        if s not in issuer['columns']:continue
        path=src.name(p);r=src.obj(path);raw=path.replace('.source.json','.csv')
        require(source_hashes.get(str(ROOT/raw))==r['sha256'],'Extension source hash mismatch')
        if r.get('first') and r.get('last'):
            upper=min(r['last'],(date.fromisoformat(joins[s])-timedelta(days=1)).isoformat())
            if r['first']<=upper:add(s,r,path,raw,r['sha256'],high=upper)
    # Explicit replacement has no observed bounds in the receipts read here.
    intervals['CHL']=[];routes['CHL']=[]
    for p in ['outputs/monthly_factor_search_v2/data_inputs.py','outputs/monthly_guidance_reset/study.py',
              'outputs/monthly_2016_extension/extend.py']:
        src.bind(p)
    return intervals,routes


def main():
    p=argparse.ArgumentParser();p.add_argument('--rules-only',action='store_true');p.add_argument('--write',action='store_true')
    p.add_argument('--rules-sha256');p.add_argument('--script-sha256');args=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    test=unittest.TextTestRunner(verbosity=1).run(suite);require(test.wasSuccessful(),'Synthetic failure')
    rules_sha=digest(RULES);script_sha=sha(ROOT/SELF)
    if args.rules_only:
        require(not args.write,'Rules-only cannot write')
        print(json.dumps(dict(rules_sha256=rules_sha,script_sha256=script_sha,synthetic_tests=test.testsRun,rules=RULES),indent=2));return
    require(args.rules_sha256==rules_sha and args.script_sha256==script_sha,'Bind rules and script before metadata')
    src=Inputs()
    for name,expected in PINS.items():src.bind(name,expected)
    delayed,exceptions,issuer,warmup=[src.obj(n,PINS[n]) for n in (D,E,I,W)]
    for report in (delayed,exceptions,issuer,warmup):src.closure(report)
    require(digest(delayed['rules'])==delayed['rules_sha256'],'Delayed rules')
    require(digest(exceptions['rules'])==exceptions['rules_sha256'],'Exception rules')
    require(digest(issuer['rules'])==issuer['rules_sha256'] and digest(issuer['columns'])==issuer['scope_sha256'],'Issuer rules/scope')
    meta=src.obj(META);sessions=src.obj(CAL)['sessions'];inventory=src.obj(INVENTORY)
    require(len(meta['dates'])==len(meta['symbols'])==122 and all(len(a)==49 for a in meta['symbols']),'122x49 grid')
    require(sessions==sorted(set(sessions)),'Calendar uniqueness')
    expected={(i,j) for i,names in enumerate(meta['symbols']) for j,s in enumerate(names) if s not in meta['etfs']}
    etfs={(i,j) for i,names in enumerate(meta['symbols']) for j,s in enumerate(names) if s in meta['etfs']}
    require(len(expected)==3904 and len(etfs)==2074,'Original denominators')
    indexed=[keyed(r['rows'],meta,'decision_date' if r is exceptions else 'signal_date') for r in (delayed,exceptions,issuer,warmup)]
    require(all(set(k)==expected for k in indexed),'Reports must retain every original stock coordinate')
    for report in (delayed,exceptions):require(set(keyed(report['preserved_etf_slots'],meta,None))==etfs,'ETF coordinates')
    original=keyed(src.obj(SOURCE_ROWS),meta,'decision_date');require(len(original)==5978,'Original cutoff grid')
    eastern=ZoneInfo.from_file(io.BytesIO((ROOT/TZ).read_bytes()),key='America/New_York');src.bind(TZ)
    # Decode only two NPZ axis members. Hashes may stream opaque market bytes.
    import numpy as np
    panel_path='outputs/monthly_2016_extension/PRICES.npz';src.bind(panel_path,warmup['input_hashes'][panel_path])
    with np.load(ROOT/panel_path,allow_pickle=False) as panel:
        require(set(panel.files)=={'dates','symbols','close','opening'},'NPZ schema')
        dates=panel['dates'].astype(str).tolist();symbols=panel['symbols'].astype(str).tolist()
    require(dates==sorted(set(dates)) and len(symbols)==len(set(symbols)),'Unique axes')
    intervals,bound_refs=bounds_metadata(src,issuer)
    proofs={};inv_snapshots={};stock_inventory={s['symbol']:s for s in inventory['stocks']}
    for item in inventory['submission_snapshots']:
        inv_snapshots[item['source']['path']]=item;proofs[item['source']['path']]=item['source']
        for page in item['advertised_pages']:
            if 'source' in page:proofs[page['source']['path']]=page['source']
    catalogs={};variants=defaultdict(list);occurrences=0
    for snapshot,saved in delayed['snapshots'].items():
        catalogs[snapshot]={e['event_id']:e for e in saved['events']}
        rawroot=src.obj(snapshot);item=inv_snapshots[snapshot]
        require(rawroot['filings'].get('files',[])==[p['advertised'] for p in item['advertised_pages']],'Descriptors differ')
        for source in saved['consumed_sources']:
            raw=src.obj(source['path'],source['sha256']);table=raw['filings']['recent'] if 'filings' in raw else raw
            require(all(isinstance(v,list) and len(v)==source['rows'] for v in table.values()),'SEC arrays/count')
            proof=proofs[source['path']]
            if proof['source_receipt_verified']:
                receipt=src.obj(proof['receipt']);require(receipt['sha256']==source['sha256'],'SEC receipt hash')
            for event in saved['events']:
                for loc in event['occurrences']:
                    if loc['source_path']!=source['path']:continue
                    k=loc['row_index'];values=[table['acceptanceDateTime' if f=='raw_acceptanceDateTime' else f][k] for f in FIELDS]
                    require(values==[event[f] for f in FIELDS] and digest([saved['cik'],values])==event['event_id'],'Raw occurrence/event ID')
                    require(proof['source_receipt_verified'] and loc['source_sha256']==source['sha256'],'Occurrence receipt')
        target={e['accessionNumber'] for e in saved['events']}
        # The second pass below uses global keys, to retain nonperiodic variants.
    target_keys={(e['cik'],e['accessionNumber']) for c in catalogs.values() for e in c.values()}
    for snapshot,saved in delayed['snapshots'].items():
        for source in saved['consumed_sources']:
            if not source['source_receipt_verified']:continue
            raw=src.obj(source['path']);table=raw['filings']['recent'] if 'filings' in raw else raw
            for k,accession in enumerate(table['accessionNumber']):
                key=(saved['cik'],accession)
                if key in target_keys:
                    variants[key].append({f:table['acceptanceDateTime' if f=='raw_acceptanceDateTime' else f][k] for f in FIELDS});occurrences+=1
    cross={(c['cik'],c['accessionNumber']):c for c in exceptions['accession_checks']}
    require(set(cross)==set(variants),'Cross-snapshot accession scope')
    for key,values in variants.items():
        require(cross[key]['metadata_conflict']==(len({tuple(v[f] for f in CONFLICT) for v in values})>1),'Cross conflict mismatch')
        require({c['raw'] for c in cross[key]['clock_sanity']}=={v['raw_acceptanceDateTime'] for v in values},'Raw clock variants missing')
    actual_links={(x['symbol'],x['event_id']):x for x in issuer['actual_delayed_event_links']}
    rows=[];label_days=0;all_dates=set(dates)
    for key in sorted(expected):
        idx,d=indexed[0][key];ei,e=indexed[1][key];ii,ir=indexed[2][key];wi,w=indexed[3][key]
        i,j=key;s=d['symbol'];day=meta['dates'][i];t=sessions.index(day);needed=sessions[t-504:t-41]
        require(t==d['signal_session_index'] and len(needed)==463,'Window alignment')
        require(w.get('required_price_start')==needed[0] and w.get('required_price_end')==needed[-1] and w.get('required_prices')==463,'Warmup boundaries')
        axis=(s in symbols and 'SPY' in symbols and set(needed)<=all_dates)
        require(axis==w['axis_ready'],'Axis warmup changed')
        prior=original[key][1]
        require(e['decision_cutoff_ns']==prior['decision_cutoff_ns'],'Original cutoff nanoseconds')
        require(datetime.fromisoformat(e['decision_cutoff'])==datetime.fromisoformat(prior['decision_cutoff']),'Original cutoff timestamp')
        selected=d['selected_snapshot'];require(e['selected_snapshot']==ir['selected_snapshot']==selected,'Selected snapshot differs')
        # Independent advertised-range closure and deterministic choice.
        lo,upper=sessions[t-273],sessions[t-43];hi=(date.fromisoformat(upper)-timedelta(days=1)).isoformat()
        require(d['half_open_filing_envelope']==[lo,upper] and d['inclusive_helper_envelope']==[lo,hi],'Inverse envelope')
        options=[]
        for cik in stock_inventory[s]['exact_current_ciks']:
            paths=stock_inventory[s]['local_snapshot_paths'][cik]
            for candidate in paths:
                item=inv_snapshots[candidate];proof=item['source']
                ok=proof['source_receipt_verified'] and proof.get('retrieved_at') is not None and proof['retrieved_at'][:10]>=hi
                for page in item['advertised_pages']:
                    a=page['advertised']
                    if a['filingFrom']<=hi and a['filingTo']>=lo:
                        ok=ok and 'source' in page and page['source']['source_receipt_verified'] and page.get('count_matches',False)
                if ok:options.append(candidate)
        require(sorted(set(options))==d['eligible_current_snapshots'] and selected==(min(options) if options else None),
            'Closure/source choice: '+str((key,s,options,d['eligible_current_snapshots'])))
        audit=d['audit'];ids=audit['event_ids'] if audit else [];context=audit['context_original_event_ids'] if audit else []
        require(e['event_ids']==ir['actual_delayed_event_ids']==ids and e['context_original_event_ids']==ir['context_original_event_ids']==context,'Actual new event set differs')
        events=[catalogs[selected][eid] for eid in dict.fromkeys(ids+context)] if selected else []
        for event in events:
            link=actual_links.get((s,event['event_id']));require(link is not None,'Missing new issuer event link')
            require(all(link[f]==event[f] for f in ('cik','accessionNumber','form','filingDate','reportDate')),'Issuer event metadata')
        part=d['partition'];partition_pass=False
        if part:
            labels,owners,full=partition_oracle(t,[catalogs[selected][eid] for eid in ids],sessions)
            require(labels==part['labels'] and owners==part['owner_anchor_indices'] and full==part['full19_original_bundles'],'Independent partition mismatch')
            require(part['counts']=={x:labels.count(x) for x in 'EBN'},'Partition counts')
            partition_pass=full>=2 and labels.count('E')>=38 and labels.count('N')>=80
            require(partition_pass==part['hypothetical_partition_pass'],'Partition floor');label_days+=210
        exception_pass=bool(selected and audit['metadata_structurally_consistent'] and all(exception_ok(v,variants[(v['cik'],v['accessionNumber'])],day,e['decision_cutoff_ns'],eastern) for v in events))
        require(exception_pass==e['exception_screen_pass'] and e['prior_partition_pass']==partition_pass,'Independent exception mismatch')
        column=issuer['columns'][s];identity=issuer_ok(s,column,needed[0],[v['cik'] for v in events])
        require(identity==ir['issuer_assumption_screen_candidate'],'Independent issuer screen mismatch')
        require(ir['consumed_price_window_metadata']==[needed[0],needed[-1]],'Issuer price interval')
        bounds=covered(needed,intervals[s]) and covered(needed,intervals['SPY'])
        reasons=[]
        if not selected:reasons.append('SOURCE_CLOSURE_UNAVAILABLE')
        if audit and not audit['metadata_structurally_consistent']:reasons+=audit['structural_issues']
        if not partition_pass:reasons+=part['failure_reasons'] if part else ['NO_STRUCTURAL_PARTITION']
        if not exception_pass:reasons+=e['reasons']
        if not identity:reasons+=ir['issuer_block_reasons']
        if not axis:reasons.append('METADATA_AXIS_WARMUP_UNAVAILABLE')
        if not covered(needed,intervals[s]):reasons.append('STOCK_RAW_SOURCE_REPORTED_BOUNDS_UNAVAILABLE')
        if not covered(needed,intervals['SPY']):reasons.append('SPY_RAW_SOURCE_REPORTED_BOUNDS_UNAVAILABLE')
        eligible=bool(selected and partition_pass and exception_pass and identity and axis and bounds)
        require(eligible==(not bool(reasons)),'Reason/eligibility mismatch')
        refs=[]
        for v in events:
            refs.append(dict(event_id=v['event_id'],is_context_original=v['event_id'] in context,cik=v['cik'],accessionNumber=v['accessionNumber'],
                occurrences=[dict(loc,receipt_path=proofs[loc['source_path']]['receipt'],receipt_sha256=src.hashes[proofs[loc['source_path']]['receipt']]) for loc in v['occurrences']]))
        rows.append(dict(signal_index=i,candidate_index=j,symbol=s,signal_date=day,signal_session_index=t,
            decision_cutoff=e['decision_cutoff'],decision_cutoff_ns=e['decision_cutoff_ns'],research_eligible=eligible,reasons=sorted(set(reasons)),
            selected_snapshot=selected,event_ids=ids,context_original_event_ids=context,source_evidence_refs=refs,
            partition_ref=dict(path=D,sha256=PINS[D],row_index=idx,labels_sha256=hashlib.sha256(part['labels'].encode()).hexdigest() if part else None,
                first_day=t-251,last_day=t-42,counts=part['counts'] if part else None,full19_original_bundles=part['full19_original_bundles'] if part else None),
            issuer_ref=dict(path=I,sha256=PINS[I],row_index=ii),exception_ref=dict(path=E,sha256=PINS[E],row_index=ei),
            warmup_ref=dict(path=W,sha256=PINS[W],row_index=wi,raw_bounds_symbols=[s,'SPY'],axis_ready=axis,reported_bounds_ready=bounds),
            expected_ciks=column['expected_current_ciks'],assumption_category=column['category'],
            checks=dict(source_closed=selected is not None,partition_minima=partition_pass,exception_clear=exception_pass,issuer_continuity_policy=identity,metadata_warmup=axis and bounds)))
    coverage={}
    for name,lo,hi in ERAS:
        group=[r for r in rows if lo<=r['signal_index']<hi];usable=[r for r in group if r['research_eligible']]
        bymonth=defaultdict(set)
        for r in usable:bymonth[r['signal_index']].update(r['expected_ciks'])
        coverage[name]=dict(original_stock_slots=len(group),research_eligible_slots=len(usable),fraction=len(usable)/len(group),
            distinct_issuers=len({c for r in usable for c in r['expected_ciks']}),distinct_symbols=len({r['symbol'] for r in usable}),
            months_with_at_least8_issuers=sum(len(bymonth[k])>=8 for k in range(lo,hi)),
            monthly_issuer_counts={str(k):len(bymonth[k]) for k in range(lo,hi)},reasons=dict(Counter(x for r in group for x in r['reasons'])))
    f=src.obj(PROTOCOL)['gates_before_any_real_factor']['broad_coverage_floor']
    floor_checks=dict(first88_fraction=coverage['first88']['fraction']>=f['minimum_usable_slot_fraction_in_first88'],
        first88_issuers=coverage['first88']['distinct_issuers']>=f['minimum_distinct_issuers_in_first88'])
    for name in ('early53','recent35'):
        floor_checks[name+'_fraction']=coverage[name]['fraction']>=f['minimum_usable_slot_fraction_in_each_early53_and_recent35']
        floor_checks[name+'_issuers']=coverage[name]['distinct_issuers']>=f['minimum_distinct_issuers_in_each_early53_and_recent35']
        floor_checks[name+'_months']=coverage[name]['months_with_at_least8_issuers']>=f['minimum_qualifying_months_in_'+name]
    for name in (SELF,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):src.bind(name,script_sha if name==SELF else None)
    src.recheck()
    result=dict(schema_version=1,stage='INDEPENDENT_F_ONLY_ASSOCIATION_METADATA_SOURCE_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(floor_checks.values()),research_eligibility_verified=True,actual_grid_verified=True,exact_acceptance_certified=False,
        status='METADATA_RESEARCH_ELIGIBILITY_ONLY_UNDER_DECLARED_RECONSTRUCTION_AND_CONTINUITY_ASSUMPTIONS',
        rules=RULES,rules_sha256=rules_sha,script_sha256=script_sha,input_hashes=dict(sorted(src.hashes.items())),
        checks=dict(original_122x49_grid_preserved=True,all_3904_stock_rows_and_2074_etfs_preserved=True,
            all_four_report_inputs_rehashed=True,new_delayed_event_sets_used=True,raw_occurrences_and_receipts_verified=True,
            source_choice_independently_verified=True,all_partitions_independently_reconstructed=True,
            original_cutoffs_and_cross_snapshot_exceptions_verified=True,issuer_policy_independently_reconstructed=True,
            original_date_symbol_axes_verified=True,raw_source_reported_bounds_checked=True,
            no_producer_imports=True,no_price_values_decoded=True,later34_not_used_for_floor=True),
        validation=dict(synthetic_tests=test.testsRun,independently_reconstructed_label_days=label_days,raw_cross_snapshot_occurrences=occurrences,input_files=len(src.hashes)),
        coverage=coverage,coverage_floor_checks=floor_checks,rows=rows,raw_source_bound_evidence=bound_refs,
        preserved_etf_slots=[dict(signal_index=i,candidate_index=j,symbol=meta['symbols'][i][j],signal_date=meta['dates'][i],status='ETF_NEUTRAL_PRESERVED',research_eligible=False,reasons=['ETF_AUXILIARY_NEUTRAL']) for i,j in sorted(etfs)],
        limitations=[RULES['definition'],'Raw source date ranges are reported metadata; interior stock/benchmark gaps and finite values remain unexamined.',
            'The exact SEC public clock and anchor-time public availability remain uncertified. Only observed metadata exceptions are excluded at original decision cutoffs.',
            'All eligible rows inherit the frozen price-panel assumptions and bounded issuer continuity assumptions; no current CIK is asserted to prove an historical epoch.',
            'Actual common finite-mask coverage must pass the same floors after prospective registration, before any selection or return evaluation.'],
        activity=dict(real_price_values_read=0,factors=0,beta_estimates=0,orders=0,returns=0,network_requests=0,registrations=0,frozen_files_modified=0))
    if args.write:
        with (ROOT/OUT).open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],coverage=coverage,coverage_floor_checks=floor_checks,validation=result['validation'],
        output=OUT if args.write else None,output_sha256=sha(ROOT/OUT) if args.write else None,rules_sha256=rules_sha,script_sha256=script_sha),indent=2))


if __name__=='__main__':main()
