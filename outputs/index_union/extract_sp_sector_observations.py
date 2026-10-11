"""Recover contemporaneous GICS observations from cached primary S&P tables.

No current-sector backfill. Company names are retained verbatim; a unique name
match is only a mapping proposal, not permanent-security identity certification.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent
SECTORS={'Information Technology':'Technology','Consumer Discretionary':'Consumer Cyclical','Consumer Staples':'Consumer Defensive','Financials':'Financial Services','Health Care':'Healthcare','Industrials':'Industrials','Energy':'Energy','Materials':'Basic Materials','Utilities':'Utilities','Real Estate':'Real Estate','Communication Services':'Communication Services','Telecommunication Services':'Communication Services'}

# Reviewed against the ticker assertions in these same primary announcements.
# Scoped to source, not a global name/ticker alias (ACT/DD/FOX are reused labels).
REVIEWED_NAMES = {
 'sp_change_2011.html': {'Sara Lee':['SLE']},
 '68b4fcd19938fb88.html': {'Eli Lilly':['LLY'],'Xerox':['XRX'],'Sprint Nextel':['S']},
 '4da7f82d9747de8c.html': {'Heinz':['HNZ']},
 '3ecafe70e25f6435.html': {'Facebook':['FB']},
 'b921ef6398a336b9.html': {'Biogen Idec':['BIIB']},
 '6da13df59135e564.html': {'Actavis':['ACT'],'National Oilwell Varco':['NOV']},
 'd9c68d2c3618c945.html': {'The Priceline Group':['PCLN']},
 '677a87372d091ddf.html': {'Hewlett-Packard':['HPQ']},
 '19dbaf92eaab5b51.html': {'du Pont':['DD']},
 '984e6dbb93d3ed2c.html': {'Twenty-First Century Fox':['TFCF','TFCFA']},
 '47d35dcab72f57c2.html': {'Thermo Fisher':['TMO']},
 '025aeaf45d142213.html': {'Salesforce.com':['CRM']},
}

def normalize(name):
    s=re.sub(r'[^a-z0-9 ]',' ',name.lower())
    s=re.sub(r'\b(incorporated|inc|corporation|corp|company|co|plc|limited|ltd|the|class a|class b|common stock)\b',' ',s)
    return ' '.join(s.split())

def extract_legacy(html):
    result=[]
    for table in BeautifulSoup(html,'html.parser').find_all('table'):
        text=' '.join(table.get_text(' ',strip=True).split())
        if not re.search(r'S&P\s*100\s+INDEX',text,re.I) or 'GICS' not in text:continue
        action=None; in_target=False
        for tr in table.find_all('tr'):
            cells=[' '.join(c.get_text(' ',strip=True).split()) for c in tr.find_all(['td','th'],recursive=False)]
            heading=' '.join(cells)
            if re.search(r'S&P.*?INDEX',heading,re.I):
                in_target=bool(re.search(r'S&P\s*100\s+INDEX',heading,re.I)); action=None
            if not in_target:continue
            for c in cells:
                if c in ('ADDED','DELETED'):action=c
            for i,c in enumerate(cells):
                if c in SECTORS and i>0 and action:
                    company=cells[i-1]
                    if company and company not in ('ADDED','DELETED'):
                        result.append(dict(company=company,gics_sector=c,sector=SECTORS[c],action='addition' if action=='ADDED' else 'deletion'))
    return result

def main():
    events=[]
    for file in ['sp100_modern_event_rows.json','sp100_legacy_reviewed_rows.json','sp100_supplemental_reviewed_rows.json']:
        events.extend(json.loads((P/file).read_text()))
    byfile=defaultdict(list)
    for e in events:byfile[e['source_file']].append(e)
    hints=json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text())
    names=defaultdict(set)
    for r in hints:
        for name in r['names']+[x['name'] for x in r.get('name_evidence',[])]:names[r['symbol']].add(normalize(name))
    qa=json.loads((P/'sources/quantiacs/all_assets.json').read_text())
    for r in qa:
        if r.get('symbol') and r.get('name'):names[r['symbol']].add(normalize(r['name']))
    for r in events:
        if r.get('company'):names[r.get('ticker',r.get('ticker_as_announced_or_effective'))].add(normalize(r['company']))
    output=[];unmapped=[]
    for filename,ev in byfile.items():
        raw=(P/filename).read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if any(r['source_sha256']!=sha for r in ev):raise ValueError('Source changed: '+filename)
        rows=extract_legacy(raw.decode(errors='replace')) if filename.endswith('.html') else []
        for r in ev:
            if r.get('sector') in SECTORS:
                rows.append(dict(company=r['company'],gics_sector=r['sector'],sector=SECTORS[r['sector']],action=r['action'],explicit_ticker=r['ticker']))
        for r in rows:
            matches=[]
            reviewed=REVIEWED_NAMES.get(Path(filename).name,{}).get(r['company'])
            for e in ev:
                symbol=e.get('ticker',e.get('ticker_as_announced_or_effective'))
                if e['action']==r['action'] and (r.get('explicit_ticker')==symbol or (symbol in reviewed if reviewed else normalize(r['company']) in names[symbol])):matches.append(e)
            matches={json.dumps(x,sort_keys=True):x for x in matches}
            matched_symbols={e.get('ticker',e.get('ticker_as_announced_or_effective')) for e in matches.values()}
            if (reviewed and matched_symbols!=set(reviewed)) or (not reviewed and len(matches)!=1):
                unmapped.append(dict(**r,source_file=filename,reason='company_to_event_not_unique',match_count=len(matches)));continue
            for e in matches.values():
                output.append(dict(symbol=e.get('ticker',e.get('ticker_as_announced_or_effective')),company=r['company'],sector=r['sector'],gics_sector=r['gics_sector'],effective_session=e['effective_session'],known_at=e['published_at'],action=e['action'],source_file=filename,source_sha256=sha,source_url=e.get('source_url'),classification_observation_verified=True,permanent_identity_verified=False))
    output=list({json.dumps(r,sort_keys=True):r for r in output}.values())
    result=dict(observations=output,unmapped=unmapped,observation_count=len(output),distinct_symbols=len({r['symbol'] for r in output}),complete_sector_history=False,warning='Deletion-table sector is not evidence for all earlier sessions. Classifications become usable only at/after publication and effective date, after permanent identity validation.')
    (P/'SP_HISTORICAL_SECTOR_OBSERVATIONS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['observations','unmapped']}));print('unmapped',len(unmapped))

if __name__=='__main__':main()
