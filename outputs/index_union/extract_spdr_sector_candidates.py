"""Map primary dated fund holdings to historical symbols, without backdating.

A unique name match is a proposal; raw company names/industry/line references
remain attached. Financial REIT and telecom classifications are separated from
the combined ETF sector, preventing the known pre-2016/pre-2018 sector error.
"""
from pathlib import Path
import json,hashlib,re
from collections import defaultdict
from extract_sp_sector_observations import normalize
P=Path(__file__).resolve().parent

def clean(s):
    return normalize(re.sub(r'\([^)]*\)|[*]', '',s).replace('’',"'").replace('–','-'))

def main():
    source=P/'sources/spdr_sectors/2011_holdings_facts.json';doc=json.loads(source.read_text());sha=hashlib.sha256(source.read_bytes()).hexdigest()
    names=defaultdict(set)
    for r in json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text()):
        for name in r['names']+[x['name'] for x in r.get('name_evidence',[])]:names[clean(name)].add(r['symbol'])
    for r in json.loads((P/'sources/quantiacs/all_assets.json').read_text()):
        if r.get('symbol') and r.get('name'):names[clean(r['name'])].add(r['symbol'])
    for r in json.loads((P/'NDX_INITIAL_RECONSTRUCTION.json').read_text())['mapped_seed']:
        names[clean(r['printed_name'])].add(r['symbol'])
    for file in ['sp100_modern_event_rows.json','sp100_legacy_reviewed_rows.json','sp100_supplemental_reviewed_rows.json']:
        for r in json.loads((P/file).read_text()):
            if r.get('company'):names[clean(r['company'])].add(r.get('ticker',r.get('ticker_as_announced_or_effective')))
    members={r['symbol'] for r in json.loads((P/'input_diagnostics/MANIFEST.json').read_text())['symbol_gaps']}
    observed=[];unmapped=[]
    for r in doc['rows']:
        candidates=sorted(names[clean(r['company'])]&members)
        overrides={'comcast':['CMCSA'],'cvs caremark':['CVS'],'wal mart stores':['WMT'],
         'berkshire hathaway':['BRK.B'],'bank of new york mellon':['BK'],
         'monster worldwide':[],'constellation energy group':[],'kraft foods':['KFT']}
        candidates=overrides.get(clean(r['company']),candidates)
        # Ambiguous aliases/reorganizations are NOT resolved from modern labels.
        if len(candidates)!=1 or (candidates==['DTV'] and clean(r['company'])==clean('DTE Energy Co.')):
            unmapped.append(dict(r,candidates=candidates));continue
        sector=r['sector'];industry=r['industry']
        if sector=='Technology' and industry=='Telecommunication Services':sector='Communication Services'
        # Fund's financial sector includes REITs at this date; preserve historical
        # financial classification, later dated GICS transition still required.
        observed.append(dict(symbol=candidates[0],company=r['company'],sector=sector,observed_industry=industry,
          effective_session=doc['observed_as_of'],known_at=doc['known_at'],source_file=str(source.relative_to(P)),
          source_sha256=sha,source_url=doc['source_url'],source_line=r['source_line'],
          mapping_method='unique_normalized_historical_name_candidate',permanent_identity_verified=False))
    result=dict(observations=observed,unmapped=unmapped,observation_count=len(observed),
      distinct_symbols=len({r['symbol'] for r in observed}),complete_sector_history=False,
      warning='SEC submissions confirms publication; unique name matching is a proposal, not a permanent security mapping. Combined ETF groups need dated benchmark/classification separation.')
    (P/'SPDR_SECTOR_CANDIDATES.json').write_text(json.dumps(result,indent=2)+'\n')
    print('observations',len(observed),'symbols',result['distinct_symbols'],'unmapped',len(unmapped))
if __name__=='__main__':main()
