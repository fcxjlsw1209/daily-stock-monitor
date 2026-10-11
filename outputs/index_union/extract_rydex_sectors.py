"""Dated 2009 primary-fund classifications; retain company/class and conflicts."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,re
from extract_sp_sector_observations import normalize
P=Path(__file__).resolve().parent

def clean(s):
 s=re.sub(r'\s*[—–-]\s*Class\s+[ABCK]|\s*[—–-]\s*SP ADR|\([^)]*\)|\*',' ',s,flags=re.I)
 return normalize(s)

# Historical printed names/classes reviewed against the 2008/2009 seed and
# issuer alias evidence. Scope is this 2009 report, not a global name dictionary.
OVERRIDES={'broadcom':['BRCM'],'electronic arts':['ERTS'],'life technologies':['LIFE'],'liberty global':['LBTYA'],'comcast':['CMCSA'],'cvs caremark':['CVS'],'wal mart stores':['WMT'],
 'berkshire hathaway':['BRK.B'],'bank of new york mellon':['BK'],
 'monster worldwide':[], 'constellation energy group':[],
 'kraft foods':['KFT'],'hansen natural':['HANS'],'seagate technology':['STX'],
 'patterson companies':['PDCO'],'j b hunt transport services':['JBHT'],
 'baidu com':['BIDU'],'foster wheeler ag':['FWLT'],'ch robinson worldwide':['CHRW'],
 'c h robinson worldwide':['CHRW'],'lowe s companies':['LOW']}

def main():
 source=P/'sources/rydex/2009_sector_facts.json';doc=json.loads(source.read_text())
 names=defaultdict(set)
 for r in json.loads((P/'NDX_INITIAL_RECONSTRUCTION.json').read_text())['mapped_seed']:
  names[clean(r['printed_name'])].add(r['symbol'])
 for r in json.loads((P/'SPDR_SECTOR_CANDIDATES.json').read_text())['observations']:
  names[clean(r['company'])].add(r['symbol'])
 for r in json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text()):
  for name in r['names']+[x['name'] for x in r.get('name_evidence',[])]:names[clean(name)].add(r['symbol'])
 members={r['symbol'] for r in json.loads((P/'input_diagnostics/MANIFEST.json').read_text())['symbol_gaps']}
 rows=[];unmapped=[]
 for r in doc['rows']:
  c=clean(r['company']);matches=OVERRIDES.get(c,sorted(names[c]&members))
  if len(matches)!=1:unmapped.append(dict(r,candidates=matches));continue
  rows.append(dict(symbol=matches[0],company=r['company'],sector=r['sector'],
   effective_session=doc['observed_as_of'],known_at=doc['known_at'],source_file=str(source.relative_to(P)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_url=doc['source_url'],source_line=r['source_line'],fund=r['fund'],classification_observation_verified=True,permanent_identity_verified=False))
 sectors=defaultdict(set)
 for r in rows:sectors[r['symbol']].add(r['sector'])
 conflicts={s:sorted(v) for s,v in sectors.items() if len(v)>1}
 rows=[r for r in rows if r['symbol'] not in conflicts]
 out=dict(observations=rows,unmapped=unmapped,conflicts=conflicts,observation_count=len(rows),distinct_symbols=len({r['symbol'] for r in rows}),warning='Report classifications, not a complete daily GICS history. No modern-sector backfill. Unique name proposals still require dated security resolution.')
 (P/'RYDEX_2009_SECTOR_OBSERVATIONS.json').write_text(json.dumps(out,indent=2)+'\n')
 print({k:v for k,v in out.items() if k not in ['observations','unmapped','warning']})
if __name__=='__main__':main()
