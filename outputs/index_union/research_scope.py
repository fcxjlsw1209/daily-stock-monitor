"""Explicit available-data research assumptions; not a full-history certificate."""
from pathlib import Path
import hashlib,json
import pandas as pd
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evidence(root,record,file_key='source_file',hash_key='source_sha256'):
 p=(root/record[file_key]).resolve()
 if root.resolve() not in p.parents:raise ValueError('Evidence escapes root')
 if sha(p)!=record[hash_key]:raise ValueError('Evidence checksum mismatch: '+str(p))
 return p

def check_nested_sources(value,root=P):
 count=0
 if isinstance(value,dict):
  for prefix in ('','timing_'):
   if prefix+'source_file' in value and prefix+'source_sha256' in value:
    evidence(root,value,prefix+'source_file',prefix+'source_sha256');count+=1
  for v in value.values():count+=check_nested_sources(v,root)
 elif isinstance(value,list):
  for v in value:count+=check_nested_sources(v,root)
 return count

# These have unresolved historical reorganizations or reused ticker identities.
# Quarantine is determined without examining strategy returns.
UNRESOLVED=set('AGN LINTA LMCA QVCA STRZA GENZ NIHD'.split())
RETIRED_NAMES={'CTRX':'catamaran','FWLT':'foster wheeler','NUAN':'nuance',
 'SGEN':'seagen','SHPG':'shire','VMED':'virgin media','WCRX':'warner chilcott',
 'SPLK':'splunk','ANSS':'ansys','WBA':'walgreens','XLNX':'xilinx','CERN':'cerner','CELG':'celgene'}

def identity_links(manifest,current):
 links={};proofs=[]
 for r in current['records']:
  matches=r['primary_current_matches'];ids={v['cik'] for v in matches}
  if len(ids)==1:
   e=matches[0];evidence(P.parent,e)
   links[r['symbol']]='SEC_CIK_'+next(iter(ids))
   proofs.append(dict(symbol=r['symbol'],basis='current_SEC_corroboration_continuity_assumption',source_file=e['source_file'],source_sha256=e['source_sha256']))
 providers={r['symbol']:r['provider'] for r in manifest['price_sources']}
 for r in json.loads((P/'LEGACY_IDENTITY_OVERRIDES.json').read_text()):
  if providers.get(r['symbol'])!=r['provider']:raise ValueError('Legacy identity wrong provider')
  evidence(P.parent,r);links[r['symbol']]=r['issuer_id'];proofs.append(dict(r,basis='retired_SEC_archive_identity'))
 for s,expected in RETIRED_NAMES.items():
  f=P.parent/'dynamic_top100/additional_sources/tiingo'/(s+'_metadata.json')
  d=json.loads(f.read_text())
  if expected not in d['name'].lower():raise ValueError('Retired metadata identity mismatch: '+s)
  if s in providers:
   links[s]='RETIRED_ISSUER_'+s
   proofs.append(dict(symbol=s,basis='dated_retired_vendor_identity',name=d['name'],source_file=str(f.relative_to(P.parent)),source_sha256=sha(f)))
 for s in UNRESOLVED:links.pop(s,None)
 return links,proofs

def apply_scope(data,links):
 out=dict(data,eligible=data['eligible'].copy());excluded=[]
 for s in out['eligible']:
  alias=s.startswith('research_equity_')
  if (s not in links and not alias) or s in UNRESOLVED:
   excluded.append(dict(symbol=s,reason='unresolved_identity',sessions=int(out['eligible'][s].sum())))
   out['eligible'][s]=False
 # Known ambiguity intervals are quarantined, not silently labelled verified.
 # New securities cannot satisfy 253 observations in these intervals anyway.
 intervals=[('KRFT','2015-07-02','2015-07-06'),('KHC','2015-07-02','2015-07-06'),
            ('SOLS','2025-10-30','2025-11-06'),('VSNT','2026-01-05','2026-01-09'),
            ('HONA','2026-06-29','2026-06-29')]
 for s,a,b in intervals:
  if s in out['eligible']:
   n=int(out['eligible'].loc[a:b,s].sum());out['eligible'].loc[a:b,s]=False
   excluded.append(dict(symbol=s,reason='unresolved_membership_interval',start=a,end=b,sessions=n))
 # EA's Aug4 deletion versus Aug5 trading suspension: prohibit Aug4 new signal.
 for s in out['eligible']:
  if s=='EA' or s=='research_equity_ERTS':
   out['eligible'].loc['2026-08-04':'2026-08-04',s]=False
   excluded.append(dict(symbol=s,reason='unresolved_membership_interval',start='2026-08-04',end='2026-08-04'))
 return out,excluded

def assumed_payment(events,dates,lag=3):
 """Research settlement convention, never an observed historical payment date."""
 import copy
 result=copy.deepcopy(events)
 for e in result:
  if e.get('cash_per_share',0)>0 and e.get('payment_session') is None:
   day=pd.Timestamp(e.get('execution_session',e.get('effective_session')))
   later=dates[dates>day]
   if len(later)<lag:raise ValueError('Settlement outside research calendar')
   e['payment_session']=str(later[lag-1].date())
   e['payment_timing_basis']='assumed_T_plus_'+str(lag)+'_exchange_sessions_not_observed'
 return result
