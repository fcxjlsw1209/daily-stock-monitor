"""Consolidate reviewed same-equity aliases without splicing price providers.

Choose one source with most observed raw closes for the entire identity; equal
coverage breaks lexically. Overlap must agree in raw units. Retrospective identity
resolution is neutral bookkeeping, not a future price/sector signal.
"""
from collections import defaultdict
import copy
import numpy as np
import pandas as pd
PRICE_FIELDS=('adj','close','opening','high','low','volume','dividends','splits','raw_per_signal')
def consolidate(data,rows,sector_observations):
 groups=defaultdict(list)
 for r in rows:
  if not r.get('retrospective_identity_link_verified'):raise ValueError('Unreviewed alias')
  if r['symbol'] in data['close']:groups[r['security_id']].append(r['symbol'])
 out={k:v.copy() if hasattr(v,'copy') else copy.deepcopy(v) for k,v in data.items()};decisions=[]
 for identity,symbols in groups.items():
  symbols=sorted(set(symbols))
  counts={}
  for s in symbols:
   epoch=pd.Series(False,index=data['close'].index)
   for r in rows:
    if r['security_id']==identity and r['symbol']==s:epoch|=(epoch.index>=r['effective_from'])&(epoch.index<r['effective_to'])
   counts[s]=int((data['close'][s].notna()&epoch).sum())
  provider=min(symbols,key=lambda s:(-int(data['close'][s].notna().sum()) if counts[s]>0 else 0,s))
  raw=data['close'][provider]*data['raw_per_signal'][provider]
  for s in symbols:
   other=data['close'][s]*data['raw_per_signal'][s]
   # A retired ticker can later be assigned to a different issuer. Compare
   # only the dates on which that label identified this reviewed equity.
   epoch=pd.Series(False,index=raw.index)
   for r in rows:
    if r['security_id']==identity and r['symbol']==s:epoch|=(raw.index>=r['effective_from'])&(raw.index<r['effective_to'])
   common=raw.notna()&other.notna()&epoch
   if common.any() and ((raw[common]/other[common]-1).abs()>0.002).any():raise ValueError('Alias raw-price conflict: '+s+' '+provider)
  for field in PRICE_FIELDS:out[field][identity]=data[field][provider]
  # Membership belongs to its actual symbol interval, even though the vendor
  # may publish the whole unchanged equity history under its latest ticker.
  member=pd.Series(False,index=data['close'].index)
  for r in rows:
   if r['security_id']!=identity or r['symbol'] not in symbols:continue
   valid=(member.index>=r['effective_from'])&(member.index<r['effective_to'])
   member.loc[valid]|=data['eligible'].loc[valid,r['symbol']]
  out['eligible'][identity]=member
  sectors=pd.Series(None,index=member.index,dtype=object)
  for r in sorted([r for r in sector_observations if r['symbol'] in symbols],key=lambda r:(r['usable_from'],r['known_at'])):sectors.loc[r['usable_from']:]=r['sector']
  out['sectors'][identity]=sectors
  # Sector benchmarks will be recomputed from dated merged observations.
  out['sector60'][identity]=np.nan
  for field in PRICE_FIELDS+('eligible','sectors','sector60'):out[field]=out[field].drop(columns=symbols)
  out['issuers'][identity]=next(r['issuer_id'] for r in rows if r['security_id']==identity)
  for s in symbols:out['issuers'].pop(s,None)
  decisions.append(dict(security_id=identity,symbols=symbols,price_provider_symbol=provider,valid_price_rows=int(raw.notna().sum())))
 columns=sorted(out['close'].columns)
 for field in PRICE_FIELDS+('eligible','sectors','sector60'):out[field]=out[field].reindex(columns=columns)
 return out,decisions
