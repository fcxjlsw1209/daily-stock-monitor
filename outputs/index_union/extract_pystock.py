"""Extract missing-symbol candidates from pinned public historical archive.

No security identity or adjustment certification is inferred from ticker text.
"""
from pathlib import Path
import hashlib,json,tarfile
import pandas as pd
P=Path(__file__).resolve().parent
S=P/'sources/pystock'
def main():
 targets=set('CEPH NIHD MICC LMCA LMCK FOX FOXA NWSA MXIM ALXN ESRX CTXS CA YHOO ACT TWX MON LIFE QRTEA QVCA LINTA KRFT SNDK DELL'.split())
 parts=[];sources=[]
 for f in sorted(S.glob('*_initial.tar.gz')):
  meta=json.loads(f.with_suffix('.provenance.json').read_text())
  if hashlib.sha256(f.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch')
  sources.append(meta)
  with tarfile.open(f) as t:
   for chunk in pd.read_csv(t.extractfile('prices.csv'),chunksize=250000):
    selected=chunk[chunk.symbol.isin(targets)]
    if len(selected):parts.append(selected)
 result=pd.concat(parts,ignore_index=True).drop_duplicates()
 folder=S/'extracted';folder.mkdir(exist_ok=True)
 inventory=[]
 for symbol,d in result.groupby('symbol'):
  d=d.sort_values('date');dest=folder/(symbol+'.csv');d.to_csv(dest,index=False)
  meta=dict(sources=sources,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),identity_verified=False,adjustment_verified=False)
  dest.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
  inventory.append(dict(symbol=symbol,rows=len(d),first=d.date.min(),last=d.date.max(),duplicate_dates=int(d.date.duplicated().sum())))
 (S/'INVENTORY.json').write_text(json.dumps(dict(files=inventory,missing=sorted(targets-set(result.symbol))),indent=2)+'\n')
 print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
