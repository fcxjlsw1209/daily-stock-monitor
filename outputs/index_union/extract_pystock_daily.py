"""Read archived daily observations without treating changing adjustment vintages as one return series."""
from pathlib import Path
import hashlib,io,json,tarfile
import pandas as pd
P=Path(__file__).resolve().parent;S=P/'sources/pystock'
def main():
 f=S/'repository.tar.gz';meta=json.loads(f.with_suffix('.provenance.json').read_text())
 if hashlib.sha256(f.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch')
 targets=set('CEPH NIHD MICC LMCA LMCK FOX FOXA NWSA MXIM ALXN ESRX CTXS CA YHOO ACT TWX MON LIFE QRTEA QVCA LINTA KRFT SNDK DELL'.split())
 parts=[];empty=[]
 with tarfile.open(f) as repo:
  for m in repo:
   name=Path(m.name).name
   if not name.endswith('.tar.gz') or 'initial' in name:continue
   with tarfile.open(fileobj=io.BytesIO(repo.extractfile(m).read())) as day:
    if 'prices.csv' not in day.getnames():continue
    try:d=pd.read_csv(day.extractfile('prices.csv'))
    except pd.errors.EmptyDataError:
     empty.append(m.name);continue
    d=d[d.symbol.isin(targets)].copy()
    if len(d):d['archive_date']=name[:8];d['archive_member']=m.name;parts.append(d)
 result=pd.concat(parts,ignore_index=True);folder=S/'daily_extracted';folder.mkdir(exist_ok=True);inventory=[]
 for symbol,d in result.groupby('symbol'):
  # Retain all vintages in the audit; candidate chooses latest observation of each day.
  dest=folder/(symbol+'.all_vintages.csv.gz');d.to_csv(dest,index=False,compression={'method':'gzip','mtime':0})
  chosen=d.sort_values('archive_date').drop_duplicates('date',keep='last').sort_values('date')
  f=folder/(symbol+'.csv');chosen.to_csv(f,index=False)
  f.with_suffix('.provenance.json').write_text(json.dumps(dict(source=meta,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),all_vintages_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),identity_verified=False,adjustment_verified=False,warning='Across-day adjustment vintages may differ; not a certified factor series.'),indent=2)+'\n')
  inventory.append(dict(symbol=symbol,rows=len(chosen),first=chosen.date.min(),last=chosen.date.max()))
 (S/'EMPTY_DAILY_FILES.json').write_text(json.dumps(empty,indent=2)+'\n')
 (S/'DAILY_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n');print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
