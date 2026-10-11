"""Read existing public archive for union candidates omitted by prior extraction."""
from pathlib import Path
import hashlib,json,zipfile,argparse
import pandas as pd
P=Path(__file__).resolve().parent
source=P.parent/'dynamic_top100/additional_sources/quandl_wiki.zip'
meta=json.loads(source.with_suffix('.provenance.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['sha256']
ap=argparse.ArgumentParser();ap.add_argument('--symbols',nargs='+');args=ap.parse_args()
targets=set(args.symbols or ['LMCA','GOLD','FOX','FOXA','LINTA','STRZA','AGN'])
folder=P/'sources/wiki_union';folder.mkdir(exist_ok=True)
parts=[]
with zipfile.ZipFile(source) as z:
 with z.open('WIKI_PRICES.csv') as stream:
  for chunk in pd.read_csv(stream,chunksize=250000):
   part=chunk[chunk.ticker.isin(targets)&chunk.date.ge('2008-01-01')]
   if len(part):parts.append(part)
result=pd.concat(parts,ignore_index=True)
for symbol,d in result.groupby('ticker'):
 f=folder/(symbol+'.csv');d.to_csv(f,index=False)
 f.with_suffix('.provenance.json').write_text(json.dumps(dict(source=meta['source'],archive_sha256=meta['sha256'],archive_member='WIKI_PRICES.csv',sha256=hashlib.sha256(f.read_bytes()).hexdigest(),identity_verified=False),indent=2)+'\n')
 print(symbol,len(d),d.date.min(),d.date.max(),flush=True)
