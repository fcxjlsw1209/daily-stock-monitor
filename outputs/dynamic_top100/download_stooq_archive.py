from pathlib import Path
import requests,hashlib,json,zipfile
p=Path(__file__).resolve().parent/'additional_sources';target=p/'kaggle_stooq_2017.zip'
u='https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/price-volume-data-for-all-us-stocks-etfs'
r=requests.get(u,stream=True,timeout=45);r.raise_for_status();total=0;h=hashlib.sha256()
with target.open('wb') as f:
 for chunk in r.iter_content(1024*1024):
  total+=len(chunk)
  if total>700*1024*1024:raise RuntimeError('700MB probe budget exceeded; partial file not usable')
  f.write(chunk);h.update(chunk)
  if total//(50*1024*1024)!=(total-len(chunk))//(50*1024*1024):print('downloaded MB',total//1024//1024,flush=True)
rec=dict(url=u,status=r.status_code,bytes=total,sha256=h.hexdigest(),zip_valid=zipfile.is_zipfile(target));(p/'kaggle_stooq_download.json').write_text(json.dumps(rec,indent=2));print(rec,flush=True)
