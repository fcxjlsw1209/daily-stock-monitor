"""Union existing broad and historical archives; no network or author holdings."""
from pathlib import Path
import json, hashlib

P=Path(__file__).resolve().parent
O=P.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    broad=O/'us_listed_top32_yahoo/broad_caps/COLLECTION.json'
    older=O/'four_factor_rebound/vendor_cap/COLLECTION.json'
    rows=json.loads(broad.read_text())['records'].copy()
    seen={r['url'] for r in rows};added=[]
    aliases={'GOOG':'GOOGL','BRK.A':'BRK-B','BRK.B':'BRK-B','BRK-A':'BRK-B'}
    for r in json.loads(older.read_text())['records']:
        if r['url'] in seen:continue
        s=aliases.get(r['vendor_ticker'],r['vendor_ticker'])
        rows.append(dict(symbol=s,url=r['url'],data_file=r['data_file'],sha256=r['data_sha256'],
                         raw_file=r['source_file'],raw_sha256=r['source_sha256'],
                         historical_publication_vintage_verified=False))
        seen.add(r['url']);added.append(s)
    for r in rows:
        assert sha(r['data_file'])==r['sha256']
        assert sha(r['raw_file'])==r['raw_sha256']
    obj=dict(records=rows,additional_symbols=added,source_manifests={str(f):sha(f) for f in [broad,older]},
             discovery='Union of all cached broad-listing and historical index-union issuer archives; independent of author holdings')
    (P/'CAP_COLLECTION.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('Cached issuer records:',len(rows),'newly included:',len(added))
if __name__=='__main__':main()
