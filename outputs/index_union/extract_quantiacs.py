"""Extract cached public NetCDF, retaining provider identity and action fields.

The close copied into adj_close is a split-adjusted candidate, NOT a certified
cash-dividend total-return series. No performance engine consumes this output.
"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'vendor_netcdf'))
import xarray as xr

def open_prices(path):
    a=xr.open_dataarray(path,decode_times=False)
    units=a.time.attrs['units'];unit,origin=units.split(' since ')
    if unit!='days' or a.time.attrs.get('calendar')!='proleptic_gregorian':
        raise ValueError('Unsupported NetCDF time convention')
    # NumPy2 int32 scalar + pandas2.3 can overflow inside old CF decoder.
    times=pd.Timestamp(origin)+pd.to_timedelta(a.time.values.astype('int64'),unit='D')
    if times.has_duplicates:raise ValueError('Duplicate source dates')
    return a.assign_coords(time=times)

def main():
    folder=P/'sources/quantiacs';out=folder/'extracted';out.mkdir(exist_ok=True)
    ids={r['id']:r for f in ['all_assets.json','ndx_assets_probe.json'] for r in json.loads((folder/f).read_text())}
    result=[]
    for f in sorted(folder.glob('*prices.nc')):
        prov=json.loads(f.with_suffix('.provenance.json').read_text())
        if hashlib.sha256(f.read_bytes()).hexdigest()!=prov['sha256']:raise ValueError('Cache changed')
        if prov['request']['type'] not in ['NDX100','SPX500']:raise ValueError('Use exact index data API contract')
        a=open_prices(f)
        for sid in a.asset.values:
            record=ids[str(sid)];d=a.sel(asset=sid).to_pandas()
            if 'close' not in d:d=d.T
            d=d[d.close.notna()].sort_index().rename(columns={'vol':'volume','divs':'dividend'})
            d.index.name='date';d=d.reset_index();d['date']=d.date.dt.strftime('%Y-%m-%d')
            d['adj_close']=d.close;d['adj_close_is_total_return']=False
            d['provider_security_id']=str(sid);d['provider_issuer_name']=record['name'];d['provider_cik']=record.get('cik')
            dest=out/(record['symbol']+'.csv');d.to_csv(dest,index=False)
            meta=dict(source_file=str(f.relative_to(P)),source_sha256=prov['sha256'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),provider_identity=record,identity_verified=False,adjustment_verified=False,rows=len(d),first=d.date.min(),last=d.date.max())
            dest.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2)+'\n');result.append(meta)
            print(record['symbol'],len(d),d.date.min(),d.date.max())
    (folder/'INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
