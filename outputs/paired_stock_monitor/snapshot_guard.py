"""Validate every session; recover omitted rows only from verified immutable snapshots."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS=['Open','High','Low','Close','Adj Close']

def missing_days(frame,calendar,cutoff):
    if frame.empty or frame.index.has_duplicates:
        raise ValueError('Empty or duplicate history')
    expected=calendar[(calendar>=frame.index.min())&(calendar<=pd.Timestamp(cutoff))]
    return expected.difference(frame.index)

def restore_rows(frame,old,missing,cutoff):
    # Refuse unit changes: five overlapping sessions must agree, including adjustment basis.
    overlap=frame.index.intersection(old.index)
    anchors=overlap[overlap<missing.min()][-5:]
    if len(anchors)<5 or not np.allclose(frame.loc[anchors,FIELDS],old.loc[anchors,FIELDS],rtol=1e-7,atol=1e-5):
        raise ValueError('Historical price/adjustment basis differs; cannot restore gap')
    if not missing.isin(old.index).all():raise ValueError('Recovery source lacks required sessions')
    after=frame.loc[(frame.index>missing.min())&(frame.index<=pd.Timestamp(cutoff))]
    if (after['Stock Splits'].fillna(0)!=0).any() or (after['Dividends'].fillna(0)!=0).any():
        raise ValueError('Corporate action after gap; manual unit verification required')
    rows=old.loc[missing]
    if not np.isfinite(rows[FIELDS]).all().all() or not (rows[FIELDS]>0).all().all():
        raise ValueError('Invalid recovery prices')
    if (rows['Stock Splits'].fillna(0)!=0).any() or (rows['Dividends'].fillna(0)!=0).any():
        raise ValueError('Corporate action on missing session; manual verification required')
    return pd.concat([frame,rows]).sort_index()

def prepare_snapshot(folder,calendar):
    folder=Path(folder);meta=json.loads((folder/'manifest.json').read_text())
    if meta['failures']:raise ValueError('Incomplete snapshot')
    frames={};gaps={}
    for symbol,digest in meta['hashes'].items():
        p=folder/(symbol+'.csv.gz')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Snapshot modified: '+symbol)
        f=pd.read_csv(p,index_col=0,parse_dates=True);frames[symbol]=f
        missing=missing_days(f,calendar,meta['as_of'])
        if len(missing):gaps[symbol]=missing
    if not gaps:return folder
    provenance=[]
    candidates=sorted(folder.parent.glob('*/manifest.json'),reverse=True)
    for symbol,missing in gaps.items():
        recovered=False
        for manifest_path in candidates:
            source=manifest_path.parent
            if source==folder or source.name.endswith('_gapfix'):continue
            prior=json.loads(manifest_path.read_text())
            if prior.get('failures') or pd.Timestamp(prior['as_of'])>pd.Timestamp(meta['as_of']):continue
            digest=prior.get('hashes',{}).get(symbol);p=source/(symbol+'.csv.gz')
            if not digest or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:continue
            old=pd.read_csv(p,index_col=0,parse_dates=True)
            if not missing.isin(old.index).all():continue
            try:restored=restore_rows(frames[symbol],old,missing,meta['as_of'])
            except ValueError:continue
            frames[symbol]=restored;recovered=True
            provenance.append({'symbol':symbol,'sessions':[str(d.date()) for d in missing],'source_snapshot':source.name,'source_sha256':digest})
            break
        if not recovered:
            from secondary_prices import recover
            try:
                frames[symbol],record=recover(frames[symbol],symbol,missing,meta['as_of'],folder.parent/'secondary_raw')
                provenance.append(record)
            except Exception as exc:
                raise RuntimeError('Missing trading sessions: '+symbol+' '+','.join(str(d.date()) for d in missing)+'; secondary fallback failed: '+str(exc)+'; no trades or new plan') from exc
    dest=folder.with_name(folder.name+'_gapfix')
    if dest.exists():return dest
    staging=folder.with_name(folder.name+'_gapfix_pending')
    if staging.exists():raise RuntimeError('Incomplete recovery directory requires inspection: '+str(staging))
    staging.mkdir();hashes={}
    for symbol,frame in frames.items():
        p=staging/(symbol+'.csv.gz')
        if symbol in gaps:frame.to_csv(p,compression='gzip')
        else:shutil.copyfile(folder/p.name,p)
        hashes[symbol]=hashlib.sha256(p.read_bytes()).hexdigest()
    meta.update(snapshot_id=dest.name,hashes=hashes,parent_snapshot=folder.name,gap_recovery=provenance)
    (staging/'manifest.json').write_text(json.dumps(meta,indent=2));staging.rename(dest)
    return dest

def fallback_seed(cache,cutoff,error):
    """Retain known history after a primary outage; missing tail still requires real quotes."""
    cache=Path(cache);source=Path(json.loads((cache/'latest.json').read_text())['folder'])
    meta=json.loads((source/'manifest.json').read_text())
    if meta['failures'] or pd.Timestamp(meta['as_of'])>pd.Timestamp(cutoff):raise RuntimeError('No usable historical seed for secondary fallback')
    stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'_secondary_seed'
    dest=cache/stamp;dest.mkdir()
    for symbol,digest in meta['hashes'].items():
        src=source/(symbol+'.csv.gz')
        if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise RuntimeError('Historical seed modified: '+symbol)
        shutil.copyfile(src,dest/src.name)
    meta.update(snapshot_id=stamp,as_of=str(pd.Timestamp(cutoff).date()),historical_seed=source.name,primary_error=str(error),required_session_validation=True)
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2))
    return dest
