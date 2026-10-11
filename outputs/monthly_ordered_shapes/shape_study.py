"""Monthly chase/volume-pattern and aligned-industry corrections, no entry gate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_ordered_listwise'))
import experiment as e
from experiment import previous,load_prices,engine
BASE_DESIGN=e.design
e.P=P
read,dump,sha=e.read,e.dump,e.sha

SECTOR={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE','Financial Services':'XLF',
        'Industrials':'XLI','Technology':'XLK','Consumer Defensive':'XLP','Real Estate':'XLRE','Utilities':'XLU',
        'Healthcare':'XLV','Consumer Cyclical':'XLY'}
FOREIGN={'ASML':'XLK','ARM':'XLK','TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY','SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF','NVS':'XLV','SNDK':'XLK'}

def shape_features(px,raw,sectors,etfs):
    r=px.pct_change(fill_method=None);v=r.rolling(63).std();scale=px/raw['close']
    high=raw['high']*scale;low=raw['low']*scale;opening=raw['opening']*scale;volume=raw['volume'];prev=px.shift(1)
    tr=np.maximum(high-low,np.maximum((high-prev).abs(),(low-prev).abs()));atr=tr.rolling(20).mean().replace(0,np.nan)
    position=((px-low)/(high-low).replace(0,np.nan)).clip(0,1)
    relative_volume=volume/volume.rolling(20).mean().shift(1).replace(0,np.nan)
    shake=(low.lt(low.shift(1).rolling(20).min())&position.gt(.6)).astype(float)*relative_volume.clip(0,4)*position
    compression=(1-(high.rolling(5).max()-low.rolling(5).min())/(4*atr)).clip(0,1)
    dry=(1-volume.rolling(5).mean()/volume.rolling(20).mean().shift(1)).clip(0,1)*compression
    failed=(high.gt(prev)&px.lt(prev)).astype(float)*((high-px)/atr).clip(0,3)
    log=np.log(px)
    out=dict(anti_recent42=-(log-log.shift(42))/(v*np.sqrt(42)),
        anti_extension200=-(log-np.log(px.rolling(200).mean()))/(v*np.sqrt(200)),
        anti_gap5=-(opening/prev-1).clip(lower=0).rolling(5).max()/v.replace(0,np.nan),
        shakeout5=shake.rolling(5).max(),dry_consolidation=dry,anti_failed_rebounds5=-failed.rolling(5).sum(),
        signed_volume_trend=(np.sign(r)*volume).rolling(126).sum()/volume.rolling(126).sum(),
        long_path_quality=r.rolling(210).sum().shift(42)/r.abs().rolling(210).sum().shift(42))
    # Classification is read at the signal date. Benchmark returns may extend
    # backward, but no future sector observation is backfilled into that date.
    for n in [126,252]:
        mom=px.shift(42)/px.shift(n)-1
        rel=pd.DataFrame(np.nan,index=px.index,columns=px.columns)
        for name,benchmark in SECTOR.items():
            bench=mom[benchmark]
            for s in px:
                if s not in sectors:continue
                mask=sectors[s].eq(name).reindex(px.index,fill_value=False)
                rel.loc[mask,s]=(1+mom.loc[mask,s])/(1+bench.loc[mask])-1
        for s,benchmark in {**{s:'SPY' for s in etfs},**FOREIGN}.items():
            if s in rel and benchmark in mom:rel[s]=(1+mom[s])/(1+mom[benchmark])-1
        out['industry_long'+str(n)]=rel
    return {k:z.replace([np.inf,-np.inf],np.nan) for k,z in out.items()}

def init():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
      hypothesis='A small penalty for recent overextension/gaps and monthly price-volume patterns may explain selling recent winners despite high older momentum. Long sector-relative momentum uses the same42-session gap as core.',
      family='Reuse22 core families. Two predeclared correction menus: chase-only and chase+volume/path patterns. All purely score corrections, never require current pullback.',
      grid='22 core families x2 normalizations x2 menus x2 temperatures(.05,.10) x2 penalties(0,.10)=352 fits. Annual market Ridge <=10% remains available but may get zero weight.',
      limits='Same>=50% long core,vol<=30%,industry<=20%,all corrections<=20%. Direction of chase penalty is fixed negative, recovery/dry-volume positive; no arbitrary ticker or month identifiers.',
      selection='Two calibration-only leaders, same similarity and ordered-match criteria. First48/later21 reused historical check, not independent OOS. Returns never used to choose leaders.',
      source_limits='Sector panel is the inherited historical proxy. Added foreign mappings are explicit broad-sector research assumptions. No claim of complete certified historical sector classifications.',
      frozen_accounts='No changes. Zero-cost research replay only. No presentation work.'))
    protected=read(O/'monthly_ordered_inverse/PROTECTED_BEFORE.json');dump('PROTECTED_BEFORE.json',{f:sha(f) for f in protected})

def prepare():
    parent=O/'monthly_ordered_listwise';m=read(parent/'INPUTS.json');px,raw,members,periods,sources,pools=load_prices()
    sf=O/'index_union/alias_resolved_inputs/sectors.csv.gz';sectors=pd.read_csv(sf,index_col=0,parse_dates=True,low_memory=False)
    shape=shape_features(px,raw,sectors,members['theme17'][1]);sources[str(sf)]=sha(sf)
    for norm in ['percentile','magnitude']:
        d=dict(np.load(parent/(norm+'.npz')),**m['packs'][norm]);ext=np.zeros((*d['X'].shape[:2],len(shape)))
        coverage=[]
        for i,(p,ss) in enumerate(zip(periods,d['symbols'])):
            tab=pd.DataFrame({k:z.loc[p['start'],ss] for k,z in shape.items()});q=engine.normalize(tab,norm)
            ext[i,:len(ss)]=q.to_numpy();coverage.append(dict(date=p['start'],missing_industry_long=[s for s in ss if pd.isna(tab.at[s,'industry_long252'])]))
        d['X']=np.concatenate([d['X'],ext],axis=-1);d['names']+=list(shape)
        np.savez_compressed(P/(norm+'.npz'),**{k:d[k] for k in ['X','eligible','labels','order']})
        m['packs'][norm]={k:d[k] for k in ['symbols','names','norm','coverage']};m['packs'][norm]['shape_coverage']=coverage
        sources[str(parent/(norm+'.npz'))]=sha(parent/(norm+'.npz'))
    sources[str(parent/'INPUTS.json')]=sha(parent/'INPUTS.json');dump('SOURCE_HASHES.json',sources);dump('INPUTS.json',m)

def design(d,core,menu):
    X,spec=BASE_DESIGN(d,core,True)
    extra=[('industry_long126',1.,'industry'),('industry_long252',1.,'industry')]
    extra +=[(k,1.,'shape') for k in ['anti_recent42','anti_extension200','anti_gap5']]
    if menu=='patterns':extra +=[(k,1.,'shape') for k in ['shakeout5','dry_consolidation','anti_failed_rebounds5','signed_volume_trend','long_path_quality']]
    xx=np.stack([d['X'][:,:,d['names'].index(k)]*s for k,s,g in extra],axis=-1)
    return np.concatenate([X,xx],axis=-1),spec+extra

e.design=design

def train():
    m=read(P/'INPUTS.json');dev=np.array(m['development']);rows=[]
    for norm in ['percentile','magnitude']:
        d=e.pack(norm,m)
        for fi,core in enumerate(m['families']):
            for menu,temp,reg in itertools.product(['chase','patterns'],[.05,.10],[0.,.10]):
                X,spec=design(d,core,menu);w,loss=e.fit(X,d,dev,spec,temp,reg);scores=np.einsum('ijk,k->ij',X,w,optimize=False)
                rows.append(dict(id=len(rows),norm=norm,family=fi,core=core,ml=menu,temp=temp,regularization=reg,spec=spec,weights=w.tolist(),loss=loss,calibration=previous.metrics(scores,d,dev)))
            if (fi+1)%4==0:print('Pattern fits',len(rows),norm,'families',fi+1,flush=True)
    assert len(rows)==352
    balanced=lambda r:(r['calibration']['selection_score'],r['calibration']['order_exact'],r['calibration']['set_exact'],-r['id'])
    ordered=lambda r:(r['calibration']['order_exact'],r['calibration']['set_exact'],r['calibration']['shared_slots'],r['calibration']['pairs_correct'],-r['id'])
    leaders=dict(balanced=max(rows,key=balanced),ordered=max(rows,key=ordered));dump('TRIALS.json',rows);dump('FROZEN.json',dict(leaders=leaders,trial_count=len(rows),training='First48 only; no return-based selection'))
    print('FROZEN',{k:dict(id=v['id'],core=v['core'],menu=v['ml'],calibration=v['calibration']) for k,v in leaders.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','prepare','train','evaluate']);a=ap.parse_args()
    {'init':init,'prepare':prepare,'train':train,'evaluate':e.evaluate}[a.phase]()
