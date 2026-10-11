"""Industry participation/strength versus excess-return scoring, bounded research."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_ordered_shapes'))
import shape_study as shapes
e=shapes.e;previous=e.previous;SHAPE_DESIGN=shapes.design;e.P=P
read,dump,sha=e.read,e.dump,e.sha
NARROW={**{s:'SOXX' for s in ['NVDA','AMD','MU','INTC','AVGO','TSM','ASML','ARM','SKHY','AMAT','LRCX','KLAC']},
        **{s:'IGV' for s in ['MSFT','ORCL','CRM','ADBE','NOW','SAP','PLTR','INTU']}}

def init():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
      hypothesis='Industry rotation may reward a strong industry as well as a stock beating its industry. Prior models mainly used excess return and can penalize strong peers.',
      new_features=['broad sector long252-skip42','broad sector recent63','narrow industry long252-skip42','narrow industry recent63','stock relative to narrow industry long252-skip42','leave-one-out peer breadth aboveMA200','negative change in industry return correlation63vs252'],
      assumptions='Broad historical classifications reused. Explicit SOXX semiconductor and IGV enterprise-software proxy mappings are research hypotheses, not author-disclosed industry taxonomy. ETF gets itself as industry benchmark.',
      core='Same22 families all skipping2months,>=50% total. All industry coefficients together<=20%. Shape/state/ML<=20%,ML<=10%. No new eligibility filter or ticker intercept.',
      grid='22 core families x2 normalizations x3 temperatures(.02,.05,.10) x2 penalties(0,.10)=264 fits; full prior pattern menu and existing market-return Ridge always available with weights allowed0.',
      selection='Two calibration-only leaders on first48, by balanced similarity and ordered-match priority. Later21 reused historical check, not independent OOS. No return selection.',
      safety='Research only; no observer rules/ledgers/plans or source prices changed.'))
    dump('NARROW_PROXY_ASSUMPTIONS.json',NARROW)
    old=read(O/'monthly_ordered_inverse/PROTECTED_BEFORE.json');dump('PROTECTED_BEFORE.json',{f:sha(f) for f in old})

def prepare():
    parent=O/'monthly_ordered_shapes';m=read(parent/'INPUTS.json');px,raw,members,periods,sources,pools=e.load_prices()
    sectors=pd.read_csv(O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False)
    returns=px.pct_change(fill_method=None);ma=px.rolling(200).mean();long=px.shift(42)/px.shift(252)-1;recent=px.pct_change(63,fill_method=None)
    correlations={}
    for b in set(shapes.SECTOR.values())|set(shapes.FOREIGN.values())|set(NARROW.values())|members['theme17'][1]:
        if b in px:correlations[b]=-(returns.rolling(63).corr(returns[b])-returns.rolling(252).corr(returns[b])).abs()
    new=['sector_strength_long','sector_strength_recent','industry_strength_long','industry_strength_recent','relative_narrow_long','peer_breadth200','industry_correlation_stability']
    for norm in ['percentile','magnitude']:
        d=dict(np.load(parent/(norm+'.npz')),**m['packs'][norm]);ext=np.zeros((*d['X'].shape[:2],len(new)));coverage=[]
        for i,(p,ss) in enumerate(zip(periods,d['symbols'])):
            date=p['start'];broad={};narrow={}
            for s in ss:
                if s in members['theme17'][1]:b=s
                elif s in shapes.FOREIGN:b=shapes.FOREIGN[s]
                else:b=shapes.SECTOR.get(sectors.at[date,s]) if s in sectors else None
                broad[s]=b;narrow[s]=NARROW.get(s,b)
            rows=[]
            for s in ss:
                b= broad[s];n=narrow[s];peers=[z for z in ss if z!=s and z not in members['theme17'][1] and narrow[z] is not None and narrow[z]==n]
                breadth=float((px.loc[date,peers]>ma.loc[date,peers]).mean()) if len(peers)>=2 else np.nan
                rows.append([long.at[date,b] if b in long else np.nan,recent.at[date,b] if b in recent else np.nan,
                  long.at[date,n] if n in long else np.nan,recent.at[date,n] if n in recent else np.nan,
                  (1+long.at[date,s])/(1+long.at[date,n])-1 if n in long else np.nan,
                  breadth,correlations[n].at[date,s] if n in correlations else np.nan])
            q=e.engine.normalize(pd.DataFrame(rows,index=ss,columns=new),norm);ext[i,:len(ss)]=q.to_numpy()
            coverage.append(dict(date=date,unknown=[s for s in ss if broad[s] is None],benchmarks=narrow))
        d['X']=np.concatenate([d['X'],ext],axis=-1);d['names']+=new
        np.savez_compressed(P/(norm+'.npz'),**{k:d[k] for k in ['X','eligible','labels','order']})
        m['packs'][norm]={k:d[k] for k in ['symbols','names','norm','coverage']};m['packs'][norm]['industry_benchmarks']=coverage
        sources[str(parent/(norm+'.npz'))]=sha(parent/(norm+'.npz'))
    m['industry_features']=new;dump('INPUTS.json',m);dump('SOURCE_HASHES.json',sources)

def design(d,core,ignored):
    X,spec=SHAPE_DESIGN(d,core,'patterns');new=['sector_strength_long','sector_strength_recent','industry_strength_long','industry_strength_recent','relative_narrow_long','peer_breadth200','industry_correlation_stability']
    extra=[(k,1.,'industry') for k in new]
    xx=np.stack([d['X'][:,:,d['names'].index(k)] for k in new],axis=-1)
    return np.concatenate([X,xx],axis=-1),spec+extra

e.design=design

def train():
    m=read(P/'INPUTS.json');dev=np.array(m['development']);rows=[]
    for norm in ['percentile','magnitude']:
        d=e.pack(norm,m)
        for fi,core in enumerate(m['families']):
            X,spec=design(d,core,True)
            for temp,reg in itertools.product([.02,.05,.10],[0.,.10]):
                w,loss=e.fit(X,d,dev,spec,temp,reg);scores=np.einsum('ijk,k->ij',X,w,optimize=False)
                rows.append(dict(id=len(rows),norm=norm,family=fi,core=core,ml=True,temp=temp,regularization=reg,spec=spec,weights=w.tolist(),loss=loss,calibration=previous.metrics(scores,d,dev)))
            if (fi+1)%4==0:print('Industry fits',len(rows),norm,'families',fi+1,flush=True)
    assert len(rows)==264
    balanced=lambda r:(r['calibration']['selection_score'],r['calibration']['order_exact'],r['calibration']['set_exact'],-r['id'])
    ordered=lambda r:(r['calibration']['order_exact'],r['calibration']['set_exact'],r['calibration']['shared_slots'],r['calibration']['pairs_correct'],-r['id'])
    leaders=dict(balanced=max(rows,key=balanced),ordered=max(rows,key=ordered));dump('TRIALS.json',rows);dump('FROZEN.json',dict(leaders=leaders,trial_count=len(rows),training='First48 only; no return-based selection'))
    print('FROZEN',{k:dict(id=v['id'],core=v['core'],calibration=v['calibration']) for k,v in leaders.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','prepare','train','evaluate']);a=ap.parse_args()
    {'init':init,'prepare':prepare,'train':train,'evaluate':e.evaluate}[a.phase]()
