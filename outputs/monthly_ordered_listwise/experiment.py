"""Sequential top-three choice reconstruction, with only pre-signal features."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,itertools,json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_ordered_inverse'))
import ordered_study as previous
from ordered_study import load_prices,actual,display,engine,market_states
from study import build_features
from search import features

def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z)+'\n')

def init():
    assert not (P/'PLAN.json').exists()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
      purpose='Continue ordered monthly reconstruction. Direct Plackett-Luce first/second/third choices instead of pairwise squared-margin loss.',
      factors='All core momentum excludes latest42 sessions or2 completed calendar months. Add long log-price trend quality and a few multi-horizon core mixtures. No daily pullback gate.',
      universe='Same reconstructed Top32/theme17; isolated ASML sector60=XLK proxy correction documented by previous control; no author pool additions.',
      model_correction='Ordinary and12-month-half-life Ridge forecasts reuse existing annual five-year market-return-trained alpha10 models, configurations2 and3. Predict with this universe features; combined weight<=10%. No new fitting of those models to author labels.',
      constraints='Nonnegative budget sums1; total core>=50%,volatility<=30%,industry<=20%,all shape/state/acceleration/ML corrections<=20%,acceleration<=10%,ML<=10%.',
      grid='22 core families x2 normalizations x3 softmax temperatures(.02,.05,.10) x2 ridge-to-prior penalties(0,.10) x2 ML settings=528 fits.',
      training='First48 through2024-12-31 only, save before later21. Later period has been seen in earlier rounds; this is research reuse, not untouched OOS.',
      selection='Freeze two leaders using calibration data only: equal-weight slot recall and internal order accuracy; separately exact ordered-month priority. Do not select on return.',
      evaluation='Same-close zero-cost69-month replay, next-open control, author/old candidate comparators; no live promotion.',
      user_steering='No visualization or presentation work; prioritize optimization and verification.'))
    prev=read(O/'monthly_ordered_inverse/PROTECTED_BEFORE.json');dump('PROTECTED_BEFORE.json',{f:sha(f) for f in prev})

def prepare():
    px,raw,members,periods,sources,pools=load_prices()
    raw['sector60']=raw['sector60'].copy();raw['sector60']['ASML']=px.XLK.pct_change(60,fill_method=None)
    feat,cores,states=previous.panels(px,raw);c,s=features(px,raw)
    for k in ['trend_quality252_skip42','trend_quality294_skip42']:
        feat[k]=c[k];cores.append(k)
    for k in ['mom189_skip42','mom189_risk_half']:feat[k]=c[k]
    families=[[k] for k in cores]+[
      ['mom189_skip42','mom252_skip42','mom294_skip42'],
      ['mom189_risk_half','mom252_risk_half','mom294_risk_half'],
      ['mom252_skip42','mom294_skip42','mom252_risk_recent63','mom294_risk_recent63'],
      ['trend_quality252_skip42','trend_quality294_skip42','mom252_skip42','mom294_skip42']]
    assert len(families)==22
    mlmeta=read(O/'monthly_guidance_reset/INPUTS.json');fits=read(O/'monthly_guidance_reset/TRAINING_AUDIT.json')
    mlfeatures,_,_,_,_,mlstates=build_features(px,raw);configs=read(O/'monthly_guidance_reset/MODEL_CONFIGS.json')
    assert configs[2]=={'target':0,'alpha':10.0,'half_life':0} and configs[3]=={'target':0,'alpha':10.0,'half_life':12}
    mm=dict(periods=periods,development=[p['end']<='2024-12-31' for p in periods],families=families,packs={},ml_sources={})
    for n in ['INPUTS.json','TRAINING_AUDIT.json','MODEL_CONFIGS.json']:
        f=O/'monthly_guidance_reset'/n;sources[str(f)]=sha(f)
    for norm in ['percentile','magnitude']:
        d=previous.make_pack(px,raw,periods,pools,members,32,'theme17',feat,cores,states,norm)
        forecasts=np.zeros((*d['X'].shape[:2],2));audit=[]
        for i,(p,ss) in enumerate(zip(periods,d['symbols'])):
            date=p['start'];year=pd.Timestamp(p['end']).year
            q=engine.normalize(pd.DataFrame({k:mlfeatures[k].loc[date,ss] for k in mlmeta['ml_names']}),'percentile')
            for k in ['spy_trend','high_market_vol']:q[k]=mlstates.at[date,k]
            for j,config in enumerate([2,3]):
                fit=next(f for f in fits if f['year']==year and f['config']==config)
                assert fit['latest_label_end']<=fit['cutoff']<=date
                val=np.einsum('ij,j->i',q.to_numpy(),fit['coef'],optimize=False)+fit['intercept']
                good=d['eligible'][i,:len(ss)]
                ranked=pd.Series(val).where(good).rank(pct=True).fillna(.5).to_numpy()-.5
                forecasts[i,:len(ss),j]=ranked
                audit.append(dict(date=date,year=year,config=config,latest_training_label=fit['latest_label_end'],cutoff=fit['cutoff']))
        d['X']=np.concatenate([d['X'],forecasts],axis=-1);d['names']+=['ordinary_ridge','recent_ridge']
        np.savez_compressed(P/(norm+'.npz'),**{k:d[k] for k in ['X','eligible','labels','order']})
        mm['packs'][norm]={k:d[k] for k in ['symbols','names','norm','coverage']};mm['ml_sources'][norm]=audit
    dump('INPUTS.json',mm);dump('SOURCE_HASHES.json',sources)
    print('Prepared22 families; market-only annual Ridge predictions verified',flush=True)

def pack(norm,m=None):
    m=m or read(P/'INPUTS.json');return dict(np.load(P/(norm+'.npz')),**m['packs'][norm])

def design(d,core,ml):
    spec=[(k,1.,'core') for k in core]
    spec+=[('vol126',s,'vol') for s in [1.,-1.]]
    spec+=[('industry_relative60',1.,'industry'),('industry_lagged60',1.,'industry')]
    for k in ['cmf21','cmf63','signed_volume63','highvol_cmf63','trend_industry60']:
        spec+=[(k,s,'shape') for s in [1.,-1.]]
    spec+=[('acceleration_skip42',1.,'accel')]
    if ml:spec+=[('ordinary_ridge',1.,'ml'),('recent_ridge',1.,'ml')]
    X=np.stack([d['X'][:,:,d['names'].index(k)]*s for k,s,g in spec],axis=-1)
    return X,spec

def limits(spec):
    group=np.array([g for _,_,g in spec]);A=[];b=[]
    for groups,cap in [({'vol'},.3),({'industry'},.2),({'shape','accel','ml'},.2),({'accel'},.1),({'ml'},.1)]:
        A.append([float(g in groups) for g in group]);b.append(cap)
    A.append(-(group=='core').astype(float));b.append(-.5)
    prior=np.zeros(len(spec));prior[group=='core']=.65/(group=='core').sum();prior[np.flatnonzero(group=='vol')[0]]=.2
    prior[group=='industry']=.10/(group=='industry').sum();prior[np.flatnonzero(group=='shape')[0]]=.05
    return np.array(A),np.array(b),prior

def events(X,d,mask):
    tensors=[];allowed=[];selected=[];weights=[]
    for i in np.flatnonzero(mask):
        seq=[int(j) for j in d['order'][i] if j>=0];remain=d['eligible'][i].copy()
        for j in seq:
            tensors.append(X[i]);allowed.append(remain.copy());selected.append(j);weights.append(1/len(seq));remain[j]=False
    weights=np.array(weights);weights/=weights.sum()
    return np.stack(tensors),np.stack(allowed),np.array(selected),weights

def likelihood(w,E,allowed,selected,pw,temp,reg,prior):
    score=np.einsum('ijk,k->ij',E,w,optimize=False)/temp
    score=np.where(allowed,score,-np.inf);lz=logsumexp(score,axis=1)
    loss=lz-score[np.arange(len(score)),selected];prob=np.exp(score-lz[:,None])
    grad=(np.einsum('ij,ijk,i->k',prob,E,pw,optimize=False)-np.einsum('ik,i->k',E[np.arange(len(score)),selected],pw,optimize=False))/temp
    return float(np.sum(pw*loss)+reg*np.sum((w-prior)**2)),grad+2*reg*(w-prior)

def fit(X,d,mask,spec,temp,reg):
    E,allowed,selected,pw=events(X,d,mask);A,b,prior=limits(spec)
    result=minimize(likelihood,prior,args=(E,allowed,selected,pw,temp,reg,prior),jac=True,method='SLSQP',bounds=[(0,1)]*len(spec),
       constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))},
         {'type':'ineq','fun':lambda w:b-np.einsum('ij,j->i',A,w,optimize=False),'jac':lambda w:-A}],
       options={'maxiter':400,'ftol':1e-10})
    assert result.success,result.message
    assert np.all(np.einsum('ij,j->i',A,result.x,optimize=False)<=b+1e-7)
    return result.x,float(result.fun)

def train():
    m=read(P/'INPUTS.json');dev=np.array(m['development']);rows=[]
    for norm in ['percentile','magnitude']:
        d=pack(norm,m)
        for fi,core in enumerate(m['families']):
            for ml in [False,True]:
                X,spec=design(d,core,ml)
                for temp,reg in itertools.product([.02,.05,.10],[0.,.10]):
                    w,loss=fit(X,d,dev,spec,temp,reg);score=np.einsum('ijk,k->ij',X,w,optimize=False)
                    rows.append(dict(id=len(rows),norm=norm,family=fi,core=core,ml=ml,temp=temp,regularization=reg,spec=spec,weights=w.tolist(),loss=loss,calibration=previous.metrics(score,d,dev)))
            if (fi+1)%4==0:print('Fits',len(rows),norm,'families',fi+1,flush=True)
    assert len(rows)==528
    balanced=lambda r:(r['calibration']['selection_score'],r['calibration']['order_exact'],r['calibration']['set_exact'],-r['id'])
    ordered=lambda r:(r['calibration']['order_exact'],r['calibration']['set_exact'],r['calibration']['shared_slots'],r['calibration']['pairs_correct'],-r['id'])
    leaders=dict(balanced=max(rows,key=balanced),ordered=max(rows,key=ordered))
    dump('TRIALS.json',rows);dump('FROZEN.json',dict(leaders=leaders,trial_count=len(rows),training='First48 only; no return-based selection'))
    print('FROZEN',{k:dict(id=v['id'],core=v['core'],ml=v['ml'],calibration=v['calibration']) for k,v in leaders.items()},flush=True)

def evaluate():
    m=read(P/'INPUTS.json');frozen=read(P/'FROZEN.json');dev=np.array(m['development']);px,raw,members,periods,sources,pools=load_prices();results={}
    for name,row in frozen['leaders'].items():
        d=pack(row['norm'],m);X,spec=design(d,row['core'],row['ml']);w=np.array(row['weights']);scores=np.einsum('ijk,k->ij',X,w,optimize=False)
        ranks,picks=previous.ranks_and_picks(scores,d['eligible']);orders=[];months=[]
        for i,p in enumerate(periods):
            ss=[d['symbols'][i][j] for j in picks[i]]
            orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=ss,scores=[float(scores[i,j]) for j in picks[i]]))
            months.append(dict(date=p['start'],end=p['end'],author=p['holdings'],ours=list(map(display,ss)),author_ranks=[int(ranks[i,j]) if j>=0 else None for j in d['order'][i]]))
        matching={k:previous.metrics(scores,d,mask) for k,mask in [('all',np.ones(len(dev),bool)),('calibration48',dev),('later21',~dev)]}
        met,eq,trades,holds=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0,execution='close',start=pd.Timestamp(periods[0]['start']),end=pd.Timestamp(periods[-1]['end']))
        np.testing.assert_allclose(eq,previous.independent(px,orders),rtol=1e-11,atol=1e-6)
        op,oe,ot,oh=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0,execution='open',start=pd.Timestamp(periods[0]['start']),end=pd.Timestamp(periods[-1]['end']))
        slices={}
        for k,a,b in [('calibration48','2020-12-31','2024-12-31'),('later21','2024-12-31','2026-09-24')]:
            z=eq.loc[a:b];slices[k]=engine.stats(z/z.iloc[0]*100000.)
        results[name]=dict(metrics=met,next_open=op,slices=slices,matching=matching,months=months,rule=row)
        dump(name+'_HOLDINGS.json',holds);dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
        print(name,'CAGR',met['cagr'],'Sharpe',met['sharpe'],'matching',matching,flush=True)
    dump('RESULTS.json',results)
    protected=read(P/'PROTECTED_BEFORE.json');changes=[f for f,h in protected.items() if sha(f)!=h];assert not changes,changes
    dump('VALIDATION.json',dict(independent_replay=True,annual_market_ridge_training_label_cutoffs_verified=True,protected_files=len(protected),protected_changes=changes))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','prepare','train','evaluate']);a=ap.parse_args();globals()[a.phase]()
