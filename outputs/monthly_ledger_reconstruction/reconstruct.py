"""Label-based monthly policy research. All fitted policies are retrospective hypotheses."""
from pathlib import Path
import json, sys, hashlib, argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog

P=Path(__file__).resolve().parent; O=P.parent
sys.path.insert(0,str(O/'monthly_path_constraints'))
from study import load,sha,read
sys.path.insert(0,str(O/'monthly_rule_inference'))
from infer import feature_frames

CORE=['mom12_2','mom9_2','mom6_2','aligned_risk12_2','aligned_risk9_2','aligned_risk6_2','arithmetic_sharpe12_2','half_risk12_2']
SIDE=['vol21','vol63','vol126','downside63','industry_relative60','cmf21','cmf63','volume_ratio21_126','near_high252','ma126_distance']
FEATURES=CORE+SIDE
FAMILIES=['universe_percentile','raw_magnitude','asset_class_percentile']
ALIAS={'META':'research_equity_FB'}
def actual(s): return ALIAS.get(s,s)
def display(s): return 'META' if s=='research_equity_FB' else s
def dump(name,obj):
    (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def market_states(px):
    ret=px.pct_change(fill_method=None);vol=ret.SPY.rolling(63).std()
    return pd.DataFrame(dict(spy_trend=px.SPY>px.SPY.rolling(200).mean(),
                             qqq_trend=px.QQQ>px.QQQ.rolling(200).mean(),
                             tech_relative=px.QQQ.pct_change(63,fill_method=None)>px.SPY.pct_change(63,fill_method=None),
                             high_market_vol=vol>vol.rolling(252).median())).astype(float)


def feature_set(px,raw):
    f=feature_frames(px,raw);ret=px.pct_change(fill_method=None)
    for n in [126,189,252]:
        vol=ret.rolling(n-42).std().shift(42)
        f[f'aligned_risk{n//21}_2']=f[f'mom{n//21}_2']/vol.replace(0,np.nan)
    histvol=ret.rolling(210).std().shift(42)
    f['arithmetic_sharpe12_2']=ret.rolling(210).mean().shift(42)/histvol.replace(0,np.nan)
    f['half_risk12_2']=f['mom12_2']/np.sqrt(histvol.replace(0,np.nan))
    return f

def load_inputs():
    px,pools,sources=load()
    raw={}
    for k in ['close','high','low','volume','sector60']:
        f=O/'index_union/alias_resolved_inputs'/f'{k}.csv.gz'
        raw[k]=pd.read_csv(f,index_col=0,parse_dates=True);sources[str(f)]=sha(f)
    manifests=[]
    for f in [O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']:
        manifests+=read(f);sources[str(f)]=sha(f)
    for ent in manifests:
        assert sha(ent['file'])==ent['sha256']
        sources[ent['file']]=ent['sha256'];z=pd.read_csv(ent['file'],index_col=0,parse_dates=True)
        px[ent['symbol']]=z['Adj Close'].reindex(px.index)
        for k,col in [('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:raw[k][ent['symbol']]=z[col].reindex(px.index)
        # Foreign industry proxies declared explicitly, ETF benchmark SPY.
        bench={'TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY'}.get(ent['symbol'],'SPY')
        raw['sector60'][ent['symbol']]=px[bench].pct_change(60,fill_method=None)
    raw={k:v.reindex(index=px.index,columns=px.columns) for k,v in raw.items()}
    f=O/'monthly_reconstruction_v2/foreign/membership.csv.gz';sources[str(f)]=sha(f)
    stock_member=pd.read_csv(f,index_col=0,parse_dates=True).reindex(index=px.index,columns=px.columns,fill_value=False).fillna(False).astype(bool)
    core=[r['symbol'] for r in read(O/'four_factor_rebound/ETF_SOURCES.json')]
    theme=[s for s in core if s not in {'DIA','XLB','XLC','XLRE','XLU','XLP'}]+['SMH','SOXX','IGV','VGT','MTUM','ARKK']
    members={}
    for label,etfs in [('core17',core),('theme17',theme)]:
        member=stock_member.copy();member[etfs]=px[etfs].notna();members[label]=(member,set(etfs))
    f=O/'monthly_supplied_ledger_audit/PERIODS.json';sources[str(f)]=sha(f)
    periods=read(f)
    return px,raw,members,periods,sources

def prepare():
    px,raw,members,periods,sources=load_inputs(); feats=feature_set(px,raw)
    warm=px.notna().rolling(253,min_periods=253).sum().eq(253)&px.gt(0)
    rows={};audits=[];snapshots=[]
    for pool,(member,etfs) in members.items():
        rows[pool]=[]
        for event in periods:
            d=pd.Timestamp(event['start']);target=sorted(actual(s) for s in event['holdings'])
            universe=sorted(px.columns[member.loc[d]])
            table=pd.DataFrame({k:feats[k].loc[d,universe] for k in FEATURES})
            eligible=[s for s in universe if warm.at[d,s] and feats['mom12_2'].at[d,s]>0]
            missing=[display(s) for s in target if s not in universe]
            ineligible=[display(s) for s in target if s in universe and s not in eligible]
            audits.append(dict(pool=pool,date=event['start'],target=[display(s) for s in target],universe_size=len(universe),eligible_size=len(eligible),missing_from_proxy=missing,failed_gate=ineligible))
            q=table.rank(pct=True).fillna(.5)-.5
            lo,hi=table.quantile(.1),table.quantile(.9)
            magnitude=((table-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)-.5
            apart=table.copy()*np.nan
            for is_etf in [True,False]:
                names=[s for s in universe if (s in etfs)==is_etf]
                apart.loc[names]=table.loc[names].rank(pct=True).fillna(.5)-.5
            X={k:t.loc[eligible].to_numpy() for k,t in [('universe_percentile',q),('raw_magnitude',magnitude),('asset_class_percentile',apart)]}
            regime=float(px.at[d,'SPY']>px.SPY.rolling(200).mean().at[d])
            row=dict(date=event['start'],end=event['end'],symbols=eligible,target=target,
                     y=np.array([s in target for s in eligible]),X=X,valid=not missing and not ineligible,regime=regime,
                     raw=table.loc[eligible],universe=universe)
            rows[pool].append(row)
            snapshots.append(dict(pool=pool,date=row['date'],target=target,symbols=eligible,valid=row['valid'],regime=regime,
                                  X={k:v.tolist() for k,v in X.items()},raw=table.loc[eligible].where(table.loc[eligible].notna(),None).to_dict('index')))
    # Store only usable numeric feature matrices; raw NaN represented as null.
    for x in snapshots:
        for vals in x['raw'].values():
            for k,v in vals.items():
                if pd.isna(v): vals[k]=None
    dump('UNIVERSE_AUDIT.json',audits);dump('SNAPSHOTS.json',snapshots);dump('SOURCE_HASHES.json',sources)
    return px,rows,feats,raw,sources

def pick(row,w,family):
    scores=np.einsum('ij,j->i',row['X'][family],w,optimize=False)
    ids=np.argsort(-scores,kind='stable')[:3]
    return [row['symbols'][i] for i in ids],scores

def agreements(rows,w,family):
    details=[]
    for row in rows:
        chosen,scores=pick(row,w,family);pairs=[]
        ids=np.flatnonzero(row['y']);other=np.flatnonzero(~row['y'])
        concordance=float((scores[ids,None]>scores[other][None,:]).mean()) if len(ids) and len(other) else None
        details.append(dict(date=row['date'],target=row['target'],chosen=chosen,shared=len(set(chosen)&set(row['target'])),exact=set(chosen)==set(row['target']),pair_accuracy=concordance,proxy_valid=row['valid']))
    return dict(months=len(rows),exact_months=sum(r['exact'] for r in details),mean_shared=float(np.mean([r['shared'] for r in details])),pair_accuracy=float(np.mean([r['pair_accuracy'] for r in details if r['pair_accuracy'] is not None])),details=details)

def baselines(rows):
    trials=[]
    for pool,rr in rows.items():
        for family in FAMILIES:
            for core in CORE:
                for variation in ['pure','space','stability']:
                    w=np.zeros(len(FEATURES));w[FEATURES.index(core)]=1 if variation=='pure' else .7
                    if variation!='pure':
                        w[FEATURES.index('vol63')]=.15 if variation=='space' else -.15
                        w[FEATURES.index('industry_relative60')]=.1;w[FEATURES.index('cmf21')]=.05
                    result=agreements(rr,w,family)
                    trials.append(dict(pool=pool,family=family,core=core,variation=variation,weights=dict(zip(FEATURES,w.tolist())),**result))
    dump('BASELINE_TRIALS.json',trials)
    best=sorted(trials,key=lambda x:(-x['exact_months'],-x['mean_shared'],-x['pair_accuracy']))[:10]
    dump('BASELINE_LEADERS.json',best)
    return best

def pair_matrix(rows,family):
    arrays=[]
    for row in rows:
        if not row['valid']:continue
        yes,no=np.flatnonzero(row['y']),np.flatnonzero(~row['y']);x=row['X'][family]
        arrays.append((x[yes,None]-x[no][None,:]).reshape(-1,len(FEATURES)))
    return np.vstack(arrays)

def max_margin(D,core_min=.6,active=None):
    c=len(CORE);s=len(SIDE);n=c+2*s+1
    A=np.column_stack([-D[:,:c],-D[:,c:],D[:,c:],np.ones(len(D))])
    b=np.zeros(len(D));dominance=np.zeros(n);dominance[:c]=-1
    A=np.vstack([A,dominance]);b=np.r_[b,-core_min]
    eq=np.ones((1,n));eq[0,-1]=0
    obj=np.zeros(n);obj[-1]=-1
    bounds=[(0,None)]*(n-1)+[(-1,1)]
    if active is not None:
        for j in range(c+s):
            if j in active:continue
            if j<c:bounds[j]=(0,0)
            else:
                bounds[j]=(0,0);bounds[c+s+(j-c)]=(0,0)
    sol=linprog(obj,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1],bounds=bounds,method='highs')
    assert sol.success,sol.message
    w=np.r_[sol.x[:c],sol.x[c:c+s]-sol.x[c+s:c+2*s]]
    margins=np.einsum('ij,j->i',D,w,optimize=False)
    assert margins.min()>=sol.x[-1]-1e-7
    return w,dict(margin=float(sol.x[-1]),core_mass=float(w[:c].sum()),side_l1=float(abs(w[c:]).sum()),weights=dict(zip(FEATURES,w.tolist())))

def diagnosis(rows):
    out=[];windows=[]
    for pool,rr in rows.items():
        for family in FAMILIES:
            good=[r for r in rr if r['valid']]
            w,fit=max_margin(pair_matrix(good,family));fit['agreement']=agreements(rr,w,family)
            locals=[]
            for r in good:
                _,local=max_margin(pair_matrix([r],family))
                locals.append(dict(date=r['date'],**local))
            out.append(dict(pool=pool,family=family,global_fit=fit,locally_explainable=sum(z['margin']>1e-6 for z in locals),local_total=len(locals),local_diagnostics=locals))
            # Uniform annual segments: explanation feasibility only, never stitch their profits.
            for y in range(2021,2027):
                part=[r for r in good if (pd.Timestamp(r['end']).year==y)]
                if not part:continue
                w,ff=max_margin(pair_matrix(part,family))
                windows.append(dict(pool=pool,family=family,holding_year=y,fit=ff,agreement=agreements(part,w,family)))
    dump('LINEAR_FEASIBILITY.json',out);dump('YEAR_FEASIBILITY.json',windows)

if __name__=='__main__':
    px,rows,feats,raw,sources=prepare()
    a=read(P/'UNIVERSE_AUDIT.json')
    print('UNIVERSE',[(k,sum(bool(x['missing_from_proxy'] or x['failed_gate']) for x in a if x['pool']==k)) for k in rows],flush=True)
    best=baselines(rows)
    for r in best[:5]:print('BASELINE',r['pool'],r['family'],r['core'],r['variation'],r['exact_months'],r['mean_shared'],flush=True)
    diagnosis(rows)
    for r in read(P/'LINEAR_FEASIBILITY.json'): print('FEASIBILITY',r['pool'],r['family'],r['locally_explainable'],r['local_total'],r['global_fit']['margin'],flush=True)
    assert all(sha(f)==h for f,h in sources.items())
