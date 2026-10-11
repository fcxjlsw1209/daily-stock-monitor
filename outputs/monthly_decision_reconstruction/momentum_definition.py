"""Test the ambiguous momentum window at the current decision frontier only."""
import numpy as np
import pandas as pd
from itertools import product
from frontier import load_first_months, solve, choices
from decisions import P, O, read, dump, sha, CORE, FEATURES, STRICT, display
from study import load


def main():
    rows=load_first_months()
    decisions=read(P/'DECISIONS.json')[:len(rows)]
    px,_,sources=load()
    vol=px.pct_change(fill_method=None).rolling(63).std()
    manifests=read(O/'four_factor_rebound/ETF_SOURCES.json')+read(O/'monthly_reconstruction_v2/ETF_MANIFEST.json')
    etfs={v['symbol'] for v in manifests}
    moments={n:px.shift(42)/px.shift(n+42)-1 for n in [126,189,252]}
    rank_changes=[]
    for row,old in zip(rows,decisions):
        date=row['date'];universe=old['universe'];names=row['symbols']
        raw=pd.DataFrame({f'mom{n//21}_2':m.loc[date,universe] for n,m in moments.items()})
        for n,m in moments.items():raw[f'risk{n//21}_2']=m.loc[date,universe]/vol.loc[date,universe]
        raw=raw[CORE]
        assert raw.loc[names].notna().all().all()
        percentile=raw.rank(pct=True)
        lo,hi=raw.quantile(.1),raw.quantile(.9)
        magnitude=((raw-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)
        separate=raw.copy()*np.nan
        for flag in [False,True]:
            members=[s for s in universe if (s in etfs)==flag]
            separate.loc[members]=raw.loc[members].rank(pct=True).fillna(.5)
        for family,values in [('universe_percentile',percentile),('raw_magnitude',magnitude),('asset_class_percentile',separate)]:
            row['X'][family][:,:len(CORE)]=values.loc[names].to_numpy()-.5
        rank_changes.append(dict(date=date,selected={s:{k:float(raw.at[s,k]) for k in CORE} for s,y in zip(names,row['y']) if y}))
    results={}
    for family,dominance,interaction in product(['universe_percentile','raw_magnitude','asset_class_percentile'],['half','largest_group'],[False,True]):
        name=f'{family}__{dominance}__'+('small_interaction' if interaction else 'linear')
        history=[];prior=None;exact=0
        for i,row in enumerate(rows):
            prediction=None
            if prior:
                picked,_=choices(row,prior,family,interaction)
                prediction=dict(trained_through=rows[i-1]['date'],picks=picked,
                    overlap=len(set(picked)&set(np.array(row['symbols'])[row['y']].tolist())))
            fit=solve(rows[:i+1],family,dominance,interaction)
            history.append(dict(date=row['date'],prior_prediction=prediction,fit=fit))
            if fit['margin']<=STRICT:break
            exact=i+1;prior=fit
            for before in rows[:i+1]:
                picked,_=choices(before,fit,family,interaction)
                assert set(picked)==set(np.array(before['symbols'])[before['y']])
        results[name]=dict(exact_prefix_months=exact,history=history,
            first_conflict=None if exact==len(rows) else history[-1]['date'])
        print(name,exact,history[-1]['fit']['margin'],flush=True)
    dump('FULL_WINDOW_LAG2_RESULTS.json',results)
    dump('FULL_WINDOW_LAG2_FACTORS.json',rank_changes)
    sources.update(read(P/'SOURCE_HASHES.json'))
    for f,h in sources.items():assert sha(f)==h,f
    dump('FULL_WINDOW_LAG2_STATUS.json',dict(maximum_common_prefix=max(r['exact_prefix_months'] for r in results.values()),
        full_period_backtest_run=False,candidate_identities_unchanged=True,side_factors_unchanged=True,
        latest_42_sessions_still_skipped=True,sources_unchanged=len(sources)))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})


if __name__=='__main__':main()
