"""Early decision factor-definition probe; no portfolio simulation."""
from itertools import product
from copy import deepcopy
import pandas as pd
import numpy as np
from frontier import load_first_months, solve, choices
from decisions import P, O, read, dump, sha, CORE, FEATURES, STRICT
from study import load


def main():
    base=load_first_months(3); events=read(P/'DECISIONS.json')[:3]
    px,_,sources=load(); ret=px.pct_change(fill_method=None)
    moments={n:px.shift(42)/px.shift(n)-1 for n in [126,189,252]}
    sector_file=O/'index_union/alias_resolved_inputs/sector60.csv.gz'
    sec=pd.read_csv(sector_file,index_col=0,parse_dates=True).reindex(index=px.index,columns=px.columns)
    manifests=read(O/'four_factor_rebound/ETF_SOURCES.json')+read(O/'monthly_reconstruction_v2/ETF_MANIFEST.json')
    etfs={v['symbol'] for v in manifests}
    for s in etfs:sec[s]=px.SPY.pct_change(60,fill_method=None)
    relative=(1+px.pct_change(60,fill_method=None))/(1+sec)-1
    sources[str(sector_file)]=sha(sector_file)
    results={}
    # The baseline was already examined; each alternative changes only the
    # volatility denominator and optionally the timing of industry comparison.
    for horizon,lag,relative_lag in product([126,252],[0,42],[0,42]):
        rows=deepcopy(base);vol=ret.rolling(horizon).std().shift(lag)
        for row,event in zip(rows,events):
            d=row['date'];universe=event['universe']
            raw=pd.DataFrame({f'risk{n//21}_2':m.loc[d,universe]/vol.loc[d,universe] for n,m in moments.items()})
            raw['industry_relative60']=relative.shift(relative_lag).loc[d,universe]
            q=raw.rank(pct=True).fillna(.5)
            lo,hi=raw.quantile(.1),raw.quantile(.9)
            scale=((raw-lo)/(hi-lo).replace(0,np.nan)).clip(0,1).fillna(.5)
            sep=raw.copy()*np.nan
            for flag in [False,True]:
                names=[s for s in universe if (s in etfs)==flag]
                sep.loc[names]=raw.loc[names].rank(pct=True).fillna(.5)
            for family,values in [('universe_percentile',q),('raw_magnitude',scale),('asset_class_percentile',sep)]:
                for k in raw:row['X'][family][:,FEATURES.index(k)]=values.loc[row['symbols'],k].to_numpy()-.5
        for family,dominance,interaction in product(['universe_percentile','raw_magnitude','asset_class_percentile'],['half','largest_group'],[False,True]):
            name=f'vol{horizon}_lag{lag}_relative{relative_lag}__{family}__{dominance}__{int(interaction)}'
            history=[];prior=None;exact=0
            for i,row in enumerate(rows):
                prediction=None
                if prior:
                    picks,_=choices(row,prior,family,interaction)
                    prediction=dict(picks=picks,overlap=len(set(picks)&set(np.array(row['symbols'])[row['y']])),trained_through=rows[i-1]['date'])
                fit=solve(rows[:i+1],family,dominance,interaction)
                history.append(dict(date=row['date'],prior_prediction=prediction,fit=fit))
                if fit['margin']<=STRICT:break
                exact=i+1;prior=fit
                for earlier in rows[:i+1]:
                    picks,_=choices(earlier,fit,family,interaction)
                    assert set(picks)==set(np.array(earlier['symbols'])[earlier['y']])
            results[name]=dict(exact_prefix_months=exact,history=history)
            print(name,exact,fit['margin'],flush=True)
        dump('DEFINITION_PROBE_RESULTS.json',results)
    sources.update(read(P/'SOURCE_HASHES.json'))
    for f,h in sources.items():assert sha(f)==h,f
    dump('DEFINITION_PROBE_STATUS.json',dict(maximum_common_prefix=max(r['exact_prefix_months'] for r in results.values()),variants=len(results),full_period_backtest_run=False,sources_unchanged=len(sources)))


if __name__=='__main__':main()
