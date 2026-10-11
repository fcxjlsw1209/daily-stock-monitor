"""Declared monthly hypotheses: strength and risk measured over matching windows.

Motivation: the original risk denominator was recent 63-day volatility while the
momentum numerator skips 42 days. Test the alternative literal interpretation
before adding arbitrary factors. No leverage, model clock change or intramonth exits.
"""
import json
import numpy as np
import pandas as pd
from research import P, REBUILD, load, raw_features, Simulator, dump, metrics, fit_errors


def trend_quality(px, window=210, lag=42):
    logp = np.log(px.where(px > 0))
    t = pd.Series(np.arange(len(px)), index=px.index)
    sy = logp.rolling(window).sum()
    sxy = logp.mul(t, axis=0).rolling(window).sum() - sy.mul(t - (window - 1) / 2, axis=0)
    sxx = window * (window**2 - 1) / 12
    syy = logp.pow(2).rolling(window).sum() - sy.pow(2) / window
    slope = sxy / sxx
    r2 = (sxy.pow(2) / (sxx * syy.where(syy > 1e-12))).clip(0, 1)
    return (slope * 252 * r2).shift(lag), r2.shift(lag)


def make_features(px, raw):
    ret = px.pct_change(fill_method=None)
    vol = ret.rolling(210).std().shift(42)
    downside = ret.clip(upper=0).pow(2).rolling(210).mean().pow(.5).shift(42)
    quality, r2 = trend_quality(px)
    out = dict(raw)
    out.update(aligned_sharpe=raw['m252_42'] / vol,
               aligned_log_sharpe=np.log1p(raw['m252_42']) / vol,
               aligned_downside=raw['m252_42'] / downside.replace(0, np.nan),
               delayed_short_risk=raw['m252_42'] / ret.rolling(63).std().shift(42),
               slope_quality=quality,
               return_quality=raw['m252_42'] * r2,
               aligned_half_risk=raw['m252_42'] / np.sqrt(vol),
               aligned_vol=vol)
    return out


def make_rules(rank):
    names=['aligned_sharpe','aligned_log_sharpe','aligned_downside','delayed_short_risk',
           'slope_quality','return_quality','aligned_half_risk']
    rules={k:rank[k] for k in names}
    for k in ['aligned_sharpe','slope_quality','return_quality']:
        rules[k+'_mix_momentum'] = .5*rank['m252_42']+.5*rank[k]
        rules[k+'_space'] = .7*rank[k]+.15*rank['vol']+.1*rank['relative']+.05*rank['cmf21']
        rules[k+'_stability'] = .7*rank[k]+.15*rank['lowvol']+.1*rank['relative']+.05*rank['cmf21']
    return rules


def main():
    config=dict(hypothesis='Align strength and risk with the same historical 12-to-2-month window; additionally test smooth-trend strength.',
                core_window='Return P[t-42]/P[t-252]-1, volatility210 endingt-42; log-price slope uses210 completed prices endingt-42.',
                bases=['aligned_sharpe','aligned_log_sharpe','aligned_downside','delayed_short_risk','slope_quality','return_quality','aligned_half_risk'],
                composites='For aligned_sharpe/slope_quality/return_quality:50%R12-2+50%feature; or70%feature+15%high/low vol+10%relative+5%CMF.',
                pools=['core17','theme17'], trial_count=32,
                execution='Same month-end close, zero costs, fixed adjusted-price units,3 equal or2 equal; only positiveR12-2; no intramonth trade.',
                selection='2016-2021 global logNAV error+2*month-shape error; previously inspected sample, not independent OOS.',
                author_has_no_future_month_end_ranking=True)
    dump(P/'ALIGNED_CONFIG.json',config)
    px,fs,pools=load();raw=raw_features(px,fs);f=make_features(px,raw)
    # A future extension cannot revise the factor prefix.
    cutoff='2020-07-01';shortpx=px.loc[:cutoff];shortfs={k:v.loc[:cutoff] for k,v in fs.items()}
    short=make_features(shortpx,raw_features(shortpx,shortfs))
    delta=max(float((v.loc[:cutoff]-short[k]).abs().max().max()) for k,v in f.items())
    assert delta < 1e-9, delta
    results={};curves={};orderlog={};cal=json.loads((REBUILD/'CALIBRATION.json').read_text());obs=pd.read_csv(REBUILD/'DIGITIZED_WEALTH.csv')
    for pool in config['pools']:
        member=pools[pool];rank={k:v.where(member).rank(axis=1,pct=True) for k,v in f.items()};sim=Simulator(px,fs,member)
        for name,score in make_rules(rank).items():
            key=pool+'__'+name
            try:
                nav,orders=sim.run(score,raw['m252_42']>0);m=metrics(nav);e=fit_errors(nav,cal,obs)
                results[key]=dict(**m,**e,pool=pool,name=name,development_selection_error=e['development_global_log_rmse']+2*e['development_month_shape_log_rmse'])
                curves[key]=nav;orderlog[key]=orders
                print(key,round(100*m['cagr_252'],3),round(100*m['max_dd'],3),flush=True)
            except Exception as e:
                results[key]=dict(failure=str(e));print(key,'FAILURE',e,flush=True)
            dump(P/'ALIGNED_RESULTS.json',results)
    assert len(results)==config['trial_count']
    good={k:v for k,v in results.items() if 'failure' not in v}
    dump(P/'ALIGNED_SELECTION.json',dict(development_selected=min(good,key=lambda k:good[k]['development_selection_error']),
         best_full_cagr_retrospective=max(good,key=lambda k:good[k]['cagr_252']),feature_prefix_max_difference=delta,
         failures=[k for k,v in results.items() if 'failure' in v],all_trials=len(results),independent_oos=False))
    dump(P/'ALIGNED_ORDERS.json',orderlog);pd.DataFrame(curves).to_csv(P/'ALIGNED_CURVES.csv.gz')


if __name__=='__main__':
    main()
