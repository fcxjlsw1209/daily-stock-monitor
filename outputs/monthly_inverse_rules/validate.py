"""Independent accounting, clock, prefix, candidate and file-integrity checks."""
import json
import numpy as np
import pandas as pd
from common import P,O,base,FEATURES,features,dump


STAGES=[('', 'MONTH_MATCHES.json'),('COHERENT_','COHERENT_MATCHES.json'),('POLISH_','POLISH_MATCHES.json'),('TREE_','TREE_MATCHES.json')]


def strict_subset(bags,mass=.95):
    total=0.;out=[]
    for b in bags:
        out.append(b);total+=b['image_weight']
        if total>=mass:break
    return out


def independent_nav(px,orders,index,cost):
    result=pd.Series(index=index,dtype=float);level=1.;previous={};last_prices=None
    for i,o in enumerate(orders):
        date=pd.Timestamp(o['date']);end=pd.Timestamp(orders[i+1]['date']) if i+1<len(orders) else index[-1]
        old={s:q*float(px.at[date,s])/level for s,q in previous.items()}
        target={s:1/len(o['holdings']) for s in o['holdings']};fraction=1.
        for _ in range(15):fraction=1-cost/20000*sum(abs(fraction*target.get(s,0)-old.get(s,0)) for s in set(old)|set(target))
        assert abs((1-fraction)-o['fee_fraction'])<1e-12
        amount=level*fraction;assert abs(amount-o['entry_equity'])<max(1.,amount)*1e-12
        shares={s:amount*w/float(px.at[date,s]) for s,w in target.items()}
        if shares:
            window=px.loc[date:end,list(shares)];part=window.mul(pd.Series(shares)).sum(axis=1)
        else:part=pd.Series(amount,index=px.loc[date:end].index)
        result.loc[part.index]=part;level=float(part.iloc[-1]);previous=shares
    return result


def run():
    px,fs,pools=base.load();metas={k:json.loads((P/f'BAGS_{k}.json').read_text()) for k in ['core17','theme17']};strict={};cases=0;navdiff=0.;metricdiff=0.;clocks=0
    for pool,meta in metas.items():
        for m in meta:
            assert base.sha(m['file'])==m['sha256'];q=np.load(m['file']);assert q['X'].shape[1]==len(FEATURES)
            assert np.isfinite(q['X']).all() and np.isclose(q['weights'].sum(),1.)
            assert str(pd.Timestamp(m['signal_date']).to_period('M')+1)==m['month']
            assert m['signal_date']<m['label_month_end']
    for prefix,matchfile in STAGES:
        results=json.loads((P/(prefix+'RESULTS.json')).read_text());orders=json.loads((P/(prefix+'ORDERS.json')).read_text());curves=pd.read_csv(P/(prefix+'CURVES.csv.gz'),index_col=0,parse_dates=True)
        matches=json.loads((P/matchfile).read_text());assert len(results)==len(orders)==len(matches)==curves.shape[1]
        for key,v in results.items():
            nav=curves[key];oo=orders[key];assert len(oo)==121;cost=v['cost_bps_roundtrip'];lag=v['signal_lag'];assert lag in [0,1]
            replay=independent_nav(px,oo,nav.index,cost);delta=float(((replay-nav)/nav).abs().max());navdiff=max(navdiff,delta);assert delta<1e-11
            rr=nav.to_numpy()[1:]/nav.to_numpy()[:-1]-1
            mm=dict(wealth=float(nav.iloc[-1]),cagr_252=float(nav.iloc[-1]**(252/(len(nav)-1))-1),sharpe=float(np.mean(rr)/np.std(rr,ddof=1)*np.sqrt(252)),max_dd=float(np.min(nav.to_numpy()/np.maximum.accumulate(np.r_[1.,nav.to_numpy()])[1:]-1)))
            for k,x in mm.items():metricdiff=max(metricdiff,abs(x-v[k]));assert abs(x-v[k])<1e-10
            pool=key.split('__')[0];rows=[]
            for o,m,match in zip(oo,metas[pool],matches[key]):
                assert px.index[px.index.get_loc(pd.Timestamp(o['date']))-lag]==pd.Timestamp(o['signal_date'])
                assert len(o['holdings']) in [0,2,3];clocks+=1
                good=strict_subset(m['candidate_triples']);chosen=set(o['holdings']);hit=any(chosen==set(q['holdings']) for q in good)
                rows.append(dict(month=m['month'],holdings=o['holdings'],anchor=m['anchor'],concentration=m['confidence'],
                    top95_fit_weight_candidate_count=len(good),strict_fit_weight_hit=hit,broad64_candidate_hit=match['candidate_hit'],
                    image_loss_percentile=match['image_loss_percentile'],candidate_groups=m['candidate_triples'][:3],
                    verified_author_trade=False))
            byera={}
            for label,part in [('early',rows[:64]),('later',rows[64:]),('all',rows)]:
                byera[label]=dict(months=len(part),anchors=sum(q['anchor'] for q in part),strict_hits=sum(q['strict_fit_weight_hit'] for q in part),strict_anchor_hits=sum(q['strict_fit_weight_hit'] and q['anchor'] for q in part),broad_hits=sum(q['broad64_candidate_hit'] for q in part))
            strict[prefix+key]=dict(summary=byera,months=rows);cases+=1
    # Prefix invariance applies to input features, not retrospective model fitting.
    cutoff=pd.Timestamp('2020-07-01');diff=0.
    for pool in ['core17','theme17']:
        _,c,f,g=features(px,fs,pools[pool]);_,cs,ff,gg=features(px.loc[:cutoff],{k:v.loc[:cutoff] for k,v in fs.items()},pools[pool].loc[:cutoff])
        assert g.loc[:cutoff].equals(gg)
        for a,b in [(c,cs),(f,ff)]:
            for k in a:
                x=a[k].loc[:cutoff];y=b[k];assert x.isna().equals(y.isna());delta=float((x-y).abs().max().max());assert delta<1e-12;diff=max(diff,delta)
    trials=0;models=0
    for prefix,directory,limit in [('COHERENT_','coherent_trials',.5),('POLISH_','polish_trials',.5),('TREE_','tree_trials',.4)]:
        fits=json.loads((P/(prefix+'MODELS.json')).read_text())
        for k,m in fits.items():
            q=np.load(P/directory/(k+'.npz'));assert len(q['loss'])==m['evaluations'];assert np.isfinite(q['loss']).all();assert np.max(np.abs(q['coefficients']).sum(axis=1))<=limit+1e-12
            assert np.isclose(q['loss'].min(),m['objective']);assert len(m['coefficients'])==15
            if '__through2021__' in k:assert m['latest_fit_label']<='2021-12-31'
            trials+=len(q['loss']);models+=1
    protected=json.loads((P/'PROTECTED_BEFORE.json').read_text());sources=json.loads((P/'SOURCE_HASHES.json').read_text())
    assert all(base.sha(k)==h for k,h in protected.items());assert all(base.sha(k)==h for k,h in sources.items())
    for log in ['prepare.log','fit.log','coherent.log','polish.log','tree.log']:
        text=(P/log).read_text();assert 'Traceback' not in text and 'RuntimeWarning' not in text
    dump('STRICT_MATCH_AUDIT.json',strict)
    result=dict(completed_replays=cases,monthly_orders_checked=clocks,independent_nav_max_relative_error=navdiff,metric_max_difference=metricdiff,
        candidate_bag_hashes_checked=sum(map(len,metas.values())),feature_prefix_cutoff=str(cutoff.date()),feature_prefix_max_difference=diff,
        curve_fit_models=models,saved_coefficient_trials=trials,protected_files_unchanged=len(protected),source_hashes_verified=len(sources),
        author_strategy_reproduced=False,independent_oos=False,model_label_availability='Image-labelled inference is retrospective, including early-period fits.',
        validation_note='Initial tree extension caught a pandas view mutation before results were written; fixed using a copy, rerun, independent accounting now verified.')
    dump('VALIDATION.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
