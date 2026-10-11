"""Replay frozen candidates with daily Sharpe, author matching and execution controls."""
import json,sqlite3
import numpy as np
import pandas as pd
from study import P,O,read,dump,sha,actual,display,load_prices,market_states,engine,picks
from search_rules import score_rule
from search import features

def independent(px,orders):
    nav=100000.;pieces=[]
    for i,o in enumerate(orders):
        z=px.loc[o['date']:o['end'],o['symbols']]
        assert np.isfinite(z.to_numpy()).all() and z.gt(0).all().all()
        e=z.div(z.iloc[0]).mean(axis=1)*nav if len(o['symbols']) else pd.Series(nav,index=z.index)
        pieces.append(e if i==0 else e.iloc[1:]);nav=float(e.iloc[-1])
    return pd.concat(pieces)

def main():
    frozen=read(P/'FROZEN_RULES.json');frozen_hash=sha(P/'FROZEN_RULES.json')
    m=read(P/'INPUTS.json');px,raw,members,periods,sources,pools=load_prices()
    pred=np.load(P/'PREDICTIONS.npz')['predictions'];states=market_states(px)*2-1
    # Re-score the same annual models in Top32; do not reuse Top35 ranks.
    p32=np.zeros_like(pred);d32=np.load(P/'top32_percentile.npz')
    xx=d32['X'][:,:,[m['names'].index(n) for n in m['ml_names']]]
    for name in ['spy_trend','high_market_vol']:
        xx=np.concatenate([xx,np.repeat(states.loc[m['dates'],name].to_numpy()[:,None,None],52,axis=1)],axis=2)
    for fit in read(P/'TRAINING_AUDIT.json'):
        ids=[i for i,e in enumerate(m['ends']) if pd.Timestamp(e).year==fit['year']]
        vals=np.einsum('ijk,k->ij',xx[ids],np.array(fit['coef']),optimize=False)+fit['intercept']
        for j,i in enumerate(ids):
            good=d32['eligible'][i];p32[fit['config'],i,good]=pd.Series(vals[j,good]).rank(pct=True).to_numpy()-.5
    preds={35:pred,32:p32};runs={};curves={};ordersets={};monthrows=[]
    def evaluate(name,orders,n=None,rule=None,mode='close'):
        start=pd.Timestamp(m['dates'][0]);end=pd.Timestamp(m['ends'][-1])
        met,eq,trades,holds=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0.,execution=mode,start=start,end=end)
        if mode=='close':np.testing.assert_allclose(eq,independent(px,orders),rtol=1e-11,atol=1e-6)
        hits=[len(set(o['symbols'])&{actual(s) for s in p['holdings']}) for o,p in zip(orders,periods)]
        slices={}
        for key,a,b in [('calibration48','2020-12-31','2024-12-31'),('later21','2024-12-31','2026-09-24')]:
            z=eq.loc[a:b];slices[key]=engine.stats(z/z.iloc[0]*100000.)
            mask=[p['end']<='2024-12-31' if key=='calibration48' else p['end']>'2024-12-31' for p in periods]
            hh=[h for h,ok in zip(hits,mask) if ok];slices[key].update(exact_months=sum(h==3 for h in hh),shared_slots=sum(hh),months=len(hh))
        met.update(exact_months=sum(h==3 for h in hits),shared_slots=sum(hits),months=len(hits))
        runs[name]=dict(metrics=met,slices=slices,pool=n,rule=rule)
        curves[name]=eq;ordersets[name]=orders
        dump(name+'_HOLDINGS.json',holds);dump(name+'_TRADES.json',trades)
        for i,(o,p) in enumerate(zip(orders,periods)):
            selected=[display(s) for s in o['symbols']]
            gross=(px.loc[p['end'],o['symbols']]/px.loc[p['start'],o['symbols']]).mean() if o['symbols'] else 1.
            au=[actual(s) for s in p['holdings']];ag=(px.loc[p['end'],au]/px.loc[p['start'],au]).mean()
            if mode=='close':monthrows.append(dict(model=name,date=p['start'],end=p['end'],selected=selected,author=p['holdings'],shared=hits[i],return_=float(gross-1),author_return=float(ag-1),log_relative_return=float(np.log(gross/ag))))
        print(name,'CAGR',round(met['cagr'],4),'Sharpe',round(met['sharpe'],3),'matches',met['exact_months'],'later',round(slices['later21']['cagr'],4),flush=True)
    def fromrule(rule,n):
        key=f'top{n}_{rule["normalization"]}';d=dict(np.load(P/(key+'.npz')),names=m['names'])
        scores=score_rule(rule,d,preds[n]);ids,good=picks(scores,d['eligible']);symbols=m[key]['symbols'];orders=[]
        for i,p in enumerate(periods):
            chosen=[int(j) for j,ok in zip(ids[i],good[i]) if ok]
            if len(chosen)<2:chosen=[]
            orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=[symbols[i][j] for j in chosen],scores=[float(scores[i,j]) for j in chosen]))
        return orders
    baseline=dict(normalization='magnitude',core='mom252_risk_recent63',weights={'mom252_risk_recent63':.5,'vol126':.35,'neg_downside63':.15})
    for n in [35,32]:evaluate(f'baseline_top{n}',fromrule(baseline,n),n,baseline)
    author=[dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=list(map(actual,p['holdings'])),scores=[]) for p in periods]
    evaluate('author',author)
    # Fixed earlier recipes are controls, not additional fits. They predate this
    # study and are replayed verbatim to distinguish model from universe changes.
    oldmeta=read(O/'monthly_factor_search_v2/INPUTS.json')
    oldspecs=[r for r in read(O/'monthly_factor_search_v2/SELECTED_CASES.json') if r['name'] in ['return_leader','development_leader']]
    core,side=features(px,raw);oldfeatures={**core,**side};warm=px.notna().rolling(295).sum().eq(295)&px.gt(0)
    for spec in oldspecs:
        etf_modes=['core17','theme17'] if spec['name']=='return_leader' else ['core17']
        for n,etf_mode in [(n,e) for n in [35,32] for e in etf_modes]+([(None,'theme17')] if spec['name']=='return_leader' else []):
            names=oldmeta[spec['key']]['feature_names'];norm=spec['key'].split('__')[1];orders=[]
            for p in periods:
                d=p['start']
                if n:ss=sorted({actual(s) for s in pools[str(n)][d]}|members[etf_mode][1])
                else:
                    member=members[etf_mode][0];ss=sorted(member.columns[member.loc[d]])
                tab=pd.DataFrame({k:oldfeatures[k].loc[d,ss] for k in names});q=engine.normalize(tab,norm)
                state,factor=spec['correction'].split('__',1)
                extra=q[factor].to_numpy()[None,:]*states.at[d,state]
                score=engine.score_of(spec,q.to_numpy()[None,:,:],extra)[0]
                eligible=(warm.loc[d,ss]&core['mom252_skip42'].loc[d,ss].gt(0)).to_numpy()
                ids=[int(j) for j in np.argsort(-np.where(eligible,score,-np.inf),kind='stable')[:3] if eligible[j]]
                if len(ids)<2:ids=[]
                orders.append(dict(date=d,end=p['end'],signal_date=d,symbols=[ss[j] for j in ids],scores=[float(score[j]) for j in ids]))
            name=f'prior_{spec["name"]}_top{n}_{etf_mode}'
            evaluate(name,orders,n,spec)
            if n is None:
                expected=read(O/'monthly_factor_search_v2/return_leader_close_zero_DAILY.json')
                prior=pd.Series({pd.Timestamp(r['date']):r['equity'] for r in expected})
                np.testing.assert_allclose(curves[name],prior,rtol=1e-11,atol=1e-6)
    for label,row in frozen['leaders'].items():
        rule=row['rule']
        for n in [35,32]:evaluate(label+f'_top{n}',fromrule(rule,n),n,rule)
        evaluate(label+'_next_open',fromrule(rule,35),35,rule,'open')
    # Development short-list only; do not pick a new winner on the later slice.
    con=sqlite3.connect(P/'TRIALS.sqlite')
    for label in ['return_selected','match_selected']:
        rule=frozen['leaders'][label]['rule']
        if rule.get('ml_weight'):
            ab=dict(rule);wc=rule['weights'][rule['core']];g=rule['ml_weight']
            if wc>=1-g-1e-10:ab['weights']={rule['core']:1.}
            else:ab['weights']={k:(v if k==rule['core'] else v*(1-wc)/(1-wc-g)) for k,v in rule['weights'].items()}
            ab.pop('ml_weight');ab.pop('ml_id');evaluate(label+'_without_ml',fromrule(ab,35),35,ab)
        while 'seed_id' in rule:
            rule=json.loads(con.execute('SELECT rule FROM trials WHERE id=?',(rule['seed_id'],)).fetchone()[0])
        evaluate(label+'_base_only',fromrule(rule,35),35,rule)
    con.close()
    q=px.loc[m['dates'][0]:m['ends'][-1],'QQQ'];curves['QQQ']=q/q.iloc[0]*100000.
    runs['QQQ']=dict(metrics=engine.stats(curves['QQQ']))
    # Prior controls must match exact daily paths, not just rounded performance.
    old=read(O/'monthly_top35_extension/DAILY_EQUITY_69.json')
    for n in [32,35]:
        prior=pd.Series({pd.Timestamp(r['date']):r[f'top{n}_extended69'] for r in old})
        np.testing.assert_allclose(curves[f'baseline_top{n}'],prior,rtol=1e-11,atol=1e-6)
    prior=pd.Series({pd.Timestamp(r['date']):r['author69'] for r in old})
    np.testing.assert_allclose(curves['author'],prior,rtol=1e-11,atol=1e-6)
    df=pd.DataFrame(curves);dump('DAILY_EQUITY.json',[dict(date=str(d.date()),**{k:float(v) for k,v in row.items()}) for d,row in df.iterrows()])
    dump('MONTHLY_COMPARISON.json',monthrows)
    dump('RESULTS.json',dict(runs=runs,trial_count=frozen['trial_count'],frozen_rule_sha256=frozen_hash,cost=0,independent_out_of_sample=False))
    assert sha(P/'FROZEN_RULES.json')==frozen_hash
    for f,h in read(P/'SOURCE_HASHES.json').items():assert sha(f)==h,('Source changed',f)
    protected=read(P/'PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(f)!=h];assert not changed,changed
    dump('REPLAY_VALIDATION.json',dict(controls_daily_nav_identical=True,independent_fixed_share_nav_matches=True,frozen_rules_unchanged=True,protected_files_unchanged=len(protected)))

if __name__=='__main__':main()
