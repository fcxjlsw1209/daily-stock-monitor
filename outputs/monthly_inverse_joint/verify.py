"""Verify all reported hard-portfolio replays and pre-entry input availability."""
from engine import *
from encoding import encoded_features
from validate import independent_nav


def run():
    px,fs,pools=base.load();results={};orders={};curves=[];metrics_error=0.;nav_error=0.;events=0;models=0;soft=0;hard=0
    for prefix,folder in [('', 'trials'),('EXPANDED_','expanded_trials'),('ENCODING_','encoding_trials'),('CORE_WEIGHT_','core_weight_trials')]:
        rr=json.loads((P/(prefix+'RESULTS.json')).read_text());oo=json.loads((P/(prefix+'ORDERS.json')).read_text());nn=pd.read_csv(P/(prefix+'CURVES.csv.gz'),index_col=0,parse_dates=True)
        mm=json.loads((P/(prefix+'MODELS.json')).read_text());assert len(rr)==len(oo)==nn.shape[1]
        for k,m in mm.items():
            q=np.load(P/folder/f'{k}.npz');assert len(q['soft_losses'])==m['soft_evaluations'];assert len(q['hard_losses'])==m['hard_refinement_evaluations'];assert np.isfinite(q['soft_losses']).all() and np.isfinite(q['hard_losses']).all()
            candidates=np.concatenate([q['soft_coefficients'],q['hard_coefficients']]);cw=m.get('core_weight',.5);limit=.9-cw if k.endswith('__portfolio') else 1-cw
            assert np.max(np.abs(candidates[:,:15]).sum(axis=1))<=limit+1e-12
            assert np.max(np.abs(candidates[:,15:]).sum(axis=1))<=(.1 if k.endswith('__portfolio') else 0)+1e-12
            assert np.isclose(q['hard_losses'].min(),m['objective'])
            if '__through2021__' in k:assert m['latest_fit_label']=='2021-12-31'
            else:assert m['latest_fit_label']=='2026-09-24'
            soft+=len(q['soft_losses']);hard+=len(q['hard_losses']);models+=1
        for key,v in rr.items():
            nav=nn[key];row=oo[key];rebuilt=independent_nav(px,row,nav.index,v['cost_bps_roundtrip']);delta=float(((rebuilt-nav)/nav).abs().max());nav_error=max(nav_error,delta);assert delta<1e-11
            r=nav.to_numpy()[1:]/nav.to_numpy()[:-1]-1
            metric=dict(cagr_252=float(nav.iloc[-1]**(252/(len(nav)-1))-1),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(252)),wealth=float(nav.iloc[-1]),max_dd=float(np.min(nav.to_numpy()/np.maximum.accumulate(np.r_[1.,nav.to_numpy()])[1:]-1)))
            for label,value in metric.items():err=abs(value-v[label]);metrics_error=max(metrics_error,err);assert err<1e-10
            assert len(row)==121
            for o in row:
                date=pd.Timestamp(o['date']);signal=pd.Timestamp(o['signal_date']);assert px.index[px.index.get_loc(date)-v['signal_lag']]==signal
                assert len(o['holdings']) in [0,2,3];assert signal<=date;events+=1
            ident=prefix+key;results[ident]=v;orders[ident]=row
        nn.columns=[prefix+k for k in nn.columns];curves.append(nn)
    cutoff=pd.Timestamp('2020-07-01');feature_delta=0.;cov_delta=0.
    for pool in ['core17','theme17']:
        member=pools[pool];smallpx=px.loc[:cutoff];smallfs={k:v.loc[:cutoff] for k,v in fs.items()};smallmember=member.loc[:cutoff]
        for function in [features,encoded_features]:
            full=function(px,fs,member);short=function(smallpx,smallfs,smallmember)
            cf,ff,gf=full[-3:];cs,fs_,gs=short[-3:];assert gf.loc[:cutoff].equals(gs)
            for a,b in [(cf,cs),(ff,fs_)]:
                for k in a:
                    assert a[k].loc[:cutoff].isna().equals(b[k].isna());delta=float((a[k].loc[:cutoff]-b[k]).abs().max().max());feature_delta=max(feature_delta,delta);assert delta<1e-12
        raw,_,_,gate=features(px,fs,member);valid=member&px.notna().rolling(253,min_periods=253).sum().eq(253)&px.gt(0)
        rank=raw['m252_42'].where(member).rank(axis=1,pct=True)
        for k,rows in orders.items():
            if pool+'__' not in k:continue
            for o in rows:
                for s in o['holdings']:
                    d=o['signal_date'];assert bool(valid.at[d,s]) and raw['m252_42'].at[d,s]>0
                    if not k.startswith('EXPANDED_'):assert rank.at[d,s]>=.5
        for signal in [pd.Timestamp('2018-08-31'),pd.Timestamp('2019-12-31'),pd.Timestamp('2020-06-30')]:
            cols=sorted(px.columns[valid.loc[signal]&gate.loc[signal]])[:8];h=np.array(list(combinations(range(len(cols)),3)))
            a=combination_features(px,signal,cols,h);b=combination_features(smallpx,signal,cols,h);delta=float(abs(a-b).max());cov_delta=max(cov_delta,delta);assert delta<1e-12
    bag_count=0
    for prefix in ['EXPANDED_','ENCODING_']:
        for pool in ['core17','theme17']:
            meta=json.loads((P/f'{prefix}BAGS_{pool}.json').read_text());assert len(meta)==121
            for m in meta:
                assert base.sha(m['file'])==m['sha256'];q=np.load(m['file']);assert np.isclose(q['weights'].sum(),1)
                assert np.isfinite(q['X']).all() and np.isfinite(q['C']).all();bag_count+=1
    hashes={}
    for manifest in ['PROTECTED_BEFORE.json','SOURCE_HASHES.json','PREVIOUS_STUDY_HASHES.json']:
        data=json.loads((P/manifest).read_text());assert all(base.sha(k)==v for k,v in data.items());hashes[manifest]=len(data)
    for log,end in [('run.log','DONE'),('expanded.log','EXPANDED_DONE'),('encoding.log','ENCODING_DONE'),('core_weight.log','CORE_WEIGHT_DONE')]:
        text=(P/log).read_text();assert text.rstrip().endswith(end);assert 'Traceback' not in text and 'RuntimeWarning' not in text
    dump('ALL_RESULTS.json',results);dump('ALL_ORDERS.json',orders);pd.concat(curves,axis=1).to_csv(P/'ALL_CURVES.csv.gz')
    selected_early=min((k for k,v in results.items() if v['fit_era']=='through2021' and v['signal_lag']==0 and v['cost_bps_roundtrip']==0),key=lambda k:results[k]['objective'])
    dump('FINAL_SELECTION.json',dict(selected_by_early_image_error=selected_early,best_full_cagr_retrospective=max(results,key=lambda k:results[k]['cagr_252']),
        best_full_sharpe_retrospective=max(results,key=lambda k:results[k]['sharpe']),independent_oos=False,author_reproduced=False))
    validation=dict(models=models,replays=len(results),monthly_events=events,soft_optimization_states_saved=soft,hard_refinement_trials_saved=hard,
        independent_nav_max_relative_error=nav_error,metric_max_difference=metrics_error,feature_prefix_cutoff=str(cutoff.date()),
        feature_prefix_max_difference=feature_delta,portfolio_correlation_prefix_max_difference=cov_delta,extra_candidate_files_verified=bag_count,
        file_hashes_verified=hashes,no_future_month_rankings=True,no_soft_mixture_performance_reported=True,
        inverse_training_label_availability='All image-based fitting is retrospective, not a historical available signal.',original_accounts_unchanged=True)
    dump('VALIDATION.json',validation);print(json.dumps(validation,indent=2))


if __name__=='__main__':run()
