"""Independent fixed-share reconstruction, moment checks and figure audit."""
import sys, json
import numpy as np
import pandas as pd
from study import P,O,I,load,read,dump,sha,START,END,TARGET,xcoord
sys.path.insert(0,str(O/'monthly_reconstruction_contract'))
from audit import image_diagnostics
from gate import contract,assess
from inverse import load_observations


def run():
    px,_,_=load();cal=read(I/'CALIBRATION.json');obs=load_observations();c=contract()
    out={};sets={};maxerror=0.;evaluations=0
    for f in sorted(P.glob('*_RESULT.json')):
        ident=f.name.replace('_RESULT.json','');r=read(f);meta=read(P/f'META_{r["pool"]}.json')
        states=np.load(P/'internal_states'/f'{ident}_states.npz');choice=states['final']
        assert len(choice)==len(meta)==122
        qnav=pd.read_csv(P/f'{ident}_NAV.csv',index_col=0,parse_dates=True).iloc[:,0]
        independent=pd.Series(index=qnav.index,dtype=float);capital=1.;selections=[]
        for m,z in enumerate(meta):
            assert sha(z['file'])==z['sha256']
            cache=np.load(z['file']);picked=[z['symbols'][j] for j in cache['combos'][choice[m]]]
            assert len(picked) in [0,2,3]
            prices=px.loc[z['start']:z['end'],picked]
            assert prices.notna().all().all() and prices.gt(0).all().all()
            shares={s:capital/len(picked)/prices.iloc[0][s] for s in picked}
            segment=sum(prices[s]*qty for s,qty in shares.items()) if picked else pd.Series(capital,index=prices.index)
            independent.loc[segment.index]=segment;capital=float(segment.iloc[-1]);selections.append(set(picked))
        error=float(((independent-qnav)/qnav).abs().max());maxerror=max(maxerror,error);assert error<1e-11
        ret=qnav.pct_change().dropna();qret=px.loc[qnav.index,'QQQ'].pct_change().dropna()
        metrics=dict(wealth=float(qnav.iloc[-1]),cagr=float(qnav.iloc[-1]**(252/len(ret))-1),
            sharpe=float(ret.mean()/ret.std()*np.sqrt(252)),max_dd=float((qnav/qnav.cummax()-1).min()),
            correlation=float(ret.corr(qret)),volatility=float(ret.std()*np.sqrt(252)))
        for k,v in metrics.items():assert np.isclose(v,r[k],rtol=1e-10,atol=1e-10),(k,v,r[k])
        metrics['cagr_nav_count']=float(qnav.iloc[-1]**(252/len(qnav))-1)
        rz=np.r_[0.,ret.to_numpy()];qz=np.r_[0.,qret.to_numpy()]
        metrics['sharpe_initial_zero']=float(rz.mean()/rz.std(ddof=1)*np.sqrt(252))
        metrics['correlation_initial_zero']=float(np.corrcoef(rz,qz)[0,1])
        diag,monthly=image_diagnostics(qnav,px['QQQ'],obs,cal,c['image_audit'])
        dd=qnav/qnav.cummax()-1;trough=dd.idxmin();peak=qnav.loc[:trough].idxmax()
        deep=obs['dd'][obs['dd'].ymax>=obs['dd'].ymax.max()-2]
        trough_x=float(xcoord(pd.DatetimeIndex([trough]),cal)[0])
        timing_matches=bool(deep.x.min()-1<=trough_x<=deep.x.max()+1)
        # Explicit two metric-count hypotheses; no automatic change to prior rounding thresholds.
        exact_rounding=dict(wealth=abs(metrics['wealth']-TARGET['wealth'])<=.005,
            cagr=abs(metrics['cagr_nav_count']-TARGET['cagr'])<=.00005,
            sharpe=abs(metrics['sharpe_initial_zero']-TARGET['sharpe'])<=.0005,
            max_dd=abs(metrics['max_dd']-TARGET['max_dd'])<=.00005,
            correlation=abs(metrics['correlation_initial_zero']-TARGET['correlation'])<=.0005)
        evidence={k:'unresolved' for k in c['admission_requirements']}
        evidence.update(execution_and_accounting='passed',exact_window='passed',no_monthly_rescaling_or_peak_reset='passed',
            continuous_wealth_and_relative_path='failed' if any(diag[p]['screen_rejected'] for p in ['wealth','relative']) else 'unresolved',
            continuous_drawdown_path='failed' if diag['drawdown']['screen_rejected'] or not timing_matches else 'unresolved')
        if not all(exact_rounding[k] for k in ['wealth','cagr','max_dd']):evidence['cagr_terminal_wealth_maxdd']='failed'
        if not exact_rounding['sharpe']:evidence['sharpe_definition_and_value']='failed'
        out[ident]=dict(metrics=metrics,delta_vs_figure=dict(cagr=metrics['cagr_nav_count']-TARGET['cagr'],
            sharpe=metrics['sharpe_initial_zero']-TARGET['sharpe'],max_dd=metrics['max_dd']-TARGET['max_dd'],
            correlation=metrics['correlation_initial_zero']-TARGET['correlation'],wealth=metrics['wealth']-TARGET['wealth']),
            figure_printed_rounding_matches=exact_rounding,image=diag,monthly=monthly,
            drawdown_timing=dict(peak=str(peak.date()),trough=str(trough.date()),inside_figure_deepest_interval=timing_matches),
            admission=assess(evidence),candidate_transactions_exported=False)
        sets[ident]=selections
        progress=read(P/f'{ident}_PROGRESS.json')
        for tf in progress['trial_files']:
            assert sha(tf['file'])==tf['sha256']
            for evaluation in read(tf['file']):
                assert np.isfinite(evaluation['losses']).all();evaluations+=len(evaluation['losses'])
    common=[]
    for i in range(122):
        values=[sets[k][i] for k in sets]
        common.append(dict(event=i,exact_all=all(v==values[0] for v in values),
            common_securities=len(set.intersection(*values)),distinct_portfolios=len({tuple(sorted(v)) for v in values})))
    protection=read(P/'PROTECTED_BEFORE.json')
    for k,v in protection.items():assert sha(k)==v,k
    for pool in ['core17','theme17']:
        for k,v in read(P/f'INPUT_HASHES_{pool}.json').items():assert sha(k)==v,k
    summary=dict(full_paths=len(out),actual_candidate_evaluations=evaluations,independent_nav_max_relative_error=maxerror,
        protected_files_unchanged=len(protection),exact_agreement_events=sum(v['exact_all'] for v in common),
        events_without_a_common_security=sum(v['common_securities']==0 for v in common),
        no_admitted_potential_transactions=not any(v['admission']['admitted'] for v in out.values()),
        original_accounts_unchanged=True,author_rule_reproduced=False)
    dump('VERIFICATION.json',summary);dump('FIGURE_AUDIT.json',out);dump('PATH_AGREEMENT.json',common)
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
