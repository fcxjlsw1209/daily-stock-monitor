"""Set-valued inverse choice fitting and a short continuous path check."""
from itertools import product
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_decision_reconstruction'))
from decisions import CORE,SIDE,FEATURES,maximum_margin,STRICT,display
from frontier import load_first_months
from study import load,read,sha,xcoord,I
sys.path.insert(0,str(O/'monthly_reconstruction_contract'))
from audit import panel_error
from inverse import load_observations

def dump(n,z): (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def core_fit(D):
    D=D[:,:len(CORE)];n=len(CORE)
    fit=linprog(np.r_[np.zeros(n),-1.],A_ub=np.column_stack([-D,np.ones(len(D))]),b_ub=np.zeros(len(D)),
        A_eq=np.array([np.r_[np.ones(n),0.]]),b_eq=[1.],bounds=[(0,None)]*n+[(-1,1)],method='highs')
    assert fit.success
    w=np.r_[fit.x[:n],np.zeros(len(SIDE))];margin=float(fit.x[-1])
    assert np.einsum('ij,j->i',D,w[:n],optimize=False).min()>=margin-1e-7
    return dict(w=w,margin=margin)

def fit_rule(rows,targets,family,kind):
    differences=[]
    for row,chosen in zip(rows,targets):
        x=row['X'][family];yes=np.array([s in chosen for s in row['symbols']]);assert yes.sum()==3
        differences.append((x[yes,None]-x[~yes][None,:]).reshape(-1,len(FEATURES)))
    d=np.vstack(differences)
    f=core_fit(d) if kind=='momentum_only' else maximum_margin(d)
    if f['margin']>STRICT:
        for row,target in zip(rows,targets):assert set(picks(row,f['w'],family))==set(target)
    return dict(weights=f['w'].tolist(),margin=f['margin'],kind=kind,family=family,
        nonzero=int((abs(f['w'])>1e-7).sum()),core_mass=float(f['w'][:len(CORE)].sum()))

def picks(row,w,family):
    score=np.einsum('ij,j->i',row['X'][family],w,optimize=False)
    assert np.isfinite(score).all()
    return [row['symbols'][i] for i in np.argsort(-score,kind='stable')[:3]]

def replay(px,events,targets,capital):
    parts=[];orders=[]
    for event,names in zip(events,targets):
        prices=px.loc[event['start']:event['end'],names]
        assert prices.notna().all().all() and prices.gt(0).all().all()
        shares=capital/3/prices.iloc[0]
        nav=prices.mul(shares).sum(axis=1)
        assert np.allclose(nav/capital,prices.div(prices.iloc[0]).mean(axis=1),rtol=1e-12,atol=1e-12)
        orders.append(dict(start=event['start'],end=event['end'],holdings=names,
            start_nav=capital,end_nav=float(nav.iloc[-1]),period_return=float(nav.iloc[-1]/capital-1)))
        capital=float(nav.iloc[-1]);parts.append(nav)
    full=pd.concat(parts);full=full[~full.index.duplicated(keep='last')]
    return full,orders

def image_check(nav,prior_peak,px,cal,obs,cfg):
    x=xcoord(nav.index,cal);logs=np.log(nav.to_numpy());dd=nav/np.maximum(prior_peak,nav.cummax())-1
    metrics={};loss=0.
    for panel in ['wealth','relative','drawdown']:
        z=obs['dd' if panel=='drawdown' else panel]
        z=z[(z.x>x[0]+.35)&(z.x<x[-1]-.35)]
        if z.empty:metrics[panel]=dict(visible_columns=0,screen_rejected=False,unresolved=True);continue
        if panel=='drawdown':y=cal['dd']['y0']-cal['dd']['scale']*dd.to_numpy();tol=cfg['drawdown_extra_y_pixels']
        else:
            levels=logs.copy()
            if panel=='relative':levels-=np.log(px.loc[nav.index,'QQQ'].to_numpy()/px.at[pd.Timestamp(cal['dates'][0]),'QQQ'])
            y=cal[panel]['y0']-cal[panel]['log_scale']*levels;tol=cfg['wealth_relative_extra_y_pixels']
        metrics[panel],_=panel_error(x,y,y,z,cfg['x_pixel_radius'],tol)
        loss+=metrics[panel]['center_rmse_px']**2
    return dict(panels=metrics,not_rejected=all(not r['screen_rejected'] for r in metrics.values()),
        all_panels_visible=all(r['visible_columns']>0 for r in metrics.values()),
        center_error_sum=loss,window_return=float(nav.iloc[-1]/nav.iloc[0]-1),
        global_peak_aware_drawdown_in_window=float(dd.min()),
        drawdown_from_window_start=float((nav/nav.cummax()-1).min()))

def main():
    window=read(P/'WINDOW_SELECTION.json')['chosen'];audit=read(P/'MONTHLY_AMBIGUITY.json')
    events=window['months'];dates={m['start'] for m in events}
    all_rows=load_first_months(121);rows=[r for r in all_rows if r['date'] in dates]
    candidates=[[c['holdings'] for c in a['candidates'] if c['near_all']] for a in audit]
    assert [len(z) for z in candidates]==[3,10,1]
    px,_,ps=load();base=pd.read_csv(O/'monthly_path_constraints/theme17__prior__draw_v2__blocks_NAV.csv',index_col=0,parse_dates=True).iloc[:,0]
    initial=float(base.at[pd.Timestamp(window['start'])]);peak=float(base.loc[:window['start']].max())
    cal=read(I/'CALIBRATION.json');obs=load_observations();cfg=read(O/'monthly_reconstruction_contract/CONTRACT.json')['image_audit']
    trials=[];path_checks=[]
    for i,targets in enumerate(product(*candidates)):
        targets=list(targets);nav,orders=replay(px,events,targets,initial)
        check=image_check(nav,peak,px,cal,obs,cfg)
        path_checks.append(dict(path_id=i,orders=orders,check=check))
        for kind,family in product(['momentum_only','momentum50_side50'],['universe_percentile','raw_magnitude','asset_class_percentile']):
            fit=fit_rule(rows,targets,family,kind)
            trials.append(dict(path_id=i,fit=fit,positive_margin=bool(fit['margin']>STRICT),short_image_not_rejected=check['not_rejected']))
        print('PATH',i,'image',check['not_rejected'],'positive',sum(t['positive_margin'] for t in trials[-6:]),flush=True)
    dump('PATH_CHECKS.json',path_checks);dump('RULE_TRIALS.json',trials)
    acceptable=[t for t in trials if t['positive_margin'] and t['short_image_not_rejected']]
    best=None
    if acceptable:
        best=min(acceptable,key=lambda t:(t['fit']['kind']!='momentum_only',t['fit']['nonzero'],-t['fit']['margin'],t['path_id']))
        fit=best['fit'];case=path_checks[best['path_id']]
        targets=[r['holdings'] for r in case['orders']]
        nav,orders=replay(px,events,targets,initial)
        next_row=next(r for r in all_rows if r['date']==window['end'])
        next_event=next(m for m in read(O/'monthly_path_constraints/META_theme17.json') if m['start']==window['end'])
        next_names=picks(next_row,fit['weights'],fit['family'])
        extension,next_orders=replay(px,[next_event],[next_names],float(nav.iloc[-1]))
        next_check=image_check(extension,max(peak,float(nav.max())),px,cal,obs,cfg)
        whole=pd.concat([nav,extension]);whole=whole[~whole.index.duplicated(keep='last')]
        combined_check=image_check(whole,peak,px,cal,obs,cfg)
        payload=dict(**best,feature_names=FEATURES,orders=orders,check=case['check'],
            next_decision=dict(order=next_orders[0],check=next_check,weights_refitted=False),
            combined_with_next_month=combined_check,
            nav=[dict(date=str(d.date()),value=float(v)) for d,v in nav.items()],
            extension_nav=[dict(date=str(d.date()),value=float(v)) for d,v in extension.items()],
            initial_nav=initial,prior_peak=peak,fit_uses_all_three_months=True,genuine_out_of_sample=False)
        dump('SELECTED_RULE.json',payload)
        print('BEST',best,case['check'],'NEXT',next_orders[0],next_check,flush=True)
    sources=read(P/'SOURCE_HASHES.json');sources.update(ps)
    for f in [O/'monthly_decision_reconstruction/SNAPSHOTS.json',O/'monthly_reconstruction_contract/CONTRACT.json']:
        sources[str(f)]=sha(f)
    for f,h in sources.items():assert sha(f)==h,f
    dump('SOURCE_HASHES.json',sources)
    dump('STATUS.json',dict(stage='short_window_rule_inference_complete',start=window['start'],end=window['end'],
        candidate_paths=len(path_checks),rule_trials=len(trials),
        positive_margin_trials=sum(t['positive_margin'] for t in trials),
        joint_positive_margin_and_image_screen_trials=len(acceptable),selected=best,
        full_period_backtest_run=False,original_sources_unchanged=len(sources),author_strategy_reproduced=False))

if __name__=='__main__':main()
