"""One-month substitution, holding all prior account history fixed."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'monthly_path_constraints'))
from study import load,read,sha,xcoord,I
sys.path.insert(0,str(P.parent/'monthly_reconstruction_contract'))
from audit import panel_error


def main():
    px,_,sources=load()
    navfile=P.parent/'monthly_path_constraints/theme17__prior__draw_v2__blocks_NAV.csv'
    contractfile=P.parent/'monthly_reconstruction_contract/CONTRACT.json'
    sources[str(navfile)]=sha(navfile);sources[str(contractfile)]=sha(contractfile)
    original=pd.read_csv(navfile,index_col=0,parse_dates=True).iloc[:,0]
    start=pd.Timestamp('2016-10-31');end=pd.Timestamp('2016-11-30')
    prefix=original.loc[:start];begin=float(prefix.iloc[-1]);peak=float(prefix.max())
    cal=read(I/'CALIBRATION.json');cfg=read(contractfile)['image_audit']
    z=pd.read_csv(I/'DIGITIZED_WEALTH.csv')
    obs={p:z[z.panel==p].copy() for p in ['wealth','relative']}
    obs['drawdown']=pd.read_csv(I/'purple_dd_pixels.csv')
    scenarios={'original':['XLE','XOM','research_equity_FB'],
               'semiconductors_keep_meta':['SMH','SOXX','research_equity_FB'],
               'previous_rule':['SMH','SOXX','INTC']}
    results={}
    for label,names in scenarios.items():
        prices=px.loc[start:end,names]
        assert prices.notna().all().all() and prices.gt(0).all().all()
        path=prices.div(prices.iloc[0]).mean(axis=1)
        nav=begin*path
        # Two independent implementations of fixed-share accounting.
        shares=begin/3/prices.iloc[0]
        assert np.allclose((prices*shares).sum(axis=1),nav,rtol=1e-12,atol=1e-12)
        if label=='original':assert np.allclose(nav,original.loc[start:end],rtol=1e-12,atol=1e-12)
        daily=path.pct_change().dropna()
        assert np.isclose(np.prod(1+daily)-1,path.iloc[-1]-1,rtol=1e-12,atol=1e-12)
        dd=nav/np.maximum(peak,nav.cummax())-1
        x=xcoord(nav.index,cal)
        diagnostics={}
        for panel in ['wealth','relative','drawdown']:
            observed=obs[panel]
            observed=observed[(observed.x>x[0]+.35)&(observed.x<x[-1]-.35)]
            if observed.empty:
                diagnostics[panel]=dict(visible_columns=0,status='no_visible_pixels');continue
            if panel=='drawdown':
                y=cal['dd']['y0']-cal['dd']['scale']*dd.to_numpy()
                tolerance=cfg['drawdown_extra_y_pixels']
            else:
                levels=np.log(nav.to_numpy())
                if panel=='relative':levels-=np.log(px.loc[nav.index,'QQQ'].to_numpy()/px.at[pd.Timestamp(cal['dates'][0]),'QQQ'])
                y=cal[panel]['y0']-cal[panel]['log_scale']*levels
                tolerance=cfg['wealth_relative_extra_y_pixels']
            d,_=panel_error(x,y,y,observed,cfg['x_pixel_radius'],tolerance)
            diagnostics[panel]=d
        ratio=nav/original.loc[start:end]
        results[label]=dict(holdings=names,
            per_security_returns={s:float(v) for s,v in (prices.iloc[-1]/prices.iloc[0]-1).items()},
            period_return=float(path.iloc[-1]-1),return_difference_pp=100*float(path.iloc[-1]-original.at[end]/begin),
            local_path_drawdown=float((path/path.cummax()-1).min()),
            global_peak_aware_drawdown_in_window=float(dd.min()),
            maximum_daily_nav_difference_percent=100*float(abs(ratio-1).max()),
            ending_nav=float(nav.iloc[-1]),ending_wealth_vertical_difference_px=float(-cal['wealth']['log_scale']*np.log(ratio.iloc[-1])),
            ending_relative_vertical_difference_px=float(-cal['relative']['log_scale']*np.log(ratio.iloc[-1])),
            image_diagnostics=diagnostics,
            daily_values=[dict(date=str(d.date()),nav=float(n),return_since_start=float(p-1),drawdown=float(dd.at[d])) for d,n,p in zip(nav.index,nav,path)])
    for f,h in sources.items():assert sha(f)==h,f
    out=dict(start=str(start.date()),end=str(end.date()),entry='same_close',weights='equal thirds then fixed units',cost=0.,
        prior_nav_unchanged=begin,prior_peak_unchanged=peak,no_monthly_rescaling=True,
        scenarios=results,source_hashes=sources,original_reconstructed_path_verified=True,
        author_exact_monthly_return_known=False,full_period_backtest_run=False,
        interpretation='A different monthly return need not be distinguishable in the compressed screenshot. Non-rejection of a local path is not confirmation of true holdings or whole-period compatibility.')
    (P/'THIRD_MONTH_SUBSTITUTION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for label,r in results.items():
        print(label,json.dumps({k:v for k,v in r.items() if k!='daily_values'},ensure_ascii=False))


if __name__=='__main__':main()
