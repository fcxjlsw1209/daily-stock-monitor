"""Select by identification stability; audit image ambiguity before fitting rules."""
from pathlib import Path
from itertools import combinations
import sys,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_path_constraints'))
from study import load,read,sha,xcoord,I
from inverse import score_paths,load_observations

def dump(name,z): (P/name).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    sources={};all_paths=[];meta=read(O/'monthly_path_constraints/META_theme17.json')
    for file in sorted((O/'monthly_path_constraints').glob('*_RESULT.json')):
        r=read(file);ident=file.name.replace('_RESULT.json','')
        mf=O/'monthly_path_constraints'/f'META_{r["pool"]}.json'
        sf=O/'monthly_path_constraints/internal_states'/f'{ident}_states.npz'
        choice=np.load(sf)['final'];m=read(mf);picked=[]
        for i,entry in enumerate(m):
            cache=np.load(entry['file']);picked.append(set(entry['symbols'][j] for j in cache['combos'][choice[i]]))
        all_paths.append(dict(id=ident,holdings=picked))
        for f in [file,mf,sf]:sources[str(f)]=sha(f)
    obs=load_observations();cal=read(I/'CALIBRATION.json');month_rows=[]
    for i,m in enumerate(meta):
        if i==0 or m['end']=='2026-09-24':continue
        hs=[r['holdings'][i] for r in all_paths]
        counts={s:sum(s in h for h in hs) for s in set.union(*hs)}
        top=sorted(counts,key=lambda s:(-counts[s],s))[:3]
        xx=xcoord(pd.DatetimeIndex([m['start'],m['end']]),cal)
        visible={p:int(((z.x>xx[0]+.35)&(z.x<xx[1]-.35)).sum()) for p,z in obs.items()}
        month_rows.append(dict(event=i,start=m['start'],end=m['end'],
            unanimous=all(h==hs[0] for h in hs),top_three_agreement=sum(counts[s] for s in top)/(3*len(hs)),
            inclusion_count=counts,path_count=len(hs),visible=visible,
            path_holdings={r['id']:sorted(r['holdings'][i]) for r in all_paths}))
    windows=[]
    for i in range(len(month_rows)-2):
        z=month_rows[i:i+3]
        windows.append(dict(start=z[0]['start'],end=z[-1]['end'],
            unanimous_months=sum(r['unanimous'] for r in z),
            mean_top_three_agreement=float(np.mean([r['top_three_agreement'] for r in z])),
            visible_columns=sum(sum(r['visible'].values()) for r in z),months=z))
    windows.sort(key=lambda r:(-r['unanimous_months'],-r['mean_top_three_agreement'],-r['visible_columns'],r['start']))
    chosen=windows[0];dump('WINDOW_SELECTION.json',dict(chosen=chosen,ranked_windows=windows,selected_by_returns=False))
    print('SELECTED',chosen['start'],chosen['end'],chosen['unanimous_months'],flush=True)
    px,pools,ps=load();sources.update(ps)
    nf=O/'monthly_path_constraints/theme17__prior__draw_v2__blocks_NAV.csv'
    baseline=pd.read_csv(nf,index_col=0,parse_dates=True).iloc[:,0];sources[str(nf)]=sha(nf)
    df=O/'monthly_decision_reconstruction/DECISIONS.json';sources[str(df)]=sha(df)
    dec={r['date']:r for r in read(df)}
    audits=[]
    for m in chosen['months']:
        start,end=pd.Timestamp(m['start']),pd.Timestamp(m['end']);names=dec[m['start']]['eligible']
        prices=px.loc[start:end,names]
        assert prices.notna().all().all() and prices.gt(0).all().all()
        combo=np.array(list(combinations(range(len(names)),3)),dtype=int)
        ratios=prices.div(prices.iloc[0]).to_numpy()
        paths=np.mean(ratios[:,combo],axis=2).T
        x=xcoord(prices.index,cal);ob={p:z[(z.x>x[0]+.35)&(z.x<x[-1]-.35)].copy() for p,z in obs.items()}
        level=float(baseline.at[start]);peak=float(baseline.loc[:start].max())
        anchor=dict(start_log_nav=float(np.log(level)),prior_log_peak=float(np.log(peak)),log_tolerance=0.)
        losses=[];shift_rows=[]
        for shift in [-.35,0.,.35]:
            res=score_paths(paths,x,ob,cal,xshift=shift,constraints=anchor)
            assert res is not None and np.isfinite(res[0]).all()
            loss=res[0];losses.append(loss);ids=np.flatnonzero(loss<=loss.min()+1)
            frequency={s:float(np.mean(np.any(combo[ids]==j,axis=1))) for j,s in enumerate(names)}
            exclusion={s:float(loss[~np.any(combo==j,axis=1)].min()-loss.min()) for j,s in enumerate(names)}
            shift_rows.append(dict(shift=shift,best_holdings=[names[j] for j in combo[np.argmin(loss)]],
                best_loss=float(loss.min()),near_count=len(ids),near_frequency=frequency,exclusion_loss_gap=exclusion))
        L=np.vstack(losses)
        near_any=np.any(L<=L.min(axis=1)[:,None]+1,axis=0)
        near_all=np.all(L<=L.min(axis=1)[:,None]+1,axis=0)
        records=[]
        for j,c in enumerate(combo):
            records.append(dict(index=j,holdings=[names[k] for k in c],loss_by_shift=L[:,j].tolist(),
                near_any=bool(near_any[j]),near_all=bool(near_all[j])))
        row=dict(start=m['start'],end=m['end'],eligible_count=len(names),triple_count=len(combo),
            anchor_level=level,anchor_peak=peak,shifts=shift_rows,
            near_any_count=int(near_any.sum()),near_all_count=int(near_all.sum()),
            minimum_near_frequency={s:min(z['near_frequency'][s] for z in shift_rows) for s in names},
            minimum_exclusion_gap={s:min(z['exclusion_loss_gap'][s] for z in shift_rows) for s in names},
            candidates=records)
        audits.append(row);print('AUDIT',row['start'],row['eligible_count'],row['near_any_count'],row['near_all_count'],flush=True)
    dump('MONTHLY_AMBIGUITY.json',audits)
    for f,h in sources.items():assert sha(f)==h,f
    dump('SOURCE_HASHES.json',sources)
    dump('STATUS.json',dict(stage='window_selected_and_ambiguity_audited',start=chosen['start'],end=chosen['end'],
        original_sources_unchanged=len(sources),full_period_backtest_run=False,verified_author_trades=False))

if __name__=='__main__':main()
