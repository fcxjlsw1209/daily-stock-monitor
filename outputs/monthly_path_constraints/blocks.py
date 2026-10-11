"""Adjacent-month inverse moves preserve endpoint while changing peak history.

Single-month coordinate updates can be trapped by the terminal-wealth constraint.
This changes the solver, not the author strategy's factors or trading rules.
"""
import json
import numpy as np
import pandas as pd
from study import P, TARGET, pack, evaluate, path_series, read, dump, sha


def run(pool):
    d=pack(pool);d['draw_refinement']=True
    previous=f'{pool}__prior__draw_v2';ident=previous+'__blocks'
    choice=np.load(P/'internal_states'/f'{previous}_states.npz')['final']
    # Independent image evidence chooses stressful months, not observed strategy profits.
    obs=d['ddobs'];cal=d['cal'];start,end=pd.to_datetime(cal['dates'])
    dates=start+pd.to_timedelta((obs.x-cal['wealth']['x0'])/cal['wealth']['width']*(end-start).days,unit='D')
    periods=pd.DatetimeIndex(dates).to_period('M');risk={}
    for period in periods.unique():
        risk[period]=float((cal['dd']['y0']-obs.loc[periods==period,'ymax'].max())/cal['dd']['scale'])
    chosen=[]
    for period,value in sorted(risk.items(),key=lambda x:x[1]):
        if value > -.20:continue
        matches=[m for m,z in enumerate(d['meta']) if pd.Timestamp(z['end']).to_period('M')==period]
        if not matches:continue
        m=matches[0]
        for a in [m-2,m-1,m]:
            if 0<=a<len(choice)-1 and (a,a+1) not in chosen:chosen.append((a,a+1))
    initial_loss,initial_details=evaluate(choice,d,4.)
    initial={k:float(v[0]) for k,v in initial_details.items()}
    trials=[];trace=[];states=[choice.copy()]
    for turn in range(2):
        evaluated=[];changed=0
        for number,(a,b) in enumerate(chosen):
            na,nb=d['counts'][a],d['counts'][b]
            ea=d['E'][a,:na];eb=d['E'][b,:nb]
            total=d['E'][np.arange(len(choice)),choice].sum()
            wanted=ea[choice[a]]+eb[choice[b]]+np.log(TARGET['wealth'])-total
            order=np.argsort(eb);places=np.searchsorted(eb[order],wanted-ea)
            left=np.repeat(np.arange(na),8)
            right=order[np.clip(places[:,None]+np.arange(-4,4),0,nb-1)].ravel()
            pairs=np.unique(np.vstack([np.column_stack([left,right]),[choice[a],choice[b]]]),axis=0)
            candidates=np.repeat(choice[None,:],len(pairs),axis=0);candidates[:,a]=pairs[:,0];candidates[:,b]=pairs[:,1]
            scores=[]
            for s in range(0,len(candidates),64):scores.extend(evaluate(candidates[s:s+64],d,4.)[0])
            scores=np.array(scores);best=int(np.argmin(scores))
            before=choice[[a,b]].tolist();choice=candidates[best].copy();states.append(choice.copy())
            changed+=before!=choice[[a,b]].tolist()
            evaluated.append(dict(months=[a,b],before=before,after=choice[[a,b]].tolist(),pairs=pairs.tolist(),losses=scores.tolist()))
            if number%4==0:print('BLOCK',ident,turn,number,'of',len(chosen),'loss',round(float(scores[best]),4),flush=True)
        loss,m=evaluate(choice,d,4.)
        entry=dict(iteration=turn,changes=changed,loss=float(loss[0]),**{k:float(v[0]) for k,v in m.items()})
        trace.append(entry)
        file=P/'internal_states'/f'{ident}_{turn}.json';file.write_text(json.dumps(evaluated,separators=(',',':')))
        trials.append(dict(file=str(file),sha256=sha(file)))
        dump(f'{ident}_PROGRESS.json',dict(initial=initial,trace=trace,trial_files=trials,paired_month_indices=chosen))
        np.savez_compressed(P/'internal_states'/f'{ident}_states.npz',states=np.array(states),final=choice)
        print('BLOCK_ROUND',ident,json.dumps(entry),flush=True)
    nav=path_series(choice,d);nav.to_csv(P/f'{ident}_NAV.csv')
    dump(f'{ident}_RESULT.json',dict(pool=pool,initialization='adjacent_month_solver',**trace[-1],initial=initial,trace=trace,
        exact_window=[str(nav.index[0].date()),str(nav.index[-1].date())],future_prices_used_for_inverse_attribution=True,
        author_strategy_reproduced=False,candidate_transactions_exported=False,uniform_selection_rule_found=False,
        metric_targets_read_from_supplied_figure=TARGET))
    print('BLOCK_DONE',ident,flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--pool',choices=['core17','theme17'],required=True)
    run(ap.parse_args().pool)
