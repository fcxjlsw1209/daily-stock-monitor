"""Alternate a common ranking explanation with image-constrained latent paths.
All fitted coefficients are retrospective. No actual author-trade export.
"""
import json
import numpy as np
from study import P,read,dump,sha,load,pack,evaluate,path_series,TARGET
from explain import frames,fit_pairwise,FEATURES


def fit_choice(choice,d,rows):
    xs=[];bs=[]
    for m,(ci,row) in enumerate(zip(choice,rows)):
        z=d['meta'][m];held={z['symbols'][j] for j in d['combos'][m][ci]}
        yes=[j for j,s in enumerate(row['symbols']) if s in held]
        no=[j for j,s in enumerate(row['symbols']) if s not in held]
        for j in yes:
            xs.extend(row['X'][j]-row['X'][no]);bs.extend(.5*(row['cores']['raw'][j]-row['cores']['raw'][no]))
    return fit_pairwise(np.array(xs),np.array(bs))


def penalties(d,rows,coef):
    pen=np.zeros_like(d['E']);deficits=np.zeros_like(pen)
    for m,row in enumerate(rows):
        z=d['meta'][m];score=.5*row['cores']['raw']+np.einsum('ij,j->i',row['X'],coef,optimize=False)
        index={s:j for j,s in enumerate(row['symbols'])}
        mapped=np.array([[index[z['symbols'][j]] for j in cc] for cc in d['combos'][m]])
        top=np.argsort(-score,kind='stable')[:4]
        for k,ids in enumerate(mapped):
            other=[j for j in top if j not in ids]
            gap=max(0.,float(score[other].max()-score[ids].min()))/2 if len(ids) and other else 0.
            deficits[m,k]=gap
            pen[m,k]=(max(0.,gap-.05)/.05)**2
    return pen,deficits


def run():
    pool='theme17';previous=f'{pool}__prior__draw_v2__blocks';ident=previous+'__joint_rule'
    px,pools,_=load();rows=frames(px,pools)[pool]
    d=pack(pool);d['draw_refinement']=True
    choice=np.load(P/'internal_states'/f'{previous}_states.npz')['final']
    initial=evaluate(choice,d,4.)[1];initial={k:float(v[0]) for k,v in initial.items()}
    traces=[];trial_files=[];states=[choice.copy()];rng=np.random.default_rng(20261010)
    for cycle in range(3):
        coef,rank_loss,fit_trace=fit_choice(choice,d,rows)
        pen,deficit=penalties(d,rows,coef);d['factor_penalty']=pen
        bad_before=int((deficit[np.arange(len(choice)),choice]>.05).sum())
        evaluated=[];changed=0
        for counter,m in enumerate(rng.permutation(len(choice))):
            candidate=np.repeat(choice[None,:],d['counts'][m],axis=0);candidate[:,m]=np.arange(d['counts'][m])
            losses=[]
            for n in range(0,len(candidate),64):losses.extend(evaluate(candidate[n:n+64],d,4.)[0])
            losses=np.array(losses);best=int(np.argmin(losses))
            evaluated.append(dict(month=int(m),before=int(choice[m]),after=best,losses=losses.tolist()))
            changed+=choice[m]!=best;choice[m]=best;states.append(choice.copy())
            if counter%40==0:print('JOINT',cycle,counter,'loss',round(float(losses[best]),4),flush=True)
        loss,metric=evaluate(choice,d,4.)
        trace=dict(iteration=cycle,loss=float(loss[0]),changes=int(changed),coefficients=dict(zip(FEATURES,coef.tolist())),
            ranking_fit_trace=fit_trace,ranking_fit_loss=rank_loss,large_correction_months_before=bad_before,
            large_correction_months_after=int((deficit[np.arange(len(choice)),choice]>.05).sum()),
            **{k:float(v[0]) for k,v in metric.items()})
        traces.append(trace);f=P/'internal_states'/f'{ident}_{cycle}.json';f.write_text(json.dumps(evaluated,separators=(',',':')))
        trial_files.append(dict(file=str(f),sha256=sha(f)))
        np.savez_compressed(P/'internal_states'/f'{ident}_states.npz',states=np.array(states),final=choice)
        dump(f'{ident}_PROGRESS.json',dict(initial=initial,trace=traces,trial_files=trial_files))
        print('JOINT_ROUND',json.dumps({k:v for k,v in trace.items() if k!='ranking_fit_trace'}),flush=True)
    nav=path_series(choice,d);nav.to_csv(P/f'{ident}_NAV.csv')
    dump(f'{ident}_RESULT.json',dict(pool=pool,initialization='joint_same_rule_and_path',**traces[-1],initial=initial,trace=traces,
        exact_window=[str(nav.index[0].date()),str(nav.index[-1].date())],future_prices_used_for_inverse_attribution=True,
        author_strategy_reproduced=False,candidate_transactions_exported=False,uniform_selection_rule_found=False,
        metric_targets_read_from_supplied_figure=TARGET))
    print('JOINT_DONE',flush=True)


if __name__=='__main__':run()
