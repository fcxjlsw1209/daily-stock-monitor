"""Diagnostic local feasibility; local oracle weights are NEVER a trading rule."""
import itertools
import numpy as np
from scipy.optimize import linprog
from ordered_study import P,read,dump,pack,basis,feasibility,sha

def minimum_correction(scores,d,i):
    if (d['order'][i]<0).any():return None
    au=list(d['order'][i]);other=np.flatnonzero(d['eligible'][i]&~d['labels'][i])
    ps=[(int(a),int(b)) for a in au for b in other]+list(itertools.combinations(au,2))
    n=len(scores);A=[];b=[]
    for a,z in ps:
        row=np.zeros(n+1);row[a]=-1;row[z]=1;A.append(row);b.append(scores[a]-scores[z]-1e-6)
    for j in range(n):
        row=np.zeros(n+1);row[j]=1;row[-1]=-1;A.append(row);b.append(0)
        row=np.zeros(n+1);row[j]=-1;row[-1]=-1;A.append(row);b.append(0)
    cost=np.r_[np.zeros(n),1.]
    z=linprog(cost,A_ub=A,b_ub=b,bounds=[(None,None)]*n+[(0,None)],method='highs')
    assert z.success,z.message
    return float(z.x[-1])

def main():
    m=read(P/'INPUTS.json');winner=read(P/'FROZEN.json')['winner'];dd=pack(winner['key'],m)
    X,spec=basis(dd,winner['core'],winner['vol_sign'],winner['acceleration']);scores=np.einsum('ijk,k->ij',X,winner['weights'],optimize=False)
    report=[dict(date=p['start'],author=p['holdings'],best_margin=None,best_spec=None,feasible_specs=0,
                 min_per_symbol_score_correction=minimum_correction(scores[i],dd,i),
                 unavailable=dd['coverage'][i]['author_unavailable']) for i,p in enumerate(m['periods'])]
    all_specs=[];solved=0
    # Same predeclared factor family, acceleration enabled but may have zero
    # weight: its feasible region contains the no-acceleration subfamily.
    for norm,core,sign in itertools.product(['percentile','magnitude'],m['cores'],[1.,-1.]):
        d=pack('top32_theme17_'+norm,m);xx,ss=basis(d,core,sign,True);spec_id=len(all_specs)
        all_specs.append(dict(normalization=norm,core=core,volatility_sign=sign,spec=ss))
        for i,r in enumerate(report):
            if r['unavailable']:continue
            z=feasibility(xx,d,ss,i,True);solved+=1
            r['feasible_specs']+=z['strictly_feasible']
            if r['best_margin'] is None or z['max_margin']>r['best_margin']:
                r.update(best_margin=z['max_margin'],best_spec=spec_id,oracle_weights=z['weights'])
        if len(all_specs)%16==0:print('Local diagnostic',len(all_specs),'architectures',solved,'LPs',flush=True)
    clean=[r for r in report if not r['unavailable']]
    summary=dict(comparable_months=len(clean),feasible_in_any_family=sum(r['best_margin']>1e-8 for r in clean),
        infeasible_dates=[r['date'] for r in clean if r['best_margin']<=1e-8],
        unavailable_months=[r['date'] for r in report if r['unavailable']],linear_programs=solved,
        arbitrary_residual_5pct_weight_sufficient=sum(r['min_per_symbol_score_correction']<=.025+1e-8 for r in clean),
        arbitrary_residual_10pct_weight_sufficient=sum(r['min_per_symbol_score_correction']<=.05+1e-8 for r in clean),
        warning='Local weights use that month author labels; not usable predictions, never replayed. Score correction is optimistic arbitrary per-symbol additive correction, NOT a trained ML forecast and NOT proof about the author model. A normalized ML score in[-.5,.5] with10% additive weight can move a symbol at most.05.')
    dump('FAMILY_FEASIBILITY.json',dict(summary=summary,months=report,specifications=all_specs))
    print(summary,flush=True)

if __name__=='__main__':main()
