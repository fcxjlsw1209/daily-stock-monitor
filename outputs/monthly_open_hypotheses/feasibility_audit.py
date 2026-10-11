"""Scope-limited feasibility diagnostic, never generates retrospective trades."""
from pathlib import Path
import itertools,json
import numpy as np
from scipy.optimize import linprog
import open_inverse as m

def feasible(X,d,groups,mask,ordered=False):
 D=[];covered=[];missing=[]
 for i in np.flatnonzero(mask):
  target=[int(k) for k in d['order'][i] if k>=0]
  if len(target)!=3:missing.append(int(i));continue
  covered.append(int(i));others=[int(k) for k in np.flatnonzero(d['eligible'][i]) if k not in target]
  ps=[(a,b) for a in target for b in others]
  if ordered:ps+=list(itertools.combinations(target,2))
  D.extend(X[i,a]-X[i,b] for a,b in ps)
 D=np.array(D);n=X.shape[-1]
 # Free side weights w=p-n; monotonic core has negative part fixed0.
 A=np.column_stack([-D,D,np.ones(len(D))]);A=np.vstack([A,np.r_[np.ones(2*n),0.]])
 b=np.r_[np.zeros(len(D)),1.];bounds=[(0,None)]*n+[(0,0) if g=='core' else (0,None) for g in groups]+[(0,None)]
 res=linprog(np.r_[np.zeros(2*n),-1.],A_ub=A,b_ub=b,bounds=bounds,method='highs')
 assert res.success,res.message
 w=res.x[:n]-res.x[n:2*n];margin=float(res.x[-1]);actual=float(np.min(np.einsum('ij,j->i',D,w,optimize=False)))
 assert actual>=margin-1e-7
 return dict(covered_months=len(covered),unavailable_month_indices=missing,max_common_margin=margin,strict_common_score_separable=margin>1e-7,constraints=len(D),weights=w.tolist(),interpretation='Only tests one common fresh-ranking linear score in the declared feature basis. Does not test arbitrary nonlinear functions, unknown inputs, retention or stateful exits. Zero is not proof that author guidance is impossible.')

def main():
 meta,packs,states=m.load();rows=[]
 for key,arch,period,ordered in itertools.product([k for k in packs if k.endswith('zclip')],['linear','regime','spline'],['training35','all69'],[False,True]):
  d=packs[key];X,names,groups=m.design(d,states,arch);mask=np.arange(69)<35 if period=='training35' else np.ones(69,bool)
  r=feasible(X,d,groups,mask,ordered);rows.append(dict(pack=key,arch=arch,period=period,ordered=ordered,**r))
 m.dump('FEASIBILITY.json',rows)
 print('Checked',len(rows),'common-score constraint systems;',sum(r['strict_common_score_separable'] for r in rows),'strictly separable.',flush=True)
if __name__=='__main__':main()
