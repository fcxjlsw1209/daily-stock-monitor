"""Sensitivity to one global scale change, never independent monthly resets."""
import sys,json
from pathlib import Path
from itertools import combinations,product
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_path_constraints'))
from study import load,read,sha,xcoord,I
from inverse import score_paths,load_observations

def main():
    px,_,_=load();audits=read(P/'MONTHLY_AMBIGUITY.json');cal=read(I/'CALIBRATION.json');obs=load_observations()
    dec={r['date']:r for r in read(O/'monthly_decision_reconstruction/DECISIONS.json')}
    out=[]
    for a in audits:
        names=dec[a['start']]['eligible'];p=px.loc[a['start']:a['end'],names];combo=np.array(list(combinations(range(len(names)),3)))
        paths=np.mean(p.div(p.iloc[0]).to_numpy()[:,combo],axis=2).T;x=xcoord(p.index,cal)
        ob={k:z[(z.x>x[0]+.35)&(z.x<x[-1]-.35)] for k,z in obs.items()};cases=[]
        for offset,shift in product([-1.,0.,1.],[-.35,0.,.35]):
            delta=offset/cal['wealth']['log_scale']
            constraint=dict(start_log_nav=np.log(a['anchor_level'])+delta,prior_log_peak=np.log(a['anchor_peak'])+delta,log_tolerance=0.)
            loss=score_paths(paths,x,ob,cal,xshift=shift,constraints=constraint)[0]
            ids=np.flatnonzero(loss<=loss.min()+1)
            cases.append(dict(global_log_level_offset_pixels=offset,xshift=shift,
                best=[names[j] for j in combo[np.argmin(loss)]],near_count=len(ids),
                common_to_all_near=sorted(set.intersection(*[set(names[j] for j in combo[i]) for i in ids]))))
        out.append(dict(start=a['start'],end=a['end'],cases=cases,
            common_in_every_case=sorted(set.intersection(*[set(c['common_to_all_near']) for c in cases]))))
    (P/'ANCHOR_STRESS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in out:print(r['start'],'common',r['common_in_every_case'],[(c['global_log_level_offset_pixels'],c['xshift'],c['near_count'],c['best']) for c in r['cases']])

if __name__=='__main__':main()
