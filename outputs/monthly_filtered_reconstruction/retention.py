"""Test monthly holding persistence as a separate hypothesis, recursively.
Past author holdings are never fed into decisions; all69periods including excluded periods update state.
"""
from filtered_study import *
from compact import compact_scores

def recursive_scores(data,scores,buffer=3,bonus=0.):
    previous=set();selected=[];scored=[];rank_rows=[]
    for i,ss in enumerate(data['symbols']):
        sc=(1-bonus)*scores[i]+bonus*(np.array([s in previous for s in ss],float)-.5)
        order=sorted(np.flatnonzero(data['eligible'][i]),key=lambda j:(-sc[j],ss[j]));within={ss[j] for j in order[:buffer]}
        held=[j for j in order if ss[j] in previous and ss[j] in within][:3] if buffer>3 else []
        chosen=(held+[j for j in order if j not in held])[:3]
        if len(chosen)<2:chosen=[]
        rank_rows.append({display(ss[j]):k+1 for k,j in enumerate(order) if data['labels'][i,j]})
        previous={ss[j] for j in chosen};selected.append(chosen)
        # Synthetic rank utility encodes decisions only for common evaluation, not another factor.
        z=np.full(len(ss),-1.);z[chosen]=1.;scored.append(z)
    result=evaluate(data,np.array(scored),'all')
    for row,ranks in zip(result['details'],rank_rows):row['target_score_ranks']=ranks
    return result,selected

def score_rule(data,r):
    scores=compact_scores(data,[r['recipe']])[0]
    if r['kind']=='compact_small_correction':
        c,a,b,wc,wa,wb=r['recipe'];donor=(data['X'][:,:,a]*wa+data['X'][:,:,b]*wb)/(1-wc) if b>=0 else data['X'][:,:,a] if a>=0 else data['X'][:,:,c]
        scores+=r['gamma']*(data['extra'][:,:,r['extra']]-donor)
    return scores

def run():
    plan,packs=prepare();allout=[];count=0
    base=read(P/'COMPACT_LEADERS.json')+read(P/'CORRECTION_LEADERS.json')
    for key,data in packs.items():
        seen=set();rules=[]
        for r in base:
            signature=(tuple(r['recipe']),r.get('extra'),r.get('gamma'))
            if r['key']==key and signature not in seen:rules.append(r);seen.add(signature)
        for r in rules:
            sc=score_rule(data,r)
            variants=[(buffer,0.) for buffer in [3,4,5,6]]+[(3,b) for b in [.025,.05,.1,.15] if r['recipe'][3]*(1-b)>=.5]
            for buffer,bonus in variants:
                full,chosen=recursive_scores(data,sc,buffer,bonus);evaluations={}
                for sl in ['all','A','B']:
                    details=[x for x,keep in zip(full['details'],mask(data,sl)) if keep]
                    evaluations[sl]=dict(periods=len(details),exact_periods=sum(x['exact'] for x in details),shared_positions=sum(x['shared'] for x in details),total_positions=3*len(details),details=details)
                allout.append(dict(key=key,kind='recursive_retention',buffer=buffer,bonus=bonus,coefficients=r['coefficients'],source=r,evaluations=evaluations));count+=1
    leaders=[]
    for sl in ['A','B']:
        ordered=sorted(allout,key=lambda r:(-r['evaluations'][sl]['exact_periods'],-r['evaluations'][sl]['shared_positions']))
        for r in ordered[:5]:leaders.append(dict(r,slice=sl))
        r=ordered[0];print('RETENTION',sl,'exact',r['evaluations'][sl]['exact_periods'],'shared',r['evaluations'][sl]['shared_positions'],'buffer',r['buffer'],'bonus',r['bonus'],'coef',r['coefficients'],flush=True)
    dump('RETENTION_LEADERS.json',leaders);dump('RETENTION_INVENTORY.json',dict(trials=count,description='Scores from matching-only compact leaders; deterministic monthly state updates through all69periods, no teacher forcing. No new factor weights fitted. Original author retention rule unknown.'))
if __name__=='__main__':run()
