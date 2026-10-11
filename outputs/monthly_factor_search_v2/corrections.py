"""Small (2.5-10%) ML / observable market-state corrections; all trials logged."""
from search import *

def choose_seeds(key,meta,data):
    rows=[]
    for suffix in ['','__refined']:
        con=sqlite3.connect(P/(key+suffix+'.sqlite'));con.row_factory=sqlite3.Row
        for criterion in ['log_later DESC','log_dev DESC','exact DESC,overlap DESC']:
            found=[];seen=set()
            rr=list(con.execute('SELECT * FROM trials WHERE valid=1 ORDER BY '+criterion+' LIMIT 3000'))
            for off in range(0,len(rr),128):
                batch=rr[off:off+128]
                rec=[tuple(r[t] for t in ['c','s1','s2','wc','w1','w2']) for r in batch]
                z=screen(data,rec,meta['dates'],meta['ends'])
                for k,r in enumerate(batch):
                    signature=z['picks'][k,data['label_mask']].tobytes()
                    if signature in seen:continue
                    seen.add(signature);found.append(dict(source=key+suffix,trial=r['id'],recipe=rec[k],criterion=criterion))
                    if len(found)>=20:break
                if len(found)>=20:break
            rows+=found
        con.close()
    seen=set();unique=[]
    for r in rows:
        signature=tuple(r['recipe'])
        if signature not in seen:unique.append(r);seen.add(signature)
    return unique

def run(key):
    meta=read(P/'INPUTS.json')[key];packed=np.load(meta['file']);assert sha(meta['file'])==meta['sha256']
    seeds=choose_seeds(key,meta,packed);mask=packed['label_mask'];d={k:packed[k][mask] for k in ['X','eligible','ratios','valid','labels','label_mask']}
    ds=np.array(meta['dates'])[mask].tolist();ends=np.array(meta['ends'])[mask].tolist();syms=[s for s,m in zip(meta['symbols'],mask) if m]
    px,raw,members,periods,sources=load_inputs();states=market_states(px).loc[ds]*2-1
    extra=[];enames=[]
    predfile=O/'monthly_ledger_reconstruction/MONTHLY_ML_PREDICTIONS.json';pred=read(predfile)
    for alpha in [.1,1.,10.]:
        rows={r['date']:r for r in pred if r['pool']==meta['pool'] and r['alpha']==alpha};arr=np.zeros(d['eligible'].shape)
        for i,date in enumerate(ds):
            r=rows[date];assert r['cutoff']<=date
            lookup=dict(zip(r['symbols'],r['centered_prediction_rank']))
            for j,s in enumerate(syms[i]):
                if d['eligible'][i,j]:assert s in lookup,(date,s)
                arr[i,j]=lookup.get(s,0.)
        enames.append('ridge_alpha'+str(alpha));extra.append(arr)
    for state in states:
        for side in meta['side_ids']:
            enames.append(state+'__'+meta['feature_names'][side]);extra.append(d['X'][:,:,side]*states[state].to_numpy()[:,None])
    ex=np.stack(extra,axis=2);np.savez_compressed(P/(key+'__correction_features.npz'),X=ex)
    plan=dict(seed_selection='20 distinct holdings paths per coarse/refined database and per objective',seeds=seeds,extra_names=enames,weights=[.025,.05,.075,.1],preserve_core='Keep core weight unchanged; take correction budget proportionally from non-core weights. Pure momentum moves correction budget from core, never below50%.',ridge_source=str(predfile),ridge_sha256=sha(predfile),dates=ds,ends=ends,feature_file=str(P/(key+'__correction_features.npz')))
    dump(key+'_CORRECTIONS_PLAN.json',plan)
    con=sqlite3.connect(P/(key+'__corrections.sqlite'))
    con.execute('CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY,seed INTEGER,extra INTEGER,gamma REAL,valid INTEGER,exact INTEGER,overlap INTEGER,log_dev REAL,log_later REAL,month_dd REAL)')
    start=con.execute('SELECT COALESCE(MAX(id),-1)+1 FROM trials').fetchone()[0]
    tasks=[(i,j,g) for i in range(len(seeds)) for j in range(len(enames)) for g in plan['weights']]
    bases=[];donors=[]
    for seed in seeds:
        c,a,b,wc,wa,wb=seed['recipe'];core=d['X'][:,:,c]*wc;side=np.zeros(core.shape)
        if a>=0:side+=d['X'][:,:,a]*wa
        if b>=0:side+=d['X'][:,:,b]*wb
        bases.append(core+side);donors.append(side/(1-wc) if wc<1-1e-10 else core)
    base=np.stack(bases);donor=np.stack(donors)
    for offset in range(start,len(tasks),128):
        batch=tasks[offset:offset+128];scores=np.stack([base[i]+g*(ex[:,:,j]-donor[i]) for i,j,g in batch],axis=2)
        dd={**d,'X':scores};rec=[(i,-1,-1,1.,0.,0.) for i in range(len(batch))];z=screen(dd,rec,ds,ends)
        vals=[(offset+k,*task,int(z['valid'][k]),int(z['exact'][k]),int(z['overlap'][k]),float(z['log_dev'][k]),float(z['log_later'][k]),float(z['month_dd'][k])) for k,task in enumerate(batch)]
        con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?)',vals);con.commit()
    print(key,'CORRECTIONS',len(tasks),con.execute('SELECT MAX(exact),MAX(log_later) FROM trials WHERE valid=1').fetchone(),flush=True);con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();run(a.key)
