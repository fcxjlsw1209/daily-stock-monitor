"""Small corrections to compact leaders chosen only by retained holdings matching."""
from compact import *

def run():
    plan,packs=prepare();allresults=[];inventory=[]
    for key,data in packs.items():
        con=sqlite3.connect(P/(key+'_compact.sqlite'));seeds=[];seen=set()
        for sl in ['A','B']:
            count=0;paths=set()
            for (encoded,) in con.execute('SELECT recipe FROM trials ORDER BY exact_'+sl+' DESC,shared_'+sl+' DESC,id LIMIT 3000'):
                rec=json.loads(encoded);s=compact_scores(data,[rec])[0];sig=pick(s,data['eligible'])[mask(data,sl)].tobytes()
                if sig in paths:continue
                paths.add(sig);count+=1
                if tuple(rec) not in seen:seeds.append(rec);seen.add(tuple(rec))
                if count>=20:break
        con.close();base=compact_scores(data,seeds);donors=[]
        for c,a,b,wc,wa,wb in seeds:
            donors.append((data['X'][:,:,a]*wa+data['X'][:,:,b]*wb)/(1-wc) if b>=0 else data['X'][:,:,a] if a>=0 else data['X'][:,:,c])
        donor=np.stack(donors);extra=data['extra'];names=read(OLD/(key+'_CORRECTIONS_PLAN.json'))['extra_names']
        tasks=[(i,j,g) for i in range(len(seeds)) for j in range(extra.shape[2]) for g in [.025,.05,.075,.1] if seeds[i][3]-g*(seeds[i][3]==1)>=.5]
        db=sqlite3.connect(P/(key+'_corrections.sqlite'));db.execute('CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY,seed INT,extra INT,gamma REAL,exact_A INT,shared_A INT,exact_B INT,shared_B INT)')
        start=db.execute('SELECT COALESCE(MAX(id),-1)+1 FROM trials').fetchone()[0]
        for offset in range(start,len(tasks),256):
            batch=tasks[offset:offset+256];sc=np.stack([base[i]+g*(extra[:,:,j]-donor[i]) for i,j,g in batch]);z=metrics(data,sc)
            db.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?)',[(offset+k,*task,int(z['A'][0][k]),int(z['A'][1][k]),int(z['B'][0][k]),int(z['B'][1][k])) for k,task in enumerate(batch)]);db.commit()
        for sl in ['A','B']:
            for id_,i,j,g in db.execute('SELECT id,seed,extra,gamma FROM trials ORDER BY exact_'+sl+' DESC,shared_'+sl+' DESC,id LIMIT 5'):
                sc=base[i]+g*(extra[:,:,j]-donor[i]);c,a,b,wc,wa,wb=seeds[i];mult=1-g/(1-wc) if wc<1 else 0
                coef={data['feature_names'][c]:wc if wc<1 else 1-g}
                if a>=0:coef[data['feature_names'][a]]=wa*mult
                if b>=0:coef[data['feature_names'][b]]=wb*mult
                coef[names[j]]=g
                allresults.append(dict(key=key,slice=sl,kind='compact_small_correction',trial=id_,recipe=seeds[i],extra=j,gamma=g,coefficients=coef,evaluations={z:evaluate(data,sc,z) for z in ['all','A','B']}))
            r=allresults[-5];print('CORRECTION',key,sl,'exact',r['evaluations'][sl]['exact_periods'],'overlap',r['evaluations'][sl]['shared_positions'],'rule',r['coefficients'],flush=True)
        db.close();inventory.append(dict(key=key,seeds=seeds,trials=len(tasks)));dump('CORRECTION_LEADERS.json',allresults);dump('CORRECTION_INVENTORY.json',inventory)
    print('DONE',flush=True)
if __name__=='__main__':run()
