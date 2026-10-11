"""Finite compact-rule grid. The objective is source holdings, never profit."""
from filtered_study import *
import itertools,sqlite3

def metrics(data,scores):
    order=np.argsort(-np.where(data['eligible'][None,:,:],scores,-np.inf),axis=2,kind='stable')[:,:,:3]
    hits=np.take_along_axis(data['labels'][None,:,:],order,axis=2).sum(axis=2)
    out={}
    for sl in ['A','B']:
        h=hits[:,mask(data,sl)];out[sl]=(np.sum(h==3,axis=1),h.sum(axis=1))
    return out

def compact_scores(data,recipes):
    x=data['X'];return np.stack([x[:,:,c]*wc+(x[:,:,a]*wa if a>=0 else 0)+(x[:,:,b]*wb if b>=0 else 0) for c,a,b,wc,wa,wb in recipes])

def run():
    plan,packs=prepare();results=[];total=0
    for key,data in packs.items():
        pairs=[(a,b) for a,b in itertools.combinations(data['side_ids'],2) if b!=a+1 or a%2!=0]
        recipes=[(c,-1,-1,1.,0.,0.) for c in data['core_ids']]
        for c in data['core_ids']:
            for wc in [.5,.6,.7,.8,.9]:
                recipes += [(c,a,-1,wc,1-wc,0.) for a in data['side_ids']]
                for a,b in pairs:
                    for share in [.25,.5,.75]:recipes.append((c,a,b,wc,(1-wc)*share,(1-wc)*(1-share)))
        con=sqlite3.connect(P/(key+'_compact.sqlite'))
        con.execute('CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY,recipe TEXT,exact_A INT,shared_A INT,exact_B INT,shared_B INT)')
        start=con.execute('SELECT COALESCE(MAX(id),-1)+1 FROM trials').fetchone()[0]
        for offset in range(start,len(recipes),256):
            batch=recipes[offset:offset+256];scores=compact_scores(data,batch);z=metrics(data,scores)
            vals=[(offset+i,json.dumps(rec),int(z['A'][0][i]),int(z['A'][1][i]),int(z['B'][0][i]),int(z['B'][1][i])) for i,rec in enumerate(batch)]
            con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?)',vals);con.commit()
        total+=len(recipes)
        for sl in ['A','B']:
            leaders=[]
            for id_,recipe in con.execute('SELECT id,recipe FROM trials ORDER BY exact_'+sl+' DESC,shared_'+sl+' DESC,id LIMIT 20'):
                recipe=json.loads(recipe);scores=compact_scores(data,[recipe])[0]
                leaders.append(dict(key=key,slice=sl,kind='compact_grid',trial=id_,recipe=recipe,coefficients={data['feature_names'][j]:w for j,w in zip(recipe[:3],recipe[3:]) if j>=0},evaluations={z:evaluate(data,scores,z) for z in ['all','A','B']}))
            results+=leaders;best=leaders[0]
            print('COMPACT',key,sl,'trials',len(recipes),'exact',best['evaluations'][sl]['exact_periods'],'overlap',best['evaluations'][sl]['shared_positions'],flush=True)
        con.close();dump('COMPACT_LEADERS.json',results)
    dump('COMPACT_INVENTORY.json',{'trials':total,'core':'One of long momentum columns7-27 retained by existing feature deduplication','core_weight':[.5,.6,.7,.8,.9,1.], 'support':'up to2 signed support factors, share25/50/75%, total weight<=50%','normalizations':['percentile','magnitude'],'objective':'exact retained periods, then total matched positions, then earlier enumerated rule'})
    print('COMPACT DONE',total,flush=True)
if __name__=='__main__':run()
