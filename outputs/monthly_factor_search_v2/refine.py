from search import *

def seed_recipes(key):
    con=sqlite3.connect(P/(key+'.sqlite'));con.row_factory=sqlite3.Row
    triples=[];provenance=[]
    for criterion in ['log_later DESC','log_dev DESC','exact DESC,overlap DESC']:
        seen=set()
        for r in con.execute('SELECT * FROM trials WHERE valid=1 ORDER BY '+criterion+' LIMIT 15000'):
            if r['s1']<0:continue
            pair=tuple(sorted([r['s1'],r['s2']])) if r['s2']>=0 else (r['s1'],-1)
            tpl=(r['c'],*pair)
            if tpl in seen:continue
            seen.add(tpl);provenance.append(dict(criterion=criterion,original_trial=r['id'],triple=tpl))
            if tpl not in triples:triples.append(tpl)
            if len(seen)>=36:break
    con.close();recipes=[]
    for c,a,b in triples:
        for wc in range(50,101):
            if b<0:recipes.append((c,a,b,wc/100,(100-wc)/100,0.));continue
            for wa in range(0,101-wc):recipes.append((c,a,b,wc/100,wa/100,(100-wc-wa)/100))
    dump(key+'_REFINEMENT_PLAN.json',dict(step=.01,selection='36distinct triples by each of full-ledger return,2021-2024return,and holdings agreement',seeds=provenance,unique_triples=len(triples),trials=len(recipes)))
    return recipes

def run_refine(key):
    meta=read(P/'INPUTS.json')[key];data=np.load(meta['file']);assert sha(meta['file'])==meta['sha256']
    rec=seed_recipes(key);con=sqlite3.connect(P/(key+'__refined.sqlite'))
    con.execute('CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY,c INTEGER,s1 INTEGER,s2 INTEGER,wc REAL,w1 REAL,w2 REAL,valid INTEGER,exact INTEGER,overlap INTEGER,log_full REAL,log_dev REAL,log_later REAL,month_dd REAL)')
    start=con.execute('SELECT COALESCE(MAX(id),-1)+1 FROM trials').fetchone()[0]
    for offset in range(start,len(rec),128):
        batch=rec[offset:offset+128];z=screen(data,batch,meta['dates'],meta['ends'])
        records=[(offset+k,*r,int(z['valid'][k]),int(z['exact'][k]),int(z['overlap'][k]),float(z['log_full'][k]),float(z['log_dev'][k]),float(z['log_later'][k]),float(z['month_dd'][k])) for k,r in enumerate(batch)]
        con.executemany('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',records);con.commit()
    print(key,'REFINED',con.execute('SELECT COUNT(*),MAX(exact),MAX(log_later) FROM trials WHERE valid=1').fetchone(),flush=True)
    con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();run_refine(a.key)
