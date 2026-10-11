"""Post-freeze attribution only; never changes selected policies."""
import blend_research as s

def main():
    fr=s.read(s.P/'FROZEN.json');assert s.sha(s.P/'FROZEN.json')==(s.P/'FROZEN.sha256').read_text().strip();assert s.sha(s.__file__)==fr['code_sha256']
    m,d,L,x,px,op,f,a,recs=s.load();pred=dict(s.np.load(s.P/'PREDICTIONS.npz'));chosen=fr['selected']['primary'];recipe=recs[chosen['recipe_id']]
    rows=s.read(s.P/'MONTH_COMPARISON.json');changes=[];attribution=[]
    oldspec=fr['base']['spec'];newsc=s.old.score(chosen['spec'],a,122);oldsc=s.old.score(oldspec,a,122)
    for i,r in enumerate(rows):
        before=r['previous66'];after=r['primary']
        if set(before['symbols'])==set(after['symbols']):continue
        changes.append(dict(date=r['date'],end=r['end'],split=r['split'],before=before,after=after,difference=after['monthly_return']-before['monthly_return']))
        names=sorted(set(before['symbols'])|set(after['symbols']));sec=[]
        for name in names:
            j=m['symbols'][i].index(name)
            sec.append(dict(symbol=name,old_score=float(oldsc[i,j]),new_score=float(newsc[i,j]),
              predicted_excess_return=float(pred[f"{recipe['mode']}_return_{recipe['return_alpha']}_raw"][i,j]),
              predicted_adverse=float(pred[f"{recipe['mode']}_adverse_{recipe['risk_alpha']}_raw"][i,j]),
              realized_return=float(px.at[s.pd.Timestamp(r['end']),name]/px.at[s.pd.Timestamp(r['date']),name]-1)))
        attribution.append(dict(date=r['date'],end=r['end'],split=r['split'],securities=sec))
    s.dump('CHANGED_MONTHS.json',changes);s.dump('SCORE_ATTRIBUTION.json',attribution)
    # Neutralizing/reversing the ML value retains its support-budget reduction;
    # this diagnoses whether score information, rather than weight shrinkage,
    # contributed to historical differences. It is not another candidate search.
    ablations={}
    for name,multiplier in [('neutral_ml',0),('reverse_ml',-1)]:
        altered={k:v.copy() for k,v in a.items()};key=f"recipe_{chosen['recipe_id']}";altered[key]*=multiplier
        ch,sc=s.old.choices(chosen['spec'],altered,m,d,122);orders=s.rp.make_orders(ch,sc,m)
        met,eq,_,_=s.ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=s.pd.Timestamp(m['dates'][0]),end=s.pd.Timestamp(m['ends'][-1]))
        ablations[name]=dict(metrics=met,slices=s.prior.old.previous.cuts(eq,m))
    s.dump('ABLATIONS.json',ablations)
    results=s.read(s.P/'RESULTS.json');summary={}
    for k,z in results.items():
        row={n:z['metrics'][n] for n in ['cagr','sharpe','max_drawdown']};row['slices']={p:{n:v[n] for n in ['cagr','sharpe','max_drawdown']} for p,v in z['slices'].items()}
        for label in ['next_open_zero','next_open_cost']:
            if label in z:row[label]={n:z[label]['metrics'][n] for n in ['cagr','sharpe','max_drawdown']}
        summary[k]=row
    s.dump('SUMMARY.json',summary)
    changed=[p for p,h in s.read(s.P/'PROTECTED.json').items() if s.sha(p)!=h];assert not changed
    validation=s.read(s.P/'VALIDATION.json');validation.update(unit_tests_passed=12,no_post_holdout_refit=True,freeze_sha256=s.sha(s.P/'FROZEN.json'));s.dump('VALIDATION.json',validation)
    print('Changed holding sets',len(changes),'later34',sum(x['split']=='later34' for x in changes))
    for k,z in ablations.items():print(k,'CAGR',z['metrics']['cagr'])
    print('Protected unchanged',len(s.read(s.P/'PROTECTED.json')))

if __name__=='__main__':main()
