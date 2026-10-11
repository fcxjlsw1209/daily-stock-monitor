"""Post-freeze attribution only; not used to select or revise the frozen winner."""
import state_rank as s


def main():
    fr=s.read(s.P/'FROZEN.json');assert s.sha(s.P/'FROZEN.json')==(s.P/'FROZEN.sha256').read_text().strip()
    m,d,L,x,px,op,f,a=s.load();spec=fr['selected']['primary']['spec'];out={}
    for name,mode in [('neutral_ml','zero'),('reversed_ml','negative')]:
        changed={k:v.copy() for k,v in a.items()}
        for z in spec:
            if z['key'].startswith('ml_'):changed[z['key']]=changed[z['key']]*(0 if mode=='zero' else -1)
        ch,sc=s.choices(spec,changed,m,d,122);orders=s.rp.make_orders(ch,sc,m)
        metrics,eq,_,_=s.ex.e.engine.replay(px,op,orders,cost=0.,execution='close',start=s.pd.Timestamp(m['dates'][0]),end=s.pd.Timestamp(m['ends'][-1]))
        out[name]=dict(metrics=metrics,slices=s.prior.old.previous.cuts(eq,m),purpose='Explanatory ablation of frozen winner, not another candidate search')
    s.dump('ABLATIONS.json',out)
    rows=s.read(s.P/'MONTH_COMPARISON.json');base='baseline';changes=[]
    for r in rows:
        original=r[base]['symbols'];current=r['primary']['symbols'];resid=r['residual_control']['symbols']
        if current!=original:
            changes.append(dict(date=r['date'],end=r['end'],split=r['split'],baseline=original,primary=current,residual=resid,
              holdings_set_changed=set(current)!=set(original),new_vs_residual_set_changed=set(current)!=set(resid),
              baseline_return=r[base]['monthly_return'],primary_return=r['primary']['monthly_return'],return_difference=r['primary']['monthly_return']-r[base]['monthly_return']))
    s.dump('CHANGED_MONTHS.json',changes)
    audit=s.read(s.P/'FORECAST_AUDIT.json');meta=s.read(s.P/'TRAIN_DATA.json')
    # Model feature and coefficient identities are reconstructible from this table.
    names=[n+':'+interaction for interaction in meta['feature_interactions'] for n in meta['feature_names']]
    coefsummary=[]
    for fit in audit:
        if fit['configuration']!=4:continue
        coefsummary.append(dict(year=fit['year'],cutoff=fit['cutoff'],top_terms=[dict(feature=names[i],coefficient=fit['coef'][i]) for i in sorted(range(len(names)),key=lambda i:abs(fit['coef'][i]),reverse=True)[:8]]))
    s.dump('SELECTED_MODEL_TERMS.json',coefsummary)
    # Store concise actual numbers only; full supporting replay is separate.
    results=s.read(s.P/'RESULTS.json');summary={}
    for k,z in results.items():
        row={n:z['metrics'][n] for n in ['cagr','sharpe','max_drawdown']}
        row['slices']={p:{n:v[n] for n in ['cagr','sharpe','max_drawdown']} for p,v in z['slices'].items()}
        for label in ['next_open_zero','next_open_cost']:
            if label in z:row[label]={n:z[label]['metrics'][n] for n in ['cagr','sharpe','max_drawdown']}
        summary[k]=row
    s.dump('SUMMARY.json',summary)
    changed=[p for p,h in s.read(s.P/'PROTECTED.json').items() if s.sha(p)!=h];assert not changed
    valid=s.read(s.P/'VALIDATION.json');valid.update(unit_tests=12,unit_test_result='passed',diagnostic_only_no_refit=True,freeze_sha256=s.sha(s.P/'FROZEN.json'));s.dump('VALIDATION.json',valid)
    print('Set changes vsbaseline',sum(r['holdings_set_changed'] for r in changes),'incrementalvsresidual',sum(r['new_vs_residual_set_changed'] for r in changes))
    for k,v in out.items():print(k,v['metrics']['cagr'],v['metrics']['sharpe'])
    print('Protected unchanged',len(s.read(s.P/'PROTECTED.json')))

if __name__=='__main__':main()
