"""Frozen candidate replay on2016-08-17--2026-09-24; no new selection or fitting."""
from validate import *

def main():
    spec=read(P/'CANDIDATE_RULES.json')['source_spec'];meta=read(P/'INPUTS.json')[spec['key']]
    packed=np.load(meta['file']);assert sha(meta['file'])==meta['sha256']
    px,raw,members,periods,sources=load_inputs();opening=load_open(px,raw)
    state,side=spec['correction'].split('__',1)
    extra=packed['X'][:,:,meta['feature_names'].index(side)]*(market_states(px).loc[meta['dates'],state].to_numpy()[:,None]*2-1)
    scores=score_of(spec,packed['X'],extra);orders=[];gross=1.
    for i,(date,end) in enumerate(zip(meta['dates'],meta['ends'])):
        eligible=np.flatnonzero(packed['eligible'][i]);ids=sorted(eligible,key=lambda j:(-scores[i,j],j))[:3]
        if len(ids)<2:ids=[]
        assert all(packed['valid'][i,j] for j in ids),(date,'missing selected close path')
        chosen=[meta['symbols'][i][j] for j in ids]
        if ids:gross*=float(np.mean(packed['ratios'][i,ids]))
        orders.append(dict(date=date,end=end,signal_date=date,symbols=chosen,scores=[float(scores[i,j]) for j in ids]))
    # All69overlapping dates must retain exactly the previously fixed candidate selections.
    old={o['date']:o for o in orders_from_pack(spec)}
    assert all(o['symbols']==old[o['date']]['symbols'] for o in orders if o['date'] in old)
    results=[]
    for execution,cost in [('close',0.),('close',.0005),('open',.0005)]:
        result,eq,trades,holdings=replay(px,opening,orders,cost,execution,start=START,end=END)
        label=execution+('_zero' if cost==0 else '_cost')
        if label=='close_zero':assert abs(result['final_equity']/100000.-gross)<1e-9
        result.update(label=label,wealth_multiple=result['final_equity']/100000.)
        for annual in result['annual_returns']:
            annual['partial']=(annual['year']==START.year and START.month!=1) or (annual['year']==END.year and END.month!=12)
        results.append(result)
        dump('FULL_PERIOD_'+label+'_DAILY.json',[dict(date=str(d.date()),equity=float(v)) for d,v in eq.items()])
        dump('FULL_PERIOD_'+label+'_HOLDINGS.json',holdings)
    qq=px.loc[START:END,'QQQ']/px.at[START,'QQQ']*100000.
    benchmarks=[]
    for cost in [0.,.0005]:
        result=stats(qq/(1+cost));benchmarks.append(dict(symbol='QQQ',buy_cost=cost,**result,wealth_multiple=result['final_equity']/100000.))
    assert all(sha(f)==h for f,h in sources.items())
    dump('FULL_PERIOD_RESULTS.json',dict(period=[str(START.date()),str(END.date())],fixed_rule='CANDIDATE_RULES.json',source_rule_sha256=sha(P/'CANDIDATE_RULES.json'),results=results,benchmarks=benchmarks,holding_periods=len(orders),checks=dict(overlapping69holdings_unchanged=True,independent_zero_cost_compounding_agrees=True,source_hashes_unchanged=True),notes=['Continuous share-ledger replay; no reset at2020/2021 and no averaging segment Sharpe.', 'Initialize at first available date2016-08-17, then normal monthend schedule; next-open version initial fill2016-08-18.', 'Same existing estimatedTop32+theme17proxy universe, same coefficients, no re-optimization.', 'Retrospective strategy search result, not author replication or independent OOS.']))
    dump('FULL_PERIOD_CODE_HASHES.json',{str(P/f):sha(P/f) for f in ['full_period.py','validate.py','search.py','data_inputs.py']})
    for r in results+benchmarks:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
