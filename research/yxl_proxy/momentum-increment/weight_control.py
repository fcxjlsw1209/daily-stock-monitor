"""Post-result attribution check; no tuning or strategy promotion."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from study import P, load_inputs, features, simulate, metrics, block_interval


def main():
    data,raw=load_inputs();base=features(data,raw)
    mom=base['base']['ranks']['long_momentum']
    score=.3*mom+(6/7)*(base['score']-.3*mom)+.1*.5
    control={**base,'score':score}
    rows=[];years=[];dailies=[];trades=[];boots=[]
    all_daily=pd.read_csv(P/'daily.csv.gz')
    for cost in [10,20]:
        d,t,o,state=simulate(data,raw,control,cost)
        for split,(a,b) in {'all':('2019-01-01','2026-09-18'),'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18')}.items():
            rows.append(dict(model='constant_control',cost_bps=cost,window=split,**metrics(d,t,o,a,b)))
        for y in range(2019,2027):
            years.append(dict(model='constant_control',cost_bps=cost,year=y,**metrics(d,t,o,f'{y}-01-01',f'{y}-12-31')))
        dd=all_daily.query("model=='momentum' and cost_bps==@cost")
        assert dd.date.tolist()==d.date.tolist()
        delta=np.log1p(dd['return'].to_numpy())-np.log1p(d['return'].to_numpy())
        boots.append(dict(cost_bps=cost,comparison='momentum_minus_constant_control',**block_interval(delta)))
        dailies.append(d.assign(cost_bps=cost));trades.append(t.assign(cost_bps=cost))
    pd.DataFrame(rows).to_csv(P/'weight_control_metrics.csv',index=False)
    pd.DataFrame(years).to_csv(P/'weight_control_yearly.csv',index=False)
    pd.concat(dailies).to_csv(P/'weight_control_daily.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'weight_control_trades.csv.gz',index=False,compression='gzip')
    (P/'weight_control_bootstrap.json').write_text(json.dumps(boots,indent=2))
    manifest=json.loads((P/'input_hashes.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    print(pd.DataFrame(rows).query("window=='all'")[['cost_bps','cagr','max_drawdown','trade_win','closed_trades']].to_string(index=False))
    print(json.dumps(boots,indent=2))


if __name__=='__main__':main()
