"""Read completed fixed-input results; no fitting or source mutations."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild'
def read(p):return json.loads(p.read_text())
def main():
    completed=read(OUT/'evaluation/results/COMPLETION.json')
    assert completed['status']=='COMPLETE'
    results=read(OUT/'evaluation/results/RESULTS.json')
    metrics={}
    for arm,paths in results['arms'].items():
        metrics[arm]={name:{split:{k:v.get(k) for k in ('start','end','cagr','sharpe','max_drawdown','total_return')}
                           for split,v in row['slices'].items()} for name,row in paths.items()}
    before=read(OUT/'control/ORDERS.json');after=read(OUT/'conditional/ORDERS.json')
    changes=[dict(index=i,date=a['date'],old=a['symbols'],new=b['symbols'],
                  composition_changed=set(a['symbols'])!=set(b['symbols']))
             for i,(a,b) in enumerate(zip(before,after)) if a['symbols']!=b['symbols']]
    print(json.dumps(dict(metrics=metrics,changes=changes,
          changed_ordered_months=len(changes),changed_composition_months=sum(c['composition_changed'] for c in changes)),
          ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
