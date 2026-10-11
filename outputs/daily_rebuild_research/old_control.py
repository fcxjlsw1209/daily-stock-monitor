"""Isolated interpreter for legacy research's module-name namespace."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_growth_research'))
import growth as g
from repair_actions import patch
outdir=P/'diagnostics';(outdir/'runs').mkdir(exist_ok=True)
data,f,base,gaps,x,ctx,audit=g.prepare(outdir/'old_control');data=patch(data)
for cost in [0,10]:
 target=outdir/'runs'/f'old_daily_cap30_cost{cost}.json'
 if target.exists():continue
 out=g.run(g.BASE,cost,data,f,base,gaps,x,ctx,start='2016-08-17',end='2026-09-24')
 target.write_text(json.dumps(out,allow_nan=False)+'\n');print('old control complete',cost,flush=True)
