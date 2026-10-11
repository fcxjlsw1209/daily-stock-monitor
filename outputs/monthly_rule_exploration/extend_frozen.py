import numpy as np
import pandas as pd
from explore import *

def main():
    env=prepare();all_results=[]
    for file in sorted(P.glob('JOINT_*_SELECTED.json')):
        src=read(file);start=src['end'];fit=src['fit'];events=[m for m in env['meta'] if m['start']>=start][:3]
        capital=src['orders'][-1]['end_nav'];peak=max(src['prior_peak'],max(r['value'] for r in src['nav']))
        pieces=[];months=[]
        for ev in events:
            names=choose(env['rows'][ev['start']],fit)
            nav,orders,check=curve_check(env,[ev],[names],ev['start'],initial=capital,peak=peak)
            capital=float(nav.iloc[-1]);peak=max(peak,float(nav.max()));pieces.append(nav)
            months.append(dict(order=orders[0],check=check))
        baseline=pd.Series({pd.Timestamp(r['date']):r['value'] for r in src['nav']})
        whole=pd.concat([baseline]+pieces);whole=whole[~whole.index.duplicated(keep='last')]
        check=image_check(whole,src['prior_peak'],env['px'],env['cal'],env['obs'],env['cfg'])
        first_fail=next((m['order']['start'] for m in months if not m['check']['not_rejected']),None)
        result=dict(start=src['start'],fit_end=src['end'],end=events[-1]['end'],fit_source=str(file),fit_sha256=sha(file),
            months=months,first_rejected_extension_decision=first_fail,combined_check=check,parameters_unchanged=True,
            nav=[dict(date=str(d.date()),value=float(v)) for d,v in whole.items()])
        all_results.append(result)
        print(src['start'],[(m['order']['end'],m['order']['holdings'],m['check']['not_rejected']) for m in months],flush=True)
    dump('FROZEN_EXTENSIONS.json',all_results)
if __name__=='__main__':main()
