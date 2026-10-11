"""Uniform local explanations of supplied labels; never stitch their NAVs."""
from fit_policy import *

def main():
    data=get_rows();all_windows=[]
    for pool,rows in data.items():
        for family in FAMILIES:
            D=[pair_matrix([r],family) if r['valid'] else None for r in rows]
            for start in range(len(rows)-2):
                end=start;best=None
                while end<len(rows) and D[end] is not None:
                    w,fit=max_margin(np.vstack(D[start:end+1]))
                    if fit['margin']<=1e-6: break
                    best=(end,w,fit);end+=1
                if best is None or best[0]-start+1<3:continue
                end,w,fit=best;segment=rows[start:end+1]
                check=agreements(segment,w,family);assert check['exact_months']==len(segment)
                future=rows[end+1:min(end+4,len(rows))]
                rng=np.random.default_rng(421)
                robust=[]
                for _ in range(100):
                    ww=w*rng.uniform(.95,1.05,len(w))
                    robust.append(agreements(segment,ww,family)['exact_months']==len(segment))
                rounded=np.round(w,2)
                record=dict(pool=pool,family=family,start=segment[0]['date'],end=segment[-1]['end'],
                            decisions=len(segment),fit=fit,exact=check,
                            rounded_weights=dict(zip(FEATURES,rounded.tolist())),rounded=agreements(segment,rounded,family),
                            perturbation_5pct_full_window_match_rate=float(np.mean(robust)),
                            subsequent_frozen=agreements(future,w,family) if future else None)
                all_windows.append(record)
            print('WINDOW',pool,family,'done',flush=True)
    dump('UNIFORM_WINDOW_RULES.json',all_windows)
    ordered=sorted(all_windows,key=lambda r:(-r['decisions'],-r['perturbation_5pct_full_window_match_rate'],-r['fit']['margin']))
    dump('UNIFORM_WINDOW_LEADERS.json',ordered[:10])
    print('TOTAL',len(all_windows),flush=True)
    for r in ordered[:5]:print({k:r[k] for k in ['pool','family','start','end','decisions','perturbation_5pct_full_window_match_rate']},'margin',r['fit']['margin'],'next',None if r['subsequent_frozen'] is None else r['subsequent_frozen']['exact_months'],flush=True)
if __name__=='__main__':main()
