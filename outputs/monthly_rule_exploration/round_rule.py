from explore import *

def main():
    env=prepare();s=read(P/'SIMPLIFY_SELECTED.json');fit=dict(s['fit']);w=np.array(fit['weights']);fit['weights']=perturb_weights(w,np.round(w,2)).tolist()
    es=[m for m in env['meta'] if s['start']<=m['start']<'2020-12-31'];ts=[choose(env['rows'][e['start']],fit) for e in es]
    targets=[o['holdings'] for o in s['orders']]
    assert all(set(a)==set(b) for a,b in zip(ts[:3],targets))
    nav,orders,check=curve_check(env,es,ts,s['start']);capital=s['initial_nav'];peak=s['prior_peak'];months=[]
    for ev,t in zip(es,ts):
        nv,no,nc=curve_check(env,[ev],[t],ev['start'],initial=capital,peak=peak)
        capital=float(nv.iloc[-1]);peak=max(peak,float(nv.max()));months.append(dict(order=no[0],check=nc))
    dump('ROUNDED_RULE.json',dict(fit=fit,all_training_choices_preserved=True,months=months,check=check,
        nav=[dict(date=str(d.date()),value=float(v)) for d,v in nav.items()],start=s['start'],fit_end=s['end'],
        initial_nav=s['initial_nav'],prior_peak=s['prior_peak'],rounded_as_nonoptimizing_sensitivity=True))
    print([(m['order']['end'],m['order']['holdings'],m['check']['not_rejected']) for m in months])
if __name__=='__main__':main()
