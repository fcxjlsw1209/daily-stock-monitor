"""Build bags of alternative monthly holdings before fitting any ranking rule.

Each bag is identified independently from that month's chart pixels. No future
month's portfolio or global best path is used to label an earlier month.
These remain retrospective, weak labels available only from today's image.
"""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from common import P,O,OLD,base,START,END,FEATURES,features,dump
from inverse import score_paths,load_observations


def run():
    cfg=dict(goal='First identify informative months and multiple admissible combinations; then infer one low-complexity ranking rule across months.',
             target=dict(cagr=.8064,sharpe=1.867,max_dd=-.3446),pools=['core17','theme17'],months=['2016-09','2026-09'],
             eligibility='PositiveR12-2 and top half,253 completed price observations;3 equal fixed-share positions per month.',
             bags='64 lowest robust image-loss triples per month; image shifts[-.2,0,.2]pixels; no later-month holdings used.',
             confidence='Entropy concentration of candidate stock inclusion, not a posterior probability; anchor threshold.65.',
             inference=dict(core_families=['raw','risk','mixed'],fixed_core_weight=.5,free_coefficient_l1_max=.5,
                            features=FEATURES,fit_eras=['through2021','all_retrospective'],restarts=3,iterations=350,
                            objective='Probability mass assigned to image-consistent alternative triples, weighted by image confidence; L2penalty. Not maximize return.',
                            fit_label_availability='Screenshot is available only now. Fitting to past image holdings is research, not historical point-in-time training.'),
             execution='Same-close zero-cost author reference plus prior-session signal/10bps roundtrip control.',
             validation='Learn early-month rule on2016-2021 and examine remaining months; previously viewed sample, not independent OOS. Full-sample inverse fit separately labelled.',
             author_no_future_month_ranking=True,original_accounts_unchanged=True)
    dump('CONFIG.json',cfg);px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations()
    (P/'bags').mkdir(exist_ok=True);allmeta={}
    for pool in cfg['pools']:
        member=pools[pool];raw,cores,ff,gate=features(px,fs,member);metas=[]
        for month in pd.period_range(*cfg['months'],freq='M'):
            days=px.index[(px.index.to_period('M')==month)&(px.index<=END)];s=px.index[px.index.get_loc(days[0])-1];e=days[-1];i=px.index.get_loc(s);frame=px.loc[s:e]
            eligible=member.loc[s]&gate.loc[s]&px.iloc[i-252:i+1].notna().all()
            # Complete paths are needed only for retrospective attribution, not entry qualification.
            cols=sorted(t for t in px if eligible[t] and frame[t].notna().all() and frame[t].gt(0).all())
            assert len(cols)>=3
            combos=np.asarray(list(combinations(range(len(cols)),3)),dtype=int)
            ratios=(frame[cols]/frame[cols].iloc[0]).to_numpy();paths=np.mean(ratios[:,combos],axis=2).T
            axis=cal['wealth'];x=axis['x0']+axis['width']*((frame.index-START).days/(END-START).days).to_numpy()
            ob={k:v[(v.x>x[0]+.8)&(v.x<x[-1]-.8)].copy() for k,v in obs.items()};loss=[];bestsets=[]
            for shift in [-.2,0,.2]:
                ref=obs['wealth'];prior=ref[ref.x<=x[0]+shift]
                constraints=dict(start_log_nav=float(np.interp(x[0]+shift,ref.x,ref.log_wealth)),prior_log_peak=max(0,float(prior.log_wealth.max())) if len(prior) else 0,log_tolerance=1.5/axis['log_scale'])
                out=score_paths(paths,x,ob,cal,xshift=shift,constraints=constraints)
                if out is None:raise ValueError(f'Insufficient visible pixels {pool} {month}')
                loss.append(out[0]);bestsets.append(set(cols[j] for j in combos[int(np.argmin(out[0]))]))
            loss=np.stack(loss);relative=loss-loss.min(axis=1,keepdims=True)
            robust=np.max(relative,axis=0)+.25*loss.mean(axis=0)
            order=np.argsort(robust,kind='stable');take=order[:min(64,len(order))];bag=combos[take]
            weights=np.exp(-(robust[take]-robust[take[0]])/.25);weights/=weights.sum()
            inclusion=np.zeros(len(cols))
            for triple,w in zip(bag,weights):inclusion[triple]+=w
            share=inclusion/3;effective=float(np.exp(-np.sum(share[share>0]*np.log(share[share>0]))))
            confidence=float(np.clip((len(cols)-effective)/(len(cols)-3),0,1))
            X=np.array([[float(ff[k].at[s,t]) for k in FEATURES] for t in cols]);C=np.array([[float(cores[k].at[s,t]) for k in cfg['inference']['core_families']] for t in cols]);assert np.isfinite(X).all() and np.isfinite(C).all()
            file=P/'bags'/f'{pool}_{month}.npz';np.savez_compressed(file,X=X,C=C,bag=bag,weights=weights,combos=combos,robust=robust)
            meta=dict(pool=pool,month=str(month),signal_date=str(s.date()),label_month_end=str(e.date()),symbols=cols,
                      confidence=confidence,effective_candidate_names=effective,anchor=confidence>=.65,
                      stable_best_names=sorted(set.intersection(*bestsets)),image_best_loss=float(robust[take[0]]),
                      candidate_triples=[dict(holdings=[cols[j] for j in triple],image_weight=float(w),robust_loss=float(robust[k])) for triple,w,k in zip(bag,weights,take)],
                      inclusion_fractions={t:float(inclusion[j]) for j,t in enumerate(cols) if inclusion[j]>0},
                      factor_percentiles={t:{k:float(X[j,n]+.5) for n,k in enumerate(FEATURES[:-2])} for j,t in enumerate(cols)},
                      file=str(file),sha256=base.sha(file),verified_author_trades=False)
            metas.append(meta);dump(f'BAGS_{pool}.json',metas)
            print(pool,month,'names',len(cols),'confidence',round(confidence,3),'persistent',meta['stable_best_names'],flush=True)
        allmeta[pool]=dict(months=len(metas),anchors=sum(m['anchor'] for m in metas),development_anchors=sum(m['anchor'] and m['month']<'2022-01' for m in metas))
    dump('BAG_SUMMARY.json',allmeta)
    dump('PROTECTED_BEFORE.json',json.loads((O/'monthly_curve_reverse/PROTECTED_BEFORE.json').read_text()))
    dump('SOURCE_HASHES.json',json.loads((OLD/'SOURCE_HASHES.json').read_text()))
    print(json.dumps(allmeta,indent=2))


if __name__=='__main__':run()
