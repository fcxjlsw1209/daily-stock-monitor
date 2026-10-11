"""Deterministic monthly selection audit. No ML, fitting or return optimization."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_guidance_reset'))
from study import load_prices,actual,display,calendar_momentum

W={'momentum':.52,'volatility':.219375,'acceleration':.185625,'state_cmf':.075}
LABEL={'momentum':'长期动量／近期波动分母','volatility':'126日高波动偏好','acceleration':'跳过两月的动量加速','state_cmf':'市场状态×CMF63'}
FEATURE={'momentum':'risk294','volatility':'vol126','acceleration':'acceleration','state_cmf':'neg_cmf63'}

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def dump(name,obj):
    (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def number(v):return float(v) if pd.notna(v) and np.isfinite(v) else None
def ranked(s,eligible):
    ss=[s0 for s0 in s.sort_values(ascending=False,kind='stable').index if eligible.at[s0]]
    return ss,ss[:3] if len(ss)>=2 else []
def pct(tab):return tab.rank(pct=True).fillna(.5)-.5

def feature_panel(px,raw):
    r=px.pct_change(fill_method=None);v63=r.rolling(63).std()
    m294=px.shift(42)/px.shift(294)-1;m252=px.shift(42)/px.shift(252)-1
    aligned=r.rolling(252).std().shift(42)
    clv=((2*raw['close']-raw['high']-raw['low'])/(raw['high']-raw['low']).replace(0,np.nan)).where(raw['high']!=raw['low'],0)
    cmf=(clv*raw['volume']).rolling(63).sum()/raw['volume'].rolling(63).sum()
    feat={'risk294':m294/v63.replace(0,np.nan),'vol126':r.rolling(126).std(),
      'acceleration':px.shift(42)/px.shift(126)-px.shift(126)/px.shift(210),'neg_cmf63':-cmf,
      'raw294':m294,'raw252':m252,'vol63':v63,'risk252':m252/v63.replace(0,np.nan),
      'risk_aligned':m294/aligned.replace(0,np.nan),'mean_sharpe':r.rolling(252).mean().shift(42)/aligned.replace(0,np.nan),
      'risk_half':m294/np.sqrt(v63.replace(0,np.nan)),
      'calendar12':calendar_momentum(px,12)/v63.replace(0,np.nan),
      'calendar14':calendar_momentum(px,14)/v63.replace(0,np.nan),
      'industry60':(1+px.pct_change(60,fill_method=None))/(1+raw['sector60'])-1}
    mv=r.SPY.rolling(63).std();med=mv.rolling(252).median()
    states=pd.DataFrame({'high_vol':mv>med,'spy_above200':px.SPY>px.SPY.rolling(200).mean(),'market_vol63':mv,'market_vol_median252':med})
    warm=px.notna().rolling(295).sum().eq(295)&px.gt(0)
    return {k:v.replace([np.inf,-np.inf],np.nan) for k,v in feat.items()},states,warm

def snapshot(date,symbols,feat,states,warm):
    ss=sorted(set(symbols));tab=pd.DataFrame({k:v.loc[date,ss] for k,v in feat.items()});q=pct(tab)
    sign=1 if states.at[date,'high_vol'] else -1
    comp=pd.DataFrame({k:q[FEATURE[k]]*w*(sign if k=='state_cmf' else 1) for k,w in W.items()})
    # Preserve original arithmetic and stable lexical tie-breaking exactly.
    side=.26*q.vol126+.22*q.acceleration
    score=.52*q.risk294+side+.075*(sign*q.neg_cmf63-side/.48)
    np.testing.assert_allclose(comp.sum(axis=1),score,rtol=0,atol=1e-14)
    eligible=warm.loc[date,ss]&tab.raw252.gt(0)
    order,chosen=ranked(score,eligible)
    raw_rank={s:i+1 for i,s in enumerate(score.sort_values(ascending=False,kind='stable').index)}
    ranks={s:i+1 for i,s in enumerate(order)}
    stages={'core':comp.momentum,'core_vol':comp.momentum+comp.volatility,
            'core_vol_accel':comp.momentum+comp.volatility+comp.acceleration,'full':score}
    stage_ranks={k:{s:i+1 for i,s in enumerate(ranked(v,eligible)[0])} for k,v in stages.items()}
    return dict(tab=tab,q=q,comp=comp,score=score,eligible=eligible,order=order,chosen=chosen,
                ranks=ranks,raw_rank=raw_rank,stage_ranks=stage_ranks,sign=sign)

def pair_difference(a,b,snap):
    # b is our selected outsider, a the missed author holding. Positive = b favored.
    delta=(snap['comp'].loc[b]-snap['comp'].loc[a])*100
    gap=(snap['score'].at[b]-snap['score'].at[a])*100
    assert abs(float(delta.sum())-gap)<1e-10
    positive=delta[delta>0]
    driver=positive.idxmax() if len(positive) else None
    return dict(author=display(a),ours=display(b),gap_points=float(gap),
        components={k:float(v) for k,v in delta.items()},driver=driver,
        driver_label=LABEL.get(driver,'并列排序'),author_rank=snap['ranks'].get(a),ours_rank=snap['ranks'].get(b),
        reversed_without=[k for k in W if gap-float(delta[k])< -1e-10])

def alternate_scores(snap):
    q=snap['q'];c=snap['comp'];full=snap['score']
    result={'core_only':c.momentum}
    for k in ['volatility','acceleration','state_cmf']:result['without_'+k]=full-c[k]
    for k in ['raw294','risk252','risk_aligned','mean_sharpe','risk_half','calendar12','calendar14']:
        result['core_'+k]=full-c.momentum+.52*q[k]
    result['industry_instead_state']=full-c.state_cmf+.075*q.industry60
    return result

def transition(previous,current):
    if previous is None:return {'added':sorted(current),'removed':[],'retained':[],'initial':True}
    return {'added':sorted(current-previous),'removed':sorted(previous-current),'retained':sorted(previous&current),'initial':False}

def cap_evidence(sources):
    f=O/'monthly_top35_extension/EARLIER_RANKS.json';sources[str(f)]=sha(f)
    ranks={e['date']:{actual(r['symbol']):dict(rank=r['rank'],market_cap_usd=r['market_cap_usd'],asof=e['cap_asof'],source=e['source_kind']) for r in e['ranks']} for e in read(f)}
    f=O/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json';sources[str(f)]=sha(f)
    for e in read(f)['periods']:
        rows=[r for r in e['top32']+e['boundary33_40'] if not(r['symbol']=='LTM' and '2020-06-22'<=e['date']<'2024-07-25')]
        ranks[e['date']]={actual(r['symbol']):dict(rank=i+1,market_cap_usd=r.get('market_cap_usd'),asof=e['date'],source='published_monthly_rank_reconstruction') for i,r in enumerate(rows)}
    return ranks

def run():
    px,raw,members,periods,sources,pools=load_prices();feat,states,warm=feature_panel(px,raw)
    ranks=cap_evidence(sources);theme=members['theme17'][1];coreetf=members['core17'][1]
    f=O/'monthly_guidance_reset/prior_return_leader_top32_theme17_HOLDINGS.json';sources[str(f)]=sha(f)
    saved={z['signal_date']:z for z in read(f)}
    allrows=[];months=[];allpairs=[];prev_author=None;prev_ours=None
    diagnostics={'rank_effects':Counter(),'pair_drivers':Counter(),'exit_classes':Counter(),'retained_classes':Counter()}
    for p in periods:
        d=p['start'];authors={actual(s) for s in p['holdings']};stocks={actual(s) for s in pools['32'][d]}
        symbols=stocks|theme;s=snapshot(d,symbols,feat,states,warm);ours=set(s['chosen'])
        assert [display(z) for z in s['chosen']]==saved[d]['symbols'],d
        np.testing.assert_allclose(s['score'].loc[s['chosen']],saved[d]['scores'],rtol=0,atol=1e-12)
        missed=authors-ours;extra=ours-authors
        au=transition(prev_author,authors);ou=transition(prev_ours,ours)
        variants={}
        for k,z in alternate_scores(s).items():
            order,picks=ranked(z,s['eligible'])
            variants[k]={'selected':list(map(display,picks)),'shared':len(authors&set(picks)),
                         'author_ranks':{display(a):order.index(a)+1 if a in order else None for a in authors}}
        for name,n,etfs in [('top35_same_rule',35,theme),('core17_same_stocks',32,coreetf),('without_arkk_soxx',32,theme-{'ARKK','SOXX'})]:
            test=snapshot(d,{actual(a) for a in pools[str(n)][d]}|etfs,feat,states,warm)
            variants[name]={'selected':list(map(display,test['chosen'])),'shared':len(authors&set(test['chosen'])),
                           'author_ranks':{display(a):test['ranks'].get(a) for a in authors}}
        _,no_gate=ranked(s['score'],warm.loc[d,sorted(symbols)])
        variants['no_positive_momentum_gate']={'selected':list(map(display,no_gate)),'shared':len(authors&set(no_gate))}
        rows={}
        for a in sorted(symbols|authors|(prev_author or set())):
            inside=a in symbols;eligible=bool(s['eligible'].at[a]) if inside else False
            reason=[]
            if not inside:reason.append('股票池外')
            if not bool(warm.at[d,a]):reason.append('295日预热不足')
            if not feat['raw252'].at[d,a]>0:reason.append('资格动量非正')
            rec=dict(symbol=display(a),in_pool=inside,eligible=eligible,blockers=reason,
                     author=a in authors,ours=a in ours,previous_author=a in (prev_author or set()),
                     asset='ETF' if a in (theme|coreetf) else 'stock',cap=ranks[d].get(a),
                     rank=s['ranks'].get(a),unfiltered_rank=s['raw_rank'].get(a),
                     score=number(100*(s['score'].at[a]+.5)) if inside else None,
                     stage_ranks={k:v.get(a) for k,v in s['stage_ranks'].items()},
                     factor_ranks={k:int(s['q'][FEATURE[k]].mul(s['sign'] if k=='state_cmf' else 1).rank(ascending=False,method='min').at[a]) if inside else None for k in W},
                     contributions={k:float(s['comp'].at[a,k]*100) if inside else None for k in W},
                     raw={k:number(v.at[d,a]) for k,v in feat.items()})
            rows[a]=rec;allrows.append(dict(signal_date=d,**rec))
        pairs=[pair_difference(a,b,s) for a in sorted(missed) if s['eligible'].get(a,False) for b in sorted(extra)]
        allpairs += [dict(date=d,**a) for a in pairs]
        for a in pairs:diagnostics['pair_drivers'][a['driver']]+=1
        lost=[]
        threshold=s['chosen'][-1] if s['chosen'] else None
        for a in sorted(missed):
            row=rows[a];reason='pool' if not row['in_pool'] else 'eligibility' if not row['eligible'] else 'ranking'
            diagnostics['rank_effects'][reason]+=1
            item={'symbol':display(a),'reason':reason,'rank':row['rank'],'cap_rank':row['cap']['rank'] if row['cap'] else None}
            if reason=='ranking' and threshold:
                # The weakest selected non-author gives the minimum displacement gap.
                b=min(extra,key=lambda z:(s['score'].at[z],z)) if extra else threshold
                item['boundary_pair']=pair_difference(a,b,s)
                item['minimal_pairwise_additive_weight']=max(0,item['boundary_pair']['gap_points']/100)
                item['helps']=[k for k,v in variants.items() if display(a) in v['selected']]
            lost.append(item)
        exit_records=[]
        for a in au['removed']:
            row=rows[a];rank=row['rank']
            cls='pool_or_gate_exit_compatible' if not row['eligible'] else 'model_would_keep_top3' if rank<=3 else 'borderline_rank4_5' if rank<=5 else 'rank_below5_exit_compatible'
            diagnostics['exit_classes'][cls]+=1
            old=next((z for z in reversed(months) if display(a) in z['author']),None)
            exit_records.append(dict(symbol=display(a),rank=rank,classification=cls,blockers=row['blockers'],
                 previous_rank=next((v['rank'] for v in old['rows'] if v['symbol']==display(a)),None) if old else None))
        retain_records=[]
        for a in au['retained']:
            rank=rows[a]['rank'];cls='kept_top3' if rank and rank<=3 else 'kept_rank4_5' if rank and rank<=5 else 'kept_below5' if rank else 'kept_outside_or_ineligible'
            diagnostics['retained_classes'][cls]+=1;retain_records.append(dict(symbol=display(a),rank=rank,classification=cls))
        comparable=all(a in symbols and s['eligible'].get(a,False) for a in authors)
        gap=None
        if comparable:
            competitors=[a for a in s['order'] if a not in authors]
            gap=max(0.,float(s['score'].loc[competitors].max()-s['score'].loc[list(authors)].min()))
        state={'high_vol':bool(states.at[d,'high_vol']),'spy_above200':bool(states.at[d,'spy_above200']),
               'market_vol63_annual':float(states.at[d,'market_vol63']*np.sqrt(252)),
               'market_vol_median252_annual':float(states.at[d,'market_vol_median252']*np.sqrt(252)),
               'cmf_direction':'偏低CMF' if s['sign']==1 else '偏高CMF'}
        m=dict(date=d,end=p['end'],holding_month=str(pd.Timestamp(p['end']).to_period('M')),
               author=[display(a) for a in sorted(authors)],ours=[display(a) for a in s['chosen']],shared=len(authors&ours),
               eligible_count=int(s['eligible'].sum()),state=state,rows=list(rows.values()),missed=lost,extra=list(map(display,sorted(extra))),
               author_transition={k:list(map(display,v)) if isinstance(v,list) else v for k,v in au.items()},
               our_transition={k:list(map(display,v)) if isinstance(v,list) else v for k,v in ou.items()},
               exits=exit_records,retained=retain_records,pairs=pairs,variants=variants,
               all_author_eligible=comparable,set_order_gap_points=gap*100 if gap is not None else None,
               theoretical_5pct_correction_sufficient=gap<=.05+1e-12 if gap is not None else None,
               theoretical_10pct_correction_sufficient=gap<=.10+1e-12 if gap is not None else None)
        months.append(m);prev_author=authors;prev_ours=ours
    variant_summary={}
    for name in months[0]['variants']:
        sets=[m['variants'][name] for m in months]
        variant_summary[name]={'exact_months':sum(v['shared']==3 for v in sets),'shared_slots':sum(v['shared'] for v in sets),
          'improved_months':[m['date'] for m,v in zip(months,sets) if v['shared']>m['shared']],
          'worsened_months':[m['date'] for m,v in zip(months,sets) if v['shared']<m['shared']]}
    comparable=[m for m in months if m['all_author_eligible']]
    missed_gaps=[a['minimal_pairwise_additive_weight'] for m in months for a in m['missed'] if 'minimal_pairwise_additive_weight' in a]
    summary={'months':len(months),'slots':3*len(months),'exact_months':sum(m['shared']==3 for m in months),
      'shared_slots':sum(m['shared'] for m in months),'author_eligible_months':len(comparable),
      'mismatch_slots_by_reason':dict(diagnostics['rank_effects']),'pairwise_dominant_factors':dict(diagnostics['pair_drivers']),
      'pair_count':len(allpairs),'exit_classes':dict(diagnostics['exit_classes']),'retained_classes':dict(diagnostics['retained_classes']),
      'unmatched_slots_with_entry_gap_gt_5pct':sum(g>.05+1e-12 for g in missed_gaps),
      'unmatched_slots_with_entry_gap_gt_10pct':sum(g>.10+1e-12 for g in missed_gaps),
      'all_eligible_months_unexplainable_by_5pct_additive':sum(not m['theoretical_5pct_correction_sufficient'] for m in comparable),
      'all_eligible_months_unexplainable_by_10pct_additive':sum(not m['theoretical_10pct_correction_sufficient'] for m in comparable),
      'variants':variant_summary,'weights':W,'factor_labels':LABEL,'ml_used':False,'returns_used_in_audit':False,
      'source_order_is_not_author_rank':True,'author_exit_reason_identified':False,
      'outside_author_months':[{'date':m['date'],'symbols':[r for r in m['missed'] if r['reason']=='pool']} for m in months if any(r['reason']=='pool' for r in m['missed'])]}
    dump('MONTHS.json',months);dump('RANK_ROWS.json',allrows);dump('PAIR_DIFFERENCES.json',allpairs);dump('SUMMARY.json',summary)
    dump('SOURCE_HASHES.json',sources)
    protected=read(P/'PROTECTED_BEFORE.json');changed=[s for s,h in protected.items() if sha(O.parent/s)!=h]
    assert not changed,changed
    dump('VALIDATION.json',{'saved_holding_parity':69,'saved_score_parity':69,'score_components_sum_exact':True,
                           'pair_contributions_sum_exact':True,'protected_files_unchanged':len(protected),'new_model_fitted':False})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['variants','outside_author_months']},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
