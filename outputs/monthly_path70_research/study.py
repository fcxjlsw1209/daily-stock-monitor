"""Finite long-trend path experiment; true incumbent selection is preserved."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous = module('path70_previous', O/'monthly_horizon70_research/study.py')
old = previous.old
rp, ex, arch, ep, read, sha = old.rp, old.ex, old.arch, old.ep, old.read, old.sha
TRAIN = 88
FACTORS = ('weakest_segment', 'balanced_growth', 'worst_segment_downside', 'centered_mad_momentum')
BUDGETS = (.025, .05, .10)


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def configs():
    rows = [dict(base='original65', factor='none', budget=0.), dict(base='incumbent66', factor='none', budget=0.)]
    rows += [dict(base=base, factor=factor, budget=budget)
             for base, factor, budget in itertools.product(('original65', 'incumbent66'), FACTORS, BUDGETS)]
    return [dict(z, id=i) for i, z in enumerate(rows)]


def implementations():
    return (P/'study.py', P/'features.py', P/'incumbent_adapter.py', Path(previous.__file__),
            Path(ep.__file__), Path(ep.old.__file__))


def initialize():
    assert not (P/'PLAN.json').exists(), 'Do not overwrite the registered experiment'
    assert len(configs()) == 26
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        objective='Continue full2016-08-17..2026-09-24 monthly research toward70%; no performance guarantee.',
        fixed='Same122 archived estimated historicalTop32+assumedtheme17; original295price/positive skipped momentum eligibility. Original50/52percent weighted core unchanged. Longonly,no leverage,threeequal/twohalf/<2cash,monthlyfixedadjustedunits.',
        hypothesis='Long growth distributed across three stages may differ from the same cumulative momentum concentrated in one stage. Compare conservative stage growth and nonlinear downside concentration; centeredabsolute deviation tests sensitivity to squared outliers.',
        factors='t-252..t-42:211 positive finite formation prices,210 returns. Three70logreturn segments: weakest=min(g1,g2,g3); balanced=mean(g)-populationSTD(g); downside=sum(g)/max(segmentRMSofnegative_logreturn). CenteredMAD=(P[t-42]/P[t-252]-1)/mean(abs(simple_return-mean(simple_return))). Zero denominator missing. Rank finite raw values within all49eventnames,pct-.5,leave missingNaN. Latest42prices cannot change rawfeatures,including missingness; separate originaleligibility unchanged.',
        grid='26fixed:original65/incumbent66 controls plus2bases*4factors*3budgets .025/.05/.10. No additionalhorizon,threshold,weight,regime orfactor search after results.',
        original_base='S65 + handbudget*(factor-originalsupport) forfinitefactor, originalsupport=(S65-weightedcore)/(1-coreweight). Missingfactor orbudget0 exactnochange; rank0 is a valid neutral score and still replaces originalsupport.',
        incumbent='Use actual pre-episode frozen66.10 global score; add the same hand support replacement BEFORE ep.select_boundary. Keep frozenepisode model2/5day utility/signal/level, budget.05 ceiling5 gap.02, originalsupport and priorcalm/weak flags. Top2 and candidate ranks3..5/gap are determined by modifiedpre-score. Never globally rerank the post-boundary score. Budget0 must reproduce all122 orderedincumbent66 holdings and scores exactly.',
        budgets='New manualfactor is NOT ML. Existingprior+episode correction slots remain<=.10; newhand<=.10, total auxiliary replacements<=.20. Coefficients sum1, core50/52%, remainingoriginalsupport>=.28. This is a newresearch budget,not a change to any frozenobserver. Missinghand makes its effectivebudget0.',
        selection='First88 ending2023-11-30. Incumbent66 trainCAGR minimum; Sharpe>=inc-.05,DD>=inc-.03,early53/recent35 CAGR>=inc-.025. HighestadmittedCAGR primary, minsegmentSharpe robust; fallbackinc66. Freeze8 besttraining base*factor diagnostics and both controls BEFORE later34. No full-period selection or ticker/dateexceptions.',
        validation='Synthetic features/prefix/missing tests; exactoriginal65/inc66 orders; independent fixedunitdaily replay; trainingfreezeparity; nextopenzero/5bpsside unchangedorders; 36/60/76prefixprocedure diagnostics and changeconcentration. Preserve all inherited protected files.',
        limitations='Previously researchedhistory is development,not pristineOOS. Sourcepool and ETF17 are proxies. Samecompleteclose signal+fill is idealized. No claim ofauthorrecovery. Observers/rules/plans/accounts/data/fingerprints/automations unchanged.',
        audit_note='Read-only dedup audit accidentally returned partial oldmonthly_rule_exploration TRIALS/PATH_CHECKS via a wideglob; no newrule was chosen using those numbers. All26newconfigs fixedbefore theirreturns.',
        hashes={str(f): sha(f) for f in implementations()}))
    dump('GRID.json', configs())
    dump('PROTECTED.json', read(previous.P/'PROTECTED.json'))


def registered():
    assert read(P/'GRID.json') == configs()
    old.assert_hashes(read(P/'PLAN.json')['hashes'])


def prepare():
    registered()
    assert not (P/'FROZEN.json').exists()
    m, inherited, px, op = old.load()
    em, d, epx, eop, a = ep.load()
    assert em['dates'] == m['dates'] and em['symbols'] == m['symbols']
    pd.testing.assert_frame_equal(epx, px)
    pd.testing.assert_frame_equal(eop, op)
    policy = read(ep.P/'FROZEN.json')['selected']['primary']
    prior_policy = read(ep.old.P/'FROZEN.json')['selected']['primary']['spec']
    ef = dict(np.load(ep.P/'FORECASTS.npz'))
    dat = {k: a[k].copy() for k in ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow')}
    dat['eligible'] = d['eligible'].copy()
    dat['pre_score'] = ep.old.score(prior_policy, a, len(m['dates']))
    dat['effects'] = ef[str(policy['model_id'])].copy()
    np.testing.assert_allclose(dat['baseline'], inherited['base_original65'],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(dat['eligible'], inherited['eligible'])
    panel, info = arch.old.panel_load(m['etfs'])
    featuremod = module('path70_features', P/'features.py')
    frames, audit = featuremod.build_features(panel['adj'])
    assert tuple(frames) == FACTORS
    for key, frame in frames.items():
        values = np.stack([frame.loc[date, ss].to_numpy(float) for date, ss in zip(m['dates'], m['symbols'])])
        assert not np.isinf(values).any()
        dat['raw_'+key] = values
        dat['rank_'+key] = np.stack([pd.Series(row).rank(pct=True).sub(.5).to_numpy() for row in values])
    adapter = module('path70_adapter', P/'incumbent_adapter.py')
    inc, sc = adapter.score_and_choose('incumbent66', 0., np.full_like(dat['baseline'], np.nan), dat, m,
        len(m['dates']), pre_score=dat['pre_score'], effects=dat['effects'], episode_policy=policy,
        boundary_selector=ep.select_boundary)
    expected, score = ep.choices(policy, em, d, a, ef, len(m['dates']))
    assert inc == expected
    np.testing.assert_array_equal(sc, score)
    assert rp.make_orders(inc, sc, m) == read(ep.P/'primary_ORDERS.json')
    orig = rp.choose(dat['baseline'], dat['eligible'], m['symbols'], 3)
    assert rp.make_orders(orig, dat['baseline'], m) == read(ep.P/'original65_ORDERS.json')
    budgets=[]
    for row in configs():
        factor=dat['rank_'+row['factor']] if row['factor']!='none' else np.full_like(dat['baseline'],np.nan)
        weights=adapter.budget_weights(row['base'],row['budget'],factor,dat,len(m['dates']))
        np.testing.assert_allclose(sum(weights.values()),1.,rtol=0,atol=1e-14)
        budgets.append(dict(spec=row,min_original_support=float(weights['original_support'].min()),
            max_hand=float(weights['hand'].max()),
            max_old_corrections=float((weights['prior_residual']+weights['prior_ml']+weights['episode']).max()),
            maximum_total_weight_error=float(np.abs(sum(weights.values())-1).max())))
    np.savez_compressed(P/'DATA.npz', **dat)
    dump('META.json', m)
    dump('INCUMBENT_POLICY.json', policy)
    dump('FEATURE_AUDIT.json', audit)
    dump('BUDGET_AUDIT.json',budgets)
    dump('PREPARE_VALIDATION.json', dict(ordered_incumbent_exact=True,incumbent_scores_exact=True,
        original65_orders_exact=True,original_eligibility_exact=True,prices_exact=True))
    hashes = dict(info['source_hashes'])
    hashes.update(read(old.P/'INPUT_HASHES.json'))
    hashes.update(audit.get('source_hashes', {}))
    for f in list(implementations()) + [old.P/'DATA.npz',old.P/'META.json',ep.P/'FROZEN.json',
        ep.P/'FORECASTS.npz',ep.P/'primary_ORDERS.json',ep.P/'original65_ORDERS.json',ep.old.P/'FROZEN.json',
        ep.old.P/'FEATURES.npz',ep.old.P/'FORECASTS.npz',Path(ex.e.engine.__file__),Path(rp.__file__),
        previous.P/'RESULTS.json',previous.P/'PROTECTED.json']:
        hashes[str(f)] = sha(f)
    dump('INPUT_HASHES.json', hashes)
    print('Prepared26fixedpathrules; exact122orderedincumbent parity', flush=True)


def load():
    registered()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    m, _, px, op = old.load()
    assert m == read(P/'META.json')
    return m, dict(np.load(P/'DATA.npz')), px, op


def choices(spec, a, m, n):
    factor = a['rank_'+spec['factor']] if spec['factor'] != 'none' else np.full_like(a['baseline'], np.nan)
    adapter = module('path70_runtime_adapter', P/'incumbent_adapter.py')
    return adapter.score_and_choose(spec['base'], spec['budget'], factor, a, m, n,
        pre_score=a['pre_score'], effects=a['effects'], episode_policy=read(P/'INCUMBENT_POLICY.json'),
        boundary_selector=ep.select_boundary)


def select(rows):
    incumbent = rows[1]
    for z in rows:
        z['admitted'] = previous.admitted(z, incumbent)
        z['old_strict_admitted'] = old.session.admitted(z, incumbent)
    best = lambda rr: max(rr, key=lambda z:(z['training']['cagr'], -z['spec']['id']))
    qualified = [z for z in rows if z['admitted']] or [incumbent]
    selected = dict(primary=best(qualified), robust=max(qualified,key=lambda z:(min(z['early']['sharpe'],z['recent']['sharpe']),z['training']['cagr'],-z['spec']['id'])),
                    original65=rows[0],incumbent66=rows[1])
    for base, factor in itertools.product(('original65','incumbent66'),FACTORS):
        selected[base+'__'+factor] = best([z for z in rows if z['spec']['base']==base and z['spec']['factor']==factor])
    return selected


def train():
    assert not (P/'FROZEN.json').exists()
    m, a, px, _ = load()
    blocks, _, bounds, valid = rp.paths(px,m,TRAIN)
    rows, returns = [], []
    for spec in read(P/'GRID.json'):
        ch, _ = choices(spec,a,m,TRAIN)
        rr = rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**old.session.measure(rr,m,bounds)))
        returns.append(rr)
    selected = select(rows)
    folds, stitched, control = [], [], []
    for n,end in ((36,60),(60,76),(76,88)):
        prefix=[dict(spec=z['spec'],**old.session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        winner=select(prefix)['primary']
        stitched.extend(returns[winner['spec']['id']][bounds[n]:bounds[end]])
        control.extend(returns[1][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json','FEATURE_AUDIT.json','BUDGET_AUDIT.json','INPUT_HASHES.json','PROTECTED.json','study.py','features.py','incumbent_adapter.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        new_admitted=sum(z['admitted'] and z['spec']['factor']!='none' for z in rows),trials=len(rows),
        hashes={file:sha(P/file) for file in files},pristine_oos=False,diagnostics_may_be_unqualified=True,author_labels_used=False))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozenprimary',selected['primary']['spec'],selected['primary']['training'],flush=True)


def evaluate():
    f=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(f['hashes'],P)
    m,a,px,op=load()
    blocks,_,bounds,valid=rp.paths(px,m,len(m['dates']))
    orders={}
    for name,z in f['selected'].items():
        ch,sc=choices(z['spec'],a,m,len(m['dates']))
        orders[name]=rp.make_orders(ch,sc,m)
    orders['qqq']=[dict(date=m['dates'][0],signal_date=m['dates'][0],end=m['ends'][-1],symbols=['QQQ'],scores=[])]
    results,memo={},{}
    for name,oo in orders.items():
        key=json.dumps(oo,sort_keys=True)
        if key not in memo:
            met,eq,tr,hold=ex.e.engine.replay(px,op,oo,cost=0.,execution='close',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for year in met['annual_returns']:
                if year['year'] in (2016,2026):year['partial']=True
            common=dict(metrics=met,slices=arch.old.prior.old.previous.cuts(eq,m))
            for label,cost in (('next_open_zero',0.),('next_open_cost',.0005)):
                mm,ee,_,_=ex.e.engine.replay(px,op,oo,cost=cost,execution='open',start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
                for year in mm['annual_returns']:
                    if year['year'] in (2016,2026):year['partial']=True
                common[label]=dict(metrics=mm,slices=arch.old.prior.old.previous.cuts(ee,m))
            memo[key]=common,eq,tr,hold
        common,eq,tr,hold=memo[key]
        if name!='qqq':
            ch=[[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(oo)]
            rr=rp.fast_returns(ch,blocks,valid)
            np.testing.assert_allclose(eq,np.r_[1.,np.cumprod(1+rr)]*100000,rtol=1e-11,atol=1e-5)
            assert abs(common['slices']['training2016_2023']['cagr']-f['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=f['selected'].get(name))
        for suffix,value in (('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)):
            dump(name+'_'+suffix+'.json',value)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    previous_results=read(previous.P/'RESULTS.json')
    for name in ('original65','incumbent66'):
        for key in ('cagr','sharpe','max_drawdown'):
            assert abs(results[name]['metrics'][key]-previous_results[name]['metrics'][key])<1e-10
    months=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,oo in orders.items():
            if name=='qqq':continue
            ss=oo[i]['symbols']
            r=px.loc[m['ends'][i],ss]/px.loc[day,ss]-1 if ss else []
            row[name]=dict(symbols=ss,monthly_return=float(np.mean(r)) if len(r) else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json',months)
    attribution={}
    for name in f['selected']:
        if name in ('original65','incumbent66'):continue
        changes=[]
        for row in months:
            x,b=row[name],row['incumbent66']
            if set(x['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],split=row['split'],old=b['symbols'],new=x['symbols'],difference=x['monthly_return']-b['monthly_return'],log_contribution=float(np.log1p(x['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        met=results[name]['metrics']
        undo=lambda n:float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name]=dict(changed_months=len(changes),changes=changes,undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json',attribution)
    protected=read(P/'PROTECTED.json')
    changed=[file for file,digest in protected.items() if sha(file)!=digest]
    assert not changed,changed
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(protected),protected_changes=changed,
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=26,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
