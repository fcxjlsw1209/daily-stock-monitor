"""Ten fixed target-definition configurations in the incumbent's weak ML slot.

Only this new directory is written. Research phases are explicit CLI actions.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import ModuleType
import argparse
import importlib.util
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


old = module('target_definition_earnings', O/'monthly_earnings_factor_research/study.py')
rp, ex, arch, ep, read, sha = old.rp, old.ex, old.arch, old.ep, old.read, old.sha
adapter = module('target_definition_adapter', O/'monthly_path70_research/incumbent_adapter.py')
model = module('target_definition_model', P/'model.py')
labels = module('target_definition_labels', P/'labels.py')
TRAIN = 88
TARGETS = model.TARGETS
POLICY_FIELDS = dict(id=7, family='level', model_id=2, budget=.05, ceiling=5, gap=.02)
PRIOR_POLICY = [dict(family='residual', key='resid_126_210_sum', amount=.025, state='calm10'),
                dict(key='ml_4', state='weak_narrow', amount=.05)]
PROTECTION_SOURCE = O/'monthly_incumbent_earnings70/PROTECTED.json'
CACHED_KEYS = ('dates','ends','event_dates','offsets','train_plain','event_plain',
               'train_good','train_eligible','eligible')


def dump(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x)+'\n')


def configs():
    rows = [dict(id=0, base='original65', target='none', blend=0.),
            dict(id=1, base='incumbent66', target='none', blend=0.)]
    for target in TARGETS:
        for blend in (.5, 1.):
            rows.append(dict(id=len(rows), base='incumbent66', target=target, blend=blend))
    return rows


def diagnostic_name(spec):
    return spec['target']+'_blend'+str(int(spec['blend']*100))


def implementations():
    """Hash the actual local import graph, including nested driver dependencies.

    Only imported module objects and three explicitly delayed input builders are
    inspected; no directory scan or results/plan wildcard is used.
    """
    paths = {P/'study.py', O/'monthly_stability_monitor/inputs.py',
             O/'monthly_core_expansion/ordinal_model.py',
             O/'monthly_earnings_repair/research_calendar.py'}
    pending, seen = [old, adapter, model, labels], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        file = getattr(value, '__file__', None)
        if file is None:
            continue
        file = Path(file).resolve()
        if not file.is_relative_to(O):
            continue
        paths.add(file)
        pending.extend(v for v in vars(value).values() if isinstance(v, ModuleType))
    return tuple(sorted(paths))


def merge_hashes(*sources):
    result = {}
    for source in sources:
        for path, digest in source.items():
            assert path not in result or result[path] == digest, 'Conflicting inherited hash: '+path
            result[path] = digest
    return result


def initialize():
    assert not (P/'PLAN.json').exists(), 'Never overwrite a registered experiment'
    protected = read(PROTECTION_SOURCE)
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        objective='Ten fixed development-research configurations toward70%; no guaranteed return or pristineOOS claim.',
        grid=configs(),
        fixed='Same122 monthly periods2016-08-17..2026-09-24, estimatedhistoricalTop32+assumedETF17, original eligibility and skip42 longcore50/52percent; original monthly exits and fixed adjusted units; longonly3equal/2half/<2cash. No new pool filters, author labels, intramonth trading or date stitching.',
        cohort='Cached original monthly plain10 features, training dates/ends/order/train_eligible/train_good from earnings DATA. Intersect positivefinite asset+SPY endpoints,64 positivefinite past prices for63daily sampleSTD>0,finiteX and all4transforms. Samecohort for allarms; at least2 names permonth or entiremonth excluded. Originalcached fullpathgood retained; no current trading eligibility depends on future label validity.',
        targets='r=assetEnd/assetSignal-1,b=SPYEnd/SPYSignal-1,e=r-b,v=std(ddof1,last63simpledailyreturns throughsignal). T1 clip(e,-1,1);T2 clip(log1p(r)-log1p(b),-1,1);T3 clip(e/(v*sqrt21),-5,5);T4 clip(e-.5*max(0,-r),-1,1). Clip first,then subtract each target same-month common-cohort mean. Continuous training targets,never target ranks. Endpoint loss differs from previous pathworstloss utility.',
        model='Annual precedingDec31 cutoff, previous5calendar years signal dates with lowerbound exclusive; allends<=cutoff. Ridgealpha10,12monthhalf-life,equal aggregate month weight thenage decay normalizedtorowcount,minimum18months; training-onlyfeature standardization. Fourarms sameX/rows/weights/scaler/availability,no tuning. Predict only signal-timeeligible finiteeventX; coldstartNaN.',
        rank='Only predictions ranked within signal-timeeligible and available names by exact old pandasrank(pct=True).fillna(.5)-.5 semantics. Ineligible/unavailable cells fallback exactoldml4. No new target-based eligibility or future-based rank membership.',
        slot='Use actual original66 pre-episode score. delta=.05*originalweak_narrow*blend*(newrank-oldml4),blend .5or1. Missing/invalid newforecast gives exactdelta0. Oldcalm residual.025 unchanged, originalsupport/core/states unchanged. Keep actual frozenepisode model2 level budget.05 ceiling5 gap.02; top2 andcandidate3..5within.02 determined by modifiedpre,not globallyreranked postscore.',
        budget='Replace only oldweakML .05 slot by convexrankblend; core50/52percent and original auxiliary coefficients unchanged. Originalprior+episode correction budgets<=.10,no additional MLslot. Effective replacement never exceeds .05; absdelta<=.05*blend. Both controls require exactscore and ordersparity.',
        comparison='Oldml4 is weekly84-feature downside_rank Ridge:21sessionexcess-.5pathadverseloss targetrank,alpha10,half12. Newmodels are monthlyplain10 andcontinuous calendar-month targets. Difference versusinc66 also changes cadence/features/labelshape; only thefournewarms internally isolate targetdefinition.',
        selection='First88 ending2023-11-30. Fixed local guards:trainCAGR>=inc66-1e-12,Sharpe>=inc66-.05,DD>=inc66-.03,early53/recent35CAGR each>=inc66-.025. HighesttrainCAGR amonginc66 andadmitted8candidates; candidateswithin1e-12ofmaximumareatietakeslowerIDandretainsinc66whenincluded. Original65 comparatoronly. Freeze all8 diagnostics regardlessadmission beforelater34. Prefix36/60/76 diagnostics use sameformula.',
        procedure='initialize,prepare,train/freeze,evaluate. Prepared allannual forecasts are PIT; selection sees onlyfirst88 returns. Hashverifiedfreeze beforelater34. Sameclosezero primary,identicalorders nextopenzero/5bp eachside,independentfixedunitreplay,originalcontrolsparity. No new variants afterevaluation.',
        protection='Copy the previouslyverified incumbentearnings PROTECTED unchanged and hash its source. Source/input/cache/frozen hashes checked for phases; fullprotected-tree verification occurs once at end ofevaluate. No oldfiles,observer,account,ledger,plan,fingerprint,automation or sourcedata changes.',
        limitations='Previouslyresearchedhistory isdevelopment,notpristineOOS; proxyuniverse and samefinalclose signal/fill idealization remain. No authorformula recovery claim.',
        hashes={str(f):sha(f) for f in implementations()},
        protection_source={str(PROTECTION_SOURCE):sha(PROTECTION_SOURCE)}))
    dump('GRID.json', configs())
    dump('PROTECTED.json', protected)


def registered():
    assert read(P/'GRID.json') == configs()
    plan = read(P/'PLAN.json')
    old.assert_hashes(plan['hashes'])
    old.assert_hashes(plan['protection_source'])


def assert_protected():
    old.assert_hashes(read(P/'PROTECTED.json'))


def validate_axes(m, meta, cached, eligible):
    assert meta['monthly'] == m, 'Monthly metadata must exactly match incumbent'
    assert len(m['dates']) == 122 and len(m['ends']) == 122
    assert m['dates'][0] == '2016-08-17' and m['ends'][87] == '2023-11-30'
    assert m['ends'][-1] == '2026-09-24'
    assert all(len(ss) == 49 for ss in m['symbols'])
    np.testing.assert_array_equal(cached['eligible'], eligible)
    tm = meta['training']
    widths = np.array([len(ss) for ss in tm['symbols']])
    offsets = np.r_[0, np.cumsum(widths)]
    np.testing.assert_array_equal(cached['offsets'], offsets)
    for key in ('dates','ends'):
        expected = np.repeat(np.array(tm[key], dtype='datetime64[D]'), widths)
        np.testing.assert_array_equal(np.asarray(cached[key], dtype='datetime64[D]'), expected)
    np.testing.assert_array_equal(np.asarray(cached['event_dates'], dtype='datetime64[D]'),
        np.repeat(np.array(m['dates'], dtype='datetime64[D]'), 49))
    assert cached['train_plain'].shape == (offsets[-1], 10)
    assert cached['event_plain'].shape == (len(m['dates'])*49, 10)
    for key in ('train_eligible','train_good'):
        assert cached[key].shape == (offsets[-1],) and cached[key].dtype.kind == 'b'


def ranked_forecasts(raw, available, eligible, old_ml4):
    """Exact old rank semantics on common signal-time forecast availability."""
    shape = np.asarray(eligible).shape
    if (len(shape) != 2 or np.asarray(eligible).dtype.kind != 'b'
            or np.asarray(available).shape != shape or np.asarray(available).dtype.kind != 'b'
            or np.asarray(old_ml4).shape != shape or not np.isfinite(old_ml4).all()):
        raise ValueError('forecast rank input shapes/boolean masks must match')
    result = {}
    scope = available & eligible
    if set(raw) != set(TARGETS):
        raise ValueError('exactly four raw forecast matrices are required')
    for key in TARGETS:
        values = np.asarray(raw[key], float)
        if values.shape != shape or not np.isfinite(values[scope]).all():
            raise ValueError('available raw forecasts must be finite and aligned')
        ranked = np.asarray(old_ml4, float).copy()
        for i in range(shape[0]):
            if scope[i].any():
                rank = ep.old.prior.ranks(pd.Series(values[i]).where(scope[i]))
                ranked[i,scope[i]] = rank[scope[i]]
        result[key] = ranked
    return result


def modified_pre_score(spec, a, n):
    """Prefix-only substitution of the old weak ML rank, with exact fallback."""
    if spec not in configs():
        raise ValueError('Only the ten registered configurations are permitted')
    shape = np.asarray(a['baseline']).shape
    if (len(shape) != 2 or isinstance(n,(bool,np.bool_)) or
            not isinstance(n,(int,np.integer)) or not 0 <= n <= shape[0]):
        raise ValueError('n must be a valid monthly prefix')
    pre = adapter._matrix(a['pre_score'],shape,n,'pre_score').copy()
    result = np.zeros(shape)
    if spec['target'] != 'none':
        weak = adapter._state(a,'state_weak_narrow',shape[0],n)
        available = adapter._matrix(a['forecast_available'],shape,n,'forecast_available',boolean=True)
        eligible = adapter._matrix(a['eligible'],shape,n,'eligible',boolean=True)
        rank = adapter._matrix(a['rank_'+spec['target']],shape,n,'new rank')
        prior = adapter._matrix(a['old_ml4'],shape,n,'old_ml4')
        active = weak[:,None] & available & eligible & np.isfinite(rank)
        if not np.isfinite(prior).all() or (np.abs(prior) > .5+1e-12).any():
            raise ValueError('old ml4 ranks must be finite and bounded')
        if (np.abs(rank[active]) > .5+1e-12).any():
            raise ValueError('new predictions must already be bounded ranks')
        delta = np.zeros((n,shape[1]))
        delta[active] = .05*spec['blend']*(rank[active]-prior[active])
        nonzero = delta != 0.
        pre[nonzero] += delta[nonzero]
    result[:n] = pre
    return result


def choices(spec, a, m, n, *, policy=None, boundary_selector=None):
    if spec not in configs():
        raise ValueError('Only the ten registered configurations are permitted')
    if spec['base'] == 'incumbent66':
        policy = read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k) == v for k,v in POLICY_FIELDS.items())
    pre = modified_pre_score(spec,a,n)
    return adapter.score_and_choose(spec['base'],0.,np.full(np.asarray(a['baseline']).shape,np.nan),
        a,m,n,pre_score=pre,effects=a['effects'],episode_policy=policy,
        boundary_selector=ep.select_boundary if boundary_selector is None else boundary_selector)


def prepare():
    registered()
    assert not (P/'FROZEN.json').exists() and not (P/'DATA.npz').exists(), 'Never overwrite prepared/frozen data'
    old.assert_hashes(read(old.P/'INPUT_HASHES.json'))
    m,d,px,op,original = ep.load()
    with np.load(old.P/'DATA.npz') as cache:
        cached = {key:cache[key].copy() for key in CACHED_KEYS}
    meta = read(old.P/'META.json')
    validate_axes(m,meta,cached,d['eligible'])
    policy = read(ep.P/'FROZEN.json')['selected']['primary']
    assert all(policy.get(k) == v for k,v in POLICY_FIELDS.items())
    prior_policy = read(ep.old.P/'FROZEN.json')['selected']['primary']['spec']
    assert prior_policy == PRIOR_POLICY, 'Only the frozen weak .05 ml4 slot may be replaced'
    panel,info = arch.old.panel_load(m['etfs'])
    symbols = np.array([s for ss in meta['training']['symbols'] for s in ss])
    target_data = labels.build_targets(panel['adj'],cached['dates'],cached['ends'],symbols,
        cached['train_eligible'],cached['train_good'],np.isfinite(cached['train_plain']).all(axis=1))
    event_good = d['eligible'].reshape(-1) & np.isfinite(cached['event_plain']).all(axis=1)
    forecast = model.annual_forecasts(cached['dates'],cached['ends'],cached['train_plain'],
        target_data['targets'],target_data['good'],cached['event_dates'],cached['event_plain'],event_good)
    shape = d['eligible'].shape
    raw = {key:forecast[key].reshape(shape) for key in TARGETS}
    available = forecast['available'].reshape(shape)
    ranks = ranked_forecasts(raw,available,d['eligible'],original['ml_4'])
    with np.load(ep.P/'FORECASTS.npz') as ef:
        effects = ef['2'].copy()
    a = {key:original[key].copy() for key in ('baseline','core','core_weight','state_calm10','state_weak_narrow')}
    a.update(eligible=d['eligible'].copy(),effects=effects,old_ml4=original['ml_4'].copy(),
        pre_score=ep.old.score(prior_policy,original,len(m['dates'])),forecast_available=available)
    a.update({'rank_'+key:ranks[key] for key in TARGETS})
    a.update({'raw_'+key:raw[key] for key in TARGETS})
    inc,score = choices(configs()[1],a,m,len(m['dates']),policy=policy)
    expected,expected_score = ep.choices(policy,m,d,original,{'2':effects},len(m['dates']))
    assert inc == expected
    np.testing.assert_array_equal(score,expected_score)
    assert rp.make_orders(inc,score,m) == read(ep.P/'primary_ORDERS.json')
    ch65,score65 = choices(configs()[0],a,m,len(m['dates']))
    np.testing.assert_array_equal(score65,original['baseline'])
    assert rp.make_orders(ch65,score65,m) == read(ep.P/'original65_ORDERS.json')
    fallback = dict(a,forecast_available=np.zeros(shape,bool))
    for spec in configs()[2:]:
        ch,z = choices(spec,fallback,m,len(m['dates']),policy=policy)
        assert ch == inc
        np.testing.assert_array_equal(z,score)
    np.savez_compressed(P/'DATA.npz',**a)
    np.savez_compressed(P/'TRAIN_INPUTS.npz',**cached,common_good=target_data['good'],
        vol63=target_data['vol63'],**{'target_'+key:value for key,value in target_data['targets'].items()})
    dump('META.json',m)
    dump('INCUMBENT_POLICY.json',policy)
    dump('LABEL_AUDIT.json',target_data['audit'])
    dump('MODEL_AUDIT.json',forecast['audit'])
    dump('PREPARE_VALIDATION.json',dict(original65_scores_orders_exact=True,
        incumbent_scores_orders_exact=True,all8_missing_forecasts_exact_incumbent=True,
        common_prediction_availability=True,common_training_cohort=True,
        forecasts_annual_prior_dec31=True,forecast_available=int(available.sum()),
        weak_eligible_available=int((available&d['eligible']&a['state_weak_narrow'][:,None]).sum()),
        original_core_support_states_unchanged=True,production_min_months=18,
        old_weekly84_vs_new_monthly10_not_pure_target_comparison=True))
    hashes = merge_hashes(read(old.P/'INPUT_HASHES.json'),read(ep.P/'INPUT_HASHES.json'),
                         read(ep.old.P/'INPUT_HASHES.json'),info['source_hashes'])
    for f in list(implementations())+[old.P/'DATA.npz',old.P/'META.json',old.P/'INPUT_HASHES.json',
            ep.P/'INPUT_HASHES.json',ep.P/'FROZEN.json',ep.P/'FORECASTS.npz',ep.P/'primary_ORDERS.json',
            ep.P/'original65_ORDERS.json',ep.P/'RESULTS.json',ep.old.P/'FROZEN.json',
            ep.old.P/'FEATURES.npz',ep.old.P/'FORECASTS.npz',ep.old.P/'INPUT_HASHES.json',
            PROTECTION_SOURCE,ex.P/'INPUTS.json',ex.P/'PRICES.npz']:
        digest = sha(f)
        assert str(f) not in hashes or hashes[str(f)] == digest, str(f)
        hashes[str(f)] = digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','TRAIN_INPUTS.npz','META.json',
        'INCUMBENT_POLICY.json','LABEL_AUDIT.json','MODEL_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared ten fixed configurations and four continuous target forecasts; exact controls verified',flush=True)


def load():
    registered()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,_,px,op,_ = ep.load()
    assert m == read(P/'META.json')
    with np.load(P/'DATA.npz') as data:
        a = {key:data[key].copy() for key in data.files}
    return m,a,px,op


def admitted(z, incumbent):
    return (z['training']['cagr'] >= incumbent['training']['cagr']-1e-12
        and z['training']['sharpe'] >= incumbent['training']['sharpe']-.05
        and z['training']['max_drawdown'] >= incumbent['training']['max_drawdown']-.03
        and z['early']['cagr'] >= incumbent['early']['cagr']-.025
        and z['recent']['cagr'] >= incumbent['recent']['cagr']-.025)


def select(rows):
    assert [z['spec'] for z in rows] == configs()
    incumbent = rows[1]
    for z in rows:
        z['admitted'] = admitted(z,incumbent)
    qualified = [incumbent]+[z for z in rows[2:] if z['admitted']]
    best = max(z['training']['cagr'] for z in qualified)
    primary = min((z for z in qualified if best-z['training']['cagr'] <= 1e-12),
                  key=lambda z:z['spec']['id'])
    selected = dict(primary=primary,original65=rows[0],incumbent66=incumbent)
    selected.update({diagnostic_name(z['spec']):z for z in rows[2:]})
    return selected

def train():
    registered()
    assert not (P/'FROZEN.json').exists(), 'Never overwrite frozen selection'
    m,a,px,_ = load()
    blocks,_,bounds,valid = rp.paths(px,m,TRAIN)
    rows,returns = [],[]
    for spec in read(P/'GRID.json'):
        ch,_ = choices(spec,a,m,TRAIN)
        rr = rp.fast_returns(ch,blocks,valid)
        rows.append(dict(spec=spec,**old.session.measure(rr,m,bounds)))
        returns.append(rr)
    selected = select(rows)
    folds,stitched,control = [],[],[]
    for n,end in ((36,60),(60,76),(76,88)):
        prefix = [dict(spec=z['spec'],**old.session.measure(rr,m,bounds,n)) for z,rr in zip(rows,returns)]
        winner = select(prefix)['primary']
        stitched.extend(returns[winner['spec']['id']][bounds[n]:bounds[end]])
        control.extend(returns[1][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('PLAN.json','GRID.json','DATA.npz','TRAIN_INPUTS.npz','META.json','INCUMBENT_POLICY.json',
           'LABEL_AUDIT.json','MODEL_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py','model.py','labels.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=10,new_admitted=sum(int(z['admitted']) for z in rows[2:]),
        primary_source=selected['primary']['spec']['target'],
        hashes={f:sha(P/f) for f in files},pristine_oos=False,author_labels_used=False,
        diagnostics_may_be_unqualified=True,all8_diagnostics_frozen=True,cagr_tie_tolerance=1e-12))
    (P/'FROZEN.sha256').write_text(sha(P/'FROZEN.json')+'\n')
    print('Frozen primary',selected['primary']['spec'],flush=True)


def evaluate():
    registered()
    frozen=read(P/'FROZEN.json')
    assert sha(P/'FROZEN.json') == (P/'FROZEN.sha256').read_text().strip()
    old.assert_hashes(frozen['hashes'],P)
    m,a,px,op=load()
    blocks,_,bounds,valid=rp.paths(px,m,len(m['dates']))
    orders={}
    for name,z in frozen['selected'].items():
        ch,score=choices(z['spec'],a,m,len(m['dates']))
        orders[name]=rp.make_orders(ch,score,m)
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
            assert abs(common['slices']['training2016_2023']['cagr']-frozen['selected'][name]['training']['cagr'])<1e-10
        results[name]=dict(common,selection=frozen['selected'].get(name))
        for suffix,value in (('ORDERS',oo),('DAILY',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]),('TRADES',tr),('HOLDINGS',hold)):
            dump(name+'_'+suffix+'.json',value)
        print(name,round(common['metrics']['cagr']*100,3),round(common['metrics']['sharpe'],3),flush=True)
    old_results=read(ep.P/'RESULTS.json')
    for name,source in (('original65','original65'),('incumbent66','primary')):
        for key in ('cagr','sharpe','max_drawdown'):
            assert abs(results[name]['metrics'][key]-old_results[source]['metrics'][key])<1e-10
    months=[]
    for i,day in enumerate(m['dates']):
        row=dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34')
        for name,oo in orders.items():
            if name=='qqq':continue
            symbols=oo[i]['symbols']
            ret=px.loc[m['ends'][i],symbols]/px.loc[day,symbols]-1 if symbols else []
            row[name]=dict(symbols=symbols,monthly_return=float(np.mean(ret)) if len(ret) else 0.)
        months.append(row)
    dump('MONTH_COMPARISON.json',months)
    attribution={}
    for name,reference in [(diagnostic_name(z),'incumbent66') for z in configs()[2:]]:
        changes=[]
        for row in months:
            z,b=row[name],row[reference]
            if set(z['symbols'])!=set(b['symbols']):
                changes.append(dict(date=row['date'],split=row['split'],old=b['symbols'],new=z['symbols'],
                    difference=z['monthly_return']-b['monthly_return'],
                    log_contribution=float(np.log1p(z['monthly_return'])-np.log1p(b['monthly_return']))))
        positive=sorted([z['log_contribution'] for z in changes if z['log_contribution']>0],reverse=True)
        met=results[name]['metrics']
        undo=lambda n:float(((1+met['total_return'])*np.exp(-sum(positive[:n])))**(1/met['years'])-1)
        attribution[name+'__vs__'+reference]=dict(changed_months=len(changes),changes=changes,
            undo_best1_diagnostic_cagr=undo(1),undo_best3_diagnostic_cagr=undo(3))
    dump('ATTRIBUTION.json',attribution)
    # The inherited large protection inventory is audited once, after all phases.
    assert_protected()
    dump('RESULTS.json',results)
    dump('VALIDATION.json',dict(original_controls_exact=True,independent_fixed_unit_replay=True,
        training_freeze_parity=True,protected_files=len(read(P/'PROTECTED.json')),protected_changes=[],
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=10,
        unchanged_core_and_total_ml_budget=True,all8_candidate_diagnostics_frozen=True,
        old_weekly84_vs_new_monthly10_not_pure_target_comparison=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
