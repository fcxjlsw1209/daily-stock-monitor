"""Fifteen frozen core-amplitude configurations, with original auxiliaries."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
O=P.parent
_SPEC=importlib.util.spec_from_file_location('amplitude70_source',O/'monthly_target_definition70/study.py')
upstream=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upstream)
old,rp,ex,arch,ep,read,sha=(upstream.old,upstream.rp,upstream.ex,upstream.arch,upstream.ep,upstream.read,upstream.sha)
core_adapter=upstream.module('amplitude70_core_adapter',O/'monthly_core_timing70_research/core_adapter.py')
TRAIN=88
POLICY_FIELDS=dict(upstream.POLICY_FIELDS)
DESIGN_SHA256='77576d23cfcb5b59a681c3e9c709ba0f8c55fa04da8d21fa1ac829caa55bacc0'
FACTORS=('z_tanh','robust_tanh','signed_square')
SOURCE_INC67_ID=5
SOURCE_INC67_SPEC=dict(id=5,base='incumbent66',target='log_excess',blend=1.)


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def configs():
    rows=[dict(id=0,base='original65',factor='none',blend=0.),
          dict(id=1,base='incumbent66',factor='none',blend=0.),
          dict(id=2,base='incumbent67',factor='none',blend=0.)]
    for base in ('original65','incumbent67'):
        for factor in FACTORS:
            for blend in (.5,1.):
                rows.append(dict(id=len(rows),base=base,factor=factor,blend=blend))
    return rows


def diagnostic_name(spec):
    return spec['base']+'_'+spec['factor']+'_blend'+str(int(spec['blend']*100))


def implementations():
    return tuple(sorted(set(upstream.implementations())|{P/'study.py',P/'features.py',
        Path(core_adapter.__file__),core_adapter.PATH_ADAPTER_PATH,
        O/'monthly_stability_monitor/base_strategy.py',O/'monthly_stability_monitor/strategy.py'}))


def design():
    assert sha(P/'DESIGN.json')==DESIGN_SHA256,'Frozen amplitude specification changed'
    value=read(P/'DESIGN.json')
    assert value['grid']==configs()
    return value


def verify_source():
    upstream.registered()
    assert upstream.configs()[SOURCE_INC67_ID]==SOURCE_INC67_SPEC
    assert sha(upstream.P/'FROZEN.json')==(upstream.P/'FROZEN.sha256').read_text().strip()
    frozen=read(upstream.P/'FROZEN.json')
    assert frozen['selected']['primary']['spec']==SOURCE_INC67_SPEC,'Frozen incumbent67 changed'
    old.assert_hashes(frozen['hashes'],upstream.P)
    old.assert_hashes(read(upstream.P/'INPUT_HASHES.json'))
    old.assert_hashes(read(upstream.P/'CACHE_HASHES.json'),upstream.P)


def source_files():
    return [upstream.P/name for name in ('PLAN.json','GRID.json','INPUT_HASHES.json',
        'CACHE_HASHES.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json',
        'INCUMBENT_POLICY.json','PROTECTED.json','LABEL_AUDIT.json','MODEL_AUDIT.json',
        'PREPARE_VALIDATION.json','primary_ORDERS.json')]


def initialize():
    assert not (P/'PLAN.json').exists(),'Never overwrite this registered experiment'
    fixed=design()
    verify_source()
    protected=read(upstream.P/'PROTECTED.json')
    assert len(protected)==21436,'Use the exact inherited protection inventory'
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        design_created_at=fixed['created_at'],design_sha256=DESIGN_SHA256,grid=configs(),
        objective=fixed['objective'],raw=fixed['raw'],mapping=fixed['mapping'],
        replacement=fixed['replacement'],selection=fixed['selection'],
        execution=fixed['execution'],novelty=fixed['novelty'],validation=fixed['validation'],
        inputs='Source target_definition70 exact preparedDATA and frozenprimaryid5 log_excess fullweakslot; original66 pre-score/effects and original65core. Full historicaladj panel only for causal samewindow amplitude features. No new MLfit, labels, rawmomentum window, pool or eligibility.',
        mechanism='Pass mappedcore values directly to frozen core_adapter without ranking. Half/full convex combination of oldweightedcore and coreweight*mappedcore. Same delta modifies originalbaseline and realpre-score; selector payload uses ORIGINALbaseline/core/coreweight to preserve auxiliaries exactly. Missingmappedvalue makes zerocoredelta; finitezero valid. Existing incumbent67 MLslot and calmresidual unchanged.',
        scope='Three controls exact65/66/67; controls65/66 never promoted. Twelve fixed candidates frombases65/67,three mappings,half/full. All12 diagnostics frozen regardlessadmission beforelater34. LowerID within1e-12oftopCAGRtie; incumbent67id2 retained when tied. Prefix36/60/76selection diagnostics.',
        protection='Copy all21436 inheritedPROTECTED paths unchanged. Verify source PLAN/INPUT/CACHE/FROZEN+sha and fixedselectedspec, code/input/cache/frozen hashes; fullprotectedinventory checked once at endevaluate. No observer,account,ledger,plan,fingerprint,automation,sourceoroldcode writes.',
        limitations='Repeatedly researched history is development,notpristineOOS; proxyhistoricaluniverse andsamefinalclose signal/fill idealization remain; no70percent guarantee.',
        hashes={str(f):sha(f) for f in implementations()},
        source_hashes={str(f):sha(f) for f in source_files()}))
    dump('GRID.json',configs())
    dump('PROTECTED.json',protected)


def registered():
    design()
    assert read(P/'GRID.json')==configs()
    plan=read(P/'PLAN.json')
    assert plan['design_sha256']==DESIGN_SHA256
    old.assert_hashes(plan['hashes'])
    old.assert_hashes(plan['source_hashes'])


def assert_protected():
    old.assert_hashes(read(P/'PROTECTED.json'))


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    """Only the fixed mapping changes core; original episode support survives."""
    if spec not in configs():
        raise ValueError('Only the fifteen registered configurations are permitted')
    shape=np.asarray(a['baseline']).shape
    if (len(shape)!=2 or isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
            or not 0<=n<=shape[0]):
        raise ValueError('n must be a valid monthly prefix')
    if spec['base']!='original65':
        policy=read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    if spec['base']=='incumbent67':
        assert upstream.configs()[SOURCE_INC67_ID]==SOURCE_INC67_SPEC
        pre=upstream.modified_pre_score(upstream.configs()[SOURCE_INC67_ID],a,n)
    else:
        pre=a['pre_score']
    core=np.full(shape,np.nan) if spec['factor']=='none' else a['core_'+spec['factor']]
    base='original65' if spec['base']=='original65' else 'incumbent66'
    return core_adapter.score_and_choose(base,spec['blend'],core,a,m,n,pre_score=pre,
        effects=a['effects'],episode_policy=policy,
        boundary_selector=ep.select_boundary if boundary_selector is None else boundary_selector)


def validate_features(arrays,shape):
    for key in FACTORS:
        values=np.asarray(arrays[key])
        if values.shape!=shape:
            raise ValueError(key+' must have the full monthly candidate shape')
        if (np.abs(values[np.isfinite(values)])>.5+1e-12).any():
            raise ValueError(key+' must already be bounded; do not rank it again')


def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists(),'Never overwrite prepared/frozen data'
    verify_source()
    m,a,px,op=upstream.load()
    policy=read(upstream.P/'INCUMBENT_POLICY.json')
    assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    assert len(m['dates'])==122 and m['dates'][0]=='2016-08-17'
    assert m['ends'][87]=='2023-11-30' and m['ends'][-1]=='2026-09-24'
    assert np.asarray(a['baseline']).shape==(122,49)
    panel,info=arch.old.panel_load(m['etfs'])
    feature_module=upstream.module('amplitude70_features',P/'features.py')
    arrays,audit=feature_module.build_features(panel['adj'],m,a['state_calm10'])
    validate_features(arrays,a['baseline'].shape)
    a.update({'core_'+key:np.array(arrays[key],dtype=float,copy=True) for key in FACTORS})
    controls={}
    for ident,source_id in ((0,0),(1,1),(2,SOURCE_INC67_ID)):
        got=choices(configs()[ident],a,m,len(m['dates']),policy=policy)
        expected=upstream.choices(upstream.configs()[source_id],a,m,len(m['dates']),policy=policy)
        assert got[0]==expected[0]
        np.testing.assert_array_equal(got[1],expected[1])
        file=(ep.P/'original65_ORDERS.json' if ident==0 else ep.P/'primary_ORDERS.json'
              if ident==1 else upstream.P/'primary_ORDERS.json')
        assert rp.make_orders(*got,m)==read(file)
        controls[ident]=got
    missing=dict(a,**{'core_'+key:np.full(a['baseline'].shape,np.nan) for key in FACTORS})
    for spec in configs()[3:]:
        got=choices(spec,missing,m,len(m['dates']),policy=policy)
        expected=controls[0 if spec['base']=='original65' else 2]
        assert got[0]==expected[0]
        np.testing.assert_array_equal(got[1],expected[1])
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m)
    dump('INCUMBENT_POLICY.json',policy)
    dump('FEATURE_AUDIT.json',audit)
    dump('PREPARE_VALIDATION.json',dict(all3_controls_scores_orders_exact=True,
        all12_missing_exact_respective_base=True,source_inc67_id=SOURCE_INC67_ID,
        no_refit=True,no_amplitude_reranking=True,core_coefficients_unchanged=True,
        original_aux_support_preserved=True,raw_window_unchanged=True,eligibility_unchanged=True))
    hashes=upstream.merge_hashes(read(upstream.P/'INPUT_HASHES.json'),info['source_hashes'],audit['source_hashes'])
    for f in list(implementations())+source_files()+[P/'DESIGN.json']:
        digest=sha(f)
        assert str(f) not in hashes or hashes[str(f)]==digest,str(f)
        hashes[str(f)]=digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','META.json','INCUMBENT_POLICY.json',
        'FEATURE_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared15 fixed core-amplitude configurations; exact65/66/67 controls verified',flush=True)


def load():
    registered()
    verify_source()
    old.assert_hashes(read(P/'INPUT_HASHES.json'))
    old.assert_hashes(read(P/'CACHE_HASHES.json'),P)
    m,_,px,op,_=ep.load()
    assert m==read(P/'META.json')
    with np.load(P/'DATA.npz') as data:
        a={key:data[key].copy() for key in data.files}
    return m,a,px,op

def admitted(z, incumbent):
    return (z['training']['cagr'] >= incumbent['training']['cagr']-1e-12
        and z['training']['sharpe'] >= incumbent['training']['sharpe']-.05
        and z['training']['max_drawdown'] >= incumbent['training']['max_drawdown']-.03
        and z['early']['cagr'] >= incumbent['early']['cagr']-.025
        and z['recent']['cagr'] >= incumbent['recent']['cagr']-.025)


def select(rows):
    assert [z['spec'] for z in rows] == configs()
    incumbent = rows[2]
    for z in rows:
        z['admitted'] = admitted(z,incumbent)
    qualified = [incumbent]+[z for z in rows[3:] if z['admitted']]
    best = max(z['training']['cagr'] for z in qualified)
    primary = min((z for z in qualified if best-z['training']['cagr'] <= 1e-12),
                  key=lambda z:z['spec']['id'])
    selected = dict(primary=primary,original65=rows[0],incumbent66=rows[1],incumbent67=incumbent)
    selected.update({diagnostic_name(z['spec']):z for z in rows[3:]})
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
        control.extend(returns[2][bounds[n]:bounds[end]])
        folds.append(dict(train_end=m['ends'][n-1],test_end=m['ends'][end-1],spec=winner['spec']))
    dump('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        metrics=rp.stats_rr(np.array(stitched),m['dates'][36],m['ends'][87]),
        incumbent=rp.stats_rr(np.array(control),m['dates'][36],m['ends'][87])))
    dump('TRAIN_TRIALS.json',rows)
    np.savez_compressed(P/'TRAIN_RETURNS.npz',returns=np.stack(returns),bounds=np.array(bounds))
    files=('DESIGN.json','PLAN.json','GRID.json','DATA.npz','META.json','INCUMBENT_POLICY.json',
           'FEATURE_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py','features.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=15,new_admitted=sum(int(z['admitted']) for z in rows[3:]),
        primary_source=selected['primary']['spec']['base']+'_'+selected['primary']['spec']['factor'],
        hashes={f:sha(P/f) for f in files},pristine_oos=False,author_labels_used=False,
        diagnostics_may_be_unqualified=True,all12_diagnostics_frozen=True,cagr_tie_tolerance=1e-12))
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
    for name,reference in [(diagnostic_name(z),'incumbent67') for z in configs()[3:]]:
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
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=15,
        unchanged_core_weights_and_total_ml_budget=True,all12_candidate_diagnostics_frozen=True,
        fixed_design_verified=True,no_refit=True,no_amplitude_reranking=True,
        incumbent67_exact=True,original_aux_support_preserved=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
