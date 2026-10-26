"""Fixed nonweak-regime extension of monthly log-excess ranking."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent
_SPEC = importlib.util.spec_from_file_location('scope70_source',O/'monthly_target_definition70/study.py')
upstream = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upstream)
old, rp, ex, arch, ep, read, sha, adapter = (upstream.old, upstream.rp, upstream.ex,
    upstream.arch, upstream.ep, upstream.read, upstream.sha, upstream.adapter)
TRAIN = 88
POLICY_FIELDS = dict(upstream.POLICY_FIELDS)
DESIGN_SHA256 = 'c52c923400606a7f89a50f0cdbb69a8eec31357a54874efc92652a49fe491924'
SCOPES = ('calm_nonweak','nonweak')


def dump(name,value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n')


def configs():
    rows=[dict(id=0,base='original65',scope='none',budget=0.),
          dict(id=1,base='incumbent66',scope='none',budget=0.),
          dict(id=2,base='incumbent67',scope='none',budget=0.)]
    for scope in SCOPES:
        for budget in (.01,.025,.05):
            rows.append(dict(id=len(rows),base='incumbent67',scope=scope,budget=budget))
    return rows


def diagnostic_name(spec):
    return spec['scope']+'_budget'+str(spec['budget']).replace('.','p')


def implementations():
    return tuple(sorted(set(upstream.implementations()) | {P/'study.py'}))


def design():
    assert sha(P/'DESIGN.json') == DESIGN_SHA256, 'The original dated ensemble specification changed'
    value = read(P/'DESIGN.json')
    assert value['grid'] == configs()
    return value


def verify_source():
    """Validate only source provenance; never consult source selected/trial returns."""
    upstream.registered()
    assert sha(upstream.P/'FROZEN.json') == (upstream.P/'FROZEN.sha256').read_text().strip()
    frozen_hashes = read(upstream.P/'FROZEN.json')['hashes']
    old.assert_hashes(frozen_hashes,upstream.P)
    old.assert_hashes(read(upstream.P/'INPUT_HASHES.json'))
    old.assert_hashes(read(upstream.P/'CACHE_HASHES.json'),upstream.P)


def source_files():
    return [upstream.P/name for name in ('PLAN.json','GRID.json','INPUT_HASHES.json',
        'CACHE_HASHES.json','FROZEN.json','FROZEN.sha256','DATA.npz','META.json',
        'INCUMBENT_POLICY.json','PROTECTED.json','LABEL_AUDIT.json','MODEL_AUDIT.json',
        'PREPARE_VALIDATION.json')]


def initialize():
    assert not (P/'PLAN.json').exists(), 'Never overwrite the registered ensemble experiment'
    fixed = design()
    verify_source()
    dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        design_created_at=fixed['created_at'],design_sha256=DESIGN_SHA256,design=fixed,
        hashes={str(f):sha(f) for f in implementations()},
        source_hashes={str(f):sha(f) for f in source_files()}))
    dump('GRID.json',configs())
    dump('PROTECTED.json',read(upstream.P/'PROTECTED.json'))


def registered():
    design()
    assert read(P/'GRID.json') == configs()
    plan=read(P/'PLAN.json')
    assert plan['design_sha256'] == DESIGN_SHA256
    old.assert_hashes(plan['hashes'])
    old.assert_hashes(plan['source_hashes'])


def assert_protected():
    old.assert_hashes(read(P/'PROTECTED.json'))


def modified_pre_score(spec,a,n):
    if spec not in configs():
        raise ValueError('Only the nine registered configurations are permitted')
    shape=np.asarray(a['baseline']).shape
    if (len(shape)!=2 or isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
            or not 0<=n<=shape[0]):
        raise ValueError('n must be a valid monthly prefix')
    source_spec=upstream.configs()[5 if spec['base']=='incumbent67' else 1]
    pre=upstream.modified_pre_score(source_spec,a,n)[:n].copy()
    if spec['scope']!='none':
        weak=adapter._state(a,'state_weak_narrow',shape[0],n)
        calm=adapter._state(a,'state_calm10',shape[0],n)
        use=~weak
        if spec['scope']=='calm_nonweak':use &= calm
        available=adapter._matrix(a['extension_available'],shape,n,'extension_available',boolean=True)
        common=adapter._matrix(a['forecast_available'],shape,n,'forecast_available',boolean=True)
        eligible=adapter._matrix(a['eligible'],shape,n,'eligible',boolean=True)
        rank=adapter._matrix(a['rank_log_excess'],shape,n,'rank_log_excess')
        baseline=adapter._matrix(a['baseline'],shape,n,'baseline')
        core=adapter._matrix(a['core'],shape,n,'core')
        cw=np.asarray(a['core_weight'],float)
        if cw.shape==(shape[0],):cw=cw[:n,None]
        elif cw.shape in ((shape[0],1),shape):cw=cw[:n]
        else:raise ValueError('invalid core_weight shape')
        if not np.isfinite(cw).all() or ((cw<.5)|(cw>=1)).any():
            raise ValueError('invalid core weights')
        support=(baseline-core)/(1-cw)
        active=use[:,None]&available&common&eligible&np.isfinite(rank)
        if (np.abs(rank[active])>.5+1e-12).any():raise ValueError('unbounded ranks')
        delta=np.zeros(pre.shape)
        delta[active]=spec['budget']*(rank[active]-support[active])
        changed=delta!=0.
        pre[changed]+=delta[changed]
    full=np.zeros(shape);full[:n]=pre
    return full


def choices(spec,a,m,n,*,policy=None,boundary_selector=None):
    if spec not in configs():
        raise ValueError('Only the nine registered configurations are permitted')
    if spec['base'] in ('incumbent66','incumbent67'):
        policy=read(P/'INCUMBENT_POLICY.json') if policy is None else policy
        assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    pre=modified_pre_score(spec,a,n)
    return adapter.score_and_choose('incumbent66' if spec['base']=='incumbent67' else spec['base'],0.,np.full(np.asarray(a['baseline']).shape,np.nan),
        a,m,n,pre_score=pre,effects=a['effects'],episode_policy=policy,
        boundary_selector=ep.select_boundary if boundary_selector is None else boundary_selector)


def prepare():
    registered()
    assert not (P/'DATA.npz').exists() and not (P/'FROZEN.json').exists(), 'Never overwrite prepared/frozen inputs'
    verify_source()
    m,a,px,op=upstream.load()
    policy=read(upstream.P/'INCUMBENT_POLICY.json')
    assert all(policy.get(k)==v for k,v in POLICY_FIELDS.items())
    assert len(m['dates'])==122 and m['dates'][0]=='2016-08-17'
    assert m['ends'][87]=='2023-11-30' and m['ends'][-1]=='2026-09-24'
    shape=np.asarray(a['baseline']).shape
    assert shape==(122,49)
    a['extension_available']=a['forecast_available'].copy()
    active=a['forecast_available']&a['eligible']
    audits={scope:dict(active_cells=int((active&((~a['state_weak_narrow']) & (a['state_calm10'] if scope=='calm_nonweak' else True))[:,None]).sum()),new_budget_max=.05) for scope in SCOPES}
    controls={}
    for i,j in ((0,0),(1,1),(2,5)):
        got=choices(configs()[i],a,m,len(m['dates']),policy=policy)
        expected=upstream.choices(upstream.configs()[j],a,m,len(m['dates']),policy=policy)
        assert got[0]==expected[0]
        np.testing.assert_array_equal(got[1],expected[1])
        filename=('original65_ORDERS.json','incumbent66_ORDERS.json','primary_ORDERS.json')[i]
        assert rp.make_orders(*got,m)==read(upstream.P/filename)
        controls[i]=got
    fallback=dict(a,extension_available=np.zeros(shape,bool))
    for spec in configs()[3:]:
        got=choices(spec,fallback,m,len(m['dates']),policy=policy)
        assert got[0]==controls[2][0]
        np.testing.assert_array_equal(got[1],controls[2][1])
        weak=a['state_weak_narrow']
        np.testing.assert_array_equal(modified_pre_score(spec,a,len(m['dates']))[weak],
            modified_pre_score(configs()[2],a,len(m['dates']))[weak])
    np.savez_compressed(P/'DATA.npz',**a)
    dump('META.json',m)
    dump('INCUMBENT_POLICY.json',policy)
    dump('SCOPE_AUDIT.json',audits)
    dump('PREPARE_VALIDATION.json',dict(original_three_controls_exact=True,all6_missing_exact_incumbent67=True,
        no_refit=True,no_new_labels=True,weak_model_unchanged=True,
        unchanged_core_support_states_and_effects=True,source_freeze_verified=True))
    hashes=upstream.merge_hashes(read(upstream.P/'INPUT_HASHES.json'))
    for f in list(implementations())+source_files()+[P/'DESIGN.json']:
        digest=sha(f)
        assert str(f) not in hashes or hashes[str(f)]==digest,str(f)
        hashes[str(f)]=digest
    dump('INPUT_HASHES.json',hashes)
    dump('CACHE_HASHES.json',{f:sha(P/f) for f in ('DATA.npz','META.json','INCUMBENT_POLICY.json',
        'SCOPE_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json')})
    print('Prepared nine fixed model-scope configurations; exact three controls verified',flush=True)


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
           'SCOPE_AUDIT.json','PREPARE_VALIDATION.json','INPUT_HASHES.json',
           'CACHE_HASHES.json','PROTECTED.json','study.py')
    dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),selected=selected,
        trials=9,new_admitted=sum(int(z['admitted']) for z in rows[3:]),
        primary_source=selected['primary']['spec']['scope'],
        hashes={f:sha(P/f) for f in files},pristine_oos=False,author_labels_used=False,
        diagnostics_may_be_unqualified=True,all6_diagnostics_frozen=True,cagr_tie_tolerance=1e-12))
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
    old_results=read(upstream.P/'RESULTS.json')
    for name,source in (('original65','original65'),('incumbent66','incumbent66'),('incumbent67','primary')):
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
        observer_accounts_changed=False,pristine_oos=False,fixed_configs=9,
        unchanged_core=True,all6_candidate_diagnostics_frozen=True,
        nonweak_aux_budget_max=.05,weak_model_unchanged=True,no_refit=True,
        primary_target70_reached=results['primary']['metrics']['cagr']>=.70))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[parser.parse_args().phase]()
