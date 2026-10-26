"""Fixed absolute versus own-sector detrended endpoint effect replacement.

Only the existing episode target changes; real phases require frozen registration.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
NAMES=('incumbent_local67','absolute_detrend','sector_detrend')
ARMS=NAMES[1:]
TRAIN=88
NEW_KEYS={'detrend_z_absolute','detrend_z_sector','detrend_q_absolute','detrend_q_sector','detrend_available'}
CACHE_FILES=('DATA.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
local=module('detrended121_local',P.parent/'monthly_local_linear70/study.py')
features=module('detrended121_features',P/'features.py')
effect_adapter=module('detrended121_effect_adapter',P.parent/'monthly_sector_reliability70/adapter.py')
rp,ex,adapter=local.rp,local.ex,local.adapter

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False,
                    default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n'
    path=P/name
    if path.exists():
        if path.read_text()!=text:raise ValueError('Refuse overwrite: '+name)
    else:
        with path.open('x') as f:f.write(text)
def verify(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('Hash changed: '+path)
def runtime():return dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def registered():
    plan=read(P/'PLAN.json');verify(plan['hashes'])
    assert plan['runtime']==runtime()
    assert plan['design']==read(P/'DESIGN.json')
    return plan

def initialize():
    assert not (P/'PLAN.json').exists()
    d=read(P/'DESIGN.json');assert d['names']==list(NAMES) and d['round']==121
    local.registered();local.verify_source()
    raw=read(local.P/'PLAN.json')['raw_source_hashes'];assert len(raw)==64
    protected=ROOT/d['protected_manifest']['path']
    assert sha(protected)==d['protected_manifest']['sha256']
    inventory=read(protected);assert len(inventory)==21444;verify(inventory)
    review=ROOT/d['source_review']
    assert sha(review)==d['source_review_sha256']
    files=set(local.implementations())|set(local.source_files())|{review}
    files.update(P/x for x in ('study.py','features.py','test_features.py','test_study.py','DESIGN.json','TESTS.txt'))
    files.update((ROOT/x).resolve() for x in d['independent_sources'])
    files.update(effect_adapter.dependency_files())
    files.update(features.dependency_files())
    independent=module('detrended121_independent',ROOT/'work/monthly_detrended121_independent.py')
    files.update(independent.source_files())
    files.add(protected)
    files.update(local.P/x for x in ('DATA.npz','META.json','CACHE_HASHES.json','INPUT_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'))
    files.update(Path(x) for x in raw)
    files.update(Path(x) for x in local.calendar_provenance()['source_hashes'])
    files.update({Path(sys.executable).resolve(),Path(np.__file__),Path(pd.__file__)})
    hashes={str(x.resolve()):sha(x) for x in files};verify(raw)
    write('PLAN.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),design=d,hashes=hashes,raw_source_hashes=raw,runtime=runtime(),actual_feature_calculations_before_registration=0,actual_model_fits_before_registration=0))
    print('Registered three fixed arms before new feature calculations or outcomes',flush=True)

def choices(name,a,m,n,policy):
    if name not in NAMES:raise ValueError('Unregistered arm')
    if name==NAMES[0]:return local.choices(local.configs()[4],a,m,n,policy=policy)
    shape=local._shape(a,n)
    key='detrend_q_absolute' if name==NAMES[1] else 'detrend_q_sector'
    factor=adapter._matrix(a[key],shape,n,key)
    if (factor[np.isfinite(factor)]>0.).any():raise ValueError('Detrended target must be nonpositive')
    pre=local.modified_pre_score(local.configs()[4],a,n)
    return effect_adapter.score_and_choose(a[key],a['detrend_available'],a,m,n,
        pre_score=pre,episode_policy=policy,boundary_selector=local.ep.select_boundary)


def source():
    m,a,px,op=local.load()
    assert len(m['dates'])==122 and a['baseline'].shape==(122,49) and len(a)==27
    return m,a,px,op

def validate_calendar(adj,m):
    last=pd.Timestamp(m['dates'][-1]);observed=adj.index[adj.index<=last]
    if not len(observed) or observed[-1]!=last:raise ValueError('Missing last signal session')
    cal=local.calendar_labels._CAL.market_calendar(start=str(observed[0].date()),end=str(last.date()))
    expected=cal.sessions_in_range(observed[0],last)
    if expected.tz is not None:expected=expected.tz_localize(None)
    if not observed.equals(expected):raise ValueError('Compressed or nonofficial price calendar')
    if not pd.DatetimeIndex(m['dates']).isin(observed).all():raise ValueError('Invalid signal sessions')
    return dict(complete=True,first=str(observed[0].date()),last=str(last.date()),count=len(observed),provenance=local.calendar_provenance())

def prepare():
    plan=registered();assert not (P/'DATA.npz').exists()
    m,a,px,op=source();original={k:v.copy() for k,v in a.items()}
    panel,info=local.arch.old.panel_load(m['etfs'])
    assert local.raw_source_hashes(info)==plan['raw_source_hashes']
    parity=local.verify_panel_parity(panel,px,op);calendar=validate_calendar(panel['adj'],m)
    cal=local.calendar_labels._CAL.market_calendar(start=str(panel['adj'].index[0].date()),end=m['dates'][-1])
    sessions=cal.sessions_in_range(panel['adj'].index[0],pd.Timestamp(m['dates'][-1]))
    if sessions.tz is not None:sessions=sessions.tz_localize(None)
    feature_arrays,audit=features.build_features(panel,m,benchmark_map=local.ex.t.strict.benchmark_map,
        official_sessions=sessions,calendar_sha256=features.calendar_digest(sessions))
    assert set(feature_arrays)==NEW_KEYS
    a.update(feature_arrays)
    assert set(a)==set(original)|NEW_KEYS
    for k in original:
        assert a[k].shape==original[k].shape and a[k].dtype==original[k].dtype and a[k].tobytes()==original[k].tobytes()
    policy=read(local.P/'INCUMBENT_POLICY.json');orders={}
    for name in NAMES:
        ch,sc=choices(name,a,m,122,policy);orders[name]=rp.make_orders(ch,sc,m)
        for n in (0,1,36,60,76,88):
            pc,ps=choices(name,a,m,n,policy);assert pc==ch[:n];np.testing.assert_array_equal(ps,sc[:n])
    assert orders[NAMES[0]]==read(local.P/'local_linear_ORDERS.json')
    missing=dict(a,detrend_available=np.zeros((122,49),bool),detrend_q_absolute=np.full((122,49),np.nan),detrend_q_sector=np.full((122,49),np.nan))
    for name in ARMS:
        ch,sc=choices(name,missing,m,122,policy);assert rp.make_orders(ch,sc,m)==orders[NAMES[0]]
    np.savez_compressed(P/'DATA.npz',**a)
    write('META.json',m);write('FEATURE_AUDIT.json',audit)
    write('PRETRAIN_ORDERS.json',orders);write('PRICE_PARITY.json',dict(prices=parity,calendar=calendar))
    write('CACHE.json',{str(P/x):sha(P/x) for x in CACHE_FILES})
    print('Three full order lists frozen before portfolio outcomes',flush=True)

def loaded():
    registered();verify(read(P/'CACHE.json'));m,old,px,op=source();assert read(P/'META.json')==m
    with np.load(P/'DATA.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    assert set(a)==set(old)|NEW_KEYS
    for k in old:assert a[k].dtype==old[k].dtype and a[k].shape==old[k].shape and a[k].tobytes()==old[k].tobytes()
    return m,a,px,op

def admitted(a,b):
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
    vals=[a[k][s] for k,s in fields];refs=[b[k][s] for k,s in fields]
    if not np.isfinite(vals+refs).all():raise ValueError('Nonfinite admission statistic')
    return all(x>=y-t for x,y,t in zip(vals,refs,(1e-12,.05,.03,.025,.025)))

def select(rows):
    assert len(rows)==3 and [r['name'] for r in rows]==list(NAMES)
    gates={NAMES[i]:admitted(rows[2],rows[i]) for i in (0,1)}
    changed=all(gates.values()) and rows[2]['training']['cagr']>rows[0]['training']['cagr']+1e-12
    return dict(primary=NAMES[2] if changed else NAMES[0],admitted=bool(changed),gates=gates)

def train():
    assert not (P/'FROZEN.json').exists()
    audit=read(P/'INDEPENDENT_PREPARE.json');assert audit['passed'] is True
    for k,f in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):assert audit[k]==sha(P/f)
    verify(audit['source_hashes']);m,a,px,op=loaded();orders=read(P/'PRETRAIN_ORDERS.json')
    blocks,_,bounds,valid=rp.paths(px,m,TRAIN);rows=[];returns=[]
    for name in NAMES:
        ch=[[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(orders[name][:TRAIN])]
        rr=rp.fast_returns(ch,blocks,valid);returns.append(rr)
        rows.append(dict(name=name,**local.old.session.measure(rr,m,bounds)))
    selection=select(rows);folds=[];stitched=[];control=[]
    for n,end in ((36,60),(60,76),(76,88)):
        prefix=[dict(name=name,**local.old.session.measure(rr,m,bounds,n)) for name,rr in zip(NAMES,returns)]
        pick=select(prefix);idx=NAMES.index(pick['primary'])
        stitched.extend(returns[idx][bounds[n]:bounds[end]]);control.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(n=n,end=end,selection=pick))
    write('TRAIN_TRIALS.json',rows)
    write('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,metrics=rp.stats_rr(np.asarray(stitched),m['dates'][36],m['ends'][87]),incumbent=rp.stats_rr(np.asarray(control),m['dates'][36],m['ends'][87])))
    files=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
    write('FROZEN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection=selection,hashes={str(P/x):sha(P/x) for x in files},pristine_oos=False,all_diagnostic_orders_frozen=True))
    print('Selected '+selection['primary'],flush=True)

def evaluate():
    frozen=read(P/'FROZEN.json');verify(frozen['hashes']);m,a,px,op=loaded()
    orders=read(P/'PRETRAIN_ORDERS.json');results={};periods=[]
    for name in NAMES:
        results[name]={}
        for mode,cost,key in (('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost')):
            met,eq,tr,hold=ex.e.engine.replay(px,op,orders[name],cost=cost,execution=mode,start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for year in met['annual_returns']:
                if year['year'] in (2016,2026):year['partial']=True
            results[name][key]=dict(metrics=met,slices=local.arch.old.prior.old.previous.cuts(eq,m))
            write(name+'_'+key+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
            write(name+'_'+key+'_TRADES.json',tr);write(name+'_'+key+'_HOLDINGS.json',hold)
        assert abs(results[name]['same_close']['slices']['training2016_2023']['cagr']-read(P/'TRAIN_TRIALS.json')[NAMES.index(name)]['training']['cagr'])<1e-10
        print(name,results[name]['same_close']['metrics']['cagr'],flush=True)
    for i,day in enumerate(m['dates']):
        ret={};held={}
        for name in NAMES:
            syms=orders[name][i]['symbols'];held[name]=syms
            ret[name]=float((px.loc[m['ends'][i],syms]/px.loc[day,syms]-1).mean()) if syms else 0.
        periods.append(dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34',holdings=held,returns=ret,sector_minus_absolute=ret[ARMS[1]]-ret[ARMS[0]]))
    attribution={}
    for name in ARMS:
        changes=[]
        for row in periods:
            if set(row['holdings'][name])!=set(row['holdings'][NAMES[0]]):
                gain=float(np.log1p(row['returns'][name])-np.log1p(row['returns'][NAMES[0]]))
                changes.append(dict(date=row['date'],split=row['split'],old=row['holdings'][NAMES[0]],new=row['holdings'][name],log_contribution=gain))
        positive=sorted((x['log_contribution'] for x in changes if x['log_contribution']>0),reverse=True);met=results[name]['same_close']['metrics']
        attribution[name]=dict(changed_months=len(changes),changes=changes,undo_best1_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:1])))**(1/met['years'])-1),undo_best3_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:3])))**(1/met['years'])-1))
    write('ATTRIBUTION.json',attribution);write('RESULTS.json',results);write('MONTH_COMPARISON.json',periods)
    verify(read(ROOT/registered()['design']['protected_manifest']['path']))
    write('VALIDATION.json',dict(protected_files=21444,protected_changes=[],fixed_arms=3,original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,primary=frozen['selection']['primary'],primary_target_reached=results[frozen['selection']['primary']]['same_close']['metrics']['cagr']>=.7,independent_outcome_review_pending=True))
    lines=['第121轮：绝对价格与行业相对价格的去趋势终点位移。',
           '固定三臂：原候选、绝对价格终点修正、行业相对终点修正；共同211收盘窗口，跳过最近42日。',
           '以非正目标替换原episode效应的一半；原负效应可能提高。没有协整或收益预测结论。',
           '2016-08-17至2026-09-24；同收盘零成本为理想化回放，复用开发历史，不是纯样本外。',
           '名称 | 年化 | Sharpe | 最大回撤 | 前88期年化 | 后34期年化 | 次日开盘5bp/边年化']
    for name,r in results.items():
        z=r['same_close'];v=z['metrics'];cut=z['slices']
        lines.append(f"{name} | {v['cagr']:.2%} | {v['sharpe']:.3f} | {v['max_drawdown']:.2%} | {cut['training2016_2023']['cagr']:.2%} | {cut['later34']['cagr']:.2%} | {r['next_open_cost']['metrics']['cagr']:.2%}")
    lines+=['冻结选择：'+frozen['selection']['primary'],'最终独立收益核验另见INDEPENDENT_EVALUATE.json。','历史Top32估算、假定17ETF和复权源限制保持；无观察账户更改。']
    with (P/'REPORT.txt').open('x') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('initialize','prepare','train','evaluate'))
    globals()[ap.parse_args().phase]()
