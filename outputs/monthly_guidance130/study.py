"""Round130: fixed reported annual guidance sign; no new predictive fitting.

Import and pure functions are inert. Only the parent runs registered real phases.
Source joins and all 366 orders are frozen before any challenger returns.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, hashlib, importlib.util, json, re, sys
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
NAMES=('incumbent_local67','zero_guidance','signed_reported_guidance')
TRAIN=88
NEW_KEYS={'guidance_hand'}
CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','NORMALIZED_NUMERIC_ROWS.json')
FROZEN_FILES=('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','TRAIN_TRIALS.json','WALKFORWARD.json','INDEPENDENT_PREPARE.json')
LOCAL_SHA='e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb'
PROPOSAL_SHA='bf9e6bc75486086736c72daacd20ad9d81f5cc37bcc876dd79d6e4006dac7315'
NUMERIC='outputs/monthly_guidance_events_union_v2/EVENTS.json'
DIRECTION='outputs/monthly_guidance_direction_join_v2/COVERAGE.json'
DEPENDENCIES=None


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(name,value):
    text=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x)+'\n'
    path=P/name
    if path.exists():
        if path.read_text()!=text:raise ValueError('Refuse overwrite '+name)
    else:
        with path.open('x') as f:f.write(text)

def absolute(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p

def verify(mapping):
    for path,digest in mapping.items():
        if sha(absolute(path))!=digest:raise ValueError('Hash changed '+path)

def dependencies():
    global DEPENDENCIES
    if P.name!='monthly_guidance130':raise ValueError('Wrong installed study directory')
    if DEPENDENCIES is None:
        source=P.parent/'monthly_local_linear70/study.py'
        if sha(source)!=LOCAL_SHA:raise ValueError('Frozen local67 changed')
        if str(ROOT/'work') not in sys.path:sys.path.insert(0,str(ROOT/'work'))
        DEPENDENCIES=(module('guidance130_local',source),
            module('guidance130_hand',ROOT/'work/monthly_guidance_hand_factor.py'),
            module('guidance130_clock',ROOT/'work/monthly_guidance_numeric_clock130.py'),
            module('guidance130_placement',ROOT/'work/monthly_guidance_placement130.py'))
    return DEPENDENCIES

def registered():
    plan=read(P/'PLAN.json');verify(plan['hashes']);verify(plan['raw_source_hashes']);verify(plan['guidance_source_hashes'])
    if (plan['python'],plan['numpy'],plan['pandas'])!=(sys.version,np.__version__,pd.__version__):raise ValueError('Registered runtime changed')
    if plan['design']!=read(P/'DESIGN.json'):raise ValueError('PLAN and DESIGN differ')
    return plan

def source_bindings(files):
    """One-level evidence closure from the explicit source packages/reviews.

    Their source manifests already enumerate raw evidence. Do not recursively
    parse historical ancestor results or execute any source financial parser.
    """
    pins={}
    def add(path,h=None,owner=None):
        raw=Path(path)
        p=(owner/raw if owner is not None and not raw.is_absolute() and len(raw.parts)==1 else absolute(raw)).resolve()
        digest=sha(p)
        if h is not None and digest!=h:raise ValueError('Source binding changed '+str(path))
        if str(p) in pins and pins[str(p)]!=digest:raise ValueError('Conflicting source pin')
        pins[str(p)]=digest
    def walk(x,owner):
        if isinstance(x,dict):
            for key,value in x.items():
                if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and ('/' in key or Path(key).suffix in ('.json','.py','.txt','.bin','.decoded','.html','.pdf')):
                    add(key,value,owner)
            for pk,hk in (('path','sha256'),('source_path','source_sha256'),('receipt_path','receipt_sha256'),('raw_path','raw_sha256')):
                if isinstance(x.get(pk),str) and isinstance(x.get(hk),str):add(x[pk],x[hk],owner)
            for value in x.values():walk(value,owner)
        elif isinstance(x,list):
            for value in x:walk(value,owner)
    for p in files:
        p=absolute(p);add(p)
        if p.suffix=='.json':walk(read(p),p.parent)
    return pins

def local_cache_files(directory):
    """Bind every inherited cache artifact read by the independent reviewer."""
    directory=Path(directory);manifest=read(directory/'CACHE_HASHES.json')
    if not isinstance(manifest,dict) or not manifest:raise ValueError('Empty local cache manifest')
    files=[]
    for name,digest in manifest.items():
        if not isinstance(name,str) or Path(name).name!=name or name in ('','.', '..'):raise ValueError('Local cache basename required')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid local cache hash')
        path=directory/name
        if sha(path)!=digest:raise ValueError('Local cache hash changed '+name)
        files.append(path.resolve())
    return files

def initialize():
    """Parent-only registration; never calculate guidance values here."""
    if (P/'PLAN.json').exists():raise ValueError('Never overwrite PLAN')
    local,hand,clock,placement=dependencies();d=read(P/'DESIGN.json')
    assert d['names']==list(NAMES) and d['round']==130 and d['cache_files']==list(CACHE_FILES)
    assert sha(ROOT/'work/monthly_guidance130_proposal.json')==PROPOSAL_SHA
    for entry in d['prerequisite_reviews']:
        receipt=read(ROOT/entry)
        if receipt.get('passed') is not True and receipt.get('status')!='PASS':raise ValueError('Missing passed prerequisite '+entry)
    local.registered();local.verify_source()
    raw=read(local.P/'PLAN.json')['raw_source_hashes'];verify(raw);assert len(raw)==64
    files=set(local.implementations())|set(local.source_files())
    files.update(P/x for x in ('study.py','DESIGN.json','test_study.py','TESTS.txt'))
    files.update(ROOT/x for x in d['implementation_sources']+d['independent_sources'])
    auditor=module('guidance130_registration_auditor',ROOT/'work/monthly_guidance130_independent.py')
    auditor_files={absolute(x).resolve() for x in auditor.source_files()}
    if not auditor_files:raise ValueError('Empty independent prepare code closure')
    files.update(auditor_files)
    files.update(local.P/x for x in ('PLAN.json','DATA.npz','META.json','CACHE_HASHES.json','INPUT_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'))
    files.update(local_cache_files(local.P))
    files.update(absolute(x) for x in raw)
    files.update(absolute(x) for x in local.calendar_provenance()['source_hashes'])
    files.add(ROOT/'outputs/monthly_2016_extension/PRICES.npz')
    files.update((Path(sys.executable).resolve(),Path(np.__file__).resolve(),Path(pd.__file__).resolve()))
    manifest=ROOT/d['protected_manifest']['path'];assert sha(manifest)==d['protected_manifest']['sha256'];files.add(manifest)
    protect=ROOT/d['protected_check'];r=read(protect)
    assert r['passed'] is True and r['checked']==21444;files.add(protect)
    source_files=[ROOT/x for x in d['source_binding_files']+d['prerequisite_reviews']]
    for directory in d['source_package_directories']:
        source_files.extend(p for p in (ROOT/directory).iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt'))
    guidance=source_bindings(source_files)
    files.update(Path(x) for x in guidance)
    write('PLAN.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),design=d,
        hashes={str(Path(x).resolve()):sha(x) for x in files},raw_source_hashes=raw,guidance_source_hashes=guidance,
        independent_prepare_source_hashes={str(x):sha(x) for x in auditor_files},
        actual_feature_calculations_before_registration=0,actual_model_fits_before_registration=0,
        new_market_downloads=0,python=sys.version,numpy=np.__version__,pandas=pd.__version__))

def build_features(meta,numeric_events,direction_coverage,sessions,hand_module,clock_module):
    """Pure source-to-hand construction; caller supplies all data explicitly."""
    numeric=clock_module.coverage(numeric_events,meta,sessions)
    direction_rows=direction_coverage['rows']
    # This fixed study uses disjoint issuers; a new overlap requires a new contract.
    numeric_symbols={e['symbol'] for e in numeric_events}
    if any(r.get('event_id') is not None and r['symbol'] in numeric_symbols for r in direction_rows):
        raise ValueError('Conflicting numeric and categorical source families')
    values,cells=hand_module.build_hand(meta,numeric['rows'],numeric_events,direction_rows)
    if values.dtype!=np.dtype('float64') or np.isinf(values).any() or not np.isin(values[np.isfinite(values)],[-.5,0.,.5]).all():raise ValueError('Invalid hand values')
    split=min(TRAIN,len(meta['dates']));finite=np.isfinite(values);nonzero=finite&(values!=0.)
    audit=dict(schema_version=1,hand_cells=cells,numeric_summary=numeric['summary'],direction_summary=direction_coverage.get('summary'),
        observed_hand_cells=int(finite.sum()),signed_hand_cells=int(nonzero.sum()),maintained_hand_cells=int((finite&(values==0.)).sum()),
        training_observed_cells=int(finite[:split].sum()),later_observed_cells=int(finite[split:].sum()),
        training_signed_cells=int(nonzero[:split].sum()),later_signed_cells=int(nonzero[split:].sum()),
        direction_counts=dict(Counter(c['direction'] for c in cells if c['direction'] is not None)),
        family_counts=dict(Counter(c['family'] for c in cells)),
        numeric_clock='original publication age<21, separate version gate; latest barrier first',
        incomplete_observed_corpus=True,historical_bytes_verified=False,new_model_fits=0,returns_computed=False)
    return {'guidance_hand':values},audit,numeric

def choose(name,a,m,n,local,placement,policy):
    if name not in NAMES:raise ValueError('Unregistered arm')
    if name==NAMES[0]:return local.choices(local.configs()[4],a,m,n,policy=policy)
    hand=np.asarray(a['guidance_hand'])
    if hand.shape!=np.asarray(a['baseline']).shape:raise ValueError('Wrong hand shape')
    prefix=hand[:n].astype(float,copy=False)
    if np.isinf(prefix).any() or not np.isin(prefix[np.isfinite(prefix)],[-.5,0.,.5]).all():raise ValueError('Invalid hand domain')
    if name==NAMES[1]:
        hand=hand.copy();hand[:n]=np.where(np.isfinite(prefix),0.,np.nan)
    pre=local.modified_pre_score(local.configs()[4],a,n)
    return placement.score_and_choose(pre,hand,a,m,n,policy=policy,boundary_selector=local.ep.select_boundary)

def unchanged(original,current):
    if set(current)!=set(original)|NEW_KEYS:raise ValueError('Unexpected data keys')
    for k,v in original.items():
        if current[k].dtype!=v.dtype or current[k].shape!=v.shape or current[k].tobytes()!=v.tobytes():raise ValueError('Original array changed '+k)

def order_panels(a,m,local,placement,policy):
    n=len(m['dates']);orders={};scores={}
    for name in NAMES:
        ch,sc=choose(name,a,m,n,local,placement,policy)
        orders[name]=local.rp.make_orders(ch,sc,m);scores[name]=sc
        for stop in sorted({0,1,min(36,n),min(60,n),min(76,n),min(88,n),n}):
            pc,ps=choose(name,a,m,stop,local,placement,policy)
            if pc!=ch[:stop] or ps.tobytes()!=sc[:stop].tobytes():raise ValueError('Order/score prefix changed')
    if orders[NAMES[1]]!=orders[NAMES[0]] or scores[NAMES[1]].tobytes()!=scores[NAMES[0]].tobytes():raise ValueError('Zero arm differs from local67')
    for absent in (np.full(np.asarray(a['baseline']).shape,np.nan),np.zeros(np.asarray(a['baseline']).shape)):
        # ETFs must remain missing even in the all-zero observed-stock probe.
        for i,names in enumerate(m['symbols']):
            for j,symbol in enumerate(names):
                if symbol in m['etfs']:absent[i,j]=np.nan
        empty=dict(a,guidance_hand=absent)
        ch,sc=choose(NAMES[2],empty,m,n,local,placement,policy)
        if local.rp.make_orders(ch,sc,m)!=orders[NAMES[0]] or sc.tobytes()!=scores[NAMES[0]].tobytes():raise ValueError('Missing/zero identity failed')
    return orders,scores

def placement_diagnostics(a,m,local,placement):
    """Audit the pre-episode increment; never claim this bounds final scores."""
    n=len(m['dates']);shape=np.asarray(a['baseline']).shape
    source=placement.old._prefix_source(a,n)
    _,_,_,_,cw,factor,finite,support=placement.ADAPTER._inputs('incumbent66',.025,a['guidance_hand'],source,n)
    eligible=placement.ADAPTER._matrix(a['eligible'],shape,n,'eligible',boolean=True)
    active=finite&(factor!=0.)&eligible
    for i,names in enumerate(m['symbols']):
        for j,name in enumerate(names):
            if name in m['etfs'] and finite[i,j]:raise ValueError('ETF guidance in audit')
    pre=local.modified_pre_score(local.configs()[4],a,n)
    delta=np.zeros(shape);delta[active]=.025*(np.clip(support[active]+factor[active],-.5,.5)-support[active])
    if not np.isfinite(delta).all() or np.any(np.abs(delta)>.0125+1e-12):raise ValueError('Audit pre-delta bound')
    cells=[]
    for i,j in zip(*np.where(finite)):
        cells.append(dict(signal_index=int(i),pool_index=int(j),symbol=m['symbols'][i][j],
            original_eligible=bool(eligible[i,j]),active_signed=bool(active[i,j]),
            original_support=float(support[i,j]),original_core_weight=float(cw[i,j]),
            original_pre_score=float(pre[i,j]),pre_delta=float(delta[i,j]),
            modified_pre_score=float(pre[i,j]+delta[i,j]) if delta[i,j]!=0. else float(pre[i,j])))
    return dict(cells=cells,nonzero_pre_deltas=int(np.count_nonzero(delta)),
        max_absolute_pre_delta=float(np.max(np.abs(delta))) if delta.size else 0.,
        minimum_reserved_original_support=float(np.min(1.-cw-.10-.025*active)),
        bound_applies_to_pre_episode_increment_only=True)

def prepare():
    plan=registered();local,hand,clock,placement=dependencies()
    if any((P/x).exists() for x in CACHE_FILES+('CACHE.json',)):raise ValueError('Never overwrite prepared artifacts')
    m,a,px,op=local.load();original={k:v.copy() for k,v in a.items()}
    assert len(m['dates'])==122 and m['train_months']==88 and all(len(x)==49 for x in m['symbols'])
    numeric_events=read(ROOT/NUMERIC)['events'];assert len(numeric_events)==32
    direction=read(ROOT/DIRECTION);assert len(direction['rows'])==5978
    dplan=read((ROOT/DIRECTION).with_name('PLAN.json'))
    assert direction['plan_sha256']==sha((ROOT/DIRECTION).with_name('PLAN.json'))
    devents=[e for p in dplan['sources'] for e in read(ROOT/p)['events']];assert len(devents)==113
    if {e['symbol'] for e in numeric_events}&{e['symbol'] for e in devents}:raise ValueError('Overlapping source corpora')
    calendar=local.calendar_labels._CAL
    sessions=[str(t.date()) for t in calendar.market_calendar('2015-01-01','2027-12-31').sessions]
    new,audit,numeric=build_features(m,numeric_events,direction,sessions,hand,clock)
    assert new['guidance_hand'].shape==(122,49)
    for stop in (1,36,60,76,88):
        meta=dict(m,dates=m['dates'][:stop],symbols=m['symbols'][:stop])
        small=dict(direction,rows=direction['rows'][:stop*49])
        nn,aa,nc=build_features(meta,numeric_events,small,sessions,hand,clock)
        if nn['guidance_hand'].tobytes()!=new['guidance_hand'][:stop].tobytes() or nc['rows']!=numeric['rows'][:stop*49]:raise ValueError('Source hand prefix changed')
    a.update(new);unchanged(original,a)
    # Existing raw-panel parity convention; no fitting or new price acquisition.
    panel,info=local.arch.old.panel_load(m['etfs']);assert local.raw_source_hashes(info)==plan['raw_source_hashes']
    parity=local.verify_panel_parity(panel,px,op)
    policy=read(local.P/'INCUMBENT_POLICY.json');orders,scores=order_panels(a,m,local,placement,policy)
    if orders[NAMES[0]]!=read(local.P/'local_linear_ORDERS.json'):raise ValueError('Original local67 orders changed')
    unchanged(original,a)
    audit['placement']=placement_diagnostics(a,m,local,placement)
    audit.update(guidance_source_hashes=plan['guidance_source_hashes'],raw_source_hashes=plan['raw_source_hashes'],calendar=local.calendar_provenance(),
        zero_control_exact=True,allmissing_control_exact=True,prefix_orders_exact=True,original_arrays_unchanged=True)
    np.savez_compressed(P/'DATA.npz',**a);np.savez_compressed(P/'SCORES.npz',**scores)
    for name,value in [('META.json',m),('FEATURE_AUDIT.json',audit),('PRETRAIN_ORDERS.json',orders),('PRICE_PARITY.json',parity),('NORMALIZED_NUMERIC_ROWS.json',numeric)]:write(name,value)
    write('CACHE.json',{str(P/x):sha(P/x) for x in CACHE_FILES})
    print('Three fixed order lists saved before challenger outcomes',flush=True)

def loaded():
    registered();verify(read(P/'CACHE.json'));local,hand,clock,placement=dependencies();m,old,px,op=local.load()
    with np.load(P/'DATA.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    unchanged(old,a)
    if m!=read(P/'META.json'):raise ValueError('Original meta changed')
    return local,m,a,px,op

def select(rows):
    if [r['name'] for r in rows]!=list(NAMES):raise ValueError('Three fixed arms in declared order required')
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
    values=[r.get(k,{}).get(s) for r in (rows[2],rows[0]) for k,s in fields]
    numeric=all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in values)
    good=False
    if numeric and np.isfinite(values).all():
        v,b=values[:5],values[5:]
        good=bool(v[0]>b[0]+1e-12 and all(x>=y-t for x,y,t in zip(v,b,(1e-12,.05,.03,.025,.025))))
    return dict(primary=NAMES[2] if good else NAMES[0],admitted=good,guards=good,guards_against={NAMES[0]:good})

PREPARE_TRUE_FIELDS=('original27_arrays_byte_identical','all_features_independently_reconstructed',
    'all_three_score_matrices_and_ordered_choices_independently_verified',
    'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')

def canonical_hashes(mapping):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Empty or invalid hash closure')
    result={}
    for path,digest in mapping.items():
        if not isinstance(path,str) or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid hash entry')
        key=str(absolute(path).resolve())
        if key in result and result[key]!=digest:raise ValueError('Conflicting normalized hash')
        result[key]=digest
    return result

def require_prepare_review(plan,review):
    if review.get('passed') is not True or review.get('stage')!='prepare':raise ValueError('Wrong independent stage/status')
    for field in PREPARE_TRUE_FIELDS:
        if review.get(field) is not True:raise ValueError('Missing independent guarantee '+field)
    for field in ('producer_modules_imported','returns_or_nav_computed'):
        if review.get(field) is not False:raise ValueError('Wrong independent isolation '+field)
    if type(review.get('monthly_label_models_fit')) is not int or review['monthly_label_models_fit']!=0:raise ValueError('Independent review fit count')
    sources=canonical_hashes(review.get('source_hashes'))
    expected=canonical_hashes(plan.get('independent_prepare_source_hashes'))
    if sources!=expected:raise ValueError('Independent code closure differs from PLAN')
    plan_hashes=canonical_hashes(plan['hashes'])
    if any(plan_hashes.get(k)!=v for k,v in sources.items()):raise ValueError('Reviewer source not PLAN-bound')
    verify(sources)
    artifacts=canonical_hashes(review.get('artifact_hashes'));cache=canonical_hashes(read(P/'CACHE.json'))
    if set(cache)!={str((P/x).resolve()) for x in CACHE_FILES}:raise ValueError('Prepared seven-artifact scope')
    required={str((P/x).resolve()):sha(P/x) for x in ('PLAN.json','CACHE.json','PRETRAIN_ORDERS.json')}
    required.update(cache)
    if any(artifacts.get(k)!=v for k,v in required.items()):raise ValueError('Incomplete independent artifact coverage')
    allowed=dict(plan_hashes);allowed.update(required)
    if any(allowed.get(k)!=v for k,v in artifacts.items()):raise ValueError('Unbound independent artifact')
    verify(artifacts)
    for key,file in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
        if review.get(key)!=sha(P/file):raise ValueError('Wrong review receipt '+key)

def require_frozen(frozen):
    hashes=canonical_hashes(frozen.get('hashes'))
    if set(hashes)!={str((P/x).resolve()) for x in FROZEN_FILES}:raise ValueError('Frozen six-input scope')
    verify(hashes)
    selection=frozen.get('selection')
    if not isinstance(selection,dict) or set(selection)!={'primary','admitted','guards','guards_against'}:raise ValueError('Frozen selection schema')
    if type(selection['admitted']) is not bool or type(selection['guards']) is not bool:raise ValueError('Frozen selection booleans')
    if not isinstance(selection['guards_against'],dict) or type(selection['guards_against'].get(NAMES[0])) is not bool:raise ValueError('Frozen comparison guard')
    if selection['guards_against']!={NAMES[0]:selection['guards']} or selection['admitted']!=selection['guards']:raise ValueError('Frozen guard consistency')
    if selection['primary']!=(NAMES[2] if selection['admitted'] else NAMES[0]):raise ValueError('Frozen primary')
    if selection!=select(read(P/'TRAIN_TRIALS.json')):raise ValueError('Frozen selection not from training')
    if frozen.get('pristine_oos') is not False or frozen.get('all_diagnostic_orders_frozen') is not True:raise ValueError('Frozen provenance flags')

def train():
    if (P/'FROZEN.json').exists():raise ValueError('Never overwrite frozen selection')
    plan=registered();review=read(P/'INDEPENDENT_PREPARE.json');require_prepare_review(plan,review)
    local,m,a,px,op=loaded();orders=read(P/'PRETRAIN_ORDERS.json');rp=local.rp
    blocks,_,bounds,valid=rp.paths(px,m,TRAIN);rows=[];returns=[]
    for name in NAMES:
        choices=[[m['symbols'][i].index(s) for s in row['symbols']] for i,row in enumerate(orders[name][:TRAIN])]
        rr=rp.fast_returns(choices,blocks,valid);returns.append(rr);rows.append(dict(name=name,**local.old.session.measure(rr,m,bounds)))
    assert rows[0]['training']==rows[1]['training'] and np.array_equal(returns[0],returns[1])
    selection=select(rows);folds=[];stitched=[];control=[]
    for n,end in ((36,60),(60,76),(76,88)):
        prefix=[dict(name=name,**local.old.session.measure(rr,m,bounds,n)) for name,rr in zip(NAMES,returns)]
        pick=select(prefix);idx=NAMES.index(pick['primary']);stitched.extend(returns[idx][bounds[n]:bounds[end]]);control.extend(returns[0][bounds[n]:bounds[end]])
        folds.append(dict(n=n,end=end,selection=pick))
    write('TRAIN_TRIALS.json',rows);write('WALKFORWARD.json',dict(folds=folds,pristine_oos=False,
        metrics=rp.stats_rr(np.asarray(stitched),m['dates'][36],m['ends'][87]),incumbent=rp.stats_rr(np.asarray(control),m['dates'][36],m['ends'][87])))
    write('FROZEN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection=selection,
        hashes={str(P/x):sha(P/x) for x in FROZEN_FILES},pristine_oos=False,all_diagnostic_orders_frozen=True))
    print('Frozen selection '+selection['primary'],flush=True)

def attribution(periods,results,new=NAMES[2],old=NAMES[0]):
    changes=[];ordered=0
    for row in periods:
        held=row['holdings'];ret=row['returns'];ordered+=held[new]!=held[old]
        if set(held[new])!=set(held[old]):
            changes.append(dict(date=row['date'],split=row['split'],old=held[old],new=held[new],
                guidance_events=row.get('guidance_events',[]),log_contribution=float(np.log1p(ret[new])-np.log1p(ret[old]))))
    positive=sorted((x['log_contribution'] for x in changes if x['log_contribution']>0),reverse=True);met=results[new]['same_close']['metrics']
    return dict(changed_months=len(changes),ordered_changed_months=ordered,changes=changes,
        undo_best1_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:1])))**(1/met['years'])-1),
        undo_best3_cagr=float(((1+met['total_return'])*np.exp(-sum(positive[:3])))**(1/met['years'])-1))

def evaluate():
    frozen=read(P/'FROZEN.json');require_frozen(frozen);local,m,a,px,op=loaded();orders=read(P/'PRETRAIN_ORDERS.json');results={}
    for name in NAMES:
        results[name]={}
        for mode,cost,key in (('close',0.,'same_close'),('open',0.,'next_open_zero'),('open',.0005,'next_open_cost')):
            met,eq,tr,hold=local.ex.e.engine.replay(px,op,orders[name],cost=cost,execution=mode,start=pd.Timestamp(m['dates'][0]),end=pd.Timestamp(m['ends'][-1]))
            for year in met['annual_returns']:
                if year['year'] in (2016,2026):year['partial']=True
            results[name][key]=dict(metrics=met,slices=local.arch.old.prior.old.previous.cuts(eq,m))
            write(name+'_'+key+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]);write(name+'_'+key+'_TRADES.json',tr);write(name+'_'+key+'_HOLDINGS.json',hold)
        assert abs(results[name]['same_close']['slices']['training2016_2023']['cagr']-read(P/'TRAIN_TRIALS.json')[NAMES.index(name)]['training']['cagr'])<1e-10
        print(name,results[name]['same_close']['metrics']['cagr'],flush=True)
    cells=read(P/'FEATURE_AUDIT.json')['hand_cells'];periods=[]
    for i,day in enumerate(m['dates']):
        ret={};held={}
        for name in NAMES:
            syms=orders[name][i]['symbols'];held[name]=syms;ret[name]=float((px.loc[m['ends'][i],syms]/px.loc[day,syms]-1).mean()) if syms else 0.
        periods.append(dict(date=day,end=m['ends'][i],split='early53' if i<53 else 'recent35' if i<88 else 'later34',holdings=held,returns=ret,
            guidance_events=[c for c in cells if c['signal_index']==i]))
    write('ATTRIBUTION.json',dict(comparisons={'signed_vs_incumbent':attribution(periods,results)}));write('RESULTS.json',results);write('MONTH_COMPARISON.json',periods)
    write('VALIDATION.json',dict(original_arrays_unchanged=True,no_observer_writes=True,no_intramonth_trades=True,
        primary=frozen['selection']['primary'],primary_target_reached=results[frozen['selection']['primary']]['same_close']['metrics']['cagr']>=.7,
        independent_outcome_review_pending=True,new_model_fits=0))
    lines=['Round130: reported annual guidance sign with fixed local67 and zero-control.',
        'Same-close zero-cost idealized research; incomplete retrospectively frozen corpus and repeatedly used history.',
        'No claim of complete PIT guidance, organic demand, consensus surprise or reliable sparse-event efficacy.',
        'name | CAGR | Sharpe | max drawdown | next-open 5bp/side CAGR']
    for name,r in results.items():
        v=r['same_close']['metrics'];lines.append(f"{name} | {v['cagr']:.2%} | {v['sharpe']:.3f} | {v['max_drawdown']:.2%} | {r['next_open_cost']['metrics']['cagr']:.2%}")
    lines+=['Frozen selection: '+frozen['selection']['primary'],'No sign reversal, weight tuning or source selection after outcomes.']
    with (P/'REPORT.txt').open('x') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('initialize','prepare','train','evaluate'));globals()[ap.parse_args().phase]()
