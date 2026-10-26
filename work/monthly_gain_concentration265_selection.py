"""Unregistered265 source/selection prototype; supplied synthetic inputs only.

The original source, full-history calibration and frozen episode helpers are
pinned264 dependencies. New statistics come solely from the separately pinned
pure265 kernel. Source admission, registration and account replay are not here.
No monkeypatching of frozen modules is used.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import math
import numpy as np
import pandas as pd
import gain_concentration265_sources as source

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_PATH=ROOT/'work/monthly_gain_survival264_kernel.py'
ORIGINAL_SHA='7537da4f0e39ab658333609d71d111086fe478dcaa5b9a358beedc4045a750fa'
MATH_PATH=ROOT/'work/monthly_gain_concentration265_kernel.py'
MATH_SHA='29a541f71b746820f384b63efc459c380c3624bd17b067ed73b5883251ceb253'

def _load_pinned(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Pinned implementation changed: '+path.name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

original=_load_pinned(ORIGINAL_PATH,ORIGINAL_SHA,'gain265_original264')
arithmetic=_load_pinned(MATH_PATH,MATH_SHA,'gain265_pure_math')
old,core_adapter=original.old,original.core_adapter
need,_exact,_finite,_positive=original.need,original._exact,original._finite,original._positive
_n,_meta,_gate,_state=original._n,original._meta,original._gate,original._state
_bound,_old_data,_old_cells,_scan=original._bound,original._old_data,original._old_cells,original._scan
_incumbent,rank=original._incumbent,original.rank
measure=arithmetic.measure
BOUNDS=original.BOUNDS
INITIAL_PARTIAL=source.INITIAL_PARTIAL
SOURCE_SCHEMA=source.SOURCE_SCHEMA
STAT_SCHEMA=arithmetic.SCHEMA
AUDIT_SCHEMA='monthly_gain_concentration265_decisions_v1'
ARMS=('incumbent_local67','daily_gain_concentration_control','block_gain_concentration_candidate')
ROLES=('daily','block')
STAT_KEYS=('schema','price_count','gain_count','block_size','block_count','daily_gains','positive_parts','positive_gain_indices','exact','exact_block_gain_sums','daily_quality','block_quality','original_raw','daily_raw','block_raw','nonpositive_original_preserved','no_positive_gains')
CELL_KEYS=original.CELL_KEYS
OBS_KEYS=original.OBS_KEYS
EVENT_KEYS=tuple(k.replace('endpoint','daily').replace('survival','block') for k in original.EVENT_KEYS)
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','calm_events','active_events','original_core_cells','original_finite_cells','required_cells','price_observations','gain_return_observations','positive_gain_days','gain_statistics','concentration_evaluations','quality_normalization_evaluations','quality_float_conversions','no_positive_gain_cells','nonpositive_original_cells','positive_raw_multiplications','empty_domain_events','available_events','daily_identity_events','block_identity_events','daily_ranked_cells','block_ranked_cells','daily_zero_quality_cells','block_zero_quality_cells','daily_rank_unchanged_events','block_rank_unchanged_events','daily_nonzero_deltas','block_nonzero_deltas','daily_changed_holdings','block_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','core_weight','branch','daily_identity','block_identity','daily_rank_unchanged','block_rank_unchanged','eligible_indices','u0_indices','original_support','daily_deltas','block_deltas','choices','pre_order')
COUNT_SCOPE={'gain_statistics':'Complete measure routine calls, including W=0; not Fraction primitive operations.',
 'concentration_evaluations':'Two exact H calculations per cell with positive W.',
 'quality_normalization_evaluations':'Two exact Q normalizations per cell with positive W.',
 'quality_float_conversions':'Two exact-Q to binary64 conversions per active cell, including W=0.',
 'positive_raw_multiplications':'Two multiplications per positive original R, including legitimate Q=0.',
 'zero_quality_cells':'All active U0 cells whose converted Q is exactly zero, including nonpositive original R; descriptive only.',
 'rank_unchanged_events':'Available events with exact target.float.hex == original_rank.float.hex for every U0 member; descriptive only, never a copy shortcut.',
 'fraction_primitives':'Not claimed or equated across producer/oracle algorithms.'}

def dependency_files():
    return tuple(dict.fromkeys([ORIGINAL_PATH,MATH_PATH,Path(__file__).resolve(),
        ROOT/'work/gain_concentration265_sources.py',*original.dependency_files(),*source.dependency_files()]))


def _setup(panel,meta,calendar,arrays,review,n):
    need(isinstance(panel,dict) and isinstance(panel.get('adj'),pd.DataFrame),'Adjusted source frame');frame=panel['adj']
    need(isinstance(frame.index,pd.DatetimeIndex) and frame.index.tz is None and frame.index.is_unique and frame.index.is_monotonic_increasing and not frame.index.hasnans and all(x==x.normalize() for x in frame.index),'Source date axes')
    need(frame.columns.is_unique and all(type(s) is str and s for s in frame.columns),'Source symbol axes')
    ss=source.calendar_days(calendar);width=np.shape(arrays['baseline'])[1]
    need(isinstance(review,dict) and review.get('schema')==SOURCE_SCHEMA and review.get('passed') is True and review.get('metadata_only') is True and review.get('source_admitted') is False and review.get('strict_pit') is False and review.get('panel_first_session')==source.ORIGIN,'Metadata-only source contract')
    _exact(review['listing_bounds'],BOUNDS,'Original listing bounds');need(isinstance(review['clock_events'],list) and len(review['clock_events'])>=n,'Metadata clocks')
    clocks=[]
    for q in range(n):
        names=meta['symbols'][q];need(isinstance(names,list) and len(names)==width and len(set(names))==width and all(type(s) is str and s for s in names),'Original names');day=old._iso(meta['dates'][q],'query');need(q==0 or day>meta['dates'][q-1],'Chronological query');need(day!=INITIAL_PARTIAL or q==0,'Named partial first')
        c=source.clock(q,day,names,ss);_exact(review['clock_events'][q],c,'Metadata official clock');clocks.append(c)
    got=[]
    for x in review['identity_cells']:
        need(isinstance(x,dict) and type(x.get('query_index')) is int and x['query_index']>=0,'Identity query index')
        if x['query_index']<n:got.append(deepcopy(x))
    _exact(got,source.identities(meta,BOUNDS,n),'Complete consumed whole issuer identities')
    return frame,ss,clocks

def _finish(cells,partial,calm,u0):
    active=[] if partial or calm else u0.copy()
    branch='initial_partial' if partial else 'calm' if calm else 'empty_domain' if not active else 'available'
    identities=[branch=='available' and all(cells[j]['statistics'][key+'_raw'].hex()==cells[j]['original_raw'].hex() for j in active) for key in ('daily','block')]
    return active,*identities,branch

def build_features(panel,meta,calendar,arrays,review,n=None):
    n=_n(arrays,n);calm,weak,cw=_state(arrays,n);frame,sessions,clocks=_setup(panel,meta,calendar,arrays,review,n);data=_old_data(frame,meta,sessions,clocks);out=[]
    for q,c in enumerate(clocks):
        partial=c['initial_partial'];cells=_old_cells(data,c,arrays,q,bool(calm[q]),cw[q]);u0=[j for j,x in enumerate(cells) if x['original_finite']];active=[] if partial or calm[q] else u0.copy()
        # Complete source scan across the full domain before any new gain arithmetic.
        for j in active:cells[j]['observations']=_scan(frame,cells[j]['symbol'],c['required_dates'],cells[j]['history_bound'])
        for j in active:cells[j]['statistics']=measure([x['value'] for x in cells[j]['observations']],cells[j]['original_raw'])
        active,ei,si,branch=_finish(cells,partial,bool(calm[q]),u0);t=c['query_position']
        e=dict(query_index=q,date=c['date'],symbols=deepcopy(c['symbols']),initial_partial=partial,calm=bool(calm[q]),weak=bool(weak[q]),core_weight=float(cw[q,0]),query_position=t,next_official_session=c['next_official_session'],clock_sessions=[] if partial else sessions[t-294:t+2],original_price_origin=source.ORIGIN,u0_indices=u0,active_indices=active)
        for key in ('numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates'):e[key]=deepcopy(c[key])
        e.update(cells=cells,daily_identity=ei,block_identity=si,branch=branch,daily_target=None,block_target=None);need(tuple(e)==EVENT_KEYS,'Event schema');out.append(e)
    return out

def _cached(features,meta,arrays,n):
    need(isinstance(features,list) and len(features)>=n,'Cached feature prefix');calm,weak,cw=_state(arrays,n);out=[];width=np.shape(arrays['baseline'])[1]
    for q,f in enumerate(features[:n]):
        need(isinstance(f,dict) and set(f)==set(EVENT_KEYS),'Cached event schema');day=meta['dates'][q];partial=day==INITIAL_PARTIAL;t=f['query_position'];need(type(t) is int and t>=0,'Cached official position')
        for key,v in dict(query_index=q,date=day,symbols=meta['symbols'][q],initial_partial=partial,calm=bool(calm[q]),weak=bool(weak[q]),core_weight=float(cw[q,0]),original_price_origin=source.ORIGIN).items():_exact(f[key],v,'Cached identity/state '+key)
        if partial:
            need(q==0 and f['clock_sessions']==[] and f['cells']==[],'Partial no new consumption');clock={k:[] for k in ('numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates')}
        else:
            cs=source.calendar_days(f['clock_sessions']);need(len(cs)==296 and cs[294]==day and cs[295]==f['next_official_session'] and cs[295][:7]!=day[:7] and t>=294,'Cached official296 clock')
            ep=[t-294,t-42];cp=list(range(t-63,t+1));fp=list(range(t-294,t-41));ds=lambda ps:[cs[p-(t-294)] for p in ps];clock=dict(numerator_positions=ep,numerator_dates=ds(ep),current_positions=cp,current_dates=ds(cp),formation_positions=fp,formation_dates=ds(fp),required_positions=fp,required_dates=ds(fp))
        for k,v in clock.items():_exact(f[k],v,'Cached clock '+k)
        need(isinstance(f['cells'],list) and len(f['cells'])==(0 if partial else width),'Cached full49 evidence');cells=deepcopy(f['cells']);u0=[]
        for j,c in enumerate(cells):
            need(isinstance(c,dict) and set(c)==set(CELL_KEYS),'Cached cell schema');need(c['column_index']==j and c['symbol']==meta['symbols'][q][j] and type(c['original_finite']) is bool,'Cached cell axes')
            if c['original_finite']:_finite(c['original_raw'],'Cached original raw');u0.append(j)
            else:need(c['original_raw'] is None,'Unavailable original raw')
            sv=c['original_sigma63'];need(sv is None or _finite(sv,'Original sigma')>=0.,'Original sigma evidence');_exact(c['history_bound'],_bound(c['symbol'],day),'Cached bound')
        vals=[c['original_raw'] if c['original_finite'] else np.nan for c in cells];ranks=pd.Series(vals,dtype=float).rank(pct=True).fillna(.5).to_numpy()-.5
        active=[] if partial or calm[q] else u0.copy()
        for j,c in enumerate(cells):
            r=float(ranks[j]);frozen=_finite(np.asarray(arrays['core'])[q,j],'Original core');rebuilt=float(cw[q,j]*r);error=abs(rebuilt-frozen);need(error<=1e-12,'Cached original core parity')
            for k,v in dict(original_rank=r,frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=error).items():_exact(c[k],v,'Cached original '+k)
            obs=c['observations'];need(isinstance(obs,list) and len(obs)==(253 if j in active else 0),'Cached required253 domain')
            for o,d in zip(obs,clock['required_dates']):
                need(isinstance(o,dict) and set(o)==set(OBS_KEYS) and o['date']==d and type(o['value']) is float,'Cached observations');_positive(o['value'],'Cached required price');need(c['history_bound'] is None or d>=c['history_bound']['not_before'],'Cached prelisting conflict')
            if j not in active:need(c['statistics'] is None,'No new statistics outside activeU0')
        # Same all-source-before-arithmetic order as physical preparation.
        for j in active:cells[j]['statistics']=measure([o['value'] for o in cells[j]['observations']],cells[j]['original_raw'])
        active,ei,si,branch=_finish(cells,partial,bool(calm[q]),u0)
        for j in active:_exact(f['cells'][j]['statistics'],cells[j]['statistics'],'Cached canonical statistic')
        rebuilt=deepcopy(f);rebuilt.update(cells=cells,u0_indices=u0,active_indices=active,daily_identity=ei,block_identity=si,branch=branch,daily_target=None,block_target=None)
        for k in ('u0_indices','active_indices','daily_identity','block_identity','branch'):_exact(f[k],rebuilt[k],'Cached domain/branch '+k)
        out.append(rebuilt)
    return out

def _ranked(features):
    out=deepcopy(features)
    for f in out:
        for key in ('daily','block'):
            target=[None]*len(f['symbols'])
            if f['branch']=='available':
                domain=f['active_indices'];cells=f['cells'];values=[cells[j]['statistics'][key+'_raw'] for j in domain]
                for j in domain:target[j]=rank(cells[j]['statistics'][key+'_raw'],values)
            f[key+'_target']=target
    return out

def _select(study,arrays,meta,bridge,features,n,orders0,ig,sg):
    width=bridge['width'];cm=_meta(meta,n);ca={k:np.asarray(v)[:n].copy() for k,v in arrays.items()};ca['core_weight']=bridge['cw'].copy();support=(ca['baseline']-ca['core'])/(1.-bridge['cw'])
    orders={ARMS[0]:orders0};scores={ARMS[0]:bridge['post'].copy()};pres={ARMS[0]:bridge['pre'].copy()};choices={ARMS[0]:deepcopy(bridge['choices'])};deltas={}
    for arm,key in zip(ARMS[1:],('daily_target','block_target')):
        target=np.full((n,width),np.nan)
        for i,f in enumerate(features):
            if f[key.replace('_target','_identity')]:continue
            for j,x in enumerate(f[key]):
                if x is not None:target[i,j]=x
        delta=np.zeros((n,width));mask=np.isfinite(target);delta[mask]=bridge['cw'][mask]*target[mask]-bridge['core'][mask];expected=bridge['pre'].copy();nz=delta!=0.;expected[nz]+=delta[nz];calls=[]
        def boundary(changed,good,names,effects,original,policy):
            need(not calls and changed.tobytes()==expected.tobytes(),'One fullcore original episode exact pre');need(np.array_equal(good,bridge['eligible']) and np.array_equal(effects,ca['effects']) and names==cm['symbols'] and policy==old.POLICY,'Original E/effects/policy')
            for k in ('baseline','core','core_weight'):need(np.array_equal(original[k],ca[k]),'Original support '+k)
            calls.append(True);pres[arm]=changed.copy();return study.ep.select_boundary(changed,good,names,effects,original,policy)
        chosen,score=core_adapter.score_and_choose('incumbent66',1.,target,ca,cm,n,pre_score=bridge['pre'].copy(),effects=ca['effects'],episode_policy=old.POLICY.copy(),boundary_selector=boundary)
        need(len(calls)==1,'Exactly one episode per arm')
        copied=[]
        for i,f in enumerate(features):
            if f['branch']!='available' or f[key.replace('_target','_identity')]:
                pres[arm][i]=bridge['pre'][i];score[i]=bridge['post'][i];chosen[i]=deepcopy(bridge['choices'][i]);copied.append(i)
        scores[arm]=score.copy();choices[arm]=chosen;orders[arm]=study.rp.make_orders(chosen,score,cm)
        for i in copied:orders[arm][i]=deepcopy(orders0[i])
        deltas[arm]=delta
    counts=dict.fromkeys(COUNT_FIELDS,0)
    counts.update(events=n,symbols_per_event=width,original_episode_calls=3)
    events=[]
    for i,f in enumerate(features):
        eligible=np.flatnonzero(bridge['eligible'][i]).tolist()
        cells=f['cells'];active=f['active_indices'];branch=f['branch'];partial=f['initial_partial']
        counts['eligible_cells']+=len(eligible)
        counts['initial_partial_events']+=int(partial)
        counts['calm_events']+=int(not partial and f['calm'])
        counts['active_events']+=int(not partial and not f['calm'])
        counts['original_core_cells']+=len(cells)
        counts['original_finite_cells']+=len(f['u0_indices'])
        counts['required_cells']+=len(active)
        counts['price_observations']+=253*len(active)
        counts['gain_return_observations']+=252*len(active)
        counts['gain_statistics']+=len(active)
        counts['quality_float_conversions']+=2*len(active)
        for j in active:
            stat=cells[j]['statistics'];positive=not stat['no_positive_gains']
            need(tuple(stat)==STAT_KEYS,'Complete265 statistic schema')
            counts['positive_gain_days']+=len(stat['positive_gain_indices'])
            counts['concentration_evaluations']+=2*int(positive)
            counts['quality_normalization_evaluations']+=2*int(positive)
            counts['no_positive_gain_cells']+=int(not positive)
            counts['nonpositive_original_cells']+=int(stat['nonpositive_original_preserved'])
            counts['positive_raw_multiplications']+=2*int(not stat['nonpositive_original_preserved'])
            for role in ROLES:counts[role+'_zero_quality_cells']+=int(stat[role+'_quality']==0.)
        counts['empty_domain_events']+=int(branch=='empty_domain')
        counts['available_events']+=int(branch=='available')
        unchanged={role:branch=='available' and all(f[role+'_target'][j].hex()==cells[j]['original_rank'].hex() for j in active) for role in ROLES}
        for arm,role in zip(ARMS[1:],ROLES):
            counts[role+'_identity_events']+=int(f[role+'_identity'])
            counts[role+'_ranked_cells']+=len(active) if branch=='available' else 0
            counts[role+'_rank_unchanged_events']+=int(unchanged[role])
            counts[role+'_nonzero_deltas']+=int(np.count_nonzero(deltas[arm][i]))
            counts[role+'_changed_holdings']+=int(set(choices[arm][i])!=set(choices[ARMS[0]][i]))
        counts['candidate_control_changed_holdings']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[2]][i]))
        event=dict(query_index=i,date=f['date'],initial_partial=partial,calm=f['calm'],weak=f['weak'],core_weight=f['core_weight'],branch=branch,daily_identity=f['daily_identity'],block_identity=f['block_identity'],daily_rank_unchanged=unchanged['daily'],block_rank_unchanged=unchanged['block'],eligible_indices=eligible,u0_indices=f['u0_indices'].copy(),original_support=support[i].tolist(),daily_deltas=deltas[ARMS[1]][i].tolist(),block_deltas=deltas[ARMS[2]][i].tolist(),choices={arm:[int(j) for j in choices[arm][i]] for arm in ARMS},pre_order=sorted(eligible,key=lambda j:(-bridge['pre'][i,j],cm['symbols'][i][j])))
        need(tuple(event)==DECISION_EVENT_KEYS,'Decision schema');events.append(event)
    return orders,scores,pres,dict(schema=AUDIT_SCHEMA,counts=counts,events=events,incumbent_gate=ig,source_gate=sg)


def prepare(study,arrays,meta,panel,calendar,review,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n);bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    features=build_features(panel,meta,calendar,arrays,review,n);sg=_gate(source_validator,(deepcopy(features),),'Source');need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals');features=_ranked(features);out=_select(study,arrays,meta,bridge,features,n,orders,ig,sg);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated');return (features,)+out

def decisions(study,arrays,meta,features,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n);bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    consumed=_cached(features,meta,arrays,n);sg=_gate(source_validator,(deepcopy(consumed),),'Source');need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals');ranked=_ranked(consumed)
    for actual,wanted in zip(features[:n],ranked):_exact(actual,wanted,'Cached full features')
    out=_select(study,arrays,meta,bridge,ranked,n,orders,ig,sg);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated');return out

def unranked_prefix(features,n):
    need(type(n) is int and isinstance(features,list) and 0<=n<=len(features),'Feature prefix');out=deepcopy(features[:n])
    for f in out:f['daily_target']=None;f['block_target']=None
    return out
