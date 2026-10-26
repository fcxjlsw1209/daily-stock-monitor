"""Prospective downside-only fixed-unit basket construction, supplied data only.

The one frozen original episode and incumbent gate precede source/statistics;
independent source validation precedes direct basket selection. No I/O loaders,
models, registrations, performance replays or writes are provided here.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LEGACY_PATH=ROOT/'work/monthly_portfolio_path247_kernel.py'
LEGACY_SHA='18c7726ae701bb3557636bfa69a0c0dc32d516286f3bb1cd27f50157d5380092'
if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()!=LEGACY_SHA:
    raise ValueError('Changed frozen247 dependency')
_spec=importlib.util.spec_from_file_location('downside258_frozen247',LEGACY_PATH)
old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(old)
f=old.stats
ARMS=('incumbent_local67','basket_downside_moment','basket_symmetric_moment')
STAT_SCHEMA='monthly_downside_basket258_moments_v1'
AUDIT_SCHEMA='monthly_downside_basket258_decisions_v1'
PRICE_COUNT=253
RETURN_COUNT=252
INITIAL_PARTIAL=old.INITIAL_PARTIAL
CAP=old.CAP
POLICY=old.POLICY.copy()
EVENT_KEYS=old.EVENT_KEYS
TRIO_KEYS=old.TRIO_KEYS
SOURCE_KEYS=old.SOURCE_KEYS
LISTING_KEYS=old.LISTING_KEYS
STAT_KEYS=('schema','price_count','return_count','normalized_prices','wealth','prior_totals','contributions',
 'portfolio_returns','direct_returns','downside_square_sum','symmetric_square_sum','upside_square_sum',
 'downside_moment','symmetric_moment','upside_moment','negative_returns','zero_returns','positive_returns',
 'zero_downside','zero_symmetric')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','active_events','shortlist_cells',
 'source_available_cells','source_price_observations','enumerated_trios','feasible_trios','trio_statistics',
 'formation_return_observations','zero_downside_trios','zero_symmetric_trios','missing_source_events','available_events',
 'downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings',
 'original_episode_calls','new_model_fits','predictive_model_fits')
BRANCHES=('initial_partial','fewer_than_three','missing_source','available')
DECISION_EVENT_KEYS=EVENT_KEYS+('choices','winner_trio_indices','tied_trio_indices')
require=old.require


def dependency_files():
    return (LEGACY_PATH,)+old.dependency_files()


def _exact(actual,expected,label):
    """Strict serialized evidence equality, including signed zero and scalar type."""
    require(type(actual) is type(expected),label+' type')
    if isinstance(expected,dict):
        require(set(actual)==set(expected),label+' keys')
        for key in expected:_exact(actual[key],expected[key],label+'.'+key)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),label+' length')
        for a,b in zip(actual,expected):_exact(a,b,label)
    elif isinstance(expected,float):
        require(struct.pack('>d',actual)==struct.pack('>d',expected),label+' float bits')
    else:require(actual==expected,label+' value')


def measure(prices):
    """Canonical binary64 3x253 fixed-unit path and three uncentered moments."""
    require(isinstance(prices,(list,tuple,np.ndarray)) and len(prices)==3,'Three price paths')
    rows=[]
    for row in prices:
        require(isinstance(row,(list,tuple,np.ndarray)) and len(row)==PRICE_COUNT,'Exactly253 prices')
        values=[old._real(v,'required price') for v in row]
        require(all(v>0. for v in values),'Positive required prices');rows.append(values)
    normalized=[[f._divide(v,row[0],'normalized price') for v in row] for row in rows]
    require(all(v>0. for row in normalized for v in row),'Positive normalized prices')
    totals=[f._sum((normalized[i][k] for i in range(3)),'normalized total') for k in range(PRICE_COUNT)]
    wealth=[f._divide(v,3.,'fixed-unit wealth') for v in totals]
    contributions=[[],[],[]];returns=[];direct=[]
    for k in range(1,PRICE_COUNT):
        for i in range(3):
            change=f._difference(normalized[i][k],normalized[i][k-1],'normalized increment')
            contributions[i].append(f._divide(change,totals[k-1],'fixed-unit contribution'))
        r=f._sum((contributions[i][-1] for i in range(3)),'fixed-unit return')
        d=f._difference(f._divide(wealth[k],wealth[k-1],'direct NAV ratio'),1.,'direct NAV return')
        f._close(r,d,2e-12,2e-12,'fixed-unit NAV side check');returns.append(r);direct.append(d)
    minus=f._sum((f._square(min(r,0.),'downside square') for r in returns),'downside square sum')
    symmetric=f._sum((f._square(r,'symmetric square') for r in returns),'symmetric square sum')
    plus=f._sum((f._square(max(r,0.),'upside square') for r in returns),'upside square sum')
    lm=f._divide(minus,252.,'downside moment');ls=f._divide(symmetric,252.,'symmetric moment');lp=f._divide(plus,252.,'upside moment')
    f._close(ls,f._sum((lm,lp),'downside plus upside'),2e-10,2e-10,'moment decomposition')
    neg=sum(r<0. for r in returns);zero=sum(r==0. for r in returns);pos=sum(r>0. for r in returns)
    require((lm==0.)==(neg==0),'Canonical downside zero identity')
    require((ls==0.)==(zero==RETURN_COUNT),'Canonical symmetric zero identity')
    out=dict(schema=STAT_SCHEMA,price_count=PRICE_COUNT,return_count=RETURN_COUNT,normalized_prices=normalized,
        wealth=wealth,prior_totals=totals[:-1],contributions=contributions,portfolio_returns=returns,direct_returns=direct,
        downside_square_sum=minus,symmetric_square_sum=symmetric,upside_square_sum=plus,
        downside_moment=lm,symmetric_moment=ls,upside_moment=lp,
        negative_returns=neg,zero_returns=zero,positive_returns=pos,zero_downside=lm==0.,zero_symmetric=ls==0.)
    require(tuple(out)==STAT_KEYS,'Statistics schema');return out


def validate_statistics(record):
    require(isinstance(record,dict) and set(record)==set(STAT_KEYS),'Cached moment schema')
    paths=record['normalized_prices']
    require(isinstance(paths,list) and len(paths)==3 and all(isinstance(row,list) and len(row)==253 for row in paths),'Cached normalized paths')
    require(all(type(row[0]) is float and row[0]==1. for row in paths),'Canonical initial normalization')
    expected=measure(paths)
    _exact(record,expected,'Cached canonical moments')
    return True


def _meta_prefix(meta,n):
    require(isinstance(meta,dict),'Metadata mapping')
    out={}
    for key in ('dates','ends','symbols'):
        require(isinstance(meta.get(key),list) and len(meta[key])>=n,'Metadata prefix '+key)
        out[key]=deepcopy(meta[key][:n])
    if 'etfs' in meta:out['etfs']=deepcopy(meta['etfs'])
    return out


def _gate(callback,args,label):
    if callback is None:return dict(passed=True,synthetic_no_callback=True)
    receipt=callback(*args)
    require(isinstance(receipt,dict) and receipt.get('passed') is True,label+' must pass')
    return deepcopy(receipt)


def _incumbent(study,meta,bridge,n,validator,expected):
    orders=study.rp.make_orders(bridge['choices'],bridge['post'],_meta_prefix(meta,n))
    if expected is not None:
        require(isinstance(expected,list) and len(expected)>=n,'Expected incumbent prefix')
        _exact(orders,expected[:n],'Frozen incumbent orders')
    return _gate(validator,(deepcopy(orders),bridge['post'].copy(),bridge['pre'].copy()),'Incumbent validator')


def _cached_event(feature,expected):
    require(isinstance(feature,dict) and set(feature)==set(EVENT_KEYS),'Cached feature schema')
    variable={'session_position','price_start','price_end','source_dates','clock_sessions','source_cells','trios','branch'}
    for key in EVENT_KEYS:
        if key not in variable:_exact(feature[key],expected[key],'Cached original binding '+key)
    require(type(feature['session_position']) is int and feature['session_position']>=294,'Cached session position')
    clock=feature['clock_sessions'];ds=feature['source_dates']
    require(isinstance(clock,list) and len(clock)==296 and all(type(v) is str for v in clock),'Cached complete clock')
    for d in clock:old._iso(d,'cached clock')
    require(clock==sorted(set(clock)) and all(date.fromisoformat(d).weekday()<5 for d in clock),'Cached increasing weekday clock')
    require(ds==clock[:253] and feature['price_start']==ds[0] and feature['price_end']==ds[-1] and clock[294]==feature['date'],'Cached exact253/query clock')
    require(feature['initial_partial'] or clock[-1][:7]!=feature['date'][:7],'Cached month-end witness')
    require(isinstance(feature['trios'],list) and len(feature['trios'])==len(expected['trios']),'Cached trio inventory')
    for actual,wanted in zip(feature['trios'],expected['trios']):
        require(isinstance(actual,dict) and set(actual)==set(TRIO_KEYS),'Cached trio schema')
        for key in TRIO_KEYS:
            if key!='statistics':_exact(actual[key],wanted[key],'Cached domain '+key)
    active=bool(expected['shortlist_indices']);cells=feature['source_cells']
    require(isinstance(cells,list) and len(cells)==len(expected['shortlist_indices']),'Cached shortlist sources')
    for cell,j in zip(cells,expected['shortlist_indices']):
        require(isinstance(cell,dict) and set(cell)==set(SOURCE_KEYS),'Cached source schema')
        _exact(cell['column_index'],j,'Cached source column');_exact(cell['symbol'],feature['symbols'][j],'Cached source symbol')
        require(type(cell['source_available']) is bool and cell['reason'] in ('available','missing_symbol','missing_official_session','missing_nan'),'Cached source reason')
        require(cell['source_available']==(cell['reason']=='available'),'Cached source gate')
        require(type(cell['observations_checked']) is int and 0<=cell['observations_checked']<=253,'Cached observations')
        if cell['source_available']:
            require(cell['observations_checked']==253 and old._real(cell['start_price'],'cached first')>0 and old._real(cell['end_price'],'cached last')>0,'Cached complete source')
        else:require(cell['start_price'] is None and cell['end_price'] is None,'Unavailable endpoints absent')
        listing=cell['listing']
        if listing is not None:
            require(isinstance(listing,dict) and set(listing)==set(LISTING_KEYS),'Cached listing schema')
            old._iso(listing['not_before'],'listing bound');old._iso(listing['available_from'],'listing clock')
            require(type(listing['issuer_key']) is str and listing['issuer_key'] and type(listing['evidence_id']) is str and listing['evidence_id'],'Cached listing evidence')
            _exact(listing['visible'],listing['available_from']<=feature['date'],'Listing visible')
            _exact(listing['start_precedes_bound'],feature['price_start']<listing['not_before'],'Listing start')
            require(not(cell['source_available'] and listing['visible'] and listing['start_precedes_bound']),'Cached listing contradiction')
    complete=active and all(c['source_available'] for c in cells)
    shared={};byid={c['column_index']:c for c in cells}
    for trio in feature['trios']:
        if complete and trio['feasible']:
            validate_statistics(trio['statistics'])
            for j,path in zip(trio['indices'],trio['statistics']['normalized_prices']):
                cell=byid[j];_exact(path[-1],f._divide(cell['end_price'],cell['start_price'],'cached normalized endpoint'),'Endpoint normalization')
                if j in shared:_exact(path,shared[j],'Shared security path')
                else:shared[j]=path
        else:require(trio['statistics'] is None,'No statistics outside common feasible source')
    branch=expected['branch'] if not active else ('available' if complete else 'missing_source')
    require(feature['branch']==branch,'Cached common branch')
    return deepcopy(feature)


def choose_trio(event,key):
    """Exact objective order; diagnostics never alter ties."""
    require(key in ('downside_moment','symmetric_moment'),'Known objective')
    hs=event['feasible_trio_indices'];require(bool(hs),'Feasible trio retained')
    values={k:old._real(event['trios'][k]['statistics'][key],'selection objective') for k in hs}
    require(all(v>=0. for v in values.values()),'Nonnegative objective')
    minimum=min(values.values());tied=[k for k in hs if values[k]==minimum]
    winner=min(tied,key=lambda k:(event['trios'][k]['indices']!=event['incumbent_trio'],tuple(event['trios'][k]['symbols'])))
    return winner,tied


def _select(study,meta,bridge,features,n,inc_gate,source_gate):
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=bridge['width'],original_episode_calls=1)
    scores={arm:bridge['post'].copy() for arm in ARMS};pres={arm:bridge['pre'].copy() for arm in ARMS}
    choices={arm:[] for arm in ARMS};history=[]
    for i,event0 in enumerate(features):
        event=deepcopy(event0);inc=event['incumbent_choices'];picked={arm:inc.copy() for arm in ARMS}
        winners={arm:None for arm in ARMS[1:]};ties={arm:[] for arm in ARMS[1:]}
        if event['branch']=='available':
            for arm,key in zip(ARMS[1:],('downside_moment','symmetric_moment')):
                best,tied=choose_trio(event,key);winners[arm]=best;ties[arm]=tied
                selected=event['trios'][best]['indices']
                picked[arm]=inc.copy() if set(selected)==set(inc) else sorted(selected,key=lambda j:(-bridge['post'][i,j],meta['symbols'][i][j]))
        event.update(choices=picked,winner_trio_indices=winners,tied_trio_indices=ties)
        for arm in ARMS:choices[arm].append(picked[arm])
        counts['eligible_cells']+=len(event['eligible_indices']);counts['initial_partial_events']+=int(event['initial_partial'])
        counts['active_events']+=int(bool(event['shortlist_indices']));counts['shortlist_cells']+=len(event['source_cells'])
        counts['source_available_cells']+=sum(c['source_available'] for c in event['source_cells'])
        counts['source_price_observations']+=sum(c['observations_checked'] for c in event['source_cells'])
        counts['enumerated_trios']+=len(event['trios']);counts['feasible_trios']+=len(event['feasible_trio_indices'])
        measured=[t['statistics'] for t in event['trios'] if t['statistics'] is not None]
        counts['trio_statistics']+=len(measured);counts['formation_return_observations']+=252*len(measured)
        counts['zero_downside_trios']+=sum(z['zero_downside'] for z in measured)
        counts['zero_symmetric_trios']+=sum(z['zero_symmetric'] for z in measured)
        for branch in ('missing_source','available'):counts[branch+'_events']+=int(event['branch']==branch)
        for arm,key in zip(ARMS[1:],('downside_changed_holdings','symmetric_changed_holdings')):counts[key]+=int(set(picked[arm])!=set(inc))
        counts['candidate_control_changed_holdings']+=int(set(picked[ARMS[1]])!=set(picked[ARMS[2]]))
        require(tuple(event)==DECISION_EVENT_KEYS,'Decision event schema');history.append(event)
    orders={arm:study.rp.make_orders(choices[arm],scores[arm],_meta_prefix(meta,n)) for arm in ARMS}
    return orders,scores,pres,dict(schema=AUDIT_SCHEMA,counts=counts,events=history,incumbent_gate=inc_gate,source_gate=source_gate)


def prepare(study,arrays,meta,panel,calendar,review,n,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    before=old._snapshot(arrays);before_meta=_meta_prefix(meta,n)
    bridge=old._bridge(study,arrays,meta,n)
    inc_gate=_incumbent(study,meta,bridge,n,incumbent_validator,expected_incumbent_orders)
    adj,sessions,queries,bounds=old._source_setup(panel,meta,calendar,review,n);features=[]
    for i,(day,names,t) in enumerate(queries):
        event=old._domain(meta,bridge,i);dates=sessions[t-294:t-41]
        event.update(session_position=t,price_start=str(dates[0].date()),price_end=str(dates[-1].date()),
            source_dates=dates.strftime('%Y-%m-%d').tolist(),clock_sessions=sessions[t-294:t+2].strftime('%Y-%m-%d').tolist())
        if event['shortlist_indices']:
            rows={}
            for j in event['shortlist_indices']:
                cell,prices=old._extract(adj,dates,j,names[j],bounds.get(names[j]),str(day.date()))
                event['source_cells'].append(cell);rows[j]=prices
            if not all(c['source_available'] for c in event['source_cells']):event['branch']='missing_source'
            else:
                for k in event['feasible_trio_indices']:
                    trio=event['trios'][k];trio['statistics']=measure([rows[j] for j in trio['indices']])
                event['branch']='available'
        require(tuple(event)==EVENT_KEYS,'Feature schema');features.append(event)
    source_gate=_gate(source_validator,(deepcopy(features),),'Source validator')
    out=_select(study,meta,bridge,features,n,inc_gate,source_gate)
    require(old._snapshot(arrays)==before and _meta_prefix(meta,n)==before_meta,'Original inputs mutated')
    return (features,)+out


def decisions(study,arrays,meta,features,n,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    before=old._snapshot(arrays);before_meta=_meta_prefix(meta,n)
    bridge=old._bridge(study,arrays,meta,n)
    inc_gate=_incumbent(study,meta,bridge,n,incumbent_validator,expected_incumbent_orders)
    require(isinstance(features,list) and len(features)>=n,'Cached feature prefix')
    consumed=[_cached_event(features[i],old._domain(meta,bridge,i)) for i in range(n)]
    source_gate=_gate(source_validator,(deepcopy(consumed),),'Source validator')
    out=_select(study,meta,bridge,consumed,n,inc_gate,source_gate)
    require(old._snapshot(arrays)==before and _meta_prefix(meta,n)==before_meta,'Original inputs mutated')
    return out
