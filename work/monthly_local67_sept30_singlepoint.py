"""Single September30 local67 signal, preserving all historical artifacts.

Only the exact inactive-(calm,weak) branch is implemented for a new query.
Both gates are independently computed; an active gate fails closed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import sys
import traceback
import numpy as np
import pandas as pd

ASOF = pd.Timestamp('2026-09-30')
PARITY_DAY = pd.Timestamp('2026-08-31')
PLAN_SHA = '6fa5d01b0911ac26d33a0834662e75ec674d0d9f1fc707bc5e7a8fceab3458fa'
PROTECTED_SHA = '89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'
FIELDS = {'adj':'Adj Close','open':'Open','close':'Close','high':'High','low':'Low','volume':'Volume'}
SOURCES = {}

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def blob(path, expected=None):
    path=Path(path).resolve();data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    require(expected is None or digest==expected, 'Input SHA mismatch: '+str(path))
    require(str(path) not in SOURCES or SOURCES[str(path)]==digest, 'Input changed: '+str(path))
    SOURCES[str(path)]=digest
    return data

def read(path, expected=None): return json.loads(blob(path,expected))
def npz(path, expected=None):
    with np.load(io.BytesIO(blob(path,expected)),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

def save(path, obj):
    with Path(path).open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def internal(s):return {'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)
def display(s):return 'META' if s=='research_equity_FB' else 'BRK-B' if s=='BRK.B' else s

def baseline(panel, day, ss, etfs, strategy, state_source):
    """Verbatim row algebra of frozen strategy.selection, exposing all49 scores."""
    vals,eligible,state,corr,tail,mom,efficiency,calm=strategy.factors(panel)
    tab=pd.DataFrame({k:v.loc[day,ss] for k,v in vals.items()})
    q=tab.rank(pct=True).fillna(.5)-.5;q['regime_cmf']*=state.at[day]
    rel={126:{},252:{}};stable={};sectors=panel['sectors']
    for s in ss:
        b=None if s in etfs else strategy.FOREIGN.get(s,strategy.SECTOR.get(sectors.at[day,s]) if s in sectors else None)
        for n in rel:
            if b is None:rel[n][s]=np.nan;continue
            a,z=mom[n].at[day,s],mom[n].at[day,b]
            rel[n][s]=(1+a)/(1+z)-1 if np.isfinite(a) and np.isfinite(z) and z>-1 else np.nan
        stable[s]=-abs(corr[b][0].at[day,s]-corr[b][1].at[day,s]) if b is not None else np.nan
    extra=pd.DataFrame({'sector126':pd.Series(rel[126]),'sector252':pd.Series(rel[252]),
        'stability':pd.Series(stable),'tail':tail.loc[day,ss],
        'raw_momentum':mom[252].loc[day,ss],'path':efficiency.loc[day,ss]})
    ranks=extra.replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5)-.5
    if calm.at[day]:
        parts=dict(core=.5*ranks.raw_momentum,volatility=.2*q.volatility,
            sector=.0875*(ranks.sector126+ranks.sector252),path=.05*ranks.path,cmf=.075*q.regime_cmf)
    else:
        support=sum(q[k]*w for k,w in strategy.WEIGHTS.items() if k!='momentum')
        budget=.10 if state.at[day]>0 else 0.
        parts=dict(core=.52*q.momentum,support=(1-budget/.48)*support,
            industry=budget*(.25*ranks.sector126+.5*ranks.stability+.25*ranks['tail']))
    states=state_source.state_row(panel['adj'],day,ss,set(etfs))
    result=dict(baseline=sum(parts.values()).reindex(ss).to_numpy()[None,:],
                core=parts['core'].reindex(ss).to_numpy()[None,:],
                core_weight=np.array([[.5 if calm.at[day] else .52]]),
                eligible=eligible.loc[day,ss].to_numpy(bool)[None,:],
                state_calm10=np.array([bool(calm.at[day])]),
                state_weak_narrow=np.array([bool(states['weak_narrow'])]))
    for k in ('baseline','core','core_weight'):require(np.isfinite(result[k]).all(),'Nonfinite '+k)
    return result,states

def episode_effects(panel, day, ss, eligible, ep):
    """Rebuild query and only mature5-session same-security event outcomes."""
    clean={k:panel[k] for k in FIELDS}
    shape,market,signal,good,vol=ep.features(clean)
    dates=panel['adj'].index.to_numpy(dtype='datetime64[D]')
    target=np.datetime64(day.date(),'D');limit=target-np.timedelta64(1827,'D')
    normalized=panel['adj']*panel['open']/panel['close']
    spy=panel['adj'].SPY.to_numpy();spy_open=normalized.SPY.to_numpy()
    config=ep.models()[2]
    require(config==dict(id=2,horizon=5,match='signal',target='utility',mode='level'),'Changed model2')
    effects=np.zeros((1,len(ss)));details=[]
    query=dict(shape=np.column_stack([v.loc[day,ss].to_numpy() for v in shape]),
               market=market.loc[day].to_numpy(),signal=signal.loc[day,ss].to_numpy(int),
               vol=vol.loc[day,ss].to_numpy())
    for j,s in enumerate(ss):
        q=dict(symbol=s,date=target,eligible=bool(eligible[0,j]),shape=query['shape'][j],
               market=query['market'],signal=int(query['signal'][j]),vol=float(query['vol'][j]))
        if not(np.isfinite(q['shape']).all() and np.isfinite(q['market']).all() and np.isfinite(q['vol'])):
            q['eligible']=False
        ev={k:[] for k in ('symbol','dates','signal','shape','market','ends_5','utility_5')}
        starts=ep.anchors(signal[s].to_numpy(),good[s].to_numpy())
        for idx in starts:
            if dates[idx]<limit or dates[idx]>=target or idx+5>=len(dates) or dates[idx+5]>target:continue
            _,utility=ep.label_path(panel['adj'][s].to_numpy(),normalized[s].to_numpy(),spy,spy_open,int(idx),5)
            ev['symbol'].append(s);ev['dates'].append(dates[idx]);ev['signal'].append(int(signal[s].iloc[idx]))
            ev['shape'].append([v[s].iloc[idx] for v in shape]);ev['market'].append(market.iloc[idx].to_numpy())
            ev['ends_5'].append(dates[idx+5]);ev['utility_5'].append(utility)
        events={k:np.asarray(v,dtype='datetime64[D]' if k in ('dates','ends_5') else None) for k,v in ev.items()}
        effects[0,j],info=ep.estimate(events,q,config)
        details.append(dict(symbol=s,signal_type=q['signal'],eligible=q['eligible'],effect=float(effects[0,j]),**info))
    return effects,query,details

def compare_float(actual, expected, label, atol=1e-12):
    require(np.shape(actual)==np.shape(expected),label+' shape mismatch')
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=0,equal_nan=True,err_msg=label)
    ok=np.isfinite(actual)&np.isfinite(expected)
    return float(np.max(np.abs(np.asarray(actual)[ok]-np.asarray(expected)[ok]))) if np.any(ok) else 0.

def run_parity(panel, meta, cached, local, strategy, query_cache, policy, original_orders):
    i=meta['dates'].index(str(PARITY_DAY.date()));ss=meta['symbols'][i]
    rebuilt,states=baseline(panel,PARITY_DAY,ss,meta['etfs'],strategy,local.arch.old)
    errors={}
    for k,v in rebuilt.items():
        original=cached[k][i:i+1]
        if v.dtype.kind=='b':require(np.array_equal(v,original),'Parity '+k)
        else:errors[k]=compare_float(v,original,k)
    effects,query,details=episode_effects(panel,PARITY_DAY,ss,rebuilt['eligible'],local.ep)
    errors['effects']=compare_float(effects,cached['effects'][i:i+1],'frozen episode effects')
    for k,v in query.items():
        expected=query_cache[k][i]
        if k=='signal':require(np.array_equal(v,expected),'Frozen query signal mismatch')
        else:errors['query_'+k]=compare_float(v,expected,'query '+k)
    a={k:v[i:i+1].copy() for k,v in cached.items()}
    require(len(a)==27,'Original27 arrays required')
    original_choices,original_score=local.choices(local.configs()[4],a,{'symbols':[ss]},1,policy=policy)
    a.update(rebuilt,effects=effects)
    ch,score=local.choices(local.configs()[4],a,{'symbols':[ss]},1,policy=policy)
    require(ch==original_choices,'Rebuilt parity ordered choice mismatch')
    errors['all_scores']=compare_float(score,original_score,'full49 scores')
    chosen=[ss[j] for j in ch[0]]
    expected=original_orders[i]
    require(chosen==expected['symbols'],'Saved8/31 ordered symbols mismatch')
    errors['saved_selected_scores']=compare_float(score[0,ch[0]],np.asarray(expected['scores']),'Saved8/31 selected scores')
    inactive=not rebuilt['state_calm10'][0] and not rebuilt['state_weak_narrow'][0]
    if inactive:
        shortcut_ch,shortcut_score=local.adapter.score_and_choose('incumbent66',0.,np.full((1,len(ss)),np.nan),
            rebuilt,{'symbols':[ss]},1,pre_score=rebuilt['baseline'],effects=effects,
            episode_policy=policy,boundary_selector=local.ep.select_boundary)
        require(shortcut_ch==ch,'Inactive branch parity choices')
        errors['inactive_branch_score']=compare_float(shortcut_score,score,'Inactive branch full49 score')
    return dict(passed=True,signal_date=str(PARITY_DAY.date()),ordered_symbols=chosen,
                inactive_branch_verified=bool(inactive),gates=dict(calm10=bool(rebuilt['state_calm10'][0]),
                weak_narrow=bool(rebuilt['state_weak_narrow'][0])),max_absolute_differences=errors,
                full49_original_features_and_episode_checked=True,full_backtest_run=False)

def merge_overlays(panel, root, ss, day):
    """Extend each original history without replacing its frozen historical rows."""
    panel={k:v.copy() for k,v in panel.items()};audit=[];missing=[]
    names=['SMH','SPCX','TSM','SKHY','SOXX','ARKK','MTUM','IGV','VGT']
    for name in names:
        if name not in ss:continue
        if all(name in panel[k] and day in panel[k].index and np.isfinite(panel[k].at[day,name]) for k in FIELDS):
            audit.append(dict(symbol=name,already_complete_in_original64_sources=True,added_rows=0))
            continue
        csv=root/'prices'/(name+'.csv');meta=root/'prices'/(name+'.source.json')
        if not csv.exists() or not meta.exists():missing += [str(p) for p in (csv,meta) if not p.exists()];continue
        md=read(meta);data=blob(csv,md['sha256']);symbol=internal(md['symbol'])
        require(symbol==name,'Overlay symbol mismatch')
        frame=pd.read_csv(io.BytesIO(data),index_col=0,parse_dates=True)
        require(frame.index.is_unique and frame.index.is_monotonic_increasing,'Overlay date axis')
        frame=frame.reindex(panel['adj'].index)
        if symbol not in panel['adj']:
            for key in FIELDS:panel[key][symbol]=np.nan
        old=panel['adj'][symbol];new=frame['Adj Close']
        overlap=(old.notna()&new.notna()&(old>0)&(new>0)&(old.index<=day))
        scale=1.;max_ulp=0.
        if overlap.any():
            # Yahoo's saved adjusted series differ by up to four float32 ULPs,
            # while every raw OHLCV observation is identical. No old value is
            # replaced and no adjustment factor is estimated or applied.
            original=old[overlap].to_numpy();incoming=new[overlap].to_numpy()
            ulp=np.abs(np.spacing(original.astype(np.float32))).astype(float)
            error=np.abs(incoming-original)
            require(np.all(error <= 4*ulp+1e-10),symbol+' adjusted history conflicts beyond float32 precision')
            max_ulp=float(np.max(error/ulp))
            for key,col in FIELDS.items():
                if key=='adj':continue
                oldfield=panel[key][symbol];newfield=frame[col]
                common=oldfield.notna()&newfield.notna()&(oldfield.index<=day)
                require(np.array_equal(oldfield[common].to_numpy(),newfield[common].to_numpy()),symbol+' raw '+key+' history differs')
        last=old.last_valid_index()
        use=new.notna() & ((new.index>last) if last is not None else True)
        # Existing rows are retained; only new tail rows enter the query panel.
        for key,col in FIELDS.items():panel[key].loc[use,symbol]=frame.loc[use,col]*(scale if key=='adj' else 1.)
        audit.append(dict(symbol=symbol,old_last_date=str(last.date()) if last is not None else None,
            added_rows=int(use.sum()),adjusted_scale=scale,overlap_max_float32_ulps=max_ulp,
            historical_rows_preserved=True,source_sha256=md['sha256'],
            overlay_last_date=str(frame['Adj Close'].last_valid_index().date())))
    require(not missing,'Missing September30 cached files: '+json.dumps(missing))
    for key in FIELDS:
        absent=[s for s in ss if s not in panel[key] or not np.isfinite(panel[key].at[day,s])]
        require(not absent,'Missing '+key+' at September30: '+json.dumps(absent))
    return panel,audit

def load_pool(input_root, etfs):
    pool=read(input_root/'POOL.json')
    require(pool['date']==str(ASOF.date()),'Wrong pool date')
    stocks=[internal(z['symbol']) for z in pool['top32']]
    require(len(stocks)==len(set(stocks))==32,'Explicit uniqueTop32 required')
    ss=sorted(set(stocks)|set(etfs));require(len(ss)==49,'Expected49 candidates')
    return pool,ss

def input_coverage(panel, ss, input_root):
    rows=[]
    for s in ss:
        rec={'symbol':s,'fields':{}}
        for field in FIELDS:
            vals=panel[field][s] if s in panel[field] else pd.Series(dtype=float)
            last=vals.last_valid_index()
            rec['fields'][field]={'asof_present':bool(ASOF in vals.index and np.isfinite(vals.at[ASOF])),
                'last_date':str(last.date()) if last is not None else None,
                'valid_rows_through_asof':int(vals.loc[:ASOF].notna().sum())}
        p=input_root/'prices'/(s+'.csv')
        if p.exists():
            d=pd.read_csv(io.BytesIO(blob(p)),index_col=0,parse_dates=True)
            rec['overlay_rows']=len(d)
            if s in panel['adj']:
                old=panel['adj'][s];new=d['Adj Close'].reindex(old.index)
                overlap=old.notna()&new.notna()&(old>0)&(new>0)&(old.index<=PARITY_DAY)
                if overlap.any():
                    anchor=old.index[overlap][-1];ratio=old.at[anchor]/new.at[anchor]
                    rec['overlay_scale']=float(ratio)
                    rec['overlap_max_error_after_scale']=float(abs(new[overlap]*ratio-old[overlap]).max())
                    error=np.abs(new[overlap].to_numpy()*ratio-old[overlap].to_numpy())
                    ulp=np.spacing(old[overlap].to_numpy().astype(np.float32)).astype(float)
                    rec['max_float32_ulp_error']=float(np.max(error/ulp))
                    rec['overlap_raw_fields']={}
                    for k,c in FIELDS.items():
                        o=panel[k][s];n=d[c].reindex(o.index)
                        ix=o.notna()&n.notna()&(o.index<=PARITY_DAY)
                        rec['overlap_raw_fields'][k]=float(abs(o[ix]-n[ix]).max()) if ix.any() else None
        rows.append(rec)
    return rows

def forecast(panel,day,ss,etfs,local,strategy,policy):
    a,states=baseline(panel,day,ss,etfs,strategy,local.arch.old)
    require(not a['state_calm10'][0] and not a['state_weak_narrow'][0],
            'Active calm/weak gate: full annual model path required; specialized result not produced')
    effects,query,details=episode_effects(panel,day,ss,a['eligible'],local.ep)
    ch,score=local.adapter.score_and_choose('incumbent66',0.,np.full((1,len(ss)),np.nan),a,{'symbols':[ss]},1,
        pre_score=a['baseline'],effects=effects,episode_policy=policy,boundary_selector=local.ep.select_boundary)
    order=sorted(np.flatnonzero(a['eligible'][0]).tolist(),key=lambda j:(-a['baseline'][0,j],ss[j]))
    candidates=[j for j in order[2:5] if a['baseline'][0,order[2]]-a['baseline'][0,j]<=.02] if len(order)>=3 else []
    rowlist=[]
    for j,s in enumerate(ss):
        detail=dict(details[j]);detail['latest_label_end']=str(detail['latest_label_end']) if detail['latest_label_end'] is not None else None
        rowlist.append(dict(symbol=s,display_symbol=display(s),eligible=bool(a['eligible'][0,j]),
            baseline_score=float(a['baseline'][0,j]),final_episode_score=float(score[0,j]),
            baseline_rank=order.index(j)+1 if j in order else None,
            boundary_candidate=j in candidates,selected=j in ch[0],episode=detail))
    chosen=[dict(symbol=display(ss[j]),internal_symbol=ss[j],weight=1/len(ch[0]),baseline_score=float(a['baseline'][0,j]),
                 final_episode_score=float(score[0,j]),reference_close=float(panel['close'].at[day,ss[j]])) for j in ch[0]]
    return dict(signal_date=str(day.date()),asof='2026-09-30 America/New_York completed trading-session close',
        strategy='incumbent_local67 / monthly_local_linear70 config4',selected=chosen,
        baseline_order=[ss[j] for j in order[:3]],boundary_candidates=[ss[j] for j in candidates],
        selection_changed_by_episode=[ss[j] for j in ch[0]]!=[ss[j] for j in order[:3]],
        same_as_saved_66_10_october_report=[display(ss[j]) for j in ch[0]]==['MU','INTC','AMD'],
        gates=dict(calm10=False,weak_narrow=False),states=states,all49=rowlist,
        new_model_fits=0,inactive_model_slots_proved_zero=True,actual_orders_or_fills_created=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--parity-only',action='store_true')
    ap.add_argument('--inspect-inputs',action='store_true');args=ap.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'outputs/monthly_sector_asymmetry138/PLAN.json',PLAN_SHA)
    def pin(rel):return plan['hashes'][str(root/rel)]
    runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__)
    require(runtime==plan['runtime'],'Frozen runtime mismatch')
    for path,digest in plan['hashes'].items():
        if path.endswith('.py'):blob(path,digest)
    local=module('singlepoint_frozen_local67',root/'outputs/monthly_local_linear70/study.py')
    # Frozen monitor imports its same-directory base_strategy by the legacy name.
    sys.path.insert(0,str(root/'outputs/monthly_stability_monitor'))
    oldbase=sys.modules.pop('base_strategy',None)
    try:strategy=module('singlepoint_frozen_original65',root/'outputs/monthly_stability_monitor/strategy.py')
    finally:
        sys.modules.pop('base_strategy',None)
        if oldbase is not None:sys.modules['base_strategy']=oldbase
    meta=read(root/'outputs/monthly_local_linear70/META.json',pin('outputs/monthly_local_linear70/META.json'))
    cached=npz(root/'outputs/monthly_local_linear70/DATA.npz',pin('outputs/monthly_local_linear70/DATA.npz'))
    policy=read(root/'outputs/monthly_local_linear70/INCUMBENT_POLICY.json',pin('outputs/monthly_local_linear70/INCUMBENT_POLICY.json'))
    orders=read(root/'outputs/monthly_local_linear70/local_linear_ORDERS.json',pin('outputs/monthly_local_linear70/local_linear_ORDERS.json'))
    protected=read(root/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json',PROTECTED_SHA)
    qp=root/'outputs/monthly_daily_episodes/QUERIES.npz';queries=npz(qp,protected[str(qp)])
    panel,info=local.arch.old.panel_load(meta['etfs'])
    require(info['source_hashes']==plan['raw_source_hashes'],'Original64 raw-source hashes mismatch')
    SOURCES.update(info['source_hashes'])
    if args.inspect_inputs:
        pool,ss=load_pool(args.input_root,meta['etfs'])
        save(out/'INPUT_COVERAGE.json',dict(pool=pool['limit'],rows=input_coverage(panel,ss,args.input_root)))
        return
    print('Loaded original64 source panel; verifying frozen8/31',flush=True)
    parity=run_parity(panel,meta,cached,local,strategy,queries,policy,orders)
    save(out/'PARITY.json',parity)
    print(json.dumps({'parity':parity}),flush=True)
    if args.parity_only:return
    pool,ss=load_pool(args.input_root,meta['etfs'])
    panel,joins=merge_overlays(panel,args.input_root,ss,ASOF)
    # Calculate once with retained later rows and once with every frame truncated.
    result=forecast(panel,ASOF,ss,meta['etfs'],local,strategy,policy)
    prefix={k:v.loc[:ASOF].copy() for k,v in panel.items()}
    trimmed=forecast(prefix,ASOF,ss,meta['etfs'],local,strategy,policy)
    require(result==trimmed,'Future-row truncation changed exact result')
    result.update(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),future_rows_removed_same_result=True,
                  parity=parity,pool_limitation=pool['limit'],pool_selection_status=pool['selection_status'],
                  overlay_joins=joins,input_sha256=dict(SOURCES),script_sha256=sha(__file__),runtime=runtime,
                  limitations=['Retrospective reconstruction from cached historical vendor sources; not original-vintage certified.',
                  'Historical price-only strategy selection, not a broker fill, account order or prospective prediction made at the time.'])
    for path,digest in SOURCES.items():require(sha(path)==digest,'Input changed during run: '+path)
    save(out/'RESULT.json',result)
    print(json.dumps({'passed':True,'selected':result['selected'],'result':str(out/'RESULT.json')}),flush=True)

if __name__=='__main__':main()
