"""Review of terminal252 saved evidence after explicit authorization.

Run via run_local.sh with the four immutable phase digests. Standard library
only: no strategy/producer/oracle imports, array decoding, prices, volume,
feature mathematics, account replay or return recomputation. Saved scalar
statistics and guard comparisons use already frozen rules. One new report.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_peer_path252'
PRIOR=ROOT/'outputs/monthly_basket_core251'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_peer_path252_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_peer_path252_CONTEXT.json'
SOURCE=ROOT/'work/monthly_peer_path252_SOURCE_CONTRACT.json'
TESTS=ROOT/'work/monthly_peer_path252_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_peer_path252_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_basket_core251_RESULT_REVIEW.json'
ARMS=('incumbent_local67','daily_peer_path_relative','endpoint_peer_relative_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','FEATURE_AUDIT.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json','DAILY_SOURCE.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={'work/monthly_peer_path252_CONTEXT.json': '1f7745e33451a028a50f649bfa016c78957f902797a234b4b376a1649d794a42', 'work/monthly_peer_path252_SOURCE_CONTRACT.json': '73d9cd9e31d3a6fe91a66bf89e7b41d0ede09cf6ca72896f680bc250168753f8', 'work/monthly_peer_path252_TESTS.json': '35ff8ef9c78f0b1346dfeb18bffdd7972924e388d0cc6c96bd945205a742e1a0', 'work/monthly_peer_path252_CODE_REVIEW.json': 'd27955a0fcecccec68db1cb0df0506caccee812a92e0c79422cefd85126dddce', 'work/monthly_basket_core251_RESULT_REVIEW.json': '770b1d61de2a199e43e5e79b70442fe6dc41d264be96fe54b88f43307a7152f3', 'work/basket_core251_saved_result_reader_v2.py': '35ab84ab4e6401e1858ac1e522c6e4c58483cce2a65b7d89d6da23014f97f555', 'work/after251_peer_path252_DRAFT_02.json': '4e6becc7e25e0b4771bdf53a605d870ef4ad54ee10508bbdc0777658c994cd94', 'work/after251_peer_draft252_REVIEW_v2.json': '0272e7777f9674b028b697df262af243ae33e8501af8a3ec4002d2c2633b45be', 'work/after251_peer_path_source_map.json': 'e15fa2a81df431e4eb3da19c76520a9f7487c3f9b54fe20e8999f54bbbb45311', 'work/monthly_calendar_clock248_v2_independent.py': 'c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
# Existing evidence: ptr169_results_independent.py:37-41,85-91 (saved-stat
# comparison); previous251 RESULT_REVIEW.incumbent_reproduction preserves
# the inherited period fee bound and exact own-ledger aggregate reconciliation.


def need(ok,message):
    if not ok:raise ValueError(message)


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need(p.is_relative_to(ROOT) and 'CloudStorage' not in p.parts,'Only local evidence')
    need(p!=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json','Do not bind mutable central')
    return p


class Evidence:
    def __init__(self):self.pins={}
    def pin(self,p,want=None):
        p=local(p);got=digest(p);need(want is None or got==want,'Digest mismatch: '+str(p))
        key=str(p.relative_to(ROOT));need(key not in self.pins or self.pins[key]==got,'Evidence changed: '+key)
        self.pins[key]=got;return got
    def read(self,p,want=None):
        p=local(p);need(p.suffix=='.json','JSON only; never decode arrays/raw bars');self.pin(p,want)
        def invalid(x):raise ValueError('Nonfinite JSON constant '+x)
        return json.loads(p.read_text(),parse_constant=invalid)
    def registered(self,plan,p):
        p=local(p)
        # Compare path spellings only: inherited pins can include external runtime
        # metadata. This reader never opens or rehashes those external files.
        found=[]
        for k,h in plan['input_sha256'].items():
            key=Path(k);key=(key if key.is_absolute() else ROOT/key).resolve()
            if key==p:found.append(h)
        need(len(found)>=1 and len(set(found))==1,'Missing/conflicting registered pin: '+str(p));return found[0]
    def phase(self,folder,want,plan_sha,names):
        c=self.read(folder/'COMPLETION.json',want)
        need(c['status']=='COMPLETE' and c['plan_sha256']==plan_sha and set(c['artifacts_sha256'])==names,'Exact phase manifest')
        need({p.name for p in folder.iterdir()}==names|{'COMPLETION.json'},'Unexpected/missing phase file')
        for name,h in c['artifacts_sha256'].items():
            need(Path(name).name==name,'Artifact traversal');self.pin(folder/name,h)
        return c


def constants(path,names):
    """Parse only literal schema tuples from verified source; never import it."""
    found={}
    def value(n):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return found[n.id]
        if isinstance(n,(ast.Tuple,ast.List)):return tuple(value(x) for x in n.elts)
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return value(n.left)+value(n.right)
        raise ValueError('Nonliteral schema')
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:found[target.id]=value(node.value)
    need(set(found)==set(names),'Missing frozen schema constants');return found


def no_replay(r):
    need(r['producer_modules_imported'] is False and r['computes_returns'] is False,'Independent non-return scope')


def source_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Original source flags')
    need(r['checked_events']==n and r['episode_calls']==0 and r['new_target_ranks_computed'] is False and r['candidate_selection_computed'] is False,'Original source before candidate')
    need(r['C_source_atol']==1e-12 and r['C_source_rtol']==0 and r['continuous_atol']==2e-10 and r['continuous_rtol']==2e-10 and r['rank_tolerance']==0,'Frozen original source tolerances');no_replay(r)
    need(finite(r['maximum_core_source_absolute_error']) and r['maximum_core_source_absolute_error']<=1e-12,'Original core receipt error')
    need(r['original_checked_cells']==49*max(0,n-1),'Original49 formal source count')


def incumbent_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Incumbent flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==n and r['numerical_tolerance']==0 and r['candidate_selection_computed'] is False,'Incumbent exact scope');no_replay(r)


def daily_gate(r,n,counts,flags):
    need(all(r.get(k) is True for k in flags),'Daily source flags')
    need(r['counts']==counts and r['checked_events']==n and r['numerical_tolerance']==0,'Daily source coverage')
    need(r['full_source_report_supplied'] is True and r['episode_calls']==0 and r['new_target_ranks_computed'] is False and r['candidate_selection_computed'] is False,'Daily source before new features');no_replay(r)


def feature_gate(r,n,counts,flags):
    need(all(r.get(k) is True for k in flags),'Feature flags')
    need(r['counts']==counts and r['events']==r['checked_events']==n and r['full_features_supplied'] is True and r['numerical_tolerance']==r['rank_tolerance']==0 and r['exact_discrete'] is True,'Exact feature scope');no_replay(r)


def selection_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Selection flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==3*n and r['full_scores_supplied'] is True and r['full_pre_scores_supplied'] is True and r['numerical_tolerance']==0,'Full exact selection');no_replay(r)
    need(r['original_source_parity_tolerance']==1e-12,'Original source tolerance')
    if n:need(finite(r['minimum_remaining_support']) and r['minimum_remaining_support']>=.33-1e-12,'Original-S funding remaining bound')
    else:need(r['minimum_remaining_support'] is None,'Empty budget scope')
    need(finite(r['maximum_budget_sum_error']) and r['maximum_budget_sum_error']<=1e-12,'Budget arithmetic only tolerance')
    need(set(r['holdings_counts'])==set(ARMS) and all(set(v)=={'0','2','3'} and sum(v.values())==n for v in r['holdings_counts'].values()),'Holding counts scope')


def counted(records,n):
    rr=[r for r in records if r['query_index']<n]
    return dict(events=n,stock_cells=len(rr),mature_cells=sum(r['history_status']=='observed_full_window' for r in rr),supported_cells=sum(r['structural_peer_support'] for r in rr),price_observations=sum(len(r['path_observations']) for r in rr),new_model_fits=0)


def preparation(e,pc,audit,daily,flags,meta,calendar):
    need(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged','source_inventory_preserved')),'Source/cache preservation')
    need(e['new_model_fits']==pc['new_model_fits']==0 and pc['new_returns_computed'] is False,'No fit/outcome prepare')
    counts=counted(daily['records'],122);need(counts==pc['counts']==audit['counts']==daily['counts'],'Full saved counts')
    def gates(item,n,c):
        source_gate(item['original_source'],n,flags['ORIGINAL_SOURCE_FLAGS']);daily_gate(item['daily_source'],n,c,flags['SOURCE_FLAGS'])
        incumbent_gate(item['incumbent'],n,flags['INCUMBENT_GATE_FLAGS']);feature_gate(item['feature'],n,c,flags['FEATURE_FLAGS']);selection_gate(item['selection'],n,flags['SELECTION_FLAGS'])
    gates(e,122,counts)
    need([r['prefix'] for r in e['physical_prefixes']]==list(PREFIXES),'Five physical prefixes')
    for item in e['physical_prefixes']:
        n=item['prefix'];c=counted(daily['records'],n);need(item['exact'] is True and item['counts']==c,'Physical exact/counts');gates(item,n,c)
        if n:
            need(item['source_last_session']==meta['dates'][n-1] and item['source_rows']>0,'Physical prices through query')
            need(item['calendar_rows']==calendar['sessions'].index(meta['dates'][n-1])+2,'Only one official next-session calendar witness')
        else:need(item['source_rows']==item['calendar_rows']==0 and item['source_last_session'] is None,'Physical n0 no source')
    need([r['prefix'] for r in e['cached_prefixes']]==list(range(123)),'All123 cached incumbent gates')
    for item in e['cached_prefixes']:
        need(item['exact'] is True,'Cached exact prefix');incumbent_gate(item['incumbent'],item['prefix'],flags['INCUMBENT_GATE_FLAGS'])
    return dict(counts=counts,physical_prefixes=list(PREFIXES),all123_order_prefixes=True,all123_cached_incumbent_gates=True,all_independent_flags_and_counts_checked=True,source_then_exact_frozen_incumbent_before_new_features=True,three_original_episodes_per_decision_call=True,minimum_remaining_support=e['selection']['minimum_remaining_support'],maximum_budget_sum_error=e['selection']['maximum_budget_sum_error'],calm_nonweak_extension_count=e['selection']['calm_nonweak_extension_count'])


def rational_metadata(value):
    """Canonical saved rational shape/sign only; no reference or rank replay."""
    need(type(value) is dict and set(value)=={'numerator','denominator'},'Saved rational pair')
    a,b=value['numerator'],value['denominator'];need(type(a) is str and type(b) is str,'Saved rational strings')
    num,den=int(a),int(b);need(str(num)==a and str(den)==b and den>0 and math.gcd(num,den)==1,'Canonical rational')
    return (num>0)-(num<0)


def source_summary(daily,audit,meta,calendar,source,orders,schemas):
    need(set(daily)==set(schemas['SOURCE_TOP_KEYS']) and daily['schema']=='monthly_peer_path252_daily_source_report_v1','Daily source schema')
    need(daily['status']=='ADMITTED' and daily['admitted'] is True and daily['gaps']==[] and daily['model_fits']==0,'New daily source admitted receipt')
    for k in ('source_only','original_inventory_preserved','original_failed_audit_preserved','daily_paths_checked'):need(daily[k] is True,'Source preservation '+k)
    for k in ('strict_pit','momentum_computed','rankings_computed','orders_computed','returns_computed'):need(daily[k] is False,'Source-only scope '+k)
    need(len(daily['events'])==len(audit['events'])==122 and audit['schema']=='monthly_peer_path252_features_v1','Feature event scope')
    need(tuple(audit['counts'])==schemas['COUNT_FIELDS'] and len(source['mapping_records'])==3904,'Source inventory')
    mapping={(r['query_index'],r['column_index']):r for r in source['mapping_records']};need(len(mapping)==3904,'Unique metadata mapping')
    need([(r['query_index'],r['column_index']) for r in daily['records']]==list(mapping),'Complete ordered direct-stock inventory')
    rows={i:[] for i in range(122)}
    for r in daily['records']:rows[r['query_index']].append(r)
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official unique calendar');positions={d:i for i,d in enumerate(axis)}
    branches=Counter();domains=Counter();missing=Counter();events=[];warmup=[];identity_seen=[];rational_pairs=0
    for i,(ev,fa,clock) in enumerate(zip(daily['events'],audit['events'],source['clock_events'])):
        need(set(ev)==set(schemas['SOURCE_EVENT_KEYS']) and tuple(fa)==schemas['EVENT_KEYS'],'Exact event schemas')
        day=meta['dates'][i];t=positions[day];partial=day=='2016-08-17';stock=[j for j,s in enumerate(meta['symbols'][i]) if s not in meta['etfs']]
        need(ev['query_index']==fa['query_index']==clock['query_index']==i and ev['date']==fa['date']==clock['date']==day and ev['initial_partial'] is partial,'Event identity')
        need(clock['symbols']==meta['symbols'][i] and clock['query_position']==t and clock['next_official_session']==axis[t+1] and clock['numeric_evaluation_required'] is (not partial),'Query clock metadata')
        need(clock['price_start']==axis[t-252] and clock['price_end']==axis[t-42] and clock['source_start_position']==t-252 and clock['source_end_position']==t-42 and clock['price_count']==211 and clock['return_count']==210,'Metadata211 clock')
        pdates=[] if partial else axis[t-252:t-41];endpoints=[] if partial else [axis[t-252],axis[t-42]];pp=[] if partial else [t-252,t-42]
        need(ev['path_dates']==pdates and ev['source_dates']==fa['source_dates']==endpoints and ev['source_positions']==fa['source_positions']==pp,'Saved official daily window')
        need(ev['stock_indices']==fa['stock_indices']==clock['stock_indices']==stock and len(stock)==32,'All32 direct stock inventory before E')
        need(len(rows[i])==len(fa['cells'])==32,'Full source/cell count')
        mature=[];supported=[]
        for r,cell in zip(rows[i],fa['cells']):
            j=r['column_index'];need(set(r)==set(schemas['SOURCE_RECORD_KEYS']) and tuple(cell)==schemas['CELL_KEYS'],'Source/feature cell schema')
            need({k:r[k] for k in source['mapping_record_keys']}==mapping[i,j],'Exact old identity/mapping projection')
            need(r['date']==day and r['symbol']==meta['symbols'][i][j] and r['initial_partial'] is partial and r['source_dates']==endpoints and r['source_positions']==pp and r['gap_reasons']==[],'Cell identity/clock')
            need(all(cell[k]==r[k] for k in schemas['CELL_KEYS'][:7]),'Feature/source identity and cohort')
            bound=next((b for b in source['listing_bounds'] if b['symbol']==r['symbol'] and b['issuer_key']==r['issuer_key'] and b['available_from']<=day),None)
            wanted=None if partial else bound
            need(r['history_bound']==wanted,'Query-visible same-issuer bound')
            state='initial_partial' if partial else ('official_pre_inception' if bound and pdates[0]<bound['not_before'] else 'observed_full_window')
            need(r['history_status']==state,'Saved maturity classification')
            obs=r['path_observations'];need(len(obs)==len(pdates),'All211 path evidence')
            for ob,date in zip(obs,pdates):
                need(set(ob)=={'date','status','reason'} and ob['date']==date,'Observed source date')
                before=bound is not None and date<bound['not_before']
                if before:need(ob['status']=='missing' and ob['reason'] in ('missing_value','missing_row','missing_column'),'Pre-bound absence remains explicit');missing[ob['reason']]+=1
                else:need(ob['status']=='valid' and ob['reason'] is None,'Every mature/post-bound slot admitted')
            if state=='observed_full_window':mature.append(j)
            if r['structural_peer_support']:supported.append(j)
            need(type(r['structural_peer_support']) is bool and (not r['structural_peer_support'] or state=='observed_full_window'),'Support requires maturity')
            if state=='official_pre_inception':warmup.append(dict(query_index=i,date=day,column_index=j,symbol=r['symbol'],not_before=bound['not_before'],prebound_missing=sum(x['status']=='missing' for x in obs)))
            for k in ('focal_gross','daily_peer_gross','endpoint_peer_gross','path_raw','endpoint_raw'):
                should=state=='observed_full_window' if k=='focal_gross' else r['structural_peer_support']
                if should:
                    sign=rational_metadata(cell[k]);rational_pairs+=1
                    if k.endswith('gross'):need(sign>0,'Positive gross reference metadata')
                else:need(cell[k] is None,'Unavailable rational remains absent')
            for k in ('path_target','endpoint_target'):
                if r['structural_peer_support']:need(finite(cell[k]) and abs(cell[k])<=.5,'Bounded available target')
                else:need(cell[k] is None,'Unsupported target absent')
        need(mature==ev['mature_indices']==fa['mature_indices'] and supported==ev['supported_indices']==fa['supported_indices'],'Source/feature common domains')
        for r in rows[i]:
            peers=[x for x in rows[i] if r['column_index'] in mature and x['column_index'] in mature and x['sector']==r['sector'] and x['issuer_key']!=r['issuer_key']]
            peers.sort(key=lambda x:(x['issuer_key'],x['symbol'],x['column_index']));ids=[x['column_index'] for x in peers]
            need(r['peer_indices']==ids and r['peer_issuer_count']==len({x['issuer_key'] for x in peers}) and r['structural_peer_support']==(len(peers)>=2),'Full metadata cohort, no E filtering')
        need(len({r['issuer_key'] for r in rows[i]})==32,'Whole issuer uniqueness')
        branch='initial_partial' if partial else ('supported' if supported else 'no_supported_domain');branches[branch]+=1;domains[str(len(supported))]+=1
        if partial or not supported:
            for a in ARMS[1:]:need(orders[a][i]==orders[ARMS[0]][i],'Partial/N0 exact incumbent orders')
        events.append(dict(query_index=i,date=day,branch=branch,mature_count=len(mature),supported_count=len(supported),warmup_symbols=[r['symbol'] for r in rows[i] if r['history_status']=='official_pre_inception']))
    for x in source['identity_cells']:
        q,j=x['query_index'],x['column_index'];r=next(r for r in rows[q] if r['column_index']==j);clock=source['clock_events'][q]
        need((x['date'],x['symbol'],x['issuer_key'])==(r['date'],r['symbol'],r['issuer_key']) and x['price_start']==clock['price_start'] and x['price_end']==clock['price_end'],'Known listing metadata identity')
        need(x['bound_visible']==(x['available_from']<=x['date']) and type(x['price_window_dates_before_bound']) is int and x['price_window_dates_before_bound']==sum(d<x['not_before'] for d in axis[clock['source_start_position']:clock['source_end_position']+1]),'Known bound clock')
        identity_seen.append(dict(query_index=q,column_index=j,symbol=r['symbol'],status=r['history_status']))
    counts=counted(daily['records'],122);need(counts==audit['counts']==daily['counts'],'New252 source counts only')
    return dict(counts=counts,branch_counts=dict(branches),supported_domain_sizes=dict(domains),events=events,warmup_cells=warmup,prebound_absence_reasons=dict(missing),known_listing_cells_checked=identity_seen,canonical_rational_pairs_checked=rational_pairs,all211_source_statuses_checked=True,exact_reference_statistics_recomputed=False,exact_ranks_recomputed=False,feature_arrays_decoded=False,proof='Complete registered independent feature/source receipts and hashes establish numeric arrays; this reader checks saved identity/clock/domain/canonical evidence only.')


def changes(orders,reference,candidate):
    rows=[]
    for i,(a,b) in enumerate(zip(orders[reference],orders[candidate])):
        if set(a['symbols'])!=set(b['symbols']):rows.append(dict(index=i,date=a['date'],end=a['end'],reference=a['symbols'],candidate=b['symbols'],removed=sorted(set(a['symbols'])-set(b['symbols'])),added=sorted(set(b['symbols'])-set(a['symbols']))))
    return dict(counts={s:sum(lo<=r['index']<hi for r in rows) for s,(lo,hi) in SLICES.items()},months=rows,by_year=dict(Counter(r['date'][:4] for r in rows)))


def guard(c,r,gain):
    return c['training']['cagr']>=r['training']['cagr']+gain-1e-12 and c['training']['sharpe']>=r['training']['sharpe']-.05 and c['training']['max_drawdown']>=r['training']['max_drawdown']-.03 and all(c[e]['cagr']>=r[e]['cagr']-.025 for e in ('early','recent'))


def required_metrics(metrics,incumbent):
    need(set(metrics)==set(PATHS[:2]),'Exactly two ZERO paths for training')
    undefined=[]
    for path,groups in metrics.items():
        need(set(groups)=={'training','early','recent'},'First88 groups only')
        for group in groups:need(finite(groups[group]['cagr']),'Finite required CAGR')
        m=groups['training'];need(finite(m['max_drawdown']),'Finite required training drawdown')
        if m['sharpe'] is None:
            need(not incumbent and finite(m.get('daily_return_sd')) and m['daily_return_sd']==0 and m.get('sharpe_status')=='undefined_sharpe' and m.get('sharpe_reason')=='zero_daily_return_sd','Experimental undefined Sharpe evidence')
            undefined.append(path)
        else:
            need(finite(m['sharpe']),'Finite required Sharpe')
            need(finite(m.get('daily_return_sd')) and m['daily_return_sd']>0 and m.get('sharpe_status')=='defined' and m.get('sharpe_reason') is None,'Defined training Sharpe evidence')
    return undefined


def saved_selection(train,changed):
    need(set(train)==set(ARMS),'Three fixed arms')
    undefined={a:required_metrics(train[a],a==ARMS[0]) for a in ARMS}
    ids=[r['index'] for r in changed[ARMS[1]+'_vs_'+ARMS[0]]['months'] if r['index']<88]
    versus=[r['index'] for r in changed[ARMS[1]+'_vs_'+ARMS[2]]['months'] if r['index']<88]
    support=lambda x:len(x)>=3 and any(i<53 for i in x) and any(i>=53 for i in x)
    guards=dict(close_vs_incumbent='same_close_zero' not in undefined[ARMS[1]] and guard(train[ARMS[1]]['same_close_zero'],train[ARMS[0]]['same_close_zero'],.005),
      open_vs_incumbent='next_open_zero' not in undefined[ARMS[1]] and guard(train[ARMS[1]]['next_open_zero'],train[ARMS[0]]['next_open_zero'],0.),
      close_vs_endpoint_control=all('same_close_zero' not in undefined[a] for a in ARMS[1:]) and guard(train[ARMS[1]]['same_close_zero'],train[ARMS[2]]['same_close_zero'],0.),
      open_vs_endpoint_control=all('next_open_zero' not in undefined[a] for a in ARMS[1:]) and guard(train[ARMS[1]]['next_open_zero'],train[ARMS[2]]['next_open_zero'],0.),changed_support=support(ids),changed_support_vs_control=support(versus))
    return dict(primary=ARMS[1] if all(guards.values()) else ARMS[0],admitted=all(guards.values()),guards=guards,changed_months=ids,changed_months_vs_control=versus,undefined_sharpe=undefined,first88_only=True,later34_used=False)


def compare_metrics(independent,saved,label,residuals):
    need(set(independent)==STAT_KEYS and STAT_KEYS<=set(saved),'Nine saved metric fields')
    for key,value in independent.items():
        other=saved[key]
        if isinstance(value,str) or value is None or key=='trading_sessions':need(other==value,'Exact metric '+label+'/'+key)
        else:near(value,other,label+'/'+key,residuals)


def near(actual,expected,label,residuals):
    need(finite(actual) and finite(expected) and abs(actual-expected)<=ACCOUNT_ATOL+ACCOUNT_RTOL*abs(expected),'Frozen saved-accounting tolerance '+label)
    if actual!=expected:residuals.append(dict(label=label,actual=actual,reference=expected,difference=actual-expected))


def accounting(ev,results,ind,training):
    need(ind['passed'] is True and ind['checked_paths']==9 and ind['checked_slices']==45 and ind['checked_training_groups']==18,'9/45/18 independent accounting')
    need(ind['independent_accounts'] is True and ind['independent_of_production_model_and_engine'] is True and ind['terminal_liquidation'] is False,'Independent fixed-unit accounting')
    need(set(ind['paths'])==set(results['arms'])==set(training)==set(ARMS),'Exactly3 result arms')
    residuals=[];fee_fields=0;metric_ops=0;exports={}
    for arm in ARMS:
        need(set(results['arms'][arm])==set(ind['paths'][arm])==set(PATHS),'Exactly3 paths')
        need(set(training[arm])==set(PATHS[:2]),'Only zero-cost training')
        for p in PATHS:
            cert=ind['paths'][arm][p];saved=results['arms'][arm][p]
            need(cert['passed'] is True and cert['monthly_executions']==122 and set(cert['slices'])==set(saved['slices'])==set(SLICES),'Five saved slices')
            compare_metrics(cert['slices']['full122'],saved['metrics'],arm+'/'+p+'/metrics',residuals);metric_ops+=9
            for split in SLICES:compare_metrics(cert['slices'][split],saved['slices'][split],arm+'/'+p+'/'+split,residuals);metric_ops+=9
            for k in ('fees_paid','dollar_turnover','trade_rows'):near(cert[k],saved['metrics'][k],arm+'/'+p+'/'+k,residuals)
            need(saved['accounting']['exported_fill_ledger_matches'] is True and saved['accounting']['terminal_liquidation'] is False,'Saved account declarations')
            exports[arm,p]={k:ev.read(OUT/'results'/(arm+'_'+p+'_'+k+'.json')) for k in KINDS}
            data=exports[arm,p]
            need(len(data['HOLDINGS'])==122 and len(data['TRADES'])==cert['trade_rows'],'Full saved ledger rows')
            need([x['date'] for x in data['DAILY']]==[x['date'] for x in data['DAILY_POSITIONS']],'Saved NAV/positions dates')
            if p!='next_open_cost':
                need(saved['metrics']['fees_paid']==cert['fees_paid']==0,'Zero-fee aggregate')
                for k,field in (('TRADES','cost'),('HOLDINGS','fee')):
                    for x in data[k]:need(type(x[field]) in (int,float) and x[field]==0,'Every zero-fee field exact0');fee_fields+=1
            if p in PATHS[:2]:
                need(set(training[arm][p])=={'training','early','recent'},'Training3 groups')
                for short,split in (('training','first88'),('early','early53'),('recent','recent35')):
                    compare_metrics(cert['slices'][split],training[arm][p][short],arm+'/'+p+'/train/'+short,residuals);metric_ops+=9
                    # Compare the two existing summaries directly using the inherited
                    # near rule with the saved full slice as the reference.
                    compare_metrics({k:training[arm][p][short][k] for k in STAT_KEYS},saved['slices'][split],arm+'/'+p+'/train-full/'+short,residuals);metric_ops+=9
    return dict(independent_paths=9,independent_slices=45,training_groups=18,metric_fields_compared=metric_ops,zero_cost_streams=6,zero_cost_fee_fields_exact_zero=fee_fields,comparison_rule='ptr169_results_independent.py:37-41,85-91: absolute1e-7+relative2e-11*abs(reference), exact strings/None/trading_sessions',actual_saved_metric_residuals=residuals,returns_recomputed=False),exports


def ordered_period_fees(rows):
    """Frozen validate.replay accumulation: initial0.0, chronological +=fee.

    This is a saved-ledger identity check, not an account or return replay.
    Do not replace with fsum or an aggregate tolerance.
    """
    need(len(rows)==122,'Full incumbent holdings')
    dates=[r['execution_date'] for r in rows]
    need(all(type(d) is str for d in dates) and all(a<b for a,b in zip(dates,dates[1:])),'Chronological saved fee accumulation')
    total=0.0
    for row in rows:
        fee=row['fee'];need(finite(fee) and fee>=0,'Finite nonnegative period fee')
        total+=fee;need(math.isfinite(total),'Finite saved fee accumulation')
    return total


def compare_incumbent_saved(before_paths,after_paths,before_exports,after_exports,before_cert,after_cert):
    """Pure comparison of saved summaries/ledgers; no file or strategy access."""
    need(set(before_paths)==set(after_paths)==set(before_exports)==set(after_exports)==set(before_cert)==set(after_cert)==set(PATHS),'Exact incumbent path domain')
    exceptions=[];aggregate=None
    for p in PATHS:
        before=before_paths[p];after=after_paths[p]
        need(set(before_exports[p])==set(after_exports[p])==set(KINDS),'Exact incumbent export domain')
        need(before_cert[p]['passed'] is True and after_cert[p]['passed'] is True,'Both saved independent incumbent certificates passed')
        for key in ('metrics','slices'):
            left=before[key];right=after[key]
            if p=='next_open_cost':
                # Remove ONLY the two explicitly reconciled aggregate fields.
                if key=='metrics':
                    need('fees_paid' in left and 'fees_paid' in right,'Cost aggregate present')
                    left={k:v for k,v in left.items() if k!='fees_paid'}
                    right={k:v for k,v in right.items() if k!='fees_paid'}
                else:
                    need(set(left)==set(right)==set(SLICES),'Exact cost slice domain')
                    need('fees_paid' in left['full122'] and 'fees_paid' in right['full122'],'Full122 cost aggregate present')
                    left={k:({f:v for f,v in m.items() if f!='fees_paid'} if k=='full122' else m) for k,m in left.items()}
                    right={k:({f:v for f,v in m.items() if f!='fees_paid'} if k=='full122' else m) for k,m in right.items()}
            need(left==right,'Incumbent '+p+'/'+key+' exact except explicitly reconciled cost aggregate')
        for kind in KINDS:
            a=before_exports[p][kind];b=after_exports[p][kind]
            if kind!='HOLDINGS':need(a==b,'Incumbent saved '+kind+' exact');continue
            need(len(a)==len(b)==122,'Full incumbent holdings')
            for i,(x,y) in enumerate(zip(a,b)):
                need(set(x)==set(y) and {k:v for k,v in x.items() if k!='fee'}=={k:v for k,v in y.items() if k!='fee'},'Holding nonfee fields exact')
                need(finite(x['fee']) and finite(y['fee']) and x['fee']>=0 and y['fee']>=0,'Finite nonnegative period fees')
                if x['fee']!=y['fee']:
                    d=y['fee']-x['fee'];need(p=='next_open_cost' and abs(d)<=PERIOD_FEE_ATOL,'Only inherited cost period fee exception')
                    exceptions.append(dict(path=p,index=i,signal_date=y['signal_date'],previous=x['fee'],current=y['fee'],difference=d))
        if p!='next_open_cost':
            need(before['metrics']['fees_paid']==after['metrics']['fees_paid']==before_cert[p]['fees_paid']==after_cert[p]['fees_paid']==0,'Both rounds zero-cost aggregate exact0')
            for data in (before_exports[p],after_exports[p]):
                for kind,field in (('HOLDINGS','fee'),('TRADES','cost')):
                    need(all(finite(row[field]) and row[field]==0 for row in data[kind]),'Both rounds every zero-fee field exact0')
            continue
        totals={}
        for label,saved,data in (('previous',before,before_exports[p]),('current',after,after_exports[p])):
            total=ordered_period_fees(data['HOLDINGS'])
            for scope,metric in (('metrics',saved['metrics']),('slices.full122',saved['slices']['full122'])):
                need(finite(metric['fees_paid']) and metric['fees_paid']==total,'Cost '+label+'/'+scope+' fees_paid exact chronological HOLDINGS accumulation')
            totals[label]=total
        previous_ind=before_cert[p]['fees_paid'];current_ind=after_cert[p]['fees_paid']
        need(finite(previous_ind) and finite(current_ind) and previous_ind>=0 and previous_ind==current_ind,'Independent incumbent cost fee totals exact across rounds')
        aggregate=dict(path=p,fields=['metrics.fees_paid','slices.full122.fees_paid'],previous=totals['previous'],current=totals['current'],difference=totals['current']-totals['previous'],
          previous_independent=previous_ind,current_independent=current_ind,independent_totals_exact=True,
          each_round_exact_to_own_ordered_holdings=True,accumulation='total=0.0; for row in chronological HOLDINGS: total+=row[fee]',
          source='outputs/monthly_factor_search_v2/validate.py:109-123 replay totalcost=0.; totalcost+=fee; fees_paid=totalcost',
          aggregate_tolerance_added=False,cross_round_aggregate_bitwise_equal=totals['previous']==totals['current'])
    return dict(prior='monthly_basket_core251',all_incumbent_metrics_slices_exact=aggregate['cross_round_aggregate_bitwise_equal'],
      zero_cost_metrics_slices_exact=True,all_other_incumbent_metrics_slices_exact=True,daily_trades_positions_exact=True,holding_nonfee_fields_exact=True,
      cost_aggregate_reconciliation=aggregate,
      period_fee_exception=dict(absolute_bound=PERIOD_FEE_ATOL,scope='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance',
        inherited_scope='251v2 own-ledger aggregate identity plus original individual cost fee bound; no new tolerance',evidence='work/monthly_basket_core251_RESULT_REVIEW.json/incumbent_reproduction/period_fee_exception',actual_differences=exceptions))


def previous_incumbent(ev,results,exports,current_ind):
    review=ev.read(PREVIOUS_REVIEW,FIXED[str(PREVIOUS_REVIEW.relative_to(ROOT))])
    inherited=review['incumbent_reproduction']['period_fee_exception']
    need(review['passed'] is True and inherited['absolute_bound']==PERIOD_FEE_ATOL and inherited['scope']=='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance','Existing prior fee exception only')
    previous_completion=ev.read(PRIOR/'results/COMPLETION.json',review['results_completion_sha256'])
    previous=ev.read(PRIOR/'results/RESULTS.json',previous_completion['artifacts_sha256']['RESULTS.json'])
    previous_ind=ev.read(PRIOR/'results/INDEPENDENT_OUTCOME.json',previous_completion['artifacts_sha256']['INDEPENDENT_OUTCOME.json'])
    need(previous_ind['passed'] is True and current_ind['passed'] is True and previous_ind['accounting_source_sha256']==current_ind['accounting_source_sha256'],'Passed independent certificates use same frozen accounting source')
    prior_exports={};zero_hashes=[];cost_exact_hashes=[]
    for p in PATHS:
        prior_exports[p]={}
        for kind in KINDS:
            name=ARMS[0]+'_'+p+'_'+kind+'.json'
            prior_exports[p][kind]=ev.read(PRIOR/'results'/name,previous_completion['artifacts_sha256'][name])
            prior_sha=ev.pins[str((PRIOR/'results'/name).relative_to(ROOT))]
            current_sha=ev.pins[str((OUT/'results'/name).relative_to(ROOT))]
            if p!='next_open_cost' or kind!='HOLDINGS':
                need(prior_sha==current_sha,'Incumbent unchanged export bytes: '+name)
                (zero_hashes if p!='next_open_cost' else cost_exact_hashes).append(name)
    check=compare_incumbent_saved(previous['arms'][ARMS[0]],results['arms'][ARMS[0]],prior_exports,{p:exports[ARMS[0],p] for p in PATHS},previous_ind['paths'][ARMS[0]],current_ind['paths'][ARMS[0]])
    check.update(zero_cost_export_sha_exact=zero_hashes,cost_daily_trades_positions_sha_exact=cost_exact_hashes)
    return check


def run(args):
    need(not REPORT.exists(),'Report write-once');need(platform.python_version()=='3.9.6','Use pinned run_local wrapper')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h)
    context=ev.read(CONTEXT);source=ev.read(SOURCE);tests=ev.read(TESTS);code=ev.read(CODE_REVIEW)
    need(source['price_sessions']==211 and source['return_intervals']==210 and source['source_admitted'] is False and source['full_daily_source_coverage']=='UNKNOWN' and source['volume_used'] is False,'Metadata did not admit211')
    need(len(source['clock_events'])==122 and len(source['identity_cells'])==5 and source['direct_stocks_per_query']==32,'Source metadata scope')
    need(plan['context']==context and plan['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))],'Registered exact context')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==85 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal85 pinned runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['tests']==85 and code['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))] and code['actual_returns_computed'] is False,'Final code review')
    need(len(tests['code_sha256'])==6,'All6 code files')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered code binding');ev.pin(p,h)
    for p,h in FIXED.items():
        if p!='work/basket_core251_saved_result_reader_v2.py':need(ev.registered(plan,p)==h,'Registered evidence binding')
    oracle=ROOT/'work/monthly_peer_path252_independent.py'
    schemas=constants(oracle,('COUNT_FIELDS','SOURCE_TOP_KEYS','SOURCE_EVENT_KEYS','SOURCE_RECORD_KEYS','EVENT_KEYS','CELL_KEYS'))
    flags=constants(oracle,('ORIGINAL_SOURCE_FLAGS','SOURCE_FLAGS','FEATURE_FLAGS','SELECTION_FLAGS'))
    flags.update(constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',)))
    need([len(schemas[k]) for k in ('COUNT_FIELDS','EVENT_KEYS','CELL_KEYS','SOURCE_TOP_KEYS','SOURCE_EVENT_KEYS','SOURCE_RECORD_KEYS')]==[6,8,14,17,9,23],'Frozen252 schema sizes')
    for p in (ROOT/'work/ptr169_results_independent.py',ROOT/'work/monthly_ownsector_gate238_host.py'):ev.pin(p,ev.registered(plan,p))
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==247 and sum(x['trials'] for x in state['completed'])==1254 and state['registered_queue']==[],'Registered pre-state')
    protection=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protection)==plan['protected_files']==21444,'Protection manifest count only')
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Frozen selection precedes diagnostics')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed interval')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Fixed training splits')
    need(set(orders)==set(ARMS),'Three order arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Saved order axis')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) for o in orders[arm]),'3/2/cash saved orders')
    oldorders=ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json'));need(orders[ARMS[0]]==oldorders,'Exact frozen original incumbent')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));originalplan=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    need(parity['raw_source_hashes']==source['raw_source_hashes']==originalplan['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64,'Original64 map identity')
    need(parity['original_arrays_unchanged'] is True and parity['original_orders_exact'] is True and len(parity['original_array_fingerprints'])==27 and parity['source_report_sha256']==source['source239']['sha256']==plan['source_binding']['source_report_sha256'],'Original27/239 metadata binding')
    daily=ev.read(OUT/'prepared/DAILY_SOURCE.json');audit=ev.read(OUT/'prepared/FEATURE_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    src=source_summary(daily,audit,meta,calendar,source,orders,schemas);prep=preparation(e,pc,audit,daily,flags,meta,calendar)
    changed={candidate+'_vs_'+ref:changes(orders,ref,candidate) for candidate,ref in ((ARMS[1],ARMS[0]),(ARMS[1],ARMS[2]),(ARMS[2],ARMS[0]))}
    selection=saved_selection(training,changed);need(selection==frozen['selection'] and selection['primary'] in ARMS[:2],'First88 recheck/control never promoted')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Result labels')
    need(ind['accounting_source_sha256']==ev.pins['work/ptr169_results_independent.py'],'Registered accounting proof')
    ac,exports=accounting(ev,result,ind,training);inc=previous_incumbent(ev,result,exports,ind)
    engine=ROOT/'outputs/monthly_factor_search_v2/validate.py';ev.pin(engine,ev.registered(plan,engine));ev.pin(Path(__file__))
    for p,h in ev.pins.items():need(digest(local(p))==h,'Input changed during review: '+p)
    summary={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_peer_path252_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='Static252 code and saved-evidence reviewer; authored252 metadata source/context contracts, none of6 implementation/test files. New exact numerical proofs supplied by independent252 oracle; no reference/rank/return replay here.',
      scope='Pre-close saved hashes, source status/identity/cohort clocks, complete independent receipts, first88 candidate-only admission guards and recorded accounting/holdings comparisons.',
      plan_sha256=args.plan_sha256,context_sha256=FIXED[str(CONTEXT.relative_to(ROOT))],prepared_completion_sha256=args.prepared_sha256,frozen_sha256=args.frozen_sha256,results_completion_sha256=args.results_sha256,
      formal_synthetic_tests=85,bindings=dict(code_files=6,prepared_files=9,result_files=38,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      preparation=prep,source_and_clock=src,selection=selection,changed_holdings=changed,accounting_review=ac,incumbent_reproduction=inc,full_results=summary,
      holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=247,registered_pre_state_trials=1254,expected_post_close_completed=248,expected_post_close_trials=1257,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],endpoint_control_never_promoted=True,prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      limitations=['Saved independent source/feature/selection/account proofs are verified, not numerically regenerated; NPZ members are hash-bound without decoding.','New211 coverage and source counts come only from252 saved records; the old239 endpoint certificate cannot establish them.','Current-query sector/identity mappings and fixed inherited universe remain declared assumptions, strictPITfalse; no new source economic certification.','Daily versus endpoint reference contrasts aggregation and weight reset, not isolated persistence; reference baskets are not traded.','Only new factors exclude recent42sessions; inherited P/core/current63/state retain their own frozen clocks.','Incumbent cross-round cost fees_paid may differ only under inherited251v2 exact own-ledger accumulation, equal independent fee certificates and original1e-10 per-period bound; every other incumbent metric/slice/export checked exact.','Only daily candidate may be admitted by first88, endpoint is a control; later34/full/cost cannot change FROZEN.','Same-close zero-cost benchmark is idealized, and all history has been reused across trials.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),selection=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},source_counts=src['counts'],changed_month_counts={k:v['counts'] for k,v in changed.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())
