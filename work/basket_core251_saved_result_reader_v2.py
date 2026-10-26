"""Version2 review of terminal251 saved evidence after explicit authorization.

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
OUT=ROOT/'outputs/monthly_basket_core251'
PRIOR=ROOT/'outputs/monthly_reference_gain250'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_basket_core251_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_basket_core251_context.json'
SOURCE=ROOT/'work/basket_core251_SOURCE_CONTRACT.json'
TESTS=ROOT/'work/monthly_basket_core251_TESTS_v2.json'
CODE_REVIEW=ROOT/'work/monthly_basket_core251_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_reference_gain250_RESULT_REVIEW.json'
ARMS=('incumbent_local67','long_momentum_basket_core','long_momentum_diagonal_core')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={
 'work/monthly_basket_core251_context.json':'d399774f4470feaa5d1943ea9acff7bb843d22fe2610b367a4e3cd8d78aa0c44',
 'work/basket_core251_SOURCE_CONTRACT.json':'16e78872660c0ee5dad15854333f6cf0d2fad8a1be179463f10cf18ae81f304c',
 'work/monthly_basket_core251_TESTS_v2.json':'59a8849f9791067e58a7f436b4ab4c184e6917acd8b9585117b0e2336ff0a2d2',
 'work/monthly_basket_core251_CODE_REVIEW.json':'03b38753ca5a505321ad2735cf377fd7e611ba4a5681ec2ec91733b8c7dabee1',
 'work/monthly_reference_gain250_RESULT_REVIEW.json':'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595',
 'work/after250_basket_core251_definition_v2.json':'3fd3acdcff6e5f0ce782b5647ed77b96921a82717ad247c91832a61a7d476a17',
 'work/basket_core251_DEFINITION_REVIEW.json':'0b31b574290c01939c74a91a6846f16865202fbdeb47b6c22df651cb339b1e4f',
 'work/basket_core251_MATH_REVIEW_v2.json':'1e74e5cd49a612f66758ca8b5a9086b7c4083454e09e7d02a8b62b1d0398824f',
 'work/basket_core251_SOURCE_REVIEW.json':'d9f7ecf71cbcd2a648b4b212546409ca81b442879c993bd36d6a0fff2e74e296',
 'work/monthly_calendar_clock248_v2_independent.py':'c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
# Existing evidence: ptr169_results_independent.py:37-41,85-91 (saved-stat
# comparison); previous250 RESULT_REVIEW.accounting_review.comparison_rule
# and incumbent_reproduction.period_fee_exception (ONLY cost HOLDINGS.fee <=1e-10).


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


def source_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Source gate flags')
    need(r['checked_events']==n and r['episode_calls']==0 and r['new_target_ranks_computed'] is False and r['new_basket_selection_computed'] is False,'Source-only before basket gate')
    need(r['C_source_atol']==1e-12 and r['C_source_rtol']==0 and r['continuous_atol']==2e-10 and r['continuous_rtol']==2e-10 and r['rank_tolerance']==0,'Source tolerance')
    need(all(r[k] is False for k in ('producer_modules_imported','computes_returns','candidate_selection_computed')),'Source scope')


def incumbent_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Incumbent gate flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==n and r['numerical_tolerance']==0,'Incumbent exact scope')
    need(all(r[k] is False for k in ('producer_modules_imported','computes_returns','candidate_selection_computed')),'Incumbent scope')


def pre_gates(r,n,flags,source):
    need(set(r)==({'source','incumbent'} if source else {'incumbent'}),'Exact pre-candidate gates')
    if source:source_gate(r['source'],n,flags['SOURCE_GATE_FLAGS'])
    incumbent_gate(r['incumbent'],n,flags['INCUMBENT_GATE_FLAGS'])


def selection_check(r,n,flags,full=False):
    need(all(r.get(k) is True for k in flags['PREPARE_FLAGS' if full else 'SELECTION_FLAGS']),'Independent full flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==3*n,'All orders/events checked')
    need(all(r[k] is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'All full arrays supplied')
    need(r['producer_modules_imported'] is False and r['computes_returns'] is False,'Independent scope')
    if full:
        need(r['full_features_supplied'] is True and r['original_core_checked'] is True and r['price_clocks_checked'] is True,'Full source scope')
        need(r['C_source_atol']==1e-12 and r['C_source_rtol']==0 and r['continuous_atol']==2e-10 and r['continuous_rtol']==2e-10 and r['rank_tolerance']==0 and r['exact_discrete'] is True,'Full tolerances')
    else:need(r['numerical_tolerance']==0,'Selection exact tolerance')


def preparation(e,pc,audit,flags,counts):
    need(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_input_unchanged')),'Source/cache preservation')
    need(e['experimental_domain_adds_positive_M252'] is True,'Declared additional annual-positive domain')
    need(e['new_model_fits']==pc['new_model_fits']==pc['new_predictive_model_fits']==0 and pc['new_returns_computed'] is False,'No fit/return preparation')
    selection_check(e['full'],122,flags,True);selection_check(e['selection'],122,flags)
    need(e['full']['counts']==e['selection']['counts']==audit['counts']==pc['descriptive_counts']==counts,'Full counts binding')
    pre_gates(e['pre_candidate_gates'],122,flags,True)
    need([x['prefix'] for x in e['all123_cached_control_gates']]==list(range(123)),'All123 incumbent gates')
    for x in e['all123_cached_control_gates']:pre_gates(x['gates'],x['prefix'],flags,False)
    need([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'Five physical prefixes')
    for x in e['physical_prefixes']:
        n=x['prefix'];need(x['exact'] is True,'Exact physical prefix')
        pre_gates(x['pre_candidate_gates'],n,flags,True);selection_check(x['full'],n,flags,True);selection_check(x['selection'],n,flags)
        need(x['counts']==x['full']['counts']==x['selection']['counts'] and x['counts']['events']==n,'Prefix count binding')
    return dict(all123_cached_control_gates=True,all123_order_prefixes=True,physical_prefixes=list(PREFIXES),all_independent_flags_and_counts_checked=True,source_gate_episode_calls=0,source_gate_new_target_ranks_computed=False,source_gate_new_basket_selection_computed=False,source_before_incumbent_before_basket_receipts=True,counts=counts)


def rational_metadata(value):
    """Inspect saved canonical-pair structure/sign; do not recompute M or risk."""
    need(type(value) is dict and set(value)=={'numerator','denominator'},'Saved rational pair')
    a,b=value['numerator'],value['denominator']
    need(type(a) is str and type(b) is str,'Saved integer strings')
    num,den=int(a),int(b)
    need(str(num)==a and str(den)==b and den>0 and math.gcd(num,den)==1,'Canonical saved rational pair')
    return (num>0)-(num<0)


def feature_summary(features,audit,meta,calendar,source,orders,schemas):
    need(len(features)==len(audit['events'])==122 and audit['schema']=='monthly_basket_core251_decisions_v1','Full feature/audit scope')
    need(list(audit['counts'])==list(schemas['COUNT_FIELDS']),'Counts schema')
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar unique');pos={d:i for i,d in enumerate(axis)}
    cells=[];branches=Counter();events=[];max_error=0.;zero_joint=zero_diag=0
    for i,e in enumerate(features):
        need(tuple(e)==schemas['EVENT_KEYS'] and e['query_index']==i and e['date']==meta['dates'][i] and e['symbols']==meta['symbols'][i],'Event identity')
        t=pos[e['date']];need(e['query_position']==t and e['next_official_session']==axis[t+1],'Official query/witness')
        need(e['price_start']==axis[t-294] and e['price_end']==axis[t-42] and e['block_boundary_dates']==[axis[t-294+21*k] for k in range(13)],'253price/12block official clocks')
        clock=source['clock_events'][i]
        need(all(clock[k]==e[k] for k in ('query_index','date','initial_partial','symbols','query_position','next_official_session','price_start','price_end','block_boundary_dates')),'Source clock binding')
        need((clock['price_count'],clock['return_count'],clock['block_count'],clock['block_sessions'])==(253,252,12,21),'Fixed source counts')
        need(clock['source_start_position']==t-294 and clock['source_end_position']==t-42,'Source positions')
        partial=e['date']=='2016-08-17';need(e['initial_partial'] is partial and clock['numeric_evaluation_required'] is (not partial),'Named partial scope')
        need(e['panel_first_session']=='2008-01-02' and type(e['calm']) is bool and e['core_weight']==(.5 if e['calm'] else .52),'Original fullpast source/state')
        need(e['trading_price_start']==axis[t-(252 if e['calm'] else 294)] and e['trading_price_end']==axis[t-42] and e['risk_price_start']==axis[t-63] and e['risk_price_end']==e['date'],'Separate original clock')
        if not partial:need(e['date'][:7]!=axis[t+1][:7],'Formal month end')
        au=audit['events'][i]
        need(tuple(au)==schemas['AUDIT_EVENT_KEYS'] and au['query_index']==i and au['date']==e['date'] and au['branch']==e['branch'] and set(au['arms'])==set(ARMS),'Decision audit identity')
        need(au['incumbent_choices']==au['arms'][ARMS[0]],'Original audit choices')
        for arm in ARMS:
            ids=au['arms'][arm]
            need(type(ids) is list and all(type(j) is int and 0<=j<49 for j in ids) and len(set(ids))==len(ids),'Saved choices indices')
            need([e['symbols'][j] for j in ids]==orders[arm][i]['symbols'],'Audit-to-order direct membership')
        branches[e['branch']]+=1
        old=e['original_eligible_indices']
        need(type(old) is list and old==sorted(set(old)) and all(type(j) is int and 0<=j<49 for j in old),'Saved original E domain')
        if partial:
            need(e['branch']=='initial_partial' and e['cells']==[] and e['u0_indices']==[] and e['required_indices']==[] and e['core_source_max_abs_error'] is None,'Partial no new sources')
            need(all(e[k] is None for k in ('new_eligible_indices','excluded_momentum_indices','shortlist_indices','trios','common_trio_indices','winners')),'Partial no new baskets')
            need(all(orders[arm][i]==orders[ARMS[0]][i] for arm in ARMS),'Partial exact incumbent')
            continue
        need(len(e['cells'])==49 and len(set(e['symbols']))==49 and e['required_indices']==old,'Required full Eold before shortlist')
        eligible=[];excluded=[];u=[]
        for j,c in enumerate(e['cells']):
            need(tuple(c)==schemas['CELL_KEYS'] and c['column_index']==j and c['symbol']==e['symbols'][j],'Cell identity')
            need(type(c['original_finite']) is bool and type(c['original_eligible']) is bool and type(c['source_available']) is bool,'Cell bools')
            need(c['original_eligible'] is (j in old) and c['source_available'] is (j in old),'Source required exactly Eold')
            if c['original_finite']:u.append(j)
            need(finite(c['core_abs_error']) and 0<=c['core_abs_error']<=1e-12,'Original source C tolerance');max_error=max(max_error,c['core_abs_error'])
            if j not in old:need(c['measurement'] is None,'Efalse no new measurement')
            else:
                m=c['measurement'];need(tuple(m)==schemas['MEASURE_KEYS'] and m['schema']=='monthly_basket_core251_measure_v1' and m['price_count']==253 and m['block_count']==12 and len(m['block_returns'])==12,'Saved exact measure metadata')
                sign=rational_metadata(m['momentum'])
                (eligible if sign>0 else excluded).append(j)
                for v in m['block_returns']:rational_metadata(v)
            cells.append(c)
        need(e['u0_indices']==u and e['core_source_max_abs_error']==max(c['core_abs_error'] for c in e['cells']),'Original finite source metadata')
        need(e['new_eligible_indices']==eligible and e['excluded_momentum_indices']==excluded,'Saved annual-positive domain')
        short=e['shortlist_indices'];S=len(short)
        need(type(short) is list and S==min(6,len(eligible)) and len(set(short))==S and set(short)<=set(eligible),'Saved Top6 inventory; order independently certified')
        triples=e['trios'];need(type(triples) is list and len(triples)==(math.comb(S,3) if S>=3 else 0),'Complete saved triple count')
        combinations_seen=[];common=[]
        for q,trio in enumerate(triples):
            need(tuple(trio)==schemas['TRIO_KEYS'],'Trio schema');ids=trio['indices'];symbols=trio['symbols']
            need(len(ids)==len(set(ids))==3 and set(ids)<=set(short) and symbols==sorted(symbols) and symbols==[e['symbols'][j] for j in ids],'Canonical triple members')
            combinations_seen.append(tuple(symbols));st=trio['statistics']
            need(tuple(st)==schemas['BASKET_KEYS'] and st['schema']=='monthly_basket_core251_basket_v1' and st['block_count']==12 and len(st['portfolio_blocks'])==12,'Basket statistic schema')
            for v in st['portfolio_blocks']:rational_metadata(v)
            need(rational_metadata(st['mean'])>0,'Saved positive mean invariant')
            jv=rational_metadata(st['joint_variance']);dv=rational_metadata(st['diagonal_variance']);need(jv>=0 and dv>=0,'Saved nonnegative risk')
            good=jv>0 and dv>0;need(st['supported'] is good,'Saved common support sign')
            reason='positive_variances' if good else 'zero_both_variances' if jv==dv==0 else 'zero_joint_variance' if jv==0 else 'zero_diagonal_variance'
            need(st['reason']==reason,'Saved zero-variance reason');zero_joint+=int(jv==0);zero_diag+=int(dv==0)
            if good:common.append(q)
        need(combinations_seen==sorted(set(combinations_seen)) and e['common_trio_indices']==common,'Complete unique ordered triple inventory/common set')
        branch='cash' if len(eligible)<2 else 'two_names' if len(eligible)==2 else 'zero_variance' if not common else 'optimized'
        need(e['branch']==branch and set(e['winners'])==set(ARMS[1:]),'Saved basket branch')
        for arm in ARMS[1:]:
            w=e['winners'][arm];need(tuple(w)==schemas['WINNER_KEYS'] and w['branch']==branch,'Winner schema/branch')
            ids=w['selected_indices'];need(ids==au['arms'][arm] and w['selected_symbols']==orders[arm][i]['symbols'],'Winner direct-order binding')
            if branch=='optimized':
                win=w['winning_trio_index'];ties=w['tied_trio_indices']
                need(win in common and ties==sorted(set(ties)) and ties and ties[0]==win and set(ties)<=set(common) and set(ids)==set(triples[win]['indices']),'Certified winning common trio/ties')
            else:
                wanted=[] if branch=='cash' else short[:2] if branch=='two_names' else short[:3]
                need(ids==wanted and w['winning_trio_index'] is None and w['tied_trio_indices']==[],'Declared small/zero domain choice')
            need(orders[arm][i]['scores']==[float(S-short.index(j)) for j in ids],'Display score S-zero_based_position only')
        events.append(dict(index=i,date=e['date'],branch=branch,original_eligible=len(old),new_eligible=len(eligible),shortlist=S,trios=len(triples),common_trios=len(common)))
    formal=[e for e in features if not e['initial_partial']];all_trios=[q for e in formal for q in e['trios']];required=sum(len(e['required_indices']) for e in formal)
    count={k:0 for k in schemas['COUNT_FIELDS']}
    count.update(events=122,symbols_per_event=49,initial_partial_events=122-len(formal),formal_events=len(formal),calm_formal_events=sum(e['calm'] for e in formal),noncalm_formal_events=sum(not e['calm'] for e in formal),original_checked_cells=len(cells),original_finite_cells=sum(c['original_finite'] for c in cells),original_eligible_cells=sum(len(e['original_eligible_indices']) for e in features),source_required_cells=required,source_available_cells=sum(c['source_available'] for c in cells),source_price_observations=253*required,momentum_estimates=required,block_return_estimates=12*required,new_eligible_cells=sum(len(e['new_eligible_indices']) for e in formal),momentum_excluded_cells=sum(len(e['excluded_momentum_indices']) for e in formal),shortlist_cells=sum(len(e['shortlist_indices']) for e in formal),enumerated_trios=len(all_trios),common_supported_trios=sum(len(e['common_trio_indices']) for e in formal),zero_joint_variance_trios=zero_joint,zero_diagonal_variance_trios=zero_diag,cash_events=branches['cash'],two_name_events=branches['two_names'],zero_variance_events=branches['zero_variance'],optimized_events=branches['optimized'],original_episode_calls=1,experimental_episode_calls=0)
    for arm,prefix in ((ARMS[1],'candidate'),(ARMS[2],'diagonal')):
        count[prefix+'_changed_holdings_events']=sum(set(orders[arm][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols']) for i in range(122))
    need(count==audit['counts'],'All31 counters derived from saved evidence')
    return count,dict(branch_counts=dict(branches),events=events,core_source_max_abs_error=max_error,price_sessions=253,return_intervals=252,block_count=12,block_sessions=21,required_domain='formal original Etrue, before new annual momentum ranking',source_counts_from_this_run_only=True,raw_sources_reloaded=False,rational_statistics_recomputed=False,basket_objectives_reranked=False,new_domain_explicit=True,volume_used=False)


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
    need(set(train)==set(ARMS),'Three fixed arms');required_metrics(train[ARMS[0]],True)
    guards={};indices={};undefined={};admissible=[]
    for candidate in ARMS[1:]:
        undefined[candidate]=required_metrics(train[candidate],False)
        ids=[r['index'] for r in changed[candidate+'_vs_'+ARMS[0]]['months'] if r['index']<88];indices[candidate]=ids
        guards[candidate]=dict(close='same_close_zero' not in undefined[candidate] and guard(train[candidate]['same_close_zero'],train[ARMS[0]]['same_close_zero'],.005),open='next_open_zero' not in undefined[candidate] and guard(train[candidate]['next_open_zero'],train[ARMS[0]]['next_open_zero'],0.),support=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids))
        if all(guards[candidate].values()):admissible.append(candidate)
    def order_key(arm):
        m=train[arm]['same_close_zero']['training'];return m['cagr'],m['sharpe'],arm==ARMS[2]
    primary=max(admissible,key=order_key) if admissible else ARMS[0]
    return dict(primary=primary,admitted=bool(admissible),admissible_arms=admissible,guards=guards,changed_months=indices,undefined_sharpe=undefined,first88_only=True,later34_used=False)


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
    return dict(prior='monthly_reference_gain250',all_incumbent_metrics_slices_exact=aggregate['cross_round_aggregate_bitwise_equal'],
      zero_cost_metrics_slices_exact=True,all_other_incumbent_metrics_slices_exact=True,daily_trades_positions_exact=True,holding_nonfee_fields_exact=True,
      cost_aggregate_reconciliation=aggregate,
      period_fee_exception=dict(absolute_bound=PERIOD_FEE_ATOL,scope='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance',
        inherited_scope='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception',evidence='work/monthly_reference_gain250_RESULT_REVIEW.json/incumbent_reproduction/period_fee_exception',actual_differences=exceptions))


def previous_incumbent(ev,results,exports,current_ind):
    review=ev.read(PREVIOUS_REVIEW,FIXED[str(PREVIOUS_REVIEW.relative_to(ROOT))])
    inherited=review['incumbent_reproduction']['period_fee_exception']
    need(review['passed'] is True and inherited['absolute_bound']==PERIOD_FEE_ATOL and inherited['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Existing prior fee exception only')
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
    need(not REPORT.exists(),'Report write-once')
    need(platform.python_version()=='3.9.6','Use pinned run_local wrapper')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h)
    context=ev.read(CONTEXT);source=ev.read(SOURCE);tests=ev.read(TESTS);code=ev.read(CODE_REVIEW)
    need(source['price_sessions']==253 and source['return_intervals']==252 and source['block_count']==12 and source['intervals_per_block']==21 and source['source_admitted'] is False and source['actual_cell_coverage']=='UNKNOWN_UNTIL_REGISTERED_PREPARE','Metadata alone did not admit new source')
    need(len(source['clock_events'])==122 and len(source['identity_cells'])==5,'Source metadata scope')
    need(plan['context']==context and plan['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))],'Registered exact context')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==140 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal140 pinned runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['tests']==140 and code['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))] and code['actual_returns_computed'] is False,'Final code review')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered10 code binding');ev.pin(p,h)
    need(len(tests['code_sha256'])==10,'All10 code files')
    for p,h in FIXED.items():need(ev.registered(plan,p)==h,'Registered contract/evidence binding')
    kernel=ROOT/'work/monthly_basket_core251_kernel.py';oracle=ROOT/'work/monthly_basket_core251_independent.py'
    schemas=constants(kernel,('EVENT_KEYS','CELL_KEYS','COUNT_FIELDS','TRIO_KEYS','WINNER_KEYS','AUDIT_EVENT_KEYS'))
    schemas.update(constants(ROOT/'work/monthly_basket_core251_math.py',('MEASURE_KEYS','BASKET_KEYS')))
    flags=constants(oracle,('SELECTION_FLAGS','PREPARE_FLAGS','SOURCE_GATE_FLAGS'))
    flags.update(constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',)))
    need([len(schemas[k]) for k in ('EVENT_KEYS','CELL_KEYS','COUNT_FIELDS')]==[28,12,31],'Frozen schema scope')
    for p in (ROOT/'work/ptr169_results_independent.py',ROOT/'work/monthly_ownsector_gate238_host.py'):
        ev.pin(p,ev.registered(plan,p))
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==246 and sum(x['trials'] for x in state['completed'])==1251 and state['registered_queue']==[],'Registered pre-state')
    protection=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protection)==plan['protected_files']==21444,'Protection manifest count only')
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Frozen selection before9path diagnostics')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed interval')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Fixed training splits')
    need(set(orders)==set(ARMS),'Three order arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Saved order axis')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) for o in orders[arm]),'Signal/3-2-cash saved orders')
    oldorders=ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json'));need(orders[ARMS[0]]==oldorders,'Original incumbent orders exact')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));originalplan=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    need(parity['raw_source_hashes']==source['raw_source_hashes']==originalplan['raw_source_hashes']==plan['raw_source_hashes'] and len(plan['raw_source_hashes'])==64,'Original64 manifest identity')
    need(parity['original_arrays_unchanged'] is True and parity['original_orders_exact'] is True and len(parity['original_array_fingerprints'])==27 and parity['source_review_sha256']==FIXED[str(SOURCE.relative_to(ROOT))],'Original27/source receipts')
    features=ev.read(OUT/'prepared/FEATURES.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    counts,src=feature_summary(features,audit,meta,calendar,source,orders,schemas)
    prep=preparation(e,pc,audit,flags,counts)
    changed={candidate+'_vs_'+ref:changes(orders,ref,candidate) for candidate,ref in ((ARMS[1],ARMS[0]),(ARMS[1],ARMS[2]),(ARMS[2],ARMS[0]))}
    selection=saved_selection(training,changed);need(selection==frozen['selection'],'Recheck first88 guards only')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Result policy labels')
    need(ind['accounting_source_sha256']==ev.pins['work/ptr169_results_independent.py'],'Accounting proof uses registered frozen source')
    ac,exports=accounting(ev,result,ind,training);inc=previous_incumbent(ev,result,exports,ind)
    ev.pin(ROOT/'work/basket_core251_saved_result_reader.py','fd1565142867e1681b60fae2885d33089648560638df874c758fc81ff95f09a3')
    ev.pin(ROOT/'work/test_basket_core251_saved_result_reader_v2.py')
    engine=ROOT/'outputs/monthly_factor_search_v2/validate.py';ev.pin(engine,ev.registered(plan,engine))
    ev.pin(Path(__file__))
    for p,h in ev.pins.items():need(digest(local(p))==h,'Input changed during review: '+p)
    summary={arm:{p:result['arms'][arm][p]['metrics'] for p in PATHS} for arm in ARMS}
    report=dict(schema='monthly_basket_core251_saved_result_review_v2',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reader_revision=dict(version=2,prior_reader_preserved=True,prior_reader_sha256='fd1565142867e1681b60fae2885d33089648560638df874c758fc81ff95f09a3',
        failed_attempt=dict(session_id=30503,exit_code=1,terminal_chunk='01b836',log='/private/tmp/basket_core251_saved_result_review.log',log_sha256='60a8affbc131a37dfd98ffd38f439b825ff15899d33288afebdd9bc30edeca6e',log_is_runtime_dependency=False,
          failed_assertion='Incumbent complete metrics exact to250',reason='V1 imposed cross-round bitwise equality on cost fees_paid beyond frozen financial reconciliation; only this cumulative display field and its full122 copy differ. Both ordered own-ledger exact sums and the cross-round independent certificate totals are checked in V2.'),
        change_scope='Only cost aggregate comparison refined; inherited per-period1e-10 bound unchanged. No aggregate tolerance, new strategy execution, or frozen input edits.'),
      reviewer_role='251 definition/code and saved-evidence reviewer; authored251metadata source/context builders and contracts but none of the10 new implementation/test files. Independent numerical and accounting attestations come from frozen251 oracle. No feature, basket-objective or return replay here.',
      scope='Pre-close saved-artifact hashes, source/control/direct-basket receipts, frozen first88 dual-arm guard arithmetic and recorded accounting/holdings comparisons only.',
      plan_sha256=args.plan_sha256,context_sha256=FIXED[str(CONTEXT.relative_to(ROOT))],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,
      formal_synthetic_tests=140,bindings=dict(code_files=10,prepared_files=8,result_files=38,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      preparation=prep,source_and_clock=src,selection=selection,changed_holdings=changed,accounting_review=ac,incumbent_reproduction=inc,full_results=summary,
      holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=246,registered_pre_state_trials=1251,expected_post_close_completed=247,expected_post_close_trials=1254,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,
      formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      limitations=['Cross-round cost fees_paid and its full122 copy need not be bitwise identical: each exactly equals its own saved chronological HOLDINGS fee sum, independent fee certificates agree exactly, and every other incumbent metric/slice and unchanged export is checked exactly.','Saved independent source/selection/account proofs are verified, not numerically regenerated.','Required253price coverage and source counts come from this251 saved receipt, not prior250 availability. Eoldfalse does not require a new window, but originalcore still uses full49 fullpast.','Adjusted-price vintage, estimated universe and other identity assumptions remain; metadata hashes do not certify economic source semantics.','Twelve21-session blocks are separately initialized descriptive holding scenarios, not statistically independent observations or a continuous252-session fixed-unit portfolio.','The new positive annual domain and complete basket selector jointly differ from Local67; joint-versus-diagonal is the specific risk cross-term contrast.','Both experiments could be admitted by first88; later34 and full history cannot change FROZEN.primary. The historical full-sample best is a separate descriptive close record.','Same-close zero-cost benchmark remains idealized; later34 is reused diagnostic history.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),selection=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},source_branches=src['branch_counts'],changed_month_counts={k:v['counts'] for k,v in changed.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())
