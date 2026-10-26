"""Review terminal250 saved evidence only, after root's explicit authorization.

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
OUT=ROOT/'outputs/monthly_reference_gain250'
PRIOR=ROOT/'outputs/monthly_volume_breadth249_v2'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_reference_gain250_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_reference_gain250_context.json'
SOURCE=ROOT/'work/reference_gain250_SOURCE_CONTRACT.json'
TESTS=ROOT/'work/monthly_reference_gain250_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_reference_gain250_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_volume_breadth249_v2_RESULT_REVIEW.json'
ARMS=('incumbent_local67','lagged_volume_reference_gain','lagged_equal_reference_gain_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={
 'work/monthly_reference_gain250_context.json':'5aa6a7a80455e226167dcb93f0f685f346c4d38e0d4682e67585d0f83b4e317a',
 'work/reference_gain250_SOURCE_CONTRACT.json':'85a277ec7de395790b4d8d82778df762e385de53f651a70b6e20980b5e6d9d19',
 'work/monthly_reference_gain250_TESTS.json':'da0af961f2d86dee31c7b7327d5e50d589d524f24e0e4448e2082f69979c9d0e',
 'work/monthly_reference_gain250_CODE_REVIEW.json':'365a43031967be9294279ee971448cbc18248d98f8a7d72567b20ec2610bcb6a',
 'work/monthly_volume_breadth249_v2_RESULT_REVIEW.json':'288c8535967ca2585727ba280f0aabeeebfa85163c2d4cd378ada410c9a8056a',
 'work/reference_gain250_METADATA_ERRATUM.json':'808c905d4e6c5289cf2ac793de6556dad0ee5cbfab0caf16e15e2b4528f2aa55',
 'work/after249_reference_gain250_definition_v2.json':'acc0c601fe2b3d7b039c380715ae7f4137ecf7ca4acecd9612a3465cf1a641c0',
 'work/reference_gain250_DEFINITION_REVIEW.json':'fd670979c37db2074efa69f3d97dfa114781af22a1564365b11e8cabb05e8411',
 'work/reference_gain250_MATH_REVIEW.json':'892d3b62d597679e8294027741383e693a7c38f61daa16c8e10527509e242655',
 'work/reference_gain250_SOURCE_REVIEW.json':'42c3cd0f5cb2548a154c8f53f6aa1c9a2a830d58647ea24dfa3951aacd5785d1',
 'work/volume_breadth249_ZERO_POLICY_REVIEW.json':'f87370db418f1f46aeafe77a298ce19a3891af63149024bae8dad929bdd600a1',
 'work/after248_volume_breadth_TRANSITION_LEDGER.json':'4b35ee2b539e83e4502741fbba2edefa41c874b2082d37f06f927f6098146db8'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
# Existing evidence: ptr169_results_independent.py:37-41,85-91 (saved-stat
# comparison); previous249_v2 RESULT_REVIEW.accounting_review.comparison_rule
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
    need(r['checked_events']==n and r['episode_calls']==0 and r['new_target_ranks_computed'] is False,'Raw-only source gate')
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
        need(r['full_features_supplied'] is True and r['original_core_checked'] is True and r['volume_clocks_checked'] is True,'Full source scope')
        need(r['C_source_atol']==1e-12 and r['C_source_rtol']==0 and r['continuous_atol']==2e-10 and r['continuous_rtol']==2e-10 and r['rank_tolerance']==0 and r['exact_discrete'] is True,'Full tolerances')
    else:need(r['numerical_tolerance']==0,'Selection exact tolerance')


def preparation(e,pc,audit,flags,counts):
    need(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Source/cache preservation')
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
    return dict(all123_cached_control_gates=True,all123_order_prefixes=True,physical_prefixes=list(PREFIXES),all_independent_flags_and_counts_checked=True,source_gate_episode_calls=0,source_gate_new_target_ranks_computed=False,source_before_incumbent_before_new_ranks_receipts=True,counts=counts)


def feature_summary(features,audit,meta,calendar,source,orders,schemas):
    need(len(features)==len(audit['events'])==122 and audit['schema']=='monthly_reference_gain250_decisions_v1','Full feature/audit scope')
    need(list(audit['counts'])==list(schemas['COUNT_FIELDS']),'Counts schema')
    axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar unique');pos={d:i for i,d in enumerate(axis)}
    cells=[];branches=Counter();zeros=[];fallback=[];max_error=0.
    for i,e in enumerate(features):
        need(tuple(e)==schemas['EVENT_KEYS'] and e['query_index']==i and e['date']==meta['dates'][i] and e['symbols']==meta['symbols'][i],'Event identity')
        t=pos[e['date']];need(e['query_position']==t and e['next_official_session']==axis[t+1],'Official query/witness')
        need(e['price_start']==e['volume_start']==axis[t-294] and e['price_end']==axis[t-42] and e['volume_end']==axis[t-43],'253/252 clocks; reference volume excluded')
        clock=source['clock_events'][i]
        need(all(clock[k]==e[k] for k in ('query_index','date','initial_partial','symbols','query_position','next_official_session','price_start','price_end','volume_start','volume_end')),'Source contract clock identity')
        need(clock['price_count']==253 and clock['volume_count']==clock['historical_count']==252 and clock['reference_volume_consumed'] is False and clock['reference_date']==e['price_end'],'Source contract reference scope')
        need(e['panel_first_session']=='2008-01-02','Full-history source origin')
        partial=e['date']=='2016-08-17';need(e['initial_partial'] is partial,'Named partial')
        if not partial:need(e['date'][:7]!=axis[t+1][:7],'Formal month end')
        a=audit['events'][i];need(a['query_index']==i and a['date']==e['date'] and a['branch']==e['branch'] and set(a['arms'])==set(ARMS),'Saved decision event')
        branches[e['branch']]+=1
        if partial:
            need(e['cells']==[] and e['branch']=='initial_partial' and e['zero_indices']==e['missing_indices']==[],'Partial no source measurements')
        else:
            need(len(e['cells'])==49 and len(set(e['symbols']))==49,'Full49 source population')
            missing=[];zero=[];u=[]
            for j,c in enumerate(e['cells']):
                need(tuple(c)==schemas['CELL_KEYS'] and c['column_index']==j and c['symbol']==e['symbols'][j],'Cell identity')
                need(type(c['original_finite']) is bool and type(c['source_available']) is bool,'Cell bools')
                if c['original_finite']:u.append(j)
                need(finite(c['core_abs_error']) and 0<=c['core_abs_error']<=1e-12,'Saved original core source bound');max_error=max(max_error,c['core_abs_error'])
                if c['price_gaps'] or c['volume_gaps']:missing.append(j)
                if c['volume_zeros']:zero.append(j)
                for field,limit,start in (('price_gaps',253,t-294),('volume_gaps',252,t-294),('volume_zeros',252,t-294)):
                    offsets=[]
                    for z in c[field]:
                        q=z['offset'];need(type(q) is int and 0<=q<limit and z['date']==axis[start+q],'Exact saved slot clock');offsets.append(q)
                        if field=='volume_zeros':
                            need(type(z['value']) is float and z['value']==0 and z['reason']=='unresolved_zero_volume','Native zero audit')
                            zeros.append(dict(query_index=i,date=e['date'],column_index=j,symbol=c['symbol'],source_date=z['date'],offset=q,value=z['value']))
                    need(offsets==sorted(set(offsets)),'Unique saved source slots')
                need(not ({x['offset'] for x in c['volume_gaps']}&{x['offset'] for x in c['volume_zeros']}),'Zero not fake missing')
                good=not(c['volume_zeros'] or c['volume_gaps'] or c['price_gaps'])
                need(c['source_available'] is good,'Common availability union')
                if c['volume_zeros']:need(c['measurement'] is None,'Zero no fake measurement')
                else:
                    m=c['measurement']
                    need(tuple(m)==schemas['MEASURE_KEYS'] and m['schema']=='monthly_reference_gain250_math_v1','Saved exact-reference schema')
                    need(m['price_count']==253 and m['volume_count']==m['historical_count']==252 and m['available'] is good,'Saved source measure scope/status')
                    need(m['missing_price_indices']==[z['offset'] for z in c['price_gaps']] and m['missing_volume_indices']==[z['offset'] for z in c['volume_gaps']],'Saved measurement missing slots')
                    need(m['reason']==('available' if good else 'missing_source'),'Saved measure reason')
                    for key in schemas['MEASURE_KEYS'][8:]:
                        value=m[key]
                        if good:
                            need(type(value) is dict and set(value)=={'numerator','denominator'} and all(type(v) is str for v in value.values()),'Saved rational pair metadata')
                            num,den=int(value['numerator']),int(value['denominator'])
                            need(str(num)==value['numerator'] and str(den)==value['denominator'] and den>0 and math.gcd(num,den)==1,'Canonical saved rational pair')
                        else:need(value is None,'No unavailable raw aggregate')
                cells.append(c)
            need(e['missing_indices']==missing and e['zero_indices']==zero and e['u0_indices']==u,'Saved source indices')
            need(e['core_source_max_abs_error']==max(c['core_abs_error'] for c in e['cells']),'Saved max core error')
            branch='zero_volume' if zero else 'missing_source' if missing else 'known_transition' if e['transition_ids'] else 'available'
            need(e['branch']==branch,'Shared branch precedence')
        need(e['transition_ids']==[],'Frozen eligible transition list empty')
        active=e['branch']=='available'
        for rankname in ('volume_rank','equal_rank'):
            rank=e[rankname]
            if active:need(rank['width']==49 and all(len(rank[k])==49 for k in ('less','equal','targets')),'Full49 rank receipt')
            else:need(rank is None,'No inactive target ranks')
        for k,arm in enumerate(ARMS):
            au=a['arms'][arm];on=active and k>0
            need(au['active'] is on and au['target_consumed'] is on,'Saved target consumption')
            need(len(au['correction'])==49 and len(au['original_support'])==49,'Saved full score support')
            need(type(au['calm']) is bool and type(au['weak']) is bool and au['calm'] is e['calm'],'Saved funding state')
            cw=.50 if e['calm'] else .52
            cap=max(0.,.10-(.025*au['calm']+.05*au['weak']));e0=min(.05,cap)
            coefficient=.02/(1.-cw) if on else 0.;alpha=1.-coefficient
            requested=alpha*e0 if on else .05
            need(au['core_weight']==e['core_weight']==cw and au['funding_budget']==(.02 if on else 0.),'Fixed .02 funding and original core weight')
            need(au['funding_coefficient']==coefficient and au['alpha']==alpha and au['original_episode_budget']==e0 and au['requested_episode_budget']==requested and au['effective_episode_budget']==min(requested,cap),'Exact saved funding/request/effective algebra')
            if on:
                rank=e['volume_rank' if k==1 else 'equal_rank']
                need(au['target']==rank['targets'] and all(finite(v) and -.5<=v<=.5 for v in au['target']),'Active full49 target binding')
            if not on:need(au['target'] is None and au['funding_budget']==0 and au['requested_episode_budget']==.05 and all(v==0 for v in au['correction']),'Inactive funding exact bypass')
            if not active:need(orders[arm][i]==orders[ARMS[0]][i],'Inactive orders exact incumbent')
        if not active:fallback.append(dict(index=i,date=e['date'],branch=e['branch'],zero_symbols=[e['symbols'][j] for j in e['zero_indices']],missing_symbols=[e['symbols'][j] for j in e['missing_indices']]))
    count={k:0 for k in schemas['COUNT_FIELDS']};formal=[e for e in features if not e['initial_partial']];good=sum(c['source_available'] for c in cells)
    count.update(events=122,symbols_per_event=49,initial_partial_events=122-len(formal),formal_events=len(formal),calm_formal_events=sum(e['calm'] for e in formal),noncalm_formal_events=sum(not e['calm'] for e in formal),original_checked_cells=len(cells),original_finite_cells=sum(c['original_finite'] for c in cells),source_checked_cells=len(cells),source_available_cells=good,source_missing_cells=sum(bool(c['price_gaps'] or c['volume_gaps']) for c in cells),source_zero_volume_cells=sum(bool(c['volume_zeros']) for c in cells),source_unavailable_cells=len(cells)-good,active_events=branches['available'],zero_volume_events=branches['zero_volume'],missing_source_events=branches['missing_source'],known_transition_events=branches['known_transition'],source_price_observations=253*len(cells),source_volume_observations=252*len(cells),source_zero_volume_observations=len(zeros),weighted_reference_estimates=good,equal_reference_estimates=good,ranked_cells=98*branches['available'],original_episode_calls=3)
    for arm,prefix in ((ARMS[1],'candidate'),(ARMS[2],'control')):
        count[prefix+'_nonzero_correction_cells']=sum(v!=0 for e in audit['events'] for v in e['arms'][arm]['correction'])
        count[prefix+'_changed_holdings_events']=sum(set(orders[arm][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols']) for i in range(122))
    need(count==audit['counts'],'All30 counters from saved evidence')
    return count,dict(branch_counts=dict(branches),fallback_events=fallback,zero_slots=zeros,unique_zero_observations=len({(z['symbol'],z['source_date']) for z in zeros}),core_source_max_abs_error=max_error,full49_before_eligibility=True,price_sessions=253,volume_sessions=252,reference_volume_consumed=False,availability_from_this_run_only=True,source_values_reloaded=False,no_zero_semantics_certificate=True)


def changes(orders,reference,candidate):
    rows=[]
    for i,(a,b) in enumerate(zip(orders[reference],orders[candidate])):
        if set(a['symbols'])!=set(b['symbols']):rows.append(dict(index=i,date=a['date'],end=a['end'],reference=a['symbols'],candidate=b['symbols'],removed=sorted(set(a['symbols'])-set(b['symbols'])),added=sorted(set(b['symbols'])-set(a['symbols']))))
    return dict(counts={s:sum(lo<=r['index']<hi for r in rows) for s,(lo,hi) in SLICES.items()},months=rows,by_year=dict(Counter(r['date'][:4] for r in rows)))


def guard(c,r,gain):
    return c['training']['cagr']>=r['training']['cagr']+gain-1e-12 and c['training']['sharpe']>=r['training']['sharpe']-.05 and c['training']['max_drawdown']>=r['training']['max_drawdown']-.03 and all(c[e]['cagr']>=r[e]['cagr']-.025 for e in ('early','recent'))


def saved_selection(train,changed):
    guards={};indices={}
    for ref in (ARMS[0],ARMS[2]):
        ids=[r['index'] for r in changed[ARMS[1]+'_vs_'+ref]['months'] if r['index']<88];indices[ref]=ids
        guards[ref]=dict(close=guard(train[ARMS[1]]['same_close_zero'],train[ref]['same_close_zero'],.005 if ref==ARMS[0] else 0.),open=guard(train[ARMS[1]]['next_open_zero'],train[ref]['next_open_zero'],0.),support=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids))
    admitted=all(all(v.values()) for v in guards.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months=indices,first88_only=True,later34_used=False)


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


def previous_incumbent(ev,results,exports):
    review=ev.read(PREVIOUS_REVIEW,FIXED[str(PREVIOUS_REVIEW.relative_to(ROOT))])
    inherited=review['incumbent_reproduction']['period_fee_exception']
    need(inherited['absolute_bound']==PERIOD_FEE_ATOL and inherited['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Existing prior fee exception only')
    previous_completion=ev.read(PRIOR/'results/COMPLETION.json',review['results_completion_sha256'])
    previous=ev.read(PRIOR/'results/RESULTS.json',previous_completion['artifacts_sha256']['RESULTS.json'])
    exceptions=[]
    for p in PATHS:
        for key in ('metrics','slices'):
            need(results['arms'][ARMS[0]][p][key]==previous['arms'][ARMS[0]][p][key],'Incumbent complete '+key+' exact to249_v2')
        for kind in KINDS:
            name=ARMS[0]+'_'+p+'_'+kind+'.json';before=ev.read(PRIOR/'results'/name,previous_completion['artifacts_sha256'][name]);after=exports[ARMS[0],p][kind]
            if kind!='HOLDINGS':need(before==after,'Incumbent saved '+kind+' exact');continue
            need(len(before)==len(after)==122,'Full incumbent holdings')
            for i,(a,b) in enumerate(zip(before,after)):
                need(set(a)==set(b) and {k:v for k,v in a.items() if k!='fee'}=={k:v for k,v in b.items() if k!='fee'},'Holding nonfee fields exact')
                need(finite(a['fee']) and finite(b['fee']),'Finite period fees')
                if a['fee']!=b['fee']:
                    d=b['fee']-a['fee'];need(p=='next_open_cost' and abs(d)<=PERIOD_FEE_ATOL,'Only inherited cost period fee exception')
                    exceptions.append(dict(path=p,index=i,signal_date=b['signal_date'],previous=a['fee'],current=b['fee'],difference=d))
    return dict(prior='monthly_volume_breadth249_v2',all_incumbent_metrics_slices_exact=True,daily_trades_positions_exact=True,holding_nonfee_fields_exact=True,period_fee_exception=dict(absolute_bound=PERIOD_FEE_ATOL,scope='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception',evidence='work/monthly_volume_breadth249_v2_RESULT_REVIEW.json/incumbent_reproduction/period_fee_exception',actual_differences=exceptions))


def run(args):
    need(not REPORT.exists(),'Report write-once')
    need(platform.python_version()=='3.9.6','Use pinned run_local wrapper')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h)
    context=ev.read(CONTEXT);source=ev.read(SOURCE);tests=ev.read(TESTS);code=ev.read(CODE_REVIEW)
    erratum=ev.read(ROOT/'work/reference_gain250_METADATA_ERRATUM.json')
    need(erratum['status']=='CLARIFIED_BEFORE_REGISTRATION' and erratum['corrections'][0]['preserved_original_text']==source['listing_scope'],'Bound descriptive clock erratum')
    need(source['price_sessions']==253 and source['volume_sessions']==source['historical_sessions']==252 and source['reference_volume_consumed'] is False and source['source_admitted'] is False and source['actual_cell_coverage']=='UNKNOWN_UNTIL_REGISTERED_PREPARE','Metadata alone did not admit new source')
    need(len(source['clock_events'])==122 and len(source['identity_cells'])==5,'Source metadata scope')
    need(plan['context']==context and plan['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))],'Registered exact context')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==206 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal206 pinned runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['tests']==206 and code['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))] and code['actual_returns_computed'] is False,'Final code review')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered12 code binding');ev.pin(p,h)
    need(len(tests['code_sha256'])==12,'All12 code files')
    for p,h in FIXED.items():need(ev.registered(plan,p)==h,'Registered contract/evidence binding')
    kernel=ROOT/'work/monthly_reference_gain250_kernel.py';oracle=ROOT/'work/monthly_reference_gain250_independent.py'
    schemas=constants(kernel,('EVENT_KEYS','CELL_KEYS','COUNT_FIELDS'))
    schemas.update(constants(ROOT/'work/monthly_reference_gain250_math.py',('MEASURE_KEYS',)))
    flags=constants(oracle,('SELECTION_FLAGS','PREPARE_FLAGS','SOURCE_GATE_FLAGS','INCUMBENT_GATE_FLAGS'))
    need([len(schemas[k]) for k in ('EVENT_KEYS','CELL_KEYS','COUNT_FIELDS')]==[27,14,30],'Frozen schema scope')
    for p in (ROOT/'work/ptr169_results_independent.py',ROOT/'work/monthly_ownsector_gate238_host.py'):
        ev.pin(p,ev.registered(plan,p))
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==245 and sum(x['trials'] for x in state['completed'])==1248 and state['registered_queue']==[],'Registered pre-state')
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
    ac,exports=accounting(ev,result,ind,training);inc=previous_incumbent(ev,result,exports)
    ev.pin(Path(__file__))
    for p,h in ev.pins.items():need(digest(local(p))==h,'Input changed during review: '+p)
    summary={arm:{p:result['arms'][arm][p]['metrics'] for p in PATHS} for arm in ARMS}
    report=dict(schema='monthly_reference_gain250_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='250 definition/code and saved-evidence reviewer; authored predecessor249 adapter and reviews, none of the12 new250 code/test files. Independent numerical and accounting attestations come from frozen250 oracle. No new feature, model or account replay here.',
      scope='Pre-close saved-artifact hashes, source/control/zero-fallback receipts, frozen first88 guard arithmetic and recorded accounting/holdings comparisons only.',
      plan_sha256=args.plan_sha256,context_sha256=FIXED[str(CONTEXT.relative_to(ROOT))],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,
      formal_synthetic_tests=206,bindings=dict(code_files=12,prepared_files=8,result_files=38,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      metadata_erratum=dict(path='work/reference_gain250_METADATA_ERRATUM.json',sha256=FIXED['work/reference_gain250_METADATA_ERRATUM.json'],structured253price252volume_clocks_checked=True),preparation=prep,source_and_clock=src,selection=selection,changed_holdings=changed,accounting_review=ac,incumbent_reproduction=inc,full_results=summary,
      holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=245,registered_pre_state_trials=1248,expected_post_close_completed=246,expected_post_close_trials=1251,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,
      formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      limitations=['Saved independent proofs are verified, not numerically regenerated.','Zero economic semantics and mixed stored-volume continuity remain UNKNOWN; this review is not source-semantic certification.','Whole-event zero/missing fallback and actual support are derived only from this250 saved feature receipts; prior249 availability counts are not copied.','Auxiliary excludes reference-day volume and latest42 sessions; the unchanged original risk/core clock and full-file hashes retain their own scope.','Both funded arms use.02; constant or all-tied targets still fund old auxiliaries and are not incumbent-neutral.','Same-close zero-cost primary remains idealized; later34 is reused diagnostic history.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),selection=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},source_branches=src['branch_counts'],changed_month_counts={k:v['counts'] for k,v in changed.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())
