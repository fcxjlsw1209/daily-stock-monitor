"""Saved review of terminal254 saved evidence after explicit authorization.

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
OUT=ROOT/'outputs/monthly_calendar_memory254'
PRIOR=ROOT/'outputs/monthly_leader_bridge253'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_calendar_memory254_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_calendar_memory254_context.json'
TESTS=ROOT/'work/monthly_calendar_memory254_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_calendar_memory254_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_leader_bridge253_RESULT_REVIEW.json'
ARMS=('incumbent_local67','calendar_memory_local67','fixed21_memory_control')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.npz','SOURCE_AUDIT.json','EFFECTS.npz','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={"work/monthly_calendar_memory254_context.json":"a94bf1b03532305fcc58e12a277cea08fddb0b8830c2e8d45781bdfb3af636ae","work/monthly_leader_bridge253_RESULT_REVIEW.json":"171583782039238103cf0daf4d34fd196d0c00873d01934ba655fb2f78458f86","work/monthly_calendar_clock248_v2_independent.py":"c7d172a9abec73f81d1d07ce8a422cd709a4a9bfebd55e6211b471c8d315663c","work/ptr169_results_independent.py":"6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59","work/monthly_ownsector_gate238_host.py":"94f70ab8c0d9fc1cd891bf1ce3eaa30212a54ef4f9cdb1eb4e237bd82bed30a5","work/monthly_calendar_memory254_TESTS.json":"be6f235500ab01515722429f95188596fb927a4435cf0098ed1439ebba73c9de","work/monthly_calendar_memory254_CODE_REVIEW.json":"e2f39cc439d53a69c9c798b5b8af6c501badf5212cc3ef3b1cb5461d2d314090"}
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


def canonical_source_map(mapping):
    """Path spelling equivalence only; exactly64 complete path/digest pairs.

    No files are opened, no basename/suffix matching, and no changed digest
    can pass. Reject ambiguous aliases even when their digests agree.
    """
    need(type(mapping) is dict and len(mapping)==64,'Exactly64 source-map entries')
    result={}
    for key,value in mapping.items():
        need(type(key) is str and bool(key),'Nonempty source-map path')
        raw=Path(key)
        need('CloudStorage' not in raw.parts and '..' not in raw.parts,'Forbidden raw source-map path')
        need(not raw.is_absolute() or raw.is_relative_to(ROOT),'Absolute source path must begin at current research root')
        path=local(key)
        need(path!=ROOT,'Source-map key is a file path')
        canonical=str(path.relative_to(ROOT))
        need(canonical not in result,'Canonical source-map collision')
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Canonical bare SHA256 source value')
        result[canonical]=value
    return result

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



def digest_record(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha_value(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def source_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Source gate flags')
    need(r.get('checked_events')==n and r.get('episode_calls')==0 and r.get('memory_effects_computed') is False and r.get('candidate_selection_computed') is False,'Source before memory and selection')
    need(r.get('full_source_audit_supplied') is True and r.get('continuous_atol')==2e-10 and r.get('continuous_rtol')==2e-10 and r.get('exact_discrete') is True,'Full source scope')
    need(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Independent source isolation')
    need(r['counts']==r['source_audit']['counts'] and r['counts']['queries']==n and sha_value(r['source_data_sha256']),'Source count and data binding')


def incumbent_gate(r,n,flags):
    need(all(r.get(k) is True for k in flags),'Incumbent gate flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==n and r['numerical_tolerance']==0,'Incumbent exact scope')
    need(r['full_scores_supplied'] is True and r['full_pre_scores_supplied'] is True,'Complete incumbent scores')
    need(all(r[k] is False for k in ('producer_modules_imported','computes_returns','candidate_selection_computed')),'Incumbent scope')


def pre_gates(r,n,flags,cached=False,full_source=None):
    need(set(r)=={'source','incumbent'},'Exact source and incumbent gates')
    if cached:
        s=r['source']
        need(s.get('passed') is True and s.get('cached_source_reference') is True and s.get('no_new_source_rebuild_claim') is True and s.get('checked_events')==n and s.get('full_source_checked_events')==122,'Cached source scope')
        need(full_source is not None and s.get('full_source_receipt_sha256')==digest_record(full_source) and s.get('full_source_data_sha256')==full_source['source_data_sha256'],'Exact cached full source binding')
    else:source_gate(r['source'],n,flags['SOURCE_FLAGS'])
    incumbent_gate(r['incumbent'],n,flags['INCUMBENT_GATE_FLAGS'])


def selection_check(r,n,flags):
    need(all(r.get(k) is True for k in flags['SELECTION_FLAGS']),'Independent selection flags')
    need(r['checked_events']==n and r['ordered_choices_checked']==3*n and r['numerical_tolerance']==0,'All exact orders/events checked')
    need(all(r[k] is True for k in ('full_scores_supplied','full_pre_scores_supplied','full_effects_supplied','full_audit_supplied')),'All full score/pre/effect/audit evidence')
    need(r['producer_modules_imported'] is False and r['computes_returns'] is False,'Independent selection scope')


def preparation(e,pc,audit,source_audit,parity,flags):
    need(all(e.get(k) is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_input_unchanged','original_full49_pre_preserved')),'Original source/cache preservation')
    need(e['new_predictive_model_fits']==pc['new_model_fits']==pc['new_predictive_model_fits']==0 and pc['new_returns_computed'] is False and pc['events']==122,'No fit/portfolio-return preparation')
    pre_gates(e['pre_candidate_gates'],122,flags)
    full=e['pre_candidate_gates']['source']
    need(full['source_audit']==source_audit and full['counts']==source_audit['counts'],'Full source audit exact')
    need(full['source_data_sha256']==parity['source_data_sha256'] and digest_record(source_audit)==parity['source_audit_sha256'],'Saved source digest and full certificate')
    selection_check(e['selection'],122,flags)
    need(e['selection']['counts']==audit['counts']==pc['descriptive_counts'],'Complete decision counts binding')
    need([x['prefix'] for x in e['all123_cached_control_gates']]==list(range(123)),'All123 cached prefixes')
    for x in e['all123_cached_control_gates']:
        need(x['exact'] is True,'Exact cached predictions')
        pre_gates(x['gates'],x['prefix'],flags,True,full)
    need([x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'Five physical prefixes')
    for x in e['physical_prefixes']:
        n=x['prefix'];need(x['exact'] is True,'Exact physical predictions')
        pre_gates(x['gates'],n,flags);selection_check(x['selection'],n,flags)
        need(x['selection']['counts']==x['counts'] and x['counts']['events']==n,'Physical count binding')
    return dict(all123_cached_control_gates=True,all123_order_prefixes=True,physical_prefixes=list(PREFIXES),all_independent_flags_and_counts_checked=True,
      source_before_original_incumbent_before_new_memory=True,source_data_sha256=full['source_data_sha256'],source_audit_sha256=parity['source_audit_sha256'],
      source_data_digest_recomputed_from_npz=False,source_digest_scope='Exact saved parity/certificate binding; NPZ bytes separately hash checked, never decoded.',counts=audit['counts'])


def schema_flags():
    flags=constants(ROOT/'work/monthly_calendar_memory254_independent.py',('SOURCE_FLAGS','SELECTION_FLAGS'))
    flags.update(constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',)))
    schemas=constants(ROOT/'work/monthly_calendar_memory254_kernel.py',('SOURCE_COUNT_FIELDS','COUNT_FIELDS','AUDIT_EVENT_KEYS'))
    need([len(schemas[k]) for k in ('SOURCE_COUNT_FIELDS','COUNT_FIELDS','AUDIT_EVENT_KEYS')]==[15,21,10],'254 source/decision schema15/21/10')
    return schemas,flags


def source_summary(audit,meta,schemas):
    need(set(audit)=={'schema','counts','anchor_rule','common_maturity_rule','numeric_cutoff','candidate_selection_computed','portfolio_returns_computed'},'Source audit schema')
    need(audit['schema']=='monthly_calendar_memory254_features_v1' and audit['anchor_rule']=='official_monthends_original_good_positive_signal_no_lockout' and audit['common_maturity_rule']=='maximum_of_actual_exits','Frozen source definitions')
    need(audit['numeric_cutoff']==meta['dates'][-1] and audit['candidate_selection_computed'] is False and audit['portfolio_returns_computed'] is False,'Source audit cutoff/scope')
    c=audit['counts'];need(tuple(c)==schemas['SOURCE_COUNT_FIELDS'] and all(type(v) is int and v>=0 for v in c.values()),'15 nonnegative source counts')
    need(c['queries']==122 and c['symbols_per_query']==49 and c['securities']==len(set(s for names in meta['symbols'] for s in names)) and c['query_cells']==122*49,'Source query axes')
    need(c['initial_partial_queries']==1 and c['formal_queries']==121 and c['new_model_fits']==c['predictive_model_fits']==0,'Source partial/no fits')
    need(c['common_valid_anchors']+c['common_invalid_anchors']==c['anchors'] and c['common_valid_anchors']<=min(c['valid21_anchors'],c['valid_calendar_anchors'])<=max(c['valid21_anchors'],c['valid_calendar_anchors'])<=c['anchors'],'Common path counts')
    need(max(c['finite_query_cells'],c['positive_signal_query_cells'])<=c['query_cells'],'Query count bounds')
    return dict(counts=c,source_numeric_recomputed=False,full_source_certificate_checked=True,
      anchor_rows_and_future_labels_not_decoded=True,labels_are_descriptive_historical_returns_not_portfolio_replay=True)


def index_list(value,width,label,ordered=False):
    need(type(value) is list and all(type(j) is int and 0<=j<width for j in value) and len(value)==len(set(value)),label)
    if not ordered:need(value==sorted(value),label+' sorted')


def decision_summary(audit,source_audit,meta,orders,schemas):
    need(audit['schema']=='monthly_calendar_memory254_decisions_v1' and len(audit['events'])==122,'Decision scope')
    c=audit['counts'];need(tuple(c)==schemas['COUNT_FIELDS'] and all(type(v) is int and v>=0 for v in c.values()),'21 nonnegative decision counts')
    total_E=formal_E=0;changes_count=[0,0,0]
    for i,e in enumerate(audit['events']):
        need(tuple(e)==schemas['AUDIT_EVENT_KEYS'] and e['query_index']==i and e['date']==meta['dates'][i],'Decision event identity')
        partial=i==0;need(e['initial_partial'] is partial and (not partial or e['date']=='2016-08-17'),'Named partial')
        ids=e['original_eligible_indices'];rank=e['pre_order'];j=e['boundary_candidates']
        index_list(ids,49,'Original eligible');index_list(rank,49,'Original pre order',True);index_list(j,49,'Boundary',True)
        need(set(ids)==set(rank),'All original eligible pre ranks')
        need((not j if len(rank)<3 else bool(j) and j[0]==rank[2] and j==[v for v in rank[2:5] if v in j]),'Original rank3to5 boundary')
        need(finite(e['episode_budget']) and 0<=e['episode_budget']<=.05,'Original effective episode budget')
        total_E+=len(ids);formal_E+=0 if partial else len(ids)
        choices=[e[k] for k in ('incumbent_choices','candidate_choices','control_choices')]
        for arm,ch in zip(ARMS,choices):
            index_list(ch,49,'Chosen indices',True)
            need(len(ch)==(3 if len(ids)>=3 else 2 if len(ids)==2 else 0) and set(ch)<=set(ids),'Original E and3/2/cash')
            if len(ids)>=3:need(set(rank[:2])<=set(ch) and len(set(ch)&set(j))==1,'Locked top2/one boundary winner')
            need(orders[arm][i]['symbols']==[meta['symbols'][i][v] for v in ch],'Saved choice/order identity')
            if partial:need(orders[arm][i]==orders[ARMS[0]][i],'Partial exact incumbent')
        for k,(a,b) in enumerate(((0,1),(0,2),(1,2))):changes_count[k]+=set(choices[a])!=set(choices[b])
    need(c['events']==122 and c['symbols_per_event']==49 and c['initial_partial_events']==1 and c['formal_events']==121 and c['original_eligible_cells']==total_E,'Decision count axes')
    need(c['anchors']==source_audit['counts']['anchors'] and c['mature_common_anchors']<=source_audit['counts']['common_valid_anchors'],'Anchor counts')
    need(c['query_signal_cells']<=121*49 and c['query_finite_cells']<=121*49,'Formal query count bounds')
    need(all(c[k]<=formal_E for k in ('calendar_nonzero_effect_cells','fixed21_nonzero_effect_cells','calendar_evidence_cells','fixed21_evidence_cells')),'Eligible effect/evidence count bounds')
    need([c[k] for k in ('candidate_changed_holdings_events','control_changed_holdings_events','candidate_vs_control_changed_holdings_events')]==changes_count,'Saved changed-set counts')
    need(all(c[k]==1 for k in ('original_episode_calls','candidate_episode_calls','control_episode_calls')) and c['new_model_fits']==c['predictive_model_fits']==0,'Exactly once each/no model fit')
    return dict(counts=c,all122_saved_event_order_domains_checked=True,all49_pre_and_effect_arrays_certified_by_independent=True,npz_arrays_decoded=False)

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
    candidate=ARMS[1]
    ids=[r['index'] for r in changed[candidate+'_vs_'+ARMS[0]]['months'] if r['index']<88]
    guards=dict(close='same_close_zero' not in undefined[candidate] and guard(train[candidate]['same_close_zero'],train[ARMS[0]]['same_close_zero'],.005),open='next_open_zero' not in undefined[candidate] and guard(train[candidate]['next_open_zero'],train[ARMS[0]]['next_open_zero'],0.),support=len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids))
    c=train[candidate];d=train[ARMS[2]]
    guards['calendar_vs_fixed21']=('same_close_zero' not in undefined[ARMS[2]] and 'same_close_zero' not in undefined[candidate]
        and all(c[p]['training']['cagr']>=d[p]['training']['cagr']-1e-12 for p in PATHS[:2])
        and c['same_close_zero']['training']['sharpe']>=d['same_close_zero']['training']['sharpe'])
    admitted=all(guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months=ids,undefined_sharpe=undefined,diagnostic_control_not_selectable=ARMS[2],first88_only=True,later34_used=False)


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
                need(all(finite(v) and v==0 for v in (saved['metrics']['fees_paid'],saved['slices']['full122']['fees_paid'],cert['fees_paid'])),'Zero-fee aggregate and full122 exact0')
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
    return dict(all_incumbent_metrics_slices_exact=aggregate['cross_round_aggregate_bitwise_equal'],
      zero_cost_metrics_slices_exact=True,all_other_incumbent_metrics_slices_exact=True,daily_trades_positions_exact=True,holding_nonfee_fields_exact=True,
      cost_aggregate_reconciliation=aggregate,
      period_fee_exception=dict(absolute_bound=PERIOD_FEE_ATOL,scope='ONLY next_open_cost individual HOLDINGS.fee; cumulative fields require exact own-ledger accumulation and exact cross-round independent certificates, not a tolerance',
        inherited_scope='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception',evidence='work/monthly_reference_gain250_RESULT_REVIEW.json/incumbent_reproduction/period_fee_exception',actual_differences=exceptions))


def previous_reference(ev,results,exports,current_ind,prior,review_path,current_arm,reference_arm):
    review=ev.read(review_path,FIXED[str(review_path.relative_to(ROOT))])
    need(review['passed'] is True and review['unresolved_blockers']==[],'Reference independently accepted')
    previous_completion=ev.read(prior/'results/COMPLETION.json',review['results_completion_sha256'])
    previous=ev.read(prior/'results/RESULTS.json',previous_completion['artifacts_sha256']['RESULTS.json'])
    previous_ind=ev.read(prior/'results/INDEPENDENT_OUTCOME.json',previous_completion['artifacts_sha256']['INDEPENDENT_OUTCOME.json'])
    need(previous_ind['passed'] is True and current_ind['passed'] is True and previous_ind['accounting_source_sha256']==current_ind['accounting_source_sha256'],'Same passed frozen accounting source')
    prior_exports={};zero_hashes=[];cost_exact_hashes=[]
    for p in PATHS:
        prior_exports[p]={}
        for kind in KINDS:
            before_name=reference_arm+'_'+p+'_'+kind+'.json';after_name=current_arm+'_'+p+'_'+kind+'.json'
            prior_exports[p][kind]=ev.read(prior/'results'/before_name,previous_completion['artifacts_sha256'][before_name])
            if p!='next_open_cost' or kind!='HOLDINGS':
                need(ev.pins[str((prior/'results'/before_name).relative_to(ROOT))]==ev.pins[str((OUT/'results'/after_name).relative_to(ROOT))],'Reference unchanged export bytes: '+after_name)
                (zero_hashes if p!='next_open_cost' else cost_exact_hashes).append(after_name)
    check=compare_incumbent_saved(previous['arms'][reference_arm],results['arms'][current_arm],prior_exports,{p:exports[current_arm,p] for p in PATHS},previous_ind['paths'][reference_arm],current_ind['paths'][current_arm])
    check.update(prior=prior.name,current_arm=current_arm,reference_arm=reference_arm,zero_cost_export_sha_exact=zero_hashes,cost_daily_trades_positions_sha_exact=cost_exact_hashes)
    return check



def run(args):
    need(not REPORT.exists(),'Report write-once');need(platform.python_version()=='3.9.6','Use pinned run_local wrapper')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h);need(ev.registered(plan,p)==h,'Registered frozen evidence '+p)
    context=ev.read(CONTEXT);tests=ev.read(TESTS);code=ev.read(CODE_REVIEW)
    need(plan['context']==context and plan['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))],'Registered exact context')
    need(context['arms']==list(ARMS) and context['promotable_arms']==[ARMS[1]] and context['registered_trials']==3,'Calendar-only promotable')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==78 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal78 runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['tests']==78 and code['code_sha256']==tests['code_sha256'] and code['context_sha256']==FIXED[str(CONTEXT.relative_to(ROOT))] and code['actual_returns_computed'] is False,'Prospective code review')
    need(len(tests['code_sha256'])==6,'All6 frozen code files')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered6code');ev.pin(p,h)
    schemas,flags=schema_flags()
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==249 and sum(x['trials'] for x in state['completed'])==1260 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Prestate249/1260 no pending')
    protection=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protection)==plan['protected_files']==21444,'Protected manifest count only')
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Frozen selection before diagnostics')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed interval')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Training split clocks')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']),'Full49 query identities')
    need(set(orders)==set(ARMS),'Exactly3 order arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order date axis')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(s) for s in o['scores']) for o in orders[arm]),'3/2/cash finite saved orders')
    need(orders[ARMS[0]]==ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Exact Local67 incumbent orders')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions']
    need(axis==sorted(set(axis)),'Official calendar metadata')
    positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):
        t=positions[d];need(i==0 or (t+1<len(axis) and axis[t+1][:7]!=d[:7]),'Formal official monthends')
    originalplan=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    maps=[canonical_source_map(x) for x in (parity['raw_source_hashes'],originalplan['raw_source_hashes'],plan['raw_source_hashes'])]
    need(all(x==maps[0] for x in maps[1:]),'Original64 canonical path/SHA identity')
    need(all(parity[k] is True for k in ('original_arrays_unchanged','original_orders_exact','original_full49_pre_preserved')) and len(parity['original_array_fingerprints'])==27,'Original27/full49 preservation')
    need(all(parity['execution_prices'][k] is True for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact')),'Saved execution source parity')
    source_audit=ev.read(OUT/'prepared/SOURCE_AUDIT.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    src=source_summary(source_audit,meta,schemas);decisions=decision_summary(audit,source_audit,meta,orders,schemas)
    prep=preparation(e,pc,audit,source_audit,parity,flags)
    changed={candidate+'_vs_'+ref:changes(orders,ref,candidate) for candidate,ref in ((ARMS[1],ARMS[0]),(ARMS[1],ARMS[2]),(ARMS[2],ARMS[0]))}
    selection=saved_selection(training,changed);need(selection==frozen['selection'],'Exact frozen first88 calendar and comparator guards')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Result policy labels')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Frozen independent accounting source')
    ac,exports=accounting(ev,result,ind,training)
    inc=previous_reference(ev,result,exports,ind,PRIOR,PREVIOUS_REVIEW,ARMS[0],ARMS[0])
    fee_path=ROOT/'work/monthly_reference_gain250_RESULT_REVIEW.json'
    fee_evidence=ev.read(fee_path,'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595')['incumbent_reproduction']['period_fee_exception']
    need(fee_evidence['absolute_bound']==PERIOD_FEE_ATOL and fee_evidence['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Inherited1e-10 single-period scope only')
    ev.pin(ROOT/'work/leader_bridge253_saved_result_reader.py','a3514e668ef0918572838c4fcc0727c4bda1ce4634658fbb2a08aae40039cb47')
    engine=ROOT/'outputs/monthly_factor_search_v2/validate.py';ev.pin(engine,ev.registered(plan,engine))
    ev.pin(Path(__file__));ev.pin(ROOT/'work/test_calendar_memory254_saved_result_reader.py')
    for p,h in ev.pins.items():need(digest(local(p))==h,'Evidence changed during review '+p)
    summary={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_calendar_memory254_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='Independent254 definition/static-code and saved-evidence reviewer; no254 producer/host/oracle code authored. Numerical/accounting certificates come from the bound independent oracle.',
      scope='Saved hashes, source and once-only episode receipts, frozen first88 calendar-only/comparator guards,9path saved accounts and historical incumbent comparison; no models, NPZ decoding, feature or portfolio return replay.',
      plan_sha256=args.plan_sha256,context_sha256=FIXED[str(CONTEXT.relative_to(ROOT))],prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,
      formal_synthetic_tests=78,bindings=dict(code_files=6,prepared_files=10,result_files=38,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_path_normalization_only=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      preparation=prep,source=src,decision=decisions,selection=selection,changed_holdings=changed,accounting_review=ac,incumbent_reproduction=inc,full_results=summary,
      segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
      holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=249,registered_pre_state_trials=1260,expected_post_close_completed=250,expected_post_close_trials=1263,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,
      formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      limitations=['Own-history label rows, source data and effect/full score arrays are not decoded. Their complete numerical verification is established by hash-bound independent prepare receipts; this reader checks saved scope/counts/identity, not labels or estimates again.',
        'Calendar versus fixed21 isolates actual exit date and resulting path utility under common sample/maturity/sqrt21. Fixed21 versus original5 also changes anchor timing,horizon,scale,maturity.',
        'Cost cumulative fees may differ only when each round exactly equals its own chronological HOLDINGS sum and the cross-round independent totals remain exact; all other metrics/slices and unchanged exports exact, inherited single-period fee limit1e-10.',
        'Only calendar_memory_local67 is promotable. Its first88 guards include both ZERO CAGR and close Sharpe against fixed21. The diagnostic control, later34, full or cost results cannot rescue a rejected candidate.',
        'Same-close benchmark is zero-cost and idealized; next-open sensitivities separate. Reused history and source vintage imply strict_pit=false; no future performance or observer promotion certified.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),selection=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},changed_month_counts={k:v['counts'] for k,v in changed.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())

