"""261 saved evidence reader: standard library, read-only, no array decoding.

Written by the producer author as an extra evidence cross-check. Numerical
independence comes from the bound separately authored verifier. This reader
neither fits nor predicts, traverses trees, recomputes NAV statistics, replays
portfolios, writes a result review, nor mutates any saved artifact. Root must
supply five exact phase digests after actual successful terminal receipts.
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
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_weak_tree261'
PRIOR=ROOT/'outputs/monthly_leader_utility260'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_weak_tree261_context.json'
CONTRACT=ROOT/'work/monthly_weak_tree261_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_weak_tree261_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_weak_tree261_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
TERMINALS=OUT/'PHASE_TERMINALS.json'
RUNTIME=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
ARMS=('incumbent_local67','matched_pooled_ridge','weak_pooled_tree')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREFIXES=(0,1,53,88,122)
PREP={'FORECASTS.npz','LABELS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
FIXED={'work/monthly_weak_tree261_context.json':'e60db38b1d0fe73fa2453cf17691b007bbb89c03fef29b4711e4e6595dfdd37c',
'work/monthly_weak_tree261_IMPLEMENTATION_CONTRACT.json':'c955aba02bb469c0b35e0499ae80f1b229d638306f72f7d1d0bfde36d9f5c4d6',
'work/monthly_weak_tree261_RUNTIME_PINS.json':'ad796b9d48f1a6dd910dfcf8d66c650359d3c2dd8851a3cd042be935079cc59d',
'work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'}
READER_PARENT='work/monthly_leader_utility260_saved_review.py'
READER_PARENT_SHA='83d0279183c4101a2fa7823aa2a870281cc9dc7eb386f8067a26a3b2b6817f93'
WORK_FIELDS=('independent_canonical_solves','independent_svd_solves','independent_tree_refits','manual_tree_traversals','original_episode_calls','original_other_target_fits','original_local_fits')
SCHEMAS={}
FLAGS={}

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

def constants(path,names,seed=None):
    """Parse only literal schema tuples from verified source; never import it."""
    found=dict(seed or {})
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
    need(set(names)<=set(found),'Missing frozen schema constants');return {k:found[k] for k in names}

def digest_record(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha_value(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def integer(value,label,expected=None):
    need(type(value) is int and value>=0,label+' nonnegative native int')
    if expected is not None:need(value==expected,label+' exact value')

def counts_check(value,fields,label):
    need(type(value) is dict and set(value)==set(fields),label+' exact fields')
    for key,item in value.items():integer(item,label+'/'+key)

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

def terminals_check(record,expected_plan,hashes):
    need(record.get('plan_sha256')==expected_plan and set(record.get('phases',{}))=={'prepare','select','diagnose'},'Three root-owned successful phase terminals')
    for phase,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        p=record['phases'][phase];need(set(p)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Terminal schema')
        need(p['session_id'] is None or (type(p['session_id']) is int and p['session_id']>0),'Terminal session identity')
        integer(p['exit_code'],'Terminal exit',0);need(type(p['chunk_id']) is str and bool(p['chunk_id'].strip()),'Terminal chunk')
        need(local(p['artifact_path'])==OUT/relative and p['artifact_sha256']==hashes[phase],'Terminal artifact exact')
    return record

def schema_flags():
    schemas=constants(ROOT/'work/monthly_weak_tree261_kernel.py',('MODEL_COUNT_FIELDS','RIDGE_ANNUAL_KEYS','TREE_ANNUAL_KEYS','MODEL_AUDIT_KEYS','TREE_KEYS','COUNT_FIELDS','EVENT_KEYS','AUDIT_KEYS','RANK_AUDIT_KEYS'))
    flags=constants(ROOT/'work/monthly_weak_tree261_independent.py',('CONTROL_FLAGS','MODEL_FLAGS','SELECTION_FLAGS'))
    flags['INCUMBENT_FLAGS']=constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',))['INCUMBENT_GATE_FLAGS']
    need([len(schemas[k]) for k in ('MODEL_COUNT_FIELDS','RIDGE_ANNUAL_KEYS','TREE_ANNUAL_KEYS','COUNT_FIELDS','EVENT_KEYS')]==[17,15,8,16,12],'Final261 schema widths')
    return schemas,flags


def model_counts(c,n,control=False):
    counts_check(c,SCHEMAS['MODEL_COUNT_FIELDS'],'Model counts');integer(c['events'],'Model events',n)
    need(c['valid_training_rows']<=c['training_rows'],'Complete usable cohort subset')
    need(c['ordinary_years']==c['supported_years']+c['cold_years'] and c['supported_years']==c['matched_ridge_fits']+c['no_prediction_years'],'Annual partition')
    need(c['tree_fits']==(0 if control else c['matched_ridge_fits']),'Paired supported models')
    need(c['actual_model_fits']==c['matched_ridge_fits']+c['tree_fits'],'Actual model fits')
    need(c['trees']==c['boosting_iterations']==100*c['tree_fits'],'100 fixed boosting iterations')
    need(c['manual_tree_traversals']==(0 if control else 100*c['prediction_rows']),'Manual inference work')
    need(c['original_other_target_fits']==c['original_local_fits']==0 and c['matched_ridge_fits']<=c['prediction_rows'],'Only required fits')
    if n<=1:need(all(v==0 for key,v in c.items() if key!='events'),'No training numeric or fits for empty/partial')


def runtime_check(rows,years):
    need(type(rows) is list and [r['year'] for r in rows]==years,'Exact fitted runtime years')
    for row in rows:
        need(set(row)=={'year','fit_threadpools','predict_threadpools'},'Runtime record keys')
        for phase in ('fit_threadpools','predict_threadpools'):
            pools=row[phase];need(type(pools) is list and bool(pools),'Effective loaded native pools')
            files=[]
            for p in pools:
                need(type(p) is dict and type(p.get('filepath')) is str and bool(p['filepath']),'Native library identity')
                integer(p.get('num_threads'),'Native thread limit',1);files.append(p['filepath'])
            need(len(files)==len(set(files)),'Unique native pools')


def receipt(r,n,kind):
    need(type(r) is dict and r.get('passed') is True,'Passed independent '+kind);integer(r.get('checked_events'),'Receipt event scope',n)
    key={'control':'CONTROL_FLAGS','models':'MODEL_FLAGS','incumbent':'INCUMBENT_FLAGS','selection':'SELECTION_FLAGS'}[kind]
    need(all(r.get(k) is True for k in FLAGS[key]),'Complete independent flags '+kind)
    need(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Independent pre-return scope')
    for key in WORK_FIELDS:integer(r.get(key),'Independent work '+key)
    need(r['original_other_target_fits']==r['original_local_fits']==0,'Other models reused')
    if kind in ('control','models'):
        c=r['counts'];model_counts(c,n,kind=='control');need(r.get('producer_equivalent_counts')==c,'Equivalent producer counts')
        for k,v in [('continuous_atol',2e-10),('continuous_rtol',2e-10),('rank_tolerance',0.)]:need(type(r.get(k)) is float and r[k]==v,'Frozen tolerance '+k)
        need(r.get('shared_tree_trainer') is True and r.get('independent_tree_training_algorithm') is False,'Shared trainer limitation')
        for k in ('exact_discrete','full_bundle_supplied','full_control_supplied','full_label_preservation_supplied','full_model_audit_supplied'):need(r.get(k) is True,'Complete supplied model '+k)
        need(r.get('candidate_models_computed') is (kind=='models') and r.get('full_candidate_supplied') is (kind=='models'),'Pre-tree/full scope')
        need(r['independent_canonical_solves']==r['independent_svd_solves']==c['matched_ridge_fits'],'Actual canonical/SVD work')
        need(r['independent_tree_refits']==c['tree_fits'] and r['manual_tree_traversals']==(0 if kind=='control' else 2*c['manual_tree_traversals']),'Refit and own/supplied inference work')
        need(r['original_episode_calls']==0,'Model verification no selector')
    else:
        for k in ('full_scores_supplied','full_pre_scores_supplied'):need(r.get(k) is True,'Complete decision proof')
        need(type(r.get('numerical_tolerance')) is float and r['numerical_tolerance']==0.,'Exact decision tolerance')
        integer(r.get('ordered_choices_checked'),'Ordered choice count',n if kind=='incumbent' else 3*n)
        need(all(r[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'No decision fits')
        integer(r['original_episode_calls'],'Independent selectors',1 if kind=='incumbent' else 3)
        if kind=='incumbent':need(r.get('candidate_selection_computed') is False,'Original before candidate')
        else:
            need(r.get('full_bundle_supplied') is True and r.get('full_audit_supplied') is True,'Full decision evidence')
            counts_check(r['counts'],SCHEMAS['COUNT_FIELDS'],'Decision receipt');integer(r['counts']['events'],'Selection prefix',n);integer(r['counts']['symbols_per_event'],'Original width',49)


def fit_binding(control,full,c,n):
    model_counts(c,n);need(full['counts']==c,'Saved full model count binding')
    before=control['counts']
    for key in c:
        expected=0 if key in ('tree_fits','boosting_iterations','trees','tree_nodes','manual_tree_traversals') else c['matched_ridge_fits'] if key=='actual_model_fits' else c[key]
        integer(before[key],'Before-tree count '+key,expected)


def decision_counts(events,n,source_eligible):
    rows=events[:n];c=dict.fromkeys(SCHEMAS['COUNT_FIELDS'],0)
    c.update(events=n,symbols_per_event=49,original_eligible_cells=source_eligible,incumbent_episode_calls=1,matched_episode_calls=1,tree_episode_calls=1)
    for i,r in enumerate(rows):
        need(set(r)==set(SCHEMAS['EVENT_KEYS']) and r['query_index']==i,'Full decision event schema')
        need(type(r['initial_partial']) is type(r['weak']) is bool and r['initial_partial'] is (i==0),'Named partial flag')
        for key in ('ranking_indices','active_indices','changed_delta_indices'):
            ids=r[key];need(type(ids) is list and ids==sorted(set(ids)) and all(type(j) is int and 0<=j<49 for j in ids),'Exact sorted '+key)
        wanted=r['ranking_indices'] if r['weak'] and not r['initial_partial'] else []
        need(r['active_indices']==wanted,'Only ordinary weak U activation')
        d=r['delta'];need(type(d) is list and len(d)==49 and all(finite(v) for v in d),'Finite full delta')
        need([j for j,v in enumerate(d) if v!=0.]==r['changed_delta_indices'] and set(r['changed_delta_indices'])<=set(wanted),'Only active nonzero updates')
        need(r['matched_choices']==r['incumbent_choices'],'Matched ordered identity')
        for key in ('incumbent_choices','matched_choices','tree_choices'):
            js=r[key];need(type(js) is list and len(js) in (0,2,3) and len(set(js))==len(js) and all(type(j) is int and 0<=j<49 for j in js),'Valid original choices')
        if not wanted:need(r['tree_choices']==r['incumbent_choices'],'Inactive choices exact')
        c['initial_partial_events']+=r['initial_partial'];c['formal_events']+=not r['initial_partial'];c['ranking_cells']+=len(r['ranking_indices']);c['weak_applicable_cells']+=len(wanted);c['changed_delta_cells']+=len(r['changed_delta_indices'])
        c['tree_changed_holdings_events']+=set(r['tree_choices'])!=set(r['incumbent_choices']);c['tree_vs_matched_changed_holdings_events']+=set(r['tree_choices'])!=set(r['matched_choices'])
    integer(source_eligible,'Receipt-backed original E count');need(c['ranking_cells']<=source_eligible<=49*n,'Original U subset of E bound')
    return c


def tree_structure(tree,parameters):
    """Inspect stored structure; never evaluate any tree or feature row."""
    need(type(tree) is dict and set(tree)==set(SCHEMAS['TREE_KEYS']),'Full serialized tree schema')
    need(tree['schema']=='monthly_weak_tree261_serialized_tree_v1' and tree['parameters']==parameters,'Fixed21 tree parameters')
    for key,v in [('n_features',10),('n_iter',100),('n_trees_per_iteration',1),('missing_bin',255)]:integer(tree[key],key,v)
    need(tree['is_categorical'] is tree['known_categories'] is tree['preprocessor'] is None,'No categorical preprocessing')
    names=['value','count','feature_idx','num_threshold','missing_go_to_left','left','right','gain','depth','is_leaf','bin_threshold','is_categorical','bitset_idx']
    need(tree['node_dtype']==[[k,d] for k,d in zip(names,['<f8','<u4','<i8','<f8','|u1','<u4','<u4','<f8','<u4','|u1','|u1','|u1','<u4'])],'Pinned node representation')
    def hexf(v,allowinf=False):
        need(type(v) is str,'Hexadecimal float');x=float.fromhex(v);need(x.hex()==v and (math.isfinite(x) or allowinf and x==math.inf),'Canonical finite float or active +inf');return x
    baseline=tree['baseline'];need(set(baseline)=={'dtype','shape','values'} and baseline['dtype']=='<f8' and baseline['shape']==[1,1] and len(baseline['values'])==1 and len(baseline['values'][0])==1,'Binary64 baseline');hexf(baseline['values'][0][0])
    need(len(tree['bin_thresholds'])==len(tree['n_bins_non_missing'])==10,'Ten binners')
    for rec,count in zip(tree['bin_thresholds'],tree['n_bins_non_missing']):
        integer(count,'Bin count');need(1<=count<=255 and rec['dtype']=='<f8' and rec['shape']==[count-1] and len(rec['values'])==count-1,'Bin representation')
        values=[hexf(x) for x in rec['values']];need(values==sorted(values),'Sorted bins')
    need(len(tree['trees'])==100,'All100 trees');nodes_total=0
    for i,tr in enumerate(tree['trees']):
        need(set(tr)=={'iteration','class_index','nodes','raw_left_cat_bitsets','binned_left_cat_bitsets'},'Full predictor record');integer(tr['iteration'],'Tree order',i);integer(tr['class_index'],'Single class',0)
        for key in ('raw_left_cat_bitsets','binned_left_cat_bitsets'):need(tr[key]=={'dtype':'<u4','shape':[0,8],'values':[]},'Empty numeric categorical bitsets')
        nodes=tr['nodes'];need(type(nodes) is list and 1<=len(nodes)<=7,'Depth2 nodes');seen=set();stack=[(0,0)]
        while stack:
            j,depth=stack.pop();need(type(j) is int and 0<=j<len(nodes) and j not in seen,'Rooted acyclic tree');seen.add(j);node=nodes[j];need(set(node)==set(names),'Every node field')
            for key in names:
                if key in ('value','gain','num_threshold'):hexf(node[key],key=='num_threshold' and node['is_leaf']==0)
                else:integer(node[key],'Node '+key)
            need(node['depth']==depth<=2 and node['is_leaf'] in (0,1) and node['missing_go_to_left'] in (0,1) and node['is_categorical']==0,'Numeric node flags')
            if not node['is_leaf']:
                need(depth<2 and node['feature_idx']<10 and node['bin_threshold']<255,'Active split range');stack.extend([(node['right'],depth+1),(node['left'],depth+1)])
        need(len(seen)==len(nodes),'All nodes reachable');nodes_total+=len(nodes)
    return nodes_total


def model_summary(ma,meta,training_meta,events,parameters):
    need(set(ma)=={'model','labels','control','candidate'},'Four saved model proof objects')
    model,lp=ma['model'],ma['labels'];need(set(model)==set(SCHEMAS['MODEL_AUDIT_KEYS']),'Model declarations schema');c=model['counts'];model_counts(c,122)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only'):need(model[key] is True,'Model preservation')
    need(model['computes_returns'] is False,'Model stage no returns');p=model['original_cache_parity'];need(p==dict(passed=True,events=122,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False),'Original calibration proof')
    need(lp['schema']=='monthly_weak_tree261_labels_preserved_v1' and lp['rows']==c['training_rows'] and c['valid_training_rows']<=lp['valid_rows']<=lp['rows'],'Label population')
    need(lp['target_keys']==['simple_excess','log_excess','vol_scaled','loss_utility'] and set(lp['target_sha256'])==set(lp['target_keys']),'Four unchanged labels')
    need(all(sha_value(v) for v in list(lp['target_sha256'].values())+[lp['common_good_sha256']]),'Original label hashes')
    need(all(lp[x] is True for x in ('labels_unchanged','common_good_unchanged','original_row_order_preserved')) and lp['recentered'] is lp['raw_target_reconstructed'] is False,'No target intervention')
    tm=training_meta['training'];rowdays=[];rowends=[]
    need(len(tm['dates'])==len(tm['ends'])==len(tm['symbols']),'Training metadata axis')
    for d,e,symbols in zip(tm['dates'],tm['ends'],tm['symbols']):
        need(d<e and len(symbols)==len(set(symbols)),'Training identities');rowdays.extend([d]*len(symbols));rowends.extend([e]*len(symbols))
    annual=ma['control']['annual'];trees=ma['candidate']['annual'];years=sorted({int(d[:4]) for d in meta['dates'][1:]})
    need([r['year'] for r in annual]==[r['year'] for r in trees]==years,'Exactly ordinary requested years')
    tree_nodes=predictions=fitcount=0;fitted=[]
    for r,tr in zip(annual,trees):
        need(set(r)==set(SCHEMAS['RIDGE_ANNUAL_KEYS']) and set(tr)==set(SCHEMAS['TREE_ANNUAL_KEYS']),'Annual evidence schemas')
        year=r['year'];need(r['cutoff']==f'{year-1}-12-31' and r['lower_bound']==f'{year-6}-12-31','Annual clock')
        ids=r['training_row_ids'];need(ids==sorted(set(ids)) and all(type(j) is int and 0<=j<len(rowdays) and r['lower_bound']<rowdays[j] and rowends[j]<=r['cutoff'] for j in ids),'Mature past-only row identities')
        months=len({rowdays[j][:7] for j in ids});need(months==r['training_months'],'Actual distinct annual months')
        qids=[i*49+j for i,d in enumerate(meta['dates']) if i and int(d[:4])==year for j in events[i]['ranking_indices']]
        need(r['query_row_ids']==qids,'Complete original query U for year')
        status='cold_start' if months<18 else 'fit' if qids else 'no_prediction_rows';need(r['status']==status,'Annual support status')
        for key in ('year','cutoff','training_row_ids','query_row_ids','status'):need(tr[key]==r[key],'Same annual Ridge/tree cohort and queries')
        if status!='fit':
            need(r['weights']==[] and r['raw_predictions']==[] and all(r[k] is None for k in ('mean','scale','beta','intercept','gram','rhs')),'No skipped fit statistics');need(tr['tree'] is None and tr['raw_predictions']==tr['manual_predictions']==[],'No skipped trees')
        else:
            fitcount+=1;fitted.append(year);predictions+=len(qids)
            need(len(r['weights'])==len(ids) and all(finite(w) and w>0 for w in r['weights']),'Annual positive weights')
            for key in ('mean','scale','beta','rhs'):need(len(r[key])==10 and all(finite(v) for v in r[key]),'Ten finite Ridge fields')
            need(all(v>0 for v in r['scale']) and finite(r['intercept']) and len(r['gram'])==10 and all(len(row)==10 and all(finite(v) for v in row) for row in r['gram']),'Saved Ridge system dimensions')
            need(len(r['raw_predictions'])==len(tr['raw_predictions'])==len(qids) and all(finite(v) for v in r['raw_predictions']+tr['raw_predictions']),'Full raw query predictions')
            need(tr['manual_predictions']==tr['raw_predictions'],'Saved exact manual inference proof');tree_nodes+=tree_structure(tr['tree'],parameters)
    need(c['matched_ridge_fits']==c['tree_fits']==fitcount and c['prediction_rows']==predictions and c['tree_nodes']==tree_nodes,'Annual source evidence work totals')
    runtime_check(model['execution_runtime'],fitted)
    need(c['training_rows']==sum(e<=f'{years[-1]-1}-12-31' for e in rowends),'Complete consumed mature row prefix')
    return dict(fitted_years=fitted,annual=[dict(year=r['year'],status=r['status'],training_months=r['training_months'],training_rows=len(r['training_row_ids']),query_rows=len(r['query_row_ids'])) for r in annual],counts=c,tree_nodes_structure_checked=tree_nodes,model_arithmetic_recomputed=False,manual_inference_recomputed=False,usable_row_membership_proven_by_bound_independent_verifier=True),rowends


def matched_check(r,n):
    integer(r['events'],'Matched strategy events',n);integer(r['producer_episode_calls'],'Matched gate original selector',1)
    for key in ('passed','matched_ranks_exact','delta_exact_zero','pre_scores_exact','final_scores_exact','ordered_choices_exact'):need(r.get(key) is True,'Matched pre-tree strategy')
    receipt(r['independent'],n,'incumbent')


def preparation_evidence(e,ma,da,meta,rowends):
    for kind in ('control','models','incumbent','selection'):receipt(e[kind],122,kind)
    need(e['main_model_counts']==ma['model'],'Main model payload binding');fit_binding(e['control'],e['models'],ma['model']['counts'],122)
    receipt(e['early_incumbent'],122,'incumbent');matched_check(e['matched_strategy'],122)
    need(e['all123_order_prefixes'] is True and e['original_arrays_unchanged'] is True and e['original_training_unchanged'] is True,'All prefixes and immutability')
    need(da['incumbent_gate']==e['incumbent'] and da['counts']==e['selection']['counts'],'Saved final decisions bind full independent receipt')
    events=da['events'];cached=e['cached_prefixes'];physical=e['physical_prefixes'];need(len(cached)==123 and len(physical)==5,'123 cached and5 source prefixes')
    previous_eligible=0
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached prefix',n);need(v['exact'] is True,'Cached exact');receipt(v['incumbent'],n,'incumbent');receipt(v['selection'],n,'selection')
        dc=v['selection']['counts'];need(dc==decision_counts(events,n,dc['original_eligible_cells']),'All derivable prefix decision counts')
        need(previous_eligible<=dc['original_eligible_cells']<=previous_eligible+49,'Receipt-backed E count monotone');previous_eligible=dc['original_eligible_cells']
    for n,v in zip(PREFIXES,physical):
        integer(v['prefix'],'Physical prefix',n);need(v['exact'] is True,'Physical exact')
        for kind in ('control','models','incumbent','selection'):receipt(v['model' if kind=='models' else kind],n,kind)
        receipt(v['early_incumbent'],n,'incumbent');matched_check(v['matched_strategy'],n)
        model=v['model_counts'];fit_binding(v['control'],v['model'],model['counts'],n)
        need(v['selection']['counts']==cached[n]['selection']['counts'],'Physical/cached counts exact')
        years=sorted({int(d[:4]) for d in meta['dates'][1:n]});annual=[r for r in ma['control']['annual'] if r['year'] in years]
        fitrows=[r for r in annual if any(i<n*49 for i in r['query_row_ids']) and r['training_months']>=18]
        mc=model['counts'];integer(mc['training_rows'],'Physical mature teacher rows',sum(day<=f'{years[-1]-1}-12-31' for day in rowends) if years else 0)
        integer(mc['ordinary_years'],'Physical ordinary years',len(years));integer(mc['matched_ridge_fits'],'Physical annual fits',len(fitrows));integer(mc['prediction_rows'],'Physical query predictions',sum(sum(i<n*49 for i in r['query_row_ids']) for r in fitrows))
        runtime_check(model['execution_runtime'],[r['year'] for r in fitrows])
        if n==122:
            need(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Full122 reuse')
            for kind in ('control','models','incumbent','selection'):need(v['model' if kind=='models' else kind]==e[kind],'Full122 exact receipts')
            need(model==ma['model'] and v['early_incumbent']==e['early_incumbent'] and v['matched_strategy']==e['matched_strategy'],'Full122 exact payload')
        else:need(v['model_recomputed'] is True and not v.get('reuses_main_verified_full',False),'Real physical prefix evidence')
    entries=[]
    for scope,n,v in [('main',122,e)]+[('physical',x['prefix'],x) for x in physical if x['prefix']!=122]:
        models=v['models'] if scope=='main' else v['model'];c=models['counts'];checks=[v['early_incumbent'],v['matched_strategy']['independent'],v['control'],models,v['incumbent'],v['selection']]
        item=dict(scope=scope,events=n,producer_matched_ridge_fits=c['matched_ridge_fits'],producer_tree_fits=c['tree_fits'],producer_boosting_iterations=c['boosting_iterations'],producer_trees=c['trees'],producer_tree_nodes=c['tree_nodes'],producer_manual_tree_traversals=c['manual_tree_traversals'],producer_episode_calls=5)
        item.update({k:sum(r[k] for r in checks) for k in WORK_FIELDS});item['total_model_fit_calls']=sum(item[k] for k in ('producer_matched_ridge_fits','producer_tree_fits','independent_canonical_solves','independent_svd_solves','independent_tree_refits'));entries.append(item)
    cc=[v[k] for v in cached for k in ('incumbent','selection')];cw=dict(invocations=123,producer_episode_calls=369,model_fit_calls=0,**{k:sum(r[k] for r in cc) for k in WORK_FIELDS})
    totals={k:sum(x[k] for x in entries) for k in entries[0] if k not in ('scope','events')};totals['producer_episode_calls']+=cw['producer_episode_calls']
    for k in WORK_FIELDS:totals[k]+=cw[k]
    expected=dict(schema='monthly_weak_tree261_execution_accounting_v1',model_invocations=5,physical122_reuses_main=True,entries=entries,cached_decisions=cw,totals=totals,original_complete_model_refits=0,matched_ridge_is_actual_calibration_refit=True,manual_traversal_is_model_fit=False)
    need(e['fit_execution_accounting']==expected,'Complete actual execution accounting')
    return expected


def saved_selection(training,orders):
    need(set(training)==set(orders)==set(ARMS),'Three fixed arms');invalid={a:required_metrics(training[a],a!=ARMS[2]) for a in ARMS}
    need(training[ARMS[1]]==training[ARMS[0]] and orders[ARMS[1]][:88]==orders[ARMS[0]][:88],'Matched exact calibration')
    c,r=ARMS[2],ARMS[0];changed=[i for i in range(88) if set(orders[c][i]['symbols'])!=set(orders[r][i]['symbols'])]
    gates=dict(close=bool(not invalid[c] and guard(training[c][PATHS[0]],training[r][PATHS[0]],.005)),open=bool(not invalid[c] and guard(training[c][PATHS[1]],training[r][PATHS[1]],0.)),support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed))
    admitted=all(gates.values());return dict(primary=c if admitted else r,admitted=admitted,guards=gates,changed_months=changed,matched_exact_incumbent=True,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)
def source_proof(ev,plan,source):
    original=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    maps=[canonical_source_map(v) for v in (source['raw_source_hashes'],plan['raw_source_hashes'],original['raw_source_hashes'])]
    need(maps[0]==maps[1]==maps[2],'Original64 source lineage')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        need(source[key]==ev.registered(plan,REBUILD/name),'Registered source pin');ev.pin(REBUILD/name,source[key])
    for key in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','matched_ridge_calibration_refits_counted'):need(source[key] is True,'Exact source declaration')
    need(source['original_other_target_fits']==source['original_local_fits']==0 and source['fresh_training_array_reconstruction'] is False,'No invented original reconstruction')
    closure=source['original169_source_closure'];need(closure==plan['original169_source_closure'],'Registered full169 closure')
    need(closure['npz_decoded'] is False and closure['source_cache_rebuilt'] is False,'Metadata-only inherited closure')
    for path,h in closure['chain_sha256'].items():need(ev.registered(plan,path)==h,'169 chain registered');ev.pin(path,h)
    parity=ev.read(REBUILD/'PARITY.json');rebuild_plan=ev.read(REBUILD.parent/'PLAN.json');need(parity['passed'] is True and parity['returns_computed'] is False and parity['source_plan_sha256']==digest(REBUILD.parent/'PLAN.json'),'Original169 source parity')
    need(len(parity['artifacts_sha256'])==12 and parity['artifacts_sha256']==closure['artifacts_sha256'],'All12 source artifacts')
    for name,h in parity['artifacts_sha256'].items():need(Path(name).name==name and ev.registered(plan,REBUILD/name)==h,'Registered full169 cache');ev.pin(REBUILD/name,h)
    need(closure['source_input_count']==len(rebuild_plan['input_sha256']),'169 input inventory')
    for path,h in rebuild_plan['input_sha256'].items():need(ev.registered(plan,path)==h,'Inherited169 source input binding')
    return dict(original64_maps_exact=True,original169_artifact_count=12,original169_artifacts_rehashed=True,npz_decoded=False,source_cache_rebuilt=False,raw64_rehashed=False,inherited_source_input_digests_bound=True)


def run(args):
    global SCHEMAS,FLAGS
    need(platform.python_version()=='3.9.6','Pinned wrapper runtime')
    for name in ('plan','prepared','frozen','results','terminals'):need(sha_value(getattr(args,name+'_sha256')),'Explicit phase digest '+name)
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():need(ev.registered(plan,p)==h,'Prospective fixed registered pin');ev.pin(p,h)
    ev.pin(READER_PARENT,READER_PARENT_SHA);context=ev.read(CONTEXT);contract=ev.read(CONTRACT)
    need(plan['context']==context and plan['context_sha256']==digest(CONTEXT) and plan['implementation_contract']==contract and plan['implementation_contract_sha256']==digest(CONTRACT),'Immutable prospective definition')
    need(context['arms']==list(ARMS) and context['only_promotable']==ARMS[2] and context['diagnostic_only']==ARMS[1] and context['registered_trials']==3,'Tree only promotion')
    tests=ev.read(TESTS,ev.registered(plan,TESTS));code=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Passed formal tests')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['actual_returns_computed'] is False and code['code_sha256']==tests['code_sha256'],'Prospective code review')
    need(code['context_sha256']==digest(CONTEXT) and code['implementation_contract_sha256']==digest(CONTRACT),'Code/definition bindings')
    expected={ROOT/'work'/name for stem in ('kernel','independent','host') for name in ('monthly_weak_tree261_'+stem+'.py','test_monthly_weak_tree261_'+stem+'.py')}
    need({local(p) for p in tests['code_sha256']}==expected,'Exactly six final files')
    for path,h in tests['code_sha256'].items():need(ev.registered(plan,path)==h,'Registered code version');ev.pin(path,h)
    flags_path=ROOT/'work/monthly_calendar_clock248_v2_independent.py';ev.pin(flags_path,ev.registered(plan,flags_path));SCHEMAS,FLAGS=schema_flags()
    runtime=ev.read(RUNTIME);need(len(runtime['input_sha256'])==3309,'Pinned package closure inventory')
    for path,h in runtime['input_sha256'].items():need(ev.registered(plan,path)==h,'Every runtime file registered')
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(state['completed'])==256 and sum(r['trials'] for r in state['completed'])==1279 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Registered prior state')
    prior=ev.read(PRIOR/'COMPLETION.json',ev.registered(plan,PRIOR/'COMPLETION.json'));need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444,'Prior verified closure')
    protected=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protected)==plan['protected_files']==21444 and all(sha_value(h) for h in protected.values()),'Protected inventory count and hashes')
    terminal=ev.read(TERMINALS,args.terminals_sha256);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP);need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared scope')
    f=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);train=ev.read(OUT/'selected/TRAIN_RESULTS.json',f['training_sha256'])
    need(f['plan_sha256']==args.plan_sha256 and f['prepared_sha256']==args.prepared_sha256 and f['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selection artifacts')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Complete saved outcome scope')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed122 intervals')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen53/35/34 split')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']) and set(orders)==set(ARMS),'Original49 and3 arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order clocks')
        for o in orders[arm]:need(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(v) for v in o['scores']),'Finite ordered source execution')
    need(orders[ARMS[0]]==orders[ARMS[1]],'Matched complete source orders')
    for path in (LOCAL/'local_linear_ORDERS.json',PRIOR/'prepared/ORDERS.json'):
        old=ev.read(path,ev.registered(plan,path));need(orders[ARMS[0]]==(old if type(old) is list else old[ARMS[0]]),'Exact prior incumbent order list')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json');source_summary=source_proof(ev,plan,source)
    metadata=ev.read(REBUILD/'MONTHLY_META.json',ev.registered(plan,REBUILD/'MONTHLY_META.json'));need(metadata['monthly']==meta,'Original full monthly identity')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar chronology');positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):need(d in positions and (i==0 or positions[d]+1<len(axis) and axis[positions[d]+1][:7]!=d[:7]),'Official monthend')
    ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');da=ev.read(OUT/'prepared/DECISION_AUDIT.json');proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    need(set(da)==set(SCHEMAS['AUDIT_KEYS']) and len(da['events'])==122 and da['computes_returns'] is False,'Complete saved decision audit')
    for key in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):need(da[key] is True,'Decision invariant')
    need(da['counts']==decision_counts(da['events'],122,da['counts']['original_eligible_cells']),'Saved decision count identities')
    for i,r in enumerate(da['events']):
        need(r['date']==meta['dates'][i],'Decision identity')
        for arm,key in zip(ARMS,('incumbent_choices','matched_choices','tree_choices')):need([meta['symbols'][i][j] for j in r[key]]==orders[arm][i]['symbols'],'Audit/order symbols')
    summary,rowends=model_summary(ma,meta,metadata,da['events'],context['tree_parameters']);work=preparation_evidence(proof,ma,da,meta,rowends)
    for v in proof['physical_prefixes']:
        if v['prefix']==122:continue
        n=v['prefix'];scope=v['input_scope'];cutoff=f'{int(meta["dates"][n-1][:4])-1}-12-31' if n>1 else None
        expected_scope=dict(events=n,training_months=sum(e<=cutoff for e in metadata['training']['ends']) if cutoff else 0,training_rows=sum(e<=cutoff for e in rowends) if cutoff else 0,cutoff=cutoff,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=n>1 and any(e<=cutoff for e in rowends),physically_shortened=True)
        need(scope==expected_scope,'Actual physically shortened source scope')
    selected=saved_selection(train,orders);need(selected==f['selection'],'Frozen first88 selection exactly reconstructed from saved summaries')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selected and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Honest result scope')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Frozen independent accounting source')
    ac,exports=accounting(ev,result,ind,train)
    need(result['arms'][ARMS[0]]==result['arms'][ARMS[1]],'Matched exact all reported metrics')
    for path in PATHS:
        for kind in KINDS:need(exports[ARMS[0],path][kind]==exports[ARMS[1],path][kind],'Matched full exported stream exact')
        for arm in ARMS:
            days=[r['date'] for r in exports[arm,path]['DAILY']];need(days==[d for d in axis if meta['dates'][0]<=d<=meta['ends'][-1]],'Full official saved daily axis')
    ev.pin(Path(__file__))
    for path,h in ev.pins.items():need(digest(local(path))==h,'Saved evidence unchanged during review')
    return dict(schema='monthly_weak_tree261_saved_reader_evidence_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='261 producer author performing extra saved-evidence cross-check; independently authored numerical evidence is supplied by the bound asset verifier, not this reader.',
      scope='Read-only hashes, complete manifests, declared model/tree/cohort/runtime/actual-work evidence, saved metrics and first88 guards. No fitting, predictions, tree traversal, NPZ decoding, source prices, return/statistic recomputation or replay.',
      plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,
      root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],source=source_summary,model_summary=summary,decision_counts=da['counts'],execution_accounting=work,selection=selected,
      changed_holdings={a+'_vs_'+b:changes(orders,b,a) for a,b in ((ARMS[2],ARMS[0]),(ARMS[1],ARMS[0]))},saved_accounting_evidence=ac,matched_calibration_exact=True,
      full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
      bindings=dict(code_files=6,prepared_files=len(PREP),result_files=len(RESULT),all_collected_pins_rechecked=True,package3309_manifest_bound=True,package_members_rehashed=False,full_registered_closure_rehashed=False),
      preparation=dict(all123_cached_prefixes=True,physical_prefixes=list(PREFIXES),physical122_reuses_main=True,source_control_before_tree_certificate_checked=True),
      protection_and_close=dict(status='pending_root_close',protected_inventory_count=21444,prior_protection_certificate_verified=True,protected_members_rehashed=False,final_root_close_verified=False,mutable_central_read_or_bound=False,registered_pre_state_completed=256,registered_pre_state_trials=1279,expected_post_close_completed=257,expected_post_close_trials=1282),
      actual_models_computed_by_reviewer=False,actual_model_solve_count=0,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
      limitations=['Original E counts and complete usable-row membership require decoded source arrays; this reader relies on bound independent numerical receipts for those facts and does not pretend to reconstruct them.',
      'The separately authored numerical verifier uses a separate canonical Ridge/SVD and input pipeline, but shares the pinned sklearn training algorithm for tree reproduction.',
      'Saved 9-path/45-slice and18-training-group metrics are compared to saved independent accounting; this reader does not recompute DAILY statistics or replay source prices.',
      'Runtime metadata is validated against saved in-scope evidence and registered package closure; a later process thread state is not substituted for fit-time evidence.',
      'Only weak_pooled_tree is promotable. First88 two ZERO paths decide; full/later/cost outputs cannot rescue rejection.',
      'Final21444 protected-member check and central CAS are root-close responsibilities pending this pre-close evidence.'],
      reader_execution=dict(script=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__)),phase_digests=vars(args),report_written=False,terminal_exit_claimed=False),input_sha256=ev.pins)


def self_test():
    global SCHEMAS,FLAGS
    SCHEMAS,FLAGS=schema_flags()
    class Checks(unittest.TestCase):
        def event(self,i):return dict(query_index=i,date='2016-08-17' if i==0 else '2016-08-31',initial_partial=i==0,weak=True,ranking_indices=[0,1,2],active_indices=[] if i==0 else [0,1,2],changed_delta_indices=[],delta=[0.]*49,incumbent_choices=[0,1,2],matched_choices=[0,1,2],tree_choices=[0,1,2],episode_budget=.05)
        def metrics(self,cagr=.2):
            m=dict(cagr=cagr,sharpe=1.,max_drawdown=-.2,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None)
            return {p:{g:dict(m) for g in ('training','early','recent')} for p in PATHS[:2]}
        def selection(self):
            m={a:self.metrics() for a in ARMS};o={a:[dict(symbols=['A','B','C']) for _ in range(122)] for a in ARMS};return m,o
        def test_integer_bool_rejected(self):
            with self.assertRaises(ValueError):integer(True,'count')
        def test_prefix_counts_from_events(self):
            rows=[self.event(0),self.event(1)];c=decision_counts(rows,2,98);self.assertEqual(c['ranking_cells'],6);self.assertEqual(c['weak_applicable_cells'],3);self.assertEqual(c['formal_events'],1)
            self.assertEqual(decision_counts(rows,0,0)['incumbent_episode_calls'],1)
        def test_nonweak_changed_delta_rejected(self):
            row=self.event(0);row['delta'][0]=.01;row['changed_delta_indices']=[0]
            with self.assertRaises(ValueError):decision_counts([row],1,49)
        def test_original_E_counter_bound(self):
            with self.assertRaises(ValueError):decision_counts([self.event(0)],1,2)
        def test_runtime_fitted_year_and_threads(self):
            row=dict(year=2016,fit_threadpools=[dict(filepath='/fake',num_threads=1)],predict_threadpools=[dict(filepath='/fake',num_threads=1)])
            runtime_check([row],[2016]);row['fit_threadpools'][0]['num_threads']=2
            with self.assertRaises(ValueError):runtime_check([row],[2016])
            with self.assertRaises(ValueError):runtime_check([],[2016])
        def test_candidate_guard_and_matched_identity(self):
            m,o=self.selection();self.assertFalse(saved_selection(m,o)['admitted']);m[ARMS[1]][PATHS[0]]['training']['cagr']+=.01
            with self.assertRaises(ValueError):saved_selection(m,o)
        def test_only_tree_promotable_and_support_both_segments(self):
            m,o=self.selection();m[ARMS[2]]=self.metrics(.21)
            for i in (2,4,60):o[ARMS[2]][i]=dict(symbols=['A','B','D'])
            self.assertTrue(saved_selection(m,o)['admitted']);self.assertEqual(saved_selection(m,o)['primary'],ARMS[2])
            o[ARMS[2]][60]=dict(symbols=['A','B','C']);self.assertFalse(saved_selection(m,o)['admitted'])
        def test_later_only_cannot_rescue(self):
            m,o=self.selection();m[ARMS[2]]=self.metrics(.3)
            for i in range(88,122):o[ARMS[2]][i]=dict(symbols=['A','B','D'])
            self.assertFalse(saved_selection(m,o)['admitted'])
        def test_model_work_counts_zero_and_fit(self):
            c=dict.fromkeys(SCHEMAS['MODEL_COUNT_FIELDS'],0);model_counts(c,0)
            c.update(events=2,training_rows=36,valid_training_rows=36,ordinary_years=1,supported_years=1,matched_ridge_fits=1,tree_fits=1,prediction_rows=3,boosting_iterations=100,trees=100,tree_nodes=100,manual_tree_traversals=300,actual_model_fits=2);model_counts(c,2)
            for key in ('tree_fits','actual_model_fits','manual_tree_traversals','original_local_fits'):
                bad=dict(c);bad[key]+=1
                with self.assertRaises(ValueError):model_counts(bad,2)
        def tiny_tree(self):
            names=['value','count','feature_idx','num_threshold','missing_go_to_left','left','right','gain','depth','is_leaf','bin_threshold','is_categorical','bitset_idx']
            types=['<f8','<u4','<i8','<f8','|u1','<u4','<u4','<f8','<u4','|u1','|u1','|u1','<u4']
            node={k:float(0).hex() if dtype=='<f8' else 0 for k,dtype in zip(names,types)};node.update(count=36,is_leaf=1)
            return dict(schema='monthly_weak_tree261_serialized_tree_v1',parameters={},n_features=10,n_iter=100,n_trees_per_iteration=1,
              baseline=dict(dtype='<f8',shape=[1,1],values=[[float(0).hex()]]),node_dtype=[[k,d] for k,d in zip(names,types)],
              trees=[dict(iteration=i,class_index=0,nodes=[dict(node)],raw_left_cat_bitsets=dict(dtype='<u4',shape=[0,8],values=[]),binned_left_cat_bitsets=dict(dtype='<u4',shape=[0,8],values=[])) for i in range(100)],
              bin_thresholds=[dict(dtype='<f8',shape=[0],values=[]) for _ in range(10)],n_bins_non_missing=[1]*10,missing_bin=255,is_categorical=None,known_categories=None,preprocessor=None)
        def test_tree_structure_without_inference(self):
            t=self.tiny_tree();self.assertEqual(tree_structure(t,{}),100)
            for field,value in [('value','nan'),('num_threshold','inf'),('gain','-inf')]:
                bad=self.tiny_tree();bad['trees'][0]['nodes'][0][field]=value
                with self.assertRaises(ValueError):tree_structure(bad,{})
        def test_tree_structural_dtype_order_and_shape_tamper(self):
            t=self.tiny_tree();t['node_dtype'][0][1]='<f4'
            with self.assertRaises(ValueError):tree_structure(t,{})
            t=self.tiny_tree();t['trees'][1]['iteration']=0
            with self.assertRaises(ValueError):tree_structure(t,{})
            t=self.tiny_tree();t['baseline']['shape']=[1]
            with self.assertRaises(ValueError):tree_structure(t,{})
        def test_missing_independent_flags_and_wrong_scope(self):
            r={k:True for k in FLAGS['INCUMBENT_FLAGS']};r.update(passed=True,checked_events=2,producer_modules_imported=False,computes_returns=False,**dict.fromkeys(WORK_FIELDS,0));r.update(full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.,ordered_choices_checked=2,original_episode_calls=1,candidate_selection_computed=False)
            receipt(r,2,'incumbent');r['checked_events']=3
            with self.assertRaises(ValueError):receipt(r,2,'incumbent')
            r['checked_events']=2;r.pop('full_pre_scores_supplied')
            with self.assertRaises(ValueError):receipt(r,2,'incumbent')
        def test_no_stage_or_model_imports(self):
            names=[]
            for n in ast.walk(ast.parse(Path(__file__).read_text())):
                if isinstance(n,ast.Import):names.extend(x.name.split('.')[0] for x in n.names)
                elif isinstance(n,ast.ImportFrom):names.append(n.module.split('.')[0])
            self.assertFalse(set(names)&{'numpy','pandas','sklearn','importlib'});self.assertFalse(any('monthly_' in n for n in names))
        def test_source_map_path_not_basename_alias(self):
            x={f'work/source/{i}.raw':'a'*64 for i in range(64)};self.assertEqual(len(canonical_source_map(x)),64)
            x['work/source/../bad.raw']=x.pop('work/source/1.raw')
            with self.assertRaises(ValueError):canonical_source_map(x)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));need(result.wasSuccessful(),'Synthetic reader self-checks');return result.testsRun


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    for name in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+name+'-sha256')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:print(json.dumps(run(args),ensure_ascii=False,indent=2,allow_nan=False),flush=True)
