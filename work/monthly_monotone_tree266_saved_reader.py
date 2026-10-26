"""266 saved evidence reader: standard library, read-only, no array decoding.

Written by the independent-verifier author as an additional saved-evidence check. Numerical
independence comes from the bound separately authored verifier. This reader
neither fits nor predicts, traverses trees, recomputes NAV statistics, replays
portfolios, writes a result review, nor mutates any saved artifact. Root must
supply five exact phase digests after actual successful terminal receipts.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_monotone_tree266'
PRIOR=ROOT/'outputs/monthly_gain_concentration265'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
CONTEXT=ROOT/'work/monthly_monotone_tree266_context.json'
CONTRACT=ROOT/'work/monthly_monotone_tree266_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_monotone_tree266_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_monotone_tree266_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
TERMINALS=OUT/'PHASE_TERMINALS.json'
RUNTIME=ROOT/'work/monthly_weak_tree261_RUNTIME_PINS.json'
ARMS=('incumbent_local67','unconstrained_weak_tree261_control','monotone_raw_weak_tree_candidate')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREFIXES=(0,1,53,88,122)
PREP={'FORECASTS.npz','LABELS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
HISTORICAL=ROOT/'outputs/monthly_weak_tree261'
ROLES=('calibration_ridge','unconstrained_control','monotone_candidate')
TREE_ROLES=ROLES[1:]
SCHEMAS={}
FLAGS={}
WORK_FIELDS=('independent_canonical_solves','independent_svd_solves','independent_tree_refits','unconstrained_tree_refits','monotone_tree_refits','manual_tree_traversals','unconstrained_manual_tree_traversals','monotone_manual_tree_traversals','shape_certificates_checked','constrained_nodes_checked','original_episode_calls','original_other_target_fits','original_local_fits')
FIXED={'work/monthly_weak_tree261_RUNTIME_PINS.json':'ad796b9d48f1a6dd910dfcf8d66c650359d3c2dd8851a3cd042be935079cc59d','work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'}
FEE_SOURCE=ROOT/'work/monthly_gain_concentration265_saved_reader.py'
FEE_SOURCE_SHA='8990f2e15e423808546da661ae36717a6dd83dd71fee377f918d55599a2decd8'
TREE_PARAMETERS=dict(loss='squared_error',quantile=None,learning_rate=.05,max_iter=100,max_leaf_nodes=31,max_depth=2,min_samples_leaf=50,l2_regularization=1.,max_features=1.,max_bins=255,categorical_features='from_dtype',monotonic_cst=None,interaction_cst=None,warm_start=False,early_stopping=False,scoring='loss',validation_fraction=.1,n_iter_no_change=10,tol=1e-7,verbose=0,random_state=1729)

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

def diagnostic_holdings_parity(current, previous):
    """Check display-only fee rounding; no tolerance for economic exports.

    Frozen engine sums at most six nonnegative turnover terms in set order,
    then scales by cost. gamma_6 bounds six binary64 rounding operations.
    This establishes compatibility with that representation error bound; it
    does not reconstruct the engine's unstored intermediate post value.
    """
    need(type(current) is list and type(previous) is list and len(current)==len(previous), 'Complete displayed holdings')
    gamma=Fraction(6,2**53-6)
    relative_bound=2*gamma/(1-gamma)
    differences=[]
    for i,(a,b) in enumerate(zip(current,previous)):
        need(type(a) is dict and type(b) is dict and set(a)==set(b) and 'fee' in a,'Holdings field schema')
        need({k:v for k,v in a.items() if k!='fee'}=={k:v for k,v in b.items() if k!='fee'},'Non-fee holdings fields must be exact')
        av,bv=a['fee'],b['fee']
        need(type(av) in (int,float) and type(bv) in (int,float) and math.isfinite(av) and math.isfinite(bv) and av>=0 and bv>=0,'Finite nonnegative display fees')
        if av==bv:continue
        need(av>0 and bv>0,'Zero display fee must remain exact')
        terms=set(a['symbols']) | (set(current[i-1]['symbols']) if i else set())
        need(len(terms)<=6,'At most six turnover terms for EW3')
        difference=abs(Fraction(av)-Fraction(bv))
        bound=relative_bound*min(Fraction(av),Fraction(bv))
        need(difference<=bound,'Display fee difference exceeds positive-sum binary64 bound')
        differences.append(dict(index=i,signal_date=a['signal_date'],execution_date=a['execution_date'],previous_fee=bv,current_fee=av,
            absolute_difference=float(difference),larger_fee_ulp=math.ulp(max(av,bv)),ulp_difference=float(difference)/math.ulp(max(av,bv)),
            exact_bound_numerator=bound.numerator,exact_bound_denominator=bound.denominator,turnover_terms_upper_bound=6))
    return dict(exact=not differences,display_fee_only=True,nonfee_fields_exact=True,differences=differences,
        gamma_operations=6,unit_roundoff='2**-53',roundoff_bound='2*gamma_6/(1-gamma_6)*min(fee_a,fee_b)',
        bound_checked_with_exact_fractions=True,intermediate_post_reconstructed=False,mechanism_compatible_not_exact_intermediate_proof=True,
        frozen_engine_path='outputs/monthly_factor_search_v2/validate.py',frozen_engine_sha256='b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4',
        frozen_engine_lines=[86,89,95],actual_portfolio_replay=False)

def diagnostic_summary_parity(current, previous, current_holdings, previous_holdings):
    """Allow only bound-checked displayed fees, with exact own-ledger totals.

    This does not recompute returns, NAV, fills or costs. Each saved total must
    equal the frozen engine's chronological binary64 ``total = 0.; total += fee``
    over its own already-saved HOLDINGS. No aggregate epsilon is used.
    """
    need(type(current) is dict and type(previous) is dict and set(current)==set(previous)==set(PATHS),'Exact summary path keys')
    display=diagnostic_holdings_parity(current_holdings,previous_holdings)
    totals={};stripped=[]
    for label,summary,holdings in (('current',current,current_holdings),('previous',previous,previous_holdings)):
        cost=summary['next_open_cost']
        need(type(cost) is dict and type(cost.get('metrics')) is dict and type(cost.get('slices')) is dict and type(cost['slices'].get('full122')) is dict,'Cost summary fee containers')
        metrics=cost['metrics'];full=cost['slices']['full122']
        need('fees_paid' in metrics and 'fees_paid' in full,'Both saved cost fee totals required')
        need(type(metrics.get('cost_per_side')) in (int,float) and metrics['cost_per_side']==.0005,'Diagnostic cost rate remains exact0.05percent per side')
        total=0.;last_signal=None;last_execution=None
        for holding in holdings:
            signal=holding.get('signal_date');execution=holding.get('execution_date')
            for value in (signal,execution):
                need(type(value) is str,'Native holdings date strings')
                try:canonical=datetime.strptime(value,'%Y-%m-%d').date().isoformat()
                except (ValueError,OverflowError):raise ValueError('Canonical holdings dates') from None
                need(canonical==value,'Canonical holdings dates')
            need(signal<=execution and (last_signal is None or last_signal<signal) and (last_execution is None or last_execution<execution),'Strict chronological holdings dates')
            last_signal=signal;last_execution=execution
            total+=holding['fee']
            need(math.isfinite(total),'Finite chronological display fee total')
        for location,container in (('metrics',metrics),('slices.full122',full)):
            value=container['fees_paid']
            need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Finite nonnegative summary fee '+label+'/'+location)
            need(value==total,'Exact own chronological display fee sum '+label+'/'+location)
        totals[label]=total
        # Remove precisely these two leaves, retaining every other key/value.
        clean=dict(summary);clean_cost=dict(cost);clean['next_open_cost']=clean_cost
        clean_cost['metrics']={k:v for k,v in metrics.items() if k!='fees_paid'}
        clean_cost['slices']=dict(cost['slices'])
        clean_cost['slices']['full122']={k:v for k,v in full.items() if k!='fees_paid'}
        stripped.append(clean)
    need(stripped[0]==stripped[1],'Every non-display-fee summary field must be exact')
    return dict(exact=current==previous,economic_summary_metrics_exact=True,nonfee_fields_exact=True,
        allowed_difference_paths=['next_open_cost.metrics.fees_paid','next_open_cost.slices.full122.fees_paid'],
        own_chronological_fee_totals=totals,own_totals_exact=True,full122_duplicate_exact=True,
        accumulation='binary64 total=0.0; chronological total+=holding[fee]',aggregate_tolerance_used=False,
        display_fee_review=display,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False)

def exact(actual,expected,label):
    """Typed saved JSON equality, preserving signed binary64 zero."""
    need(type(actual) is type(expected),label+' type')
    if type(actual) is float:need(actual.hex()==expected.hex(),label+' float bits')
    elif type(actual) is dict:
        need(set(actual)==set(expected),label+' keys')
        for key in expected:exact(actual[key],expected[key],label+'/'+str(key))
    elif type(actual) in (list,tuple):
        need(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):exact(a,b,label+'/'+str(i))
    else:need(actual==expected,label+' value')


def schema_flags():
    # The independent file supplies literal declarations, not executable code.
    names=('MODEL_COUNT_FIELDS','RIDGE_ANNUAL_KEYS','TREE_ANNUAL_KEYS','TREE_KEYS','COUNT_FIELDS','EVENT_KEYS','RANK_AUDIT_KEYS','WORK_FIELDS')
    schemas=constants(ROOT/'work/monthly_monotone_tree266_independent.py',names)
    schemas.update(constants(ROOT/'work/monthly_weak_tree261_kernel.py',('MODEL_AUDIT_KEYS','AUDIT_KEYS')))
    flags=constants(ROOT/'work/monthly_monotone_tree266_independent.py',('CALIBRATION_FLAGS','CONTROL_FLAGS','MODEL_FLAGS','CONTROL_SELECTION_FLAGS','SELECTION_FLAGS'))
    flags['INCUMBENT_FLAGS']=constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',))['INCUMBENT_GATE_FLAGS']
    schemas['REFERENCE_COUNT_FIELDS']=constants(ROOT/'work/monthly_monotone_tree266_reference.py',('COUNT_FIELDS',))['COUNT_FIELDS']
    need([len(schemas[k]) for k in ('MODEL_COUNT_FIELDS','COUNT_FIELDS','EVENT_KEYS','WORK_FIELDS')]==[22,17,14,13],'Final266 literal schema widths')
    need(schemas['WORK_FIELDS']==WORK_FIELDS,'All actual work fields')
    return schemas,flags


def model_counts(c,n,stage='full'):
    need(stage in ('calibration','control','full'),'Known model stage')
    counts_check(c,SCHEMAS['MODEL_COUNT_FIELDS'],'Model counts');integer(c['events'],'Model events',n)
    need(c['valid_training_rows']<=c['training_rows'],'Usable cohort subset')
    need(c['ordinary_years']==c['supported_years']+c['cold_years'] and c['supported_years']==c['matched_ridge_fits']+c['no_prediction_years'],'Annual partition')
    fits=c['matched_ridge_fits']
    for prefix,on in [('unconstrained',stage!='calibration'),('monotone',stage=='full')]:
        integer(c[prefix+'_tree_fits'],prefix+' paired fits',fits if on else 0)
        need(c[prefix+'_trees']==c[prefix+'_boosting_iterations']==100*c[prefix+'_tree_fits'],'100 fixed trees')
        integer(c[prefix+'_manual_tree_traversals'],prefix+' traversal work',100*c['prediction_rows'] if on else 0)
        need(c[prefix+'_trees']<=c[prefix+'_tree_nodes']<=7*c[prefix+'_trees'],'Depth2 node work')
    need(c['actual_model_fits']==fits+c['unconstrained_tree_fits']+c['monotone_tree_fits'],'Actual producer fits')
    need(c['original_other_target_fits']==c['original_local_fits']==0 and fits<=c['prediction_rows'],'Only calibration and trees fitted')
    if n<=1:need(all(v==0 for k,v in c.items() if k!='events'),'Empty/partial no model inputs consumed')


def stage_counts(c,stage):
    out=dict(c)
    for prefix in ('unconstrained','monotone'):
        if stage=='calibration' or stage=='control' and prefix=='monotone':
            for suffix in ('tree_fits','boosting_iterations','trees','tree_nodes','manual_tree_traversals'):out[prefix+'_'+suffix]=0
    out['actual_model_fits']=out['matched_ridge_fits']+out['unconstrained_tree_fits']+out['monotone_tree_fits']
    return out


def runtime_check(rows,years,stage='full'):
    wanted=[(role,year) for role in TREE_ROLES[:0 if stage=='calibration' else 1 if stage=='control' else 2] for year in years]
    need(type(rows) is list and [(r['role'],r['year']) for r in rows]==wanted,'Exact ordered runtime roles/years')
    for row in rows:
        need(set(row)=={'role','year','fit_threadpools','predict_threadpools'},'Runtime record keys')
        for phase in ('fit_threadpools','predict_threadpools'):
            pools=row[phase];need(type(pools) is list and pools,'Effective native pools')
            identities=[]
            for pool in pools:
                need(type(pool) is dict and type(pool.get('filepath')) is str and pool['filepath'],'Native library identity')
                integer(pool.get('num_threads'),'Native thread limit',1);identities.append(pool['filepath'])
            need(len(identities)==len(set(identities)),'Unique loaded native libraries')


def receipt(r,n,kind,constrained_nodes=None):
    need(type(r) is dict and r.get('passed') is True and 'synthetic_no_callback' not in r,'Passed actual independent '+kind)
    integer(r.get('checked_events'),'Receipt event scope',n)
    flag={'calibration':'CALIBRATION_FLAGS','control':'CONTROL_FLAGS','models':'MODEL_FLAGS','incumbent':'INCUMBENT_FLAGS','control_strategy':'CONTROL_SELECTION_FLAGS','selection':'SELECTION_FLAGS'}[kind]
    need(all(r.get(k) is True for k in FLAGS[flag]),'All independent flags '+kind)
    need(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Independent pre-return scope')
    for key in WORK_FIELDS:integer(r.get(key),'Work '+key)
    need(r['original_other_target_fits']==r['original_local_fits']==0,'Original other models reused')
    if kind in ('calibration','control','models'):
        stage='full' if kind=='models' else kind;c=r['counts'];model_counts(c,n,stage);exact(r['producer_equivalent_counts'],c,'Equivalent counts')
        for key,value in [('continuous_atol',2e-10),('continuous_rtol',2e-10),('rank_tolerance',0.)]:exact(r.get(key),value,'Frozen tolerance '+key)
        for key in ('exact_discrete','full_bundle_supplied','full_calibration_supplied','full_label_preservation_supplied','full_model_audit_supplied','shared_tree_trainer'):need(r.get(key) is True,'Complete scope '+key)
        need(r.get('independent_tree_training_algorithm') is False,'Shared trainer limitation')
        need(r.get('full_control_supplied') is (stage!='calibration') and r.get('full_candidate_supplied') is (stage=='full') and r.get('candidate_models_computed') is (stage=='full'),'Stage gate before new models')
        need(r['independent_canonical_solves']==r['independent_svd_solves']==c['matched_ridge_fits'],'Actual independent normal/SVD counts')
        for prefix in ('unconstrained','monotone'):
            integer(r[prefix+'_tree_refits'],prefix+' refits',c[prefix+'_tree_fits'])
            integer(r[prefix+'_manual_tree_traversals'],prefix+' own and supplied walks',2*c[prefix+'_manual_tree_traversals'])
        need(r['independent_tree_refits']==r['unconstrained_tree_refits']+r['monotone_tree_refits'],'Total tree refits')
        need(r['manual_tree_traversals']==r['unconstrained_manual_tree_traversals']+r['monotone_manual_tree_traversals'],'Total walks')
        integer(r['shape_certificates_checked'],'Actual structural checks',3*r['independent_tree_refits'])
        if constrained_nodes is not None:integer(r['constrained_nodes_checked'],'Actual constrained node checks',3*constrained_nodes if stage=='full' else 0)
        elif stage!='full':integer(r['constrained_nodes_checked'],'No constrained control nodes',0)
        integer(r['original_episode_calls'],'No model selectors',0)
    else:
        for key in ('full_scores_supplied','full_pre_scores_supplied'):need(r.get(key) is True,'Complete saved scores')
        exact(r.get('numerical_tolerance'),0.,'Exact decision tolerance')
        integer(r.get('ordered_choices_checked'),'Ordered choice scope',3*n if kind=='selection' else n)
        need(all(r[k]==0 for k in WORK_FIELDS if k!='original_episode_calls'),'Decision stages no models/trees')
        integer(r['original_episode_calls'],'Actual independent selectors',3 if kind=='selection' else 1)
        if kind in ('incumbent','control_strategy'):need(r.get('candidate_selection_computed') is False,'No candidate selection before gate')
        if kind=='control_strategy':need(r.get('full_control_supplied') is True,'Complete independent control strategy')
        if kind=='selection':
            need(r.get('full_bundle_supplied') is True and r.get('full_audit_supplied') is True,'Complete final selection evidence')
            counts_check(r['counts'],SCHEMAS['COUNT_FIELDS'],'Decision counts');integer(r['counts']['events'],'Decision events',n)


def decision_counts(events,n,source_eligible,width=49):
    rows=events[:n];c=dict.fromkeys(SCHEMAS['COUNT_FIELDS'],0)
    c.update(events=n,symbols_per_event=width,original_eligible_cells=source_eligible,incumbent_episode_calls=1,control_episode_calls=1,candidate_episode_calls=1)
    for i,r in enumerate(rows):
        need(set(r)==set(SCHEMAS['EVENT_KEYS']) and type(r['query_index']) is int and r['query_index']==i,'Full decision event schema')
        need(type(r['initial_partial']) is type(r['weak']) is bool and r['initial_partial'] is (i==0),'Named initial partial')
        for key in ('ranking_indices','active_indices','control_changed_delta_indices','candidate_changed_delta_indices'):
            ids=r[key];need(type(ids) is list and ids==sorted(set(ids)) and all(type(j) is int and 0<=j<width for j in ids),'Exact sorted indices '+key)
        wanted=r['ranking_indices'] if r['weak'] and not r['initial_partial'] else []
        need(r['active_indices']==wanted,'Only original U ordinary weak activation')
        for prefix in ('control','candidate'):
            d=r[prefix+'_delta'];need(type(d) is list and len(d)==width and all(finite(v) for v in d),'Finite full delta')
            changed=r[prefix+'_changed_delta_indices'];need([j for j,v in enumerate(d) if v!=0.]==changed and set(changed)<=set(wanted),'Only active delta cells')
            need(all(abs(v)<=.0375 for v in d),'Fixed pooled three-quarter slot bound')
            if not wanted:need(r[prefix+'_choices']==r['incumbent_choices'],'Nonweak/partial choices exact')
            c[prefix+'_changed_delta_cells']+=len(changed)
            c[prefix+'_changed_holdings_events']+=set(r[prefix+'_choices'])!=set(r['incumbent_choices'])
        for key in ('incumbent_choices','control_choices','candidate_choices'):
            js=r[key];need(type(js) is list and len(js) in (0,2,3) and len(set(js))==len(js) and all(type(j) is int and 0<=j<width for j in js),'Original ordered selector choices')
        c['initial_partial_events']+=r['initial_partial'];c['formal_events']+=not r['initial_partial'];c['ranking_cells']+=len(r['ranking_indices']);c['weak_applicable_cells']+=len(wanted)
        c['candidate_control_changed_holdings_events']+=set(r['candidate_choices'])!=set(r['control_choices'])
    integer(source_eligible,'Receipt-backed E count');need(c['ranking_cells']<=source_eligible<=width*n,'Original U subset of original E')
    return c


def rank_audit_check(a,events,counts):
    need(set(a)==set(SCHEMAS['RANK_AUDIT_KEYS']) and a['schema']=='monthly_monotone_tree266_rank_delta_v1','Rank audit schema')
    integer(a['events'],'Rank events',len(events));integer(a['ranking_cells'],'Rank population',counts['ranking_cells'])
    exact(a['active_indices'],[r['active_indices'] for r in events],'Saved activation rows')
    for prefix in ('control','candidate'):integer(a[prefix+'_changed_cells'],'Rank changed cells',counts[prefix+'_changed_delta_cells'])
    for key in set(SCHEMAS['RANK_AUDIT_KEYS'])-{'schema','events','ranking_cells','active_indices','control_changed_cells','candidate_changed_cells'}:need(a[key] is True,'Rank preservation '+key)


def saved_selection(training,orders):
    need(set(training)==set(orders)==set(ARMS),'Three economic arms')
    invalid={a:required_metrics(training[a],a==ARMS[0]) for a in ARMS}
    need(all(len(orders[a])>=88 for a in ARMS),'Complete first88 orders')
    candidate=ARMS[2];guards={};changed={}
    for ref in ARMS[:2]:
        rows=[i for i in range(88) if set(orders[candidate][i]['symbols'])!=set(orders[ref][i]['symbols'])];changed[ref]=rows
        valid=not invalid[candidate] and not invalid[ref]
        guards[ref]=dict(close=bool(valid and guard(training[candidate][PATHS[0]],training[ref][PATHS[0]],.005 if ref==ARMS[0] else 0.)),open=bool(valid and guard(training[candidate][PATHS[1]],training[ref][PATHS[1]],0.)),support=len(rows)>=3 and any(i<53 for i in rows) and any(i>=53 for i in rows))
    admitted=all(all(g.values()) for g in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=changed,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)
def legacy_tree_structure(tree,parameters):
    """Inspect stored structure; never evaluate any tree or feature row."""
    need(type(tree) is dict and set(tree)==set(SCHEMAS['TREE_KEYS'])-{'role','shape_certificate'},'Full serialized tree schema')
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

def tree_structure(tree,role):
    need(type(tree) is dict and set(tree)==set(SCHEMAS['TREE_KEYS']),'Complete dual-tree serialized schema')
    need(role in TREE_ROLES and tree['role']==role and tree['schema']=='monthly_monotone_tree266_serialized_tree_v1','Serialized role/schema')
    parameters=deepcopy(TREE_PARAMETERS)
    if role==TREE_ROLES[1]:parameters['monotonic_cst']=[1,0,0,0,0,0,0,0,0,0]
    exact(tree['parameters'],parameters,'Fixed complete21 role parameters')
    legacy={k:v for k,v in tree.items() if k not in ('role','shape_certificate')};legacy['schema']='monthly_weak_tree261_serialized_tree_v1'
    nodes=legacy_tree_structure(legacy,parameters);records=[]
    if role==TREE_ROLES[1]:
        for predictor in tree['trees']:
            ns=predictor['nodes']
            def descendants(start):
                todo=[start];out=[]
                while todo:
                    j=todo.pop()
                    if ns[j]['is_leaf']:out.append(j)
                    else:todo.extend((ns[j]['left'],ns[j]['right']))
                return sorted(out)
            for j,node in enumerate(ns):
                if node['is_leaf'] or node['feature_idx']!=0:continue
                left=descendants(node['left']);right=descendants(node['right'])
                lv=max(float.fromhex(ns[k]['value']) for k in left);rv=min(float.fromhex(ns[k]['value']) for k in right)
                need(lv<=rv,'Every feature0 split all-descendant monotonicity, without epsilon')
                records.append(dict(iteration=predictor['iteration'],node_index=j,left_leaf_indices=left,right_leaf_indices=right,left_max=lv.hex(),right_min=rv.hex()))
    wanted=dict(schema='monthly_monotone_tree266_shape_certificate_v1',role=role,feature_index=0,required=role==TREE_ROLES[1],constrained_nodes=records,passed=True)
    exact(tree['shape_certificate'],wanted,'Recomputed structural monotonicity certificate')
    return dict(nodes=nodes,constrained_nodes=len(records))


def training_axis(metadata):
    tm=metadata['training'];dates=[];ends=[]
    need(len(tm['dates'])==len(tm['ends'])==len(tm['symbols']),'Original training metadata axes')
    need(tm['dates']==sorted(set(tm['dates'])),'Ordered training months')
    for d,e,names in zip(tm['dates'],tm['ends'],tm['symbols']):
        need(type(d) is type(e) is str and d<e and len(names)==len(set(names)),'Original training identity')
        dates.extend([d]*len(names));ends.extend([e]*len(names))
    return dates,ends


def annual_query_ids(meta,events,year,n,width):
    return [i*width+j for i,d in enumerate(meta['dates'][:n]) if i and int(d[:4])==year for j in events[i]['ranking_indices']]


def model_summary(ma,meta,training_meta,events,width=49):
    need(set(ma)=={'model','labels',*ROLES},'Five complete model proof objects')
    model,lp=ma['model'],ma['labels'];need(set(model)==set(SCHEMAS['MODEL_AUDIT_KEYS']) and model['schema']=='monthly_monotone_tree266_model_audit_v1','Model schema')
    n=len(meta['dates']);c=model['counts'];model_counts(c,n)
    for key in ('original_row_order_preserved','original_common_good_preserved','all_targets_preserved','local_models_reused','other_targets_not_fitted','weak_only'):need(model[key] is True,'Original model preservation '+key)
    need(model['computes_returns'] is False,'No returns in models')
    exact(model['original_cache_parity'],dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False),'Original Ridge calibration gate')
    need(set(lp)=={'schema','rows','valid_rows','target_keys','target_sha256','common_good_sha256','labels_unchanged','common_good_unchanged','original_row_order_preserved','recentered','raw_target_reconstructed'},'Label proof schema')
    need(lp['schema']=='monthly_monotone_tree266_labels_preserved_v1' and lp['rows']==c['training_rows'] and c['valid_training_rows']<=lp['valid_rows']<=lp['rows'],'Label consumed scope')
    need(lp['target_keys']==['simple_excess','log_excess','vol_scaled','loss_utility'] and set(lp['target_sha256'])==set(lp['target_keys']),'All four unchanged targets')
    need(all(sha_value(h) for h in list(lp['target_sha256'].values())+[lp['common_good_sha256']]),'Complete target/source hashes')
    need(all(lp[k] is True for k in ('labels_unchanged','common_good_unchanged','original_row_order_preserved')) and lp['recentered'] is lp['raw_target_reconstructed'] is False,'No target rewriting')
    rowdays,rowends=training_axis(training_meta);years=sorted({int(d[:4]) for d in meta['dates'][1:]})
    annual=ma[ROLES[0]]['annual']
    for role,suffix in zip(ROLES,('ridge_audit','tree_audit','tree_audit')):
        need(set(ma[role])=={'schema','annual'} and ma[role]['schema']=='monthly_monotone_tree266_'+suffix+'_v1','Role audit schema')
        need([r['year'] for r in ma[role]['annual']]==years,'Exactly requested ordinary years')
    fitted=[];predictions=0;node_counts={r:0 for r in TREE_ROLES};constrained=0
    for i,r in enumerate(annual):
        need(set(r)==set(SCHEMAS['RIDGE_ANNUAL_KEYS']),'Ridge annual schema');year=r['year']
        need(type(year) is int and r['cutoff']==f'{year-1}-12-31' and r['lower_bound']==f'{year-6}-12-31','Prior December five-year clock')
        ids=r['training_row_ids'];need(type(ids) is list and ids==sorted(set(ids)) and all(type(j) is int and 0<=j<len(rowdays) and r['lower_bound']<rowdays[j] and rowends[j]<=r['cutoff'] for j in ids),'Mature global training IDs')
        groups=Counter(rowdays[j][:7] for j in ids);need(all(k>=2 for k in groups.values()),'Original common cohort month support')
        months=len(groups);integer(r['training_months'],'Distinct mature training months',months)
        qids=annual_query_ids(meta,events,year,n,width);exact(r['query_row_ids'],qids,'Complete original U year queries')
        status='cold_start' if months<18 else 'fit' if qids else 'no_prediction_rows';need(r['status']==status,'Status from maturity and source U, not caller fit claim')
        if status=='fit':
            fitted.append(year);predictions+=len(qids)
            need(len(r['weights'])==len(ids) and all(finite(w) and w>0 for w in r['weights']),'Positive finite complete weights')
            # Algebraic equality within each month is exact; exp2 source arithmetic
            # is checked by the bound canonical/SVD verifier, not stdlib pow.
            month_weights={}
            for rowid,w in zip(ids,r['weights']):
                month=rowdays[rowid][:7]
                if month in month_weights:exact(w,month_weights[month],'Equal weight within original month')
                else:month_weights[month]=w
            for key in ('mean','scale','beta','rhs'):need(type(r[key]) is list and len(r[key])==10 and all(finite(v) for v in r[key]),'Ten finite Ridge fields')
            need(all(x>0 for x in r['scale']) and finite(r['intercept']) and len(r['gram'])==10 and all(len(x)==10 and all(finite(v) for v in x) for x in r['gram']),'Finite Ridge system')
            need(len(r['raw_predictions'])==len(qids) and all(finite(v) for v in r['raw_predictions']),'Saved Ridge query predictions')
        else:need(r['weights']==r['raw_predictions']==[] and all(r[k] is None for k in ('mean','scale','beta','intercept','gram','rhs')),'No coefficients or predictions in skipped year')
        for role in TREE_ROLES:
            tr=ma[role]['annual'][i];need(set(tr)==set(SCHEMAS['TREE_ANNUAL_KEYS']),'Tree annual schema')
            for key in ('year','cutoff','training_row_ids','query_row_ids','status'):exact(tr[key],r[key],'Common role cohort/query '+key)
            if status!='fit':need(tr['tree'] is None and tr['raw_predictions']==tr['manual_predictions']==[],'No tree borrowed from later queries')
            else:
                need(len(tr['raw_predictions'])==len(qids) and all(finite(v) for v in tr['raw_predictions']),'Complete saved raw tree predictions')
                exact(tr['manual_predictions'],tr['raw_predictions'],'Exact saved manual-versus-library predictions')
                summary=tree_structure(tr['tree'],role);node_counts[role]+=summary['nodes'];constrained+=summary['constrained_nodes']
    integer(c['ordinary_years'],'Ordinary year count',len(years));integer(c['matched_ridge_fits'],'Annual fits',len(fitted));integer(c['prediction_rows'],'Annual query predictions',predictions)
    integer(c['cold_years'],'Cold years',sum(r['status']=='cold_start' for r in annual));integer(c['no_prediction_years'],'No-query years',sum(r['status']=='no_prediction_rows' for r in annual))
    for role,prefix in zip(TREE_ROLES,('unconstrained','monotone')):integer(c[prefix+'_tree_nodes'],'Stored tree node totals',node_counts[role])
    runtime_check(model['execution_runtime'],fitted)
    expected_rows=sum(day<=f'{years[-1]-1}-12-31' for day in rowends) if years else 0
    integer(c['training_rows'],'Consumed mature source rows',expected_rows)
    return dict(fitted_years=fitted,annual=[dict(year=r['year'],status=r['status'],training_months=r['training_months'],training_rows=len(r['training_row_ids']),query_rows=len(r['query_row_ids'])) for r in annual],counts=c,tree_nodes_structure_checked=node_counts,constrained_nodes_structure_checked=constrained,shape_certificates_recomputed_from_saved_nodes=True,model_arithmetic_recomputed=False,manual_inference_recomputed=False,usable_row_membership_proven_by_bound_independent_verifier=True),rowends
def meta_prefix(meta,n):
    out=dict(meta)
    for k in ('dates','ends','symbols'):out[k]=meta[k][:n]
    return out


def layout_digest(n,width,meta,cohorts,u_indices):
    """Reproduce only the host's JSON-layout + Boolean matrix fingerprint.

    The ndarray representation is bool/C-order bytes built from saved integer
    indices. No NPZ is opened, and this does not prove those indices equal E&U.
    That numerical fact belongs to the authenticated preparation receipts.
    """
    h=hashlib.sha256()
    def walk(v):
        h.update(type(v).__name__.encode()+b':')
        if isinstance(v,dict):
            for k in sorted(v):walk(k);walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        else:h.update(repr(v).encode())
    data=dict(n=n,width=width,monthly=meta_prefix(meta,n),annual_cohorts=cohorts)
    h.update(b'dict:')
    for key in sorted(list(data)+['original_u']):
        walk(key)
        if key=='original_u':
            h.update(b'ndarray:bool');h.update(str((n,width)).encode())
            h.update(bytes(int(j in row) for row in u_indices for j in range(width)))
        else:walk(data[key])
    return h.hexdigest()


def reference_chain(ev,plan,binding):
    need(set(binding)=={'schema','files_sha256','completion_sha256','prepared_completion_sha256','plan_sha256','prepared_members','npz_decoded','hash_scope'} and binding['schema']=='monthly_monotone_tree266_historical_binding_v1','Historical binding schema')
    expected={HISTORICAL/'PLAN.json',HISTORICAL/'COMPLETION.json',HISTORICAL/'prepared/COMPLETION.json'}|{HISTORICAL/'prepared'/name for name in PREP}
    mapping={local(p):h for p,h in binding['files_sha256'].items()}
    need(len(binding['files_sha256'])==len(mapping)==13 and set(mapping)==expected,'All13 exact historical files')
    for path,h in mapping.items():need(h==ev.registered(plan,path),'Historical file is registered');ev.pin(path,h)
    need(binding['prepared_members']==sorted(PREP) and binding['npz_decoded'] is False,'Complete opaque reference manifest scope')
    for key,path in [('completion_sha256',HISTORICAL/'COMPLETION.json'),('prepared_completion_sha256',HISTORICAL/'prepared/COMPLETION.json'),('plan_sha256',HISTORICAL/'PLAN.json')]:need(binding[key]==mapping[path],'Reference binding '+key)
    c=ev.read(HISTORICAL/'COMPLETION.json');pc=ev.read(HISTORICAL/'prepared/COMPLETION.json')
    need(c['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and c['protected_files']==21444,'Historical completed independent verification')
    need(pc['status']=='COMPLETE' and pc['events']==122 and pc['new_returns_computed'] is False and set(pc['artifacts_sha256'])==PREP,'Historical full preparation')
    need(c['plan_sha256']==pc['plan_sha256']==binding['plan_sha256'],'Historical PLAN links')
    for name,h in pc['artifacts_sha256'].items():need(h==mapping[HISTORICAL/'prepared'/name],'Historical complete member binding')
    need(c['model_audit_sha256']==pc['artifacts_sha256']['MODEL_AUDIT.json'] and c['independent_prepare_sha256']==pc['artifacts_sha256']['INDEPENDENT_PREPARE.json'],'Closed model and independent proof binding')
    return {k:ev.read(HISTORICAL/'prepared'/name) for k,name in [('model','MODEL_AUDIT.json'),('meta','META.json'),('orders','ORDERS.json'),('decision','DECISION_AUDIT.json')]}


def project_annual(row,qi,status,ridge):
    """Crop metadata before any model body; skipped prefix never borrows fit."""
    keys=('year','cutoff','training_row_ids')+(('lower_bound','training_months') if ridge else ())
    out={k:deepcopy(row[k]) for k in keys};out.update(query_row_ids=list(qi),status=status)
    if status!='fit':
        out['raw_predictions']=[]
        if ridge:out.update(weights=[],mean=None,scale=None,beta=None,intercept=None,gram=None,rhs=None)
        else:out.update(tree=None,manual_predictions=[])
        return out
    need(row['status']=='fit','A prefix fit requires a historical/full-year fit')
    oldids=row['query_row_ids'];need(oldids==sorted(set(oldids)),'Sorted global query IDs')
    positions={q:i for i,q in enumerate(oldids)};need(all(q in positions for q in qi),'Consumed query identity subset')
    for key in ('raw_predictions',) if ridge else ('raw_predictions','manual_predictions'):
        values=row[key];need(type(values) is list and len(values)==len(oldids),'Prediction axis')
        out[key]=[values[positions[q]] for q in qi]
    if ridge:
        for key in ('weights','mean','scale','beta','intercept','gram','rhs'):out[key]=deepcopy(row[key])
    else:out['tree']=deepcopy(row['tree'])
    return out


def historical_models_check(ma,old,meta,events,orders):
    exact(old['meta'],meta,'Historical monthly identities')
    need(set(old['model'])=={'model','labels','control','candidate'},'Complete historical model JSON')
    lp=deepcopy(ma['labels']);lp['schema']='monthly_weak_tree261_labels_preserved_v1';exact(lp,old['model']['labels'],'Historical original label preservation')
    for newrole,oldrole in ((ROLES[0],'control'),(ROLES[1],'candidate')):
        expected_years=[r['year'] for r in ma[newrole]['annual']]
        source={r['year']:r for r in old['model'][oldrole]['annual']};need(len(source)==len(old['model'][oldrole]['annual']) and set(source)==set(expected_years),'Historical requested years')
        for current in ma[newrole]['annual']:
            original=source[current['year']]
            for key in ('year','cutoff','training_row_ids')+(('lower_bound','training_months') if newrole==ROLES[0] else ()):exact(current[key],original[key],'Historical same cohort '+key)
            wanted=project_annual(original,current['query_row_ids'],current['status'],newrole==ROLES[0]);actual=deepcopy(current)
            if newrole==ROLES[1] and actual['tree'] is not None:
                actual['tree']={k:v for k,v in actual['tree'].items() if k not in ('role','shape_certificate')};actual['tree']['schema']='monthly_weak_tree261_serialized_tree_v1'
            exact(actual,wanted,'Historical complete annual model/prediction projection')
    exact(orders[ARMS[0]],old['orders']['incumbent_local67'],'Historical incumbent ordered scores/choices')
    exact(old['orders']['matched_pooled_ridge'],old['orders']['incumbent_local67'],'Historical Ridge calibration ordered identity')
    exact(orders[ARMS[1]],old['orders']['weak_pooled_tree'],'Actual unconstrained economic control')
    oldevents=old['decision']['events'];need(len(oldevents)==len(events),'Historical decision event count')
    for current,prior in zip(events,oldevents):
        for key in ('query_index','date','initial_partial','weak','ranking_indices','active_indices','incumbent_choices','episode_budget'):exact(current[key],prior[key],'Historical original decision '+key)
        exact(current['control_delta'],prior['delta'],'Historical unconstrained delta')
        exact(current['control_changed_delta_indices'],prior['changed_delta_indices'],'Historical changed delta cells')
        exact(current['control_choices'],prior['tree_choices'],'Historical unconstrained selected indices')
    return dict(files_hashed=13,complete_prepared_members=10,saved_annual_models_and_predictions_exact=True,complete_tree_projection_exact=True,labels_metadata_exact=True,control_orders_and_ordered_scores_exact=True,control_delta_json_exact=True,npz_decoded=False,forecast_arrays_scores_and_labels_numerically_recomputed=False,opaque_arrays_verified_by_bound_independent_and_reference_gates=True)


def reference_envelope_check(value,n,width,stage,binding,ma,meta,events,mc):
    need(set(value)=={'reference_prepared_sha256','source_layout_sha256','consumed_years','annual_cohorts','original_u_indices','comparison'},'Reference envelope exact schema')
    need(value['reference_prepared_sha256']==binding['prepared_completion_sha256'],'Reference prepared digest')
    years=sorted({int(d[:4]) for d in meta['dates'][1:n]});source={r['year']:r for r in ma[ROLES[0]]['annual']}
    cohort_keys=('year','cutoff','lower_bound','training_row_ids','training_months')
    expected=[{k:source[y][k] for k in cohort_keys} for y in years]
    exact(value['consumed_years'],years,'Reference consumed years');exact(value['annual_cohorts'],expected,'Prefix cohort projection')
    u=[r['ranking_indices'] for r in events[:n]];exact(value['original_u_indices'],u,'Reference original U prefix')
    need(value['source_layout_sha256']==layout_digest(n,width,meta,expected,u),'Source layout fingerprint from saved Boolean indices')
    r=value['comparison'];need(r['schema']=='monthly_monotone_tree266_historical_'+stage+'_reference_v1','Reference stage')
    integer(r['checked_events'],'Reference events',n);integer(r['width'],'Reference width',width)
    for key in ('passed','source_layout_independently_verified_by_caller','reference_manifest_verified_by_caller','original_query_domain_used','prefix_fit_state_from_verified_source','full_annual_fit_not_borrowed','exact_bits_and_types'):need(r[key] is True,'Historical reference check '+key)
    need(r['receipt_authentication_performed'] is r['candidate_models_computed'] is r['computes_returns'] is False and r['model_fits']==0,'Reference no fitting/authentication claim')
    exact(r['numeric_tolerance'],0.,'Exact historical numeric tolerance')
    need(r['reference_role']==('control' if stage=='calibration' else 'candidate') and r['current_role']==(ROLES[0] if stage=='calibration' else ROLES[1]),'Genuine historical economic role mapping')
    counts=dict.fromkeys(SCHEMAS['REFERENCE_COUNT_FIELDS'],0);qids={y:annual_query_ids(meta,events,y,n,width) for y in years}
    fit=[y for y in years if source[y]['training_months']>=18 and qids[y]]
    counts.update(forecast_numeric_cells_compared=2*n*width,availability_cells_compared=n*width,annual_metadata_rows_compared=len(years),annual_training_row_ids_compared=sum(len(source[y]['training_row_ids']) for y in years),annual_query_row_ids_compared=sum(len(qids[y]) for y in years),query_prediction_values_compared=sum(len(qids[y]) for y in fit)*(1 if stage=='calibration' else 2),score_numeric_cells_compared=2*n*width,ordered_events_compared=n,verification_binding_checks=1)
    if stage=='calibration':counts.update(label_values_compared=mc['training_rows'] if years else 0,label_row_ids_compared=mc['training_rows'] if years else 0,ridge_models_compared=len(fit),calibration_incumbent_score_cells_compared=2*n*width,calibration_incumbent_ordered_events_compared=n)
    else:counts['tree_models_compared']=len(fit)
    counts_check(r['counts'],SCHEMAS['REFERENCE_COUNT_FIELDS'],'Historical exact comparison count schema');exact(r['counts'],counts,'All17 historical comparison counts')
    return counts
def strategy_check(value,n,stage):
    integer(value['events'],'Strategy events',n);integer(value['producer_episode_calls'],'Strategy producer selector',1)
    for key in ('passed','pre_scores_exact','final_scores_exact','ordered_choices_exact'):need(value[key] is True,'Complete pre-candidate strategy proof')
    if stage=='calibration':
        need(value['matched_ranks_exact'] is True and value['delta_exact_zero'] is True,'Internal Ridge equals original')
    receipt(value['independent'],n,'incumbent' if stage=='calibration' else 'control_strategy')


def execution_accounting(e):
    entries=[];invocations=[('main',122,e)]+[('physical',v['prefix'],v) for v in e['physical_prefixes'] if v['prefix']!=122]
    producer_fields=('matched_ridge_fits','unconstrained_tree_fits','unconstrained_boosting_iterations','unconstrained_trees','unconstrained_tree_nodes','unconstrained_manual_tree_traversals','monotone_tree_fits','monotone_boosting_iterations','monotone_trees','monotone_tree_nodes','monotone_manual_tree_traversals')
    for scope,n,v in invocations:
        models=v['models'] if scope=='main' else v['model'];c=models['counts']
        checks=[v['early_incumbent'],v['calibration'],v['calibration_strategy']['independent'],v['control'],v['control_strategy']['independent'],models,v['incumbent'],v['selection']]
        item=dict(scope=scope,events=n,**{'producer_'+k:c[k] for k in producer_fields});item['producer_episode_calls']=6
        item.update({k:sum(q[k] for q in checks) for k in WORK_FIELDS})
        item['total_model_fit_calls']=sum(item['producer_'+k] for k in ('matched_ridge_fits','unconstrained_tree_fits','monotone_tree_fits'))+item['independent_canonical_solves']+item['independent_svd_solves']+item['independent_tree_refits'];entries.append(item)
    cached=e['cached_prefixes'];checks=[v[k] for v in cached for k in ('incumbent','selection')]
    cw=dict(invocations=len(cached),producer_episode_calls=3*len(cached),model_fit_calls=0,**{k:sum(q[k] for q in checks) for k in WORK_FIELDS})
    totals={k:sum(v[k] for v in entries) for k in entries[0] if k not in ('scope','events')};totals['producer_episode_calls']+=cw['producer_episode_calls']
    for key in WORK_FIELDS:totals[key]+=cw[key]
    rw={k:sum(v['reference261_'+stage]['comparison']['counts'][k] for _,_,v in invocations for stage in ('calibration','control')) for k in SCHEMAS['REFERENCE_COUNT_FIELDS']}
    return dict(schema='monthly_monotone_tree266_execution_accounting_v1',model_invocations=len(entries),physical122_reuses_main=True,entries=entries,cached_decisions=cw,totals=totals,historical_reference_comparison_work=rw,original_complete_model_refits=0,matched_ridge_is_actual_calibration_refit=True,manual_traversal_is_model_fit=False,source_load_scope='Full pinned reference JSON/NPZ decoded once; opaque full input hashes and immutable snapshots are not model fits or zero-read claims')


def preparation_evidence(e,ma,da,meta,training_meta,rowends,width=49):
    need(e['all123_order_prefixes'] is e['original_arrays_unchanged'] is e['original_training_unchanged'] is e['historical_reference_unchanged'] is True,'Preserved original and historical source')
    exact(e['main_model_counts'],ma['model'],'Saved main model audit');exact(da['incumbent_gate'],e['incumbent'],'Final original gate');exact(da['counts'],e['selection']['counts'],'Final independent decision binding')
    cached=e['cached_prefixes'];physical=e['physical_prefixes'];events=da['events']
    need(type(cached) is list and len(cached)==123 and type(physical) is list and [v['prefix'] for v in physical]==list(PREFIXES),'All123 cached/five physical prefixes')
    eligible=0
    for n,v in enumerate(cached):
        integer(v['prefix'],'Cached index',n);need(v['exact'] is True,'Cached exact')
        receipt(v['incumbent'],n,'incumbent');receipt(v['selection'],n,'selection')
        dc=v['selection']['counts'];exact(dc,decision_counts(events,n,dc['original_eligible_cells'],width),'Derived cached decision counts')
        need(eligible<=dc['original_eligible_cells']<=eligible+width,'E prefix count bound');eligible=dc['original_eligible_cells']
    source={r['year']:r for r in ma[ROLES[0]]['annual']};mono={r['year']:r for r in ma[ROLES[2]]['annual']}
    for n,v,isphysical in [(122,e,False)]+[(v['prefix'],v,True) for v in physical]:
        if isphysical:need(v['exact'] is True,'Exact physical prefix')
        audit=v['model_counts'] if isphysical else v['main_model_counts'];c=audit['counts'];model_counts(c,n)
        years=sorted({int(d[:4]) for d in meta['dates'][1:n]});fit=[y for y in years if source[y]['training_months']>=18 and annual_query_ids(meta,events,y,n,width)]
        cn=sum(len(mono[y]['tree']['shape_certificate']['constrained_nodes']) for y in fit)
        for kind in ('calibration','control','models','incumbent','selection'):
            record=v['model' if kind=='models' and isphysical else kind];receipt(record,n,kind,cn)
        exact(v['model' if isphysical else 'models']['counts'],c,'Full stage counts')
        for stage in ('calibration','control'):
            exact(v[stage]['counts'],stage_counts(c,stage),'Pre-candidate stage counts')
            strategy_check(v[stage+'_strategy'],n,stage)
            reference_envelope_check(v['reference261_'+stage],n,width,stage,e['historical_reference_binding'],ma,meta,events,c)
        receipt(v['early_incumbent'],n,'incumbent');exact(v['historical_reference_binding'],e['historical_reference_binding'],'Every physical historical chain')
        original=v['original_reference'];integer(original['events'],'Early original scope',n)
        for key in ('passed','source_incumbent_orders_exact','prior_incumbent_exact','reference261_incumbent_exact'):need(original[key] is True,'Early original source chain')
        exact(v['selection']['counts'],cached[n]['selection']['counts'],'Physical/cached decision domain')
        cutoff=f'{years[-1]-1}-12-31' if years else None
        integer(c['training_rows'],'Physical mature source rows',sum(d<=cutoff for d in rowends) if years else 0)
        integer(c['ordinary_years'],'Physical years',len(years));integer(c['matched_ridge_fits'],'Actual prefix annual fits',len(fit))
        integer(c['cold_years'],'Physical cold years',sum(source[y]['training_months']<18 for y in years));integer(c['no_prediction_years'],'Physical no-query years',sum(source[y]['training_months']>=18 and not annual_query_ids(meta,events,y,n,width) for y in years))
        integer(c['prediction_rows'],'Physical query rows',sum(len(annual_query_ids(meta,events,y,n,width)) for y in fit))
        for role,prefix in zip(TREE_ROLES,('unconstrained','monotone')):
            rows={r['year']:r for r in ma[role]['annual']}
            integer(c[prefix+'_tree_nodes'],'Physical fitted tree nodes',sum(sum(len(t['nodes']) for t in rows[y]['tree']['trees']) for y in fit))
        runtime_check(audit['execution_runtime'],fit)
        for key in set(SCHEMAS['MODEL_AUDIT_KEYS'])-{'counts','execution_runtime','original_cache_parity'}:exact(audit[key],ma['model'][key],'Physical model declarations')
        expected=dict(ma['model']['original_cache_parity']);expected['events']=n;exact(audit['original_cache_parity'],expected,'Physical original cache proof')
        if isphysical and n==122:
            need(v['model_recomputed'] is False and v.get('reuses_main_verified_full') is True,'Physical122 exact main reuse')
            for key in ('early_incumbent','original_reference','calibration','calibration_strategy','reference261_calibration','control','control_strategy','reference261_control','incumbent','selection','historical_reference_binding'):exact(v[key],e[key],'Reused full evidence')
            exact(v['model'],e['models'],'Reused full model receipt');exact(audit,ma['model'],'Reused model audit')
        elif isphysical:
            need(v['model_recomputed'] is True and not v.get('reuses_main_verified_full',False),'Real physical prefix recomputation')
            expected_scope=dict(events=n,training_months=sum(d<=cutoff for d in training_meta['training']['ends']) if years else 0,training_rows=c['training_rows'],cutoff=cutoff,global_row_ids_start=0,global_row_ids_preserved=True,training_numeric_consumed=bool(c['training_rows']),physically_shortened=True)
            exact(v['input_scope'],expected_scope,'Physically restricted numeric input boundary')
    wanted=execution_accounting(e);exact(e['fit_execution_accounting'],wanted,'Every actual fit/selector/reference count once')
    return wanted


def formal_bindings(ev,plan):
    for p,h in FIXED.items():need(ev.registered(plan,p)==h,'Fixed source registered');ev.pin(p,h)
    context=ev.read(CONTEXT,ev.registered(plan,CONTEXT));contract=ev.read(CONTRACT,ev.registered(plan,CONTRACT))
    exact(plan['context'],context,'Embedded context');exact(plan['implementation_contract'],contract,'Embedded contract')
    need(plan['context_sha256']==digest(CONTEXT) and plan['implementation_contract_sha256']==digest(CONTRACT) and contract['context_sha256']==digest(CONTEXT),'Final top-level definition binding')
    for record in (context,contract):need(record['frozen'] is True and record['registration_authorized'] is True and record['actual_data_started'] is False,'Prospective frozen contracts')
    need(context['arms']==list(ARMS) and context['only_promotable']==ARMS[2] and context['diagnostic_only']==ARMS[1] and context['registered_trials']==3,'Three economic arms and only candidate promotion')
    tests=ev.read(TESTS,ev.registered(plan,TESTS));review=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0,'Formal synthetic success')
    need(tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Pinned runtime')
    need(review['passed'] is True and review['unresolved_blockers']==[] and review['actual_returns_computed'] is False and review['code_sha256']==tests['code_sha256'],'Final code review and tests')
    need(review['context_sha256']==digest(CONTEXT) and review['implementation_contract_sha256']==digest(CONTRACT),'Review definitions')
    expected={ROOT/'work'/name for stem in ('host','kernel','independent','admission','reference') for name in ('monthly_monotone_tree266_'+stem+'.py','test_monthly_monotone_tree266_'+stem+'.py')}
    mapping={local(p):h for p,h in tests['code_sha256'].items()};need(len(mapping)==len(tests['code_sha256'])==10 and set(mapping)==expected,'Exactly ten formal source/test files')
    for p,h in mapping.items():need(ev.registered(plan,p)==h,'Registered formal source');ev.pin(p,h)
    for p in (ROOT/'work/monthly_weak_tree261_kernel.py',ROOT/'work/monthly_calendar_clock248_v2_independent.py',FEE_SOURCE,Path(__file__),ROOT/'work/test_monthly_monotone_tree266_saved_reader.py'):ev.pin(p,ev.registered(plan,p))
    need(digest(FEE_SOURCE)==FEE_SOURCE_SHA,'Immutable fee helper source')
    old=ast.parse(FEE_SOURCE.read_text());new=ast.parse(Path(__file__).read_text())
    for name in ('diagnostic_holdings_parity','diagnostic_summary_parity'):
        a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
        need(ast.dump(a)==ast.dump(b),'Unchanged frozen fee helper AST')
    return context,contract,tests
def full_original_chain(ev,plan):
    folder=REBUILD.parent
    parity=ev.read(REBUILD/'PARITY.json',ev.registered(plan,REBUILD/'PARITY.json'))
    ep=ev.read(folder/'evaluation/PLAN.json',ev.registered(plan,folder/'evaluation/PLAN.json'))
    ec=ev.read(folder/'evaluation/results/COMPLETION.json',ev.registered(plan,folder/'evaluation/results/COMPLETION.json'))
    ic=ev.read(folder/'independent_review/COMPLETION.json',ev.registered(plan,folder/'independent_review/COMPLETION.json'))
    need(parity['arm']=='control' and parity['passed'] is True and parity['returns_computed'] is False,'Original169 control source')
    need(ep['rebuild_plan_sha256']==parity['source_plan_sha256']==digest(folder/'PLAN.json'),'Original169 evaluation/source PLAN link')
    for path in [REBUILD/name for name in parity['artifacts_sha256']]+[REBUILD/'PARITY.json',folder/'PLAN.json']:need(ev.registered(ep,path)==ev.registered(plan,path),'Complete inherited evaluation cache closure')
    need(ec['status']=='COMPLETE' and ec['evaluation_plan_sha256']==digest(folder/'evaluation/PLAN.json') and ec['rebuild_plan_sha256']==parity['source_plan_sha256'],'Original169 result links')
    need(ec['exact_control_rechecked'] is ec['source_hashes_verified_before_after'] is True,'Original169 exact cache recheck')
    need(ic['status']=='PASS' and ic['checked_paths']==6 and ic['checked_slices']==30 and ic['evaluation_completion_sha256']==digest(folder/'evaluation/results/COMPLETION.json'),'Original169 independent outcome chain')


def zero_rates(results):
    for arm in ARMS:
        for path in PATHS[:2]:
            value=results['arms'][arm][path]
            for metrics in (value['metrics'],value['slices']['full122']):
                need(finite(metrics.get('cost_per_side')) and metrics['cost_per_side']==0.,'Six ZERO cost rates exact0')
                need(finite(metrics.get('fees_paid')) and metrics['fees_paid']==0.,'Six ZERO fee totals exact0')


def run(args):
    global SCHEMAS,FLAGS
    need(platform.python_version()=='3.9.6','Pinned wrapper runtime')
    for name in ('plan','prepared','frozen','results','terminals'):need(sha_value(getattr(args,name+'_sha256')),'Explicit phase digest '+name)
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256);context,contract,tests=formal_bindings(ev,plan);SCHEMAS,FLAGS=schema_flags()
    runtime=ev.read(RUNTIME,ev.registered(plan,RUNTIME));need(len(runtime['input_sha256'])==3309,'Pinned numeric package inventory')
    for path,h in runtime['input_sha256'].items():need(ev.registered(plan,path)==h,'Registered runtime file closure')
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(state['completed'])==261 and sum(r['trials'] for r in state['completed'])==1294 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Registered predecessor state')
    prior=ev.read(PRIOR/'COMPLETION.json',ev.registered(plan,PRIOR/'COMPLETION.json'));need(prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior['protected_files']==21444 and prior['target_reached'] is False,'Closed predecessor with unmet goal')
    need(prior['plan_sha256']==ev.pin(PRIOR/'PLAN.json',ev.registered(plan,PRIOR/'PLAN.json')) and context['predecessor']['completion_sha256']==digest(PRIOR/'COMPLETION.json'),'Predecessor PLAN/context chain')
    priorreview=ev.read(ROOT/'work/monthly_gain_concentration265_RESULT_REVIEW.json',ev.registered(plan,ROOT/'work/monthly_gain_concentration265_RESULT_REVIEW.json'))
    need(priorreview['passed'] is True and priorreview['unresolved_blockers']==[] and prior['result_review_sha256']==context['predecessor']['result_review_sha256']==digest(ROOT/'work/monthly_gain_concentration265_RESULT_REVIEW.json'),'Bound predecessor independent saved review')
    protected=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protected)==plan['protected_files']==21444 and all(sha_value(v) for v in protected.values()),'Protected inventory')
    terminal=ev.read(TERMINALS,args.terminals_sha256);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP);need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared scope before returns')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'First88 immutable chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selected artifacts')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Nine complete outcome paths')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed full122 chronology')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Fixed53/35/34 split')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']) and set(orders)==set(ARMS),'Original49 and three arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order clock')
        for o in orders[arm]:need(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(v) for v in o['scores']),'Original finite ordered source selections')
    source=ev.read(OUT/'prepared/SOURCE_PARITY.json');source_summary=source_proof(ev,plan,source);full_original_chain(ev,plan)
    exact(source['historical261_reference'],plan['historical261_reference'],'Registered historical source binding')
    need(source['prior_prepared_sha256']==ev.registered(plan,PRIOR/'prepared/COMPLETION.json'),'Immediate predecessor source manifest')
    for k in ('historical_control_before_candidate_fit','full_independent_before_candidate_selection','reference_numeric_files_fully_decoded'):need(source[k] is True,'Actual source scope '+k)
    need(source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Source reuse disclosed')
    historical=reference_chain(ev,plan,source['historical261_reference'])
    priorprep=ev.read(PRIOR/'prepared/COMPLETION.json',source['prior_prepared_sha256']);need(priorprep['status']=='COMPLETE' and priorprep['plan_sha256']==prior['plan_sha256'],'Immediate predecessor preparation chain')
    for name in ('META.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz'):ev.pin(PRIOR/'prepared'/name,priorprep['artifacts_sha256'][name])
    exact(meta,ev.read(PRIOR/'prepared/META.json'),'Immediate predecessor identities');exact(orders[ARMS[0]],ev.read(PRIOR/'prepared/ORDERS.json')[ARMS[0]],'Immediate predecessor incumbent orders')
    exact(orders[ARMS[0]],ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Original Local67 ordered source')
    metadata=ev.read(REBUILD/'MONTHLY_META.json',ev.registered(plan,REBUILD/'MONTHLY_META.json'));exact(meta,metadata['monthly'],'Original169 monthly identities')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar');positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):need(d in positions and (i==0 or positions[d]+1<len(axis) and axis[positions[d]+1][:7]!=d[:7]),'Original official monthends')
    ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');da=ev.read(OUT/'prepared/DECISION_AUDIT.json');proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    need(set(da)==set(SCHEMAS['AUDIT_KEYS']) and da['schema']=='monthly_monotone_tree266_decisions_v1' and len(da['events'])==122 and da['computes_returns'] is False,'Complete saved decision audit')
    for k in ('original_arrays_unchanged','original_eligibility_unchanged','original_core_support_preserved','single_original_episode_per_arm','calm_slot_preserved','partial_exact'):need(da[k] is True,'Original decision preservation')
    exact(da['counts'],decision_counts(da['events'],122,da['counts']['original_eligible_cells']),'Derived decision counts');rank_audit_check(da['rank_delta'],da['events'],da['counts'])
    for i,event in enumerate(da['events']):
        need(event['date']==meta['dates'][i],'Decision dates')
        for arm,key in zip(ARMS,('incumbent_choices','control_choices','candidate_choices')):exact([meta['symbols'][i][j] for j in event[key]],orders[arm][i]['symbols'],'Saved choices and execution orders')
    summary,rowends=model_summary(ma,meta,metadata,da['events']);historical_summary=historical_models_check(ma,historical,meta,da['events'],orders)
    exact(proof['historical_reference_binding'],source['historical261_reference'],'Independent preparation source chain')
    work=preparation_evidence(proof,ma,da,meta,metadata,rowends)
    selected=saved_selection(training,orders);exact(selected,frozen['selection'],'Frozen first88 two-reference selection')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    exact(result['selection'],selected,'Outcome immutable selection')
    need(result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Declared-assumption scope')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Bound separate accounting source')
    zero_rates(result);account,exports=accounting(ev,result,ind,training)
    for arm in ARMS:
        for path in PATHS:exact([r['date'] for r in exports[arm,path]['DAILY']],[d for d in axis if meta['dates'][0]<=d<=meta['ends'][-1]],'Complete official daily dates')
    prc=ev.read(PRIOR/'results/COMPLETION.json',prior['result_completion_sha256']);need(prc['status']=='COMPLETE' and prc['plan_sha256']==prior['plan_sha256'] and prc['paths']==9 and prc['slices']==45,'Prior complete economic manifest')
    oldresult=ev.read(PRIOR/'results/RESULTS.json',prc['artifacts_sha256']['RESULTS.json']);priorcost=None;exact_economic=True
    for path in PATHS:
        for kind in KINDS:
            filename=ARMS[0]+'_'+path+'_'+kind+'.json';previous=ev.read(PRIOR/'results'/filename,prc['artifacts_sha256'][filename]);current=exports[ARMS[0],path][kind]
            if path=='next_open_cost' and kind=='HOLDINGS':priorcost=previous
            else:exact(current,previous,'Original incumbent economic export '+path+'/'+kind)
    fees=diagnostic_summary_parity(result['arms'][ARMS[0]],oldresult['arms'][ARMS[0]],exports[ARMS[0],'next_open_cost']['HOLDINGS'],priorcost)
    for path,h in ev.pins.items():need(digest(local(path))==h,'All saved evidence unchanged during reader')
    return dict(schema='monthly_monotone_tree266_saved_reader_evidence_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
        reviewer_role='Independent numerical verifier author performing additional saved-evidence review; own implementation authorship disclosed, no fresh independent training in this reader.',
        scope=dict(opaque_npz_full_bytes_hashed=True,npz_decoded=False,saved_receipt_scope_and_work_checked=True,saved_json_tree_structures_and_shape_certificates_independently_checked=True,saved_json_historical_models_predictions_and_orders_exact=True,forecast_availability_rank_and_full_P_F_array_values_rely_on_bound_independent_and_reference_gates=True,complete_usable_row_membership_recomputed=False,actual_fit_predict_or_nav_replay=False),
        plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,root_owned_terminals=terminal,formal_synthetic_tests=tests['tests'],source=source_summary,historical_reference=historical_summary,model_summary=summary,decision_counts=da['counts'],execution_accounting=work,selection=selected,
        changed_holdings={a+'_vs_'+b:changes(orders,b,a) for a,b in ((ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]))},saved_accounting_evidence=account,incumbent_economic_exports_exact=exact_economic,incumbent_summary_exact=fees['exact'],summary_fee_display_review=fees,
        full_results={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS},segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
        bindings=dict(code_files=10,prepared_files=len(PREP),result_files=len(RESULT),historical_reference_files=13,all_collected_pins_rechecked=True,package3309_manifest_bound=True,package_members_rehashed=False,full_registered_closure_rehashed=False),
        preparation=dict(all123_cached_prefixes=True,physical_prefixes=list(PREFIXES),physical122_reuses_main=True,original_calibration_and_historical_control_before_candidate_certificate_checked=True),
        protection_and_close=dict(status='pending_root_close',protected_inventory_count=21444,protected_members_rehashed=False,final_root_close_verified=False,mutable_central_read_or_bound=False,registered_pre_state_completed=261,registered_pre_state_trials=1294,expected_post_close_completed=262,expected_post_close_trials=1297),
        actual_models_computed_by_reviewer=False,actual_model_solve_count=0,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
        limitations=['Complete finite/common-good training membership, original E/availability, forecast arrays and P/F NPZ values are proved by the bound actual independent verification and reference stages; the reader hashes those opaque arrays, it does not decode or re-derive their numeric contents.',
        'Tree training reproduction shares pinned sklearn. Saved JSON trees and structural feature0 certificates are independently inspected here, but no features are evaluated and no inference is repeated.',
        'Conditional shape only within one annual model holding the other nine features fixed; no cross-year, cross-security unconditional rank or final portfolio monotonicity is claimed.',
        'Saved9/45/18 accounting is reconciled to the separately authored account certificate; no new returns, Sharpe, NAV or source-price account is computed.',
        'Full/later/cost cannot rescue first88 rejection. After admission, full122 participates in cross-round best/target bookkeeping; repeated sample is not untouched out-of-sample.',
        'Root alone completes21444 protected-file hashing, central CAS, execution-terminal reporting and closure.'],
        reader_execution=dict(script=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__)),phase_digests=vars(args),report_written=False,terminal_exit_claimed=False),input_sha256=ev.pins)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+name+'-sha256',required=True)
    print(json.dumps(run(parser.parse_args()),sort_keys=True,indent=2,allow_nan=False))
