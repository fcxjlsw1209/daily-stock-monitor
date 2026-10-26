"""Saved review of terminal258 saved evidence after explicit authorization.

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
OUT=ROOT/'outputs/monthly_downside_basket258'
PRIOR=ROOT/'outputs/monthly_correlation_distance257_v2'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REPORT=ROOT/'work/monthly_downside_basket258_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_downside_basket258_context.json'
SOURCE=ROOT/'work/after246_portfolio_source_preflight.json'
CONTRACT=ROOT/'work/monthly_downside_basket258_IMPLEMENTATION_CONTRACT.json'
TERMINALS=ROOT/'work/monthly_downside_basket258_PHASE_TERMINALS.json'
TESTS=ROOT/'work/monthly_downside_basket258_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_downside_basket258_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_correlation_distance257_v2_RESULT_REVIEW.json'
ARMS=('incumbent_local67','basket_downside_moment','basket_symmetric_moment')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREP={'FEATURES.json','SCORES.npz','PRE_SCORES.npz','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
PREFIXES=(0,1,53,88,122)
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
FIXED={'work/monthly_downside_basket258_context.json': 'a2dd06142a4c3dd51e6abe57980b1ca35b3717bf4a19eeaeffceb542c9304c3a', 'work/monthly_downside_basket258_IMPLEMENTATION_CONTRACT.json': '29ae0d30401996bd5bd242976ebbef8e85a4e76972f45cacd48a35e9433b0d00', 'work/after246_portfolio_source_preflight.json': 'f3c800cbbb0b2a4677874815b74158d805879cc177ee50f4406df2acf66122a9', 'work/monthly_correlation_distance257_v2_RESULT_REVIEW.json': '0318d8768599d98c016b1f1520785451365d76dcbe7e951d50a1310cd78558ab', 'work/monthly_portfolio_path247_kernel.py': '18c7726ae701bb3557636bfa69a0c0dc32d516286f3bb1cd27f50157d5380092', 'work/ptr169_results_independent.py': '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59', 'work/monthly_ownsector_gate238_host.py': '94f70ab8c0d9fc1cd891bf1ce3eaa30212a54ef4f9cdb1eb4e237bd82bed30a5'}
EXPECTED_FORMAL_TESTS=99  # Root-owned terminal75938/23cbc6, formal99 PASS.
FIXED.update({
    "work/monthly_downside_basket258_TESTS.json":"51f178212acd92d8ba1d7c8db2fa1c77dfd2abc9e6fd87a540138495a733c22a",
    "work/monthly_downside_basket258_CODE_REVIEW.json":"8161189e60cc9399a4f44193599ab912ebe60bae4afb15f9dcda9283bfd70b04",
})
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

def index_list(value,width,label,ordered=False):
    need(type(value) is list and all(type(j) is int and 0<=j<width for j in value) and len(value)==len(set(value)),label)
    if not ordered:need(value==sorted(value),label+' sorted')
    return value

def object_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def integer(value,label,want=None):
    need(type(value) is int and value>=0,'Nonnegative integer '+label)
    if want is not None:need(value==want,'Integer scope '+label)
    return value


def schema_flags():
    legacy=constants(ROOT/'work/monthly_portfolio_path247_kernel.py',('EVENT_KEYS','TRIO_KEYS','SOURCE_KEYS','LISTING_KEYS'))
    new=constants(ROOT/'work/monthly_downside_basket258_kernel.py',('COUNT_FIELDS','STAT_KEYS','BRANCHES'))
    flags=constants(ROOT/'work/monthly_downside_basket258_independent.py',('FEATURE_FLAGS','INCUMBENT_FLAGS','SELECTION_FLAGS'))
    flags['PREPARE_FLAGS']=tuple(dict.fromkeys(flags['FEATURE_FLAGS']+flags['SELECTION_FLAGS']+('independent_source_rebuild',)))
    legacy.update(new);legacy['DECISION_EVENT_KEYS']=legacy['EVENT_KEYS']+('choices','winner_trio_indices','tied_trio_indices')
    need([len(legacy[k]) for k in ('EVENT_KEYS','TRIO_KEYS','SOURCE_KEYS','COUNT_FIELDS','STAT_KEYS','DECISION_EVENT_KEYS')]==[26,10,8,22,20,29],'Frozen258 schema dimensions')
    return legacy,flags


def count_check(c,n,fields):
    need(set(c)==set(fields),'Complete22 counts')
    for k,v in c.items():integer(v,k)
    need(c['events']==n and c['symbols_per_event']==49 and c['original_episode_calls']==1,'122x49/single original episode counts')
    need(c['new_model_fits']==c['predictive_model_fits']==0 and c['formation_return_observations']==252*c['trio_statistics'],'No models;252 formation steps')


def receipt(r,n,kind,flags,fields):
    f={'incumbent':'INCUMBENT_FLAGS','source':'FEATURE_FLAGS','cached_source':'FEATURE_FLAGS','selection':'SELECTION_FLAGS','full':'PREPARE_FLAGS'}[kind]
    need(r.get('passed') is True and all(r.get(k) is True for k in flags[f]),'Independent '+kind+' flags')
    integer(r.get('checked_events'),'checked_events',n)
    need(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Independent receipt isolation')
    for k in ('original_episode_calls','descriptive_path_evaluations','cached_path_validations'):integer(r.get(k),k)
    integer(r['original_episode_calls'],'Independent bridge count',2 if kind=='full' else 1)
    if kind in ('source','cached_source','full'):
        need(r['continuous_atol']==r['continuous_rtol']==2e-10 and r['nav_atol']==r['nav_rtol']==2e-12 and r['rank_tolerance']==0. and r['exact_discrete'] is True,'Canonical versus diagnostic tolerances')
    else:need(r['numerical_tolerance']==0.,'Exact decisions tolerance')
    if kind=='incumbent':
        need(r['full_scores_supplied'] is True and r['full_pre_scores_supplied'] is True and r['candidate_selection_computed'] is False,'Original gate before new moments')
        integer(r['ordered_choices_checked'],'Original orders',n)
        need(r['descriptive_path_evaluations']==r['cached_path_validations']==0,'No new path in incumbent gate')
    elif kind in ('source','cached_source'):
        c=r['feature_counts'];count_check(c,n,fields)
        need(r['full_features_supplied'] is True and r['candidate_selection_computed'] is False and r['source_rebuilt'] is (kind=='source'),'Source-only gate scope')
        need(all(c[k]==0 for k in ('downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings')),'No source-stage choice counts')
        need(r['descriptive_path_evaluations']==(c['trio_statistics'] if kind=='source' else 0) and r['cached_path_validations']==(c['trio_statistics'] if kind=='cached_source' else 0),'Source computational counts')
    else:
        c=r['counts'];count_check(c,n,fields)
        need(all(r[k] is True for k in ('full_features_supplied','full_scores_supplied','full_pre_scores_supplied','full_audit_supplied')),'Complete saved array/audit proof')
        integer(r['ordered_choices_checked'],'Three-arm choices',3*n)
        need(r['cached_path_validations']==c['trio_statistics'] and r['descriptive_path_evaluations']==(c['trio_statistics'] if kind=='full' else 0),'Selection computational counts')
        if kind=='full':need(r['descriptive_path_returns_checked'] is True,'Full formation paths verified')


def execution_counts(e):
    physical=[e['main']]+[x for x in e['physical_prefixes'] if not x['reused_full']];cached=e['cached_prefixes'];proofs=[]
    for x in physical:proofs.extend((x['gates']['incumbent'],x['gates']['source'],x['full']))
    for x in cached:proofs.extend((x['gates']['incumbent'],x['gates']['source'],x['selection']))
    ps=sum(x['counts']['trio_statistics'] for x in physical);pc=sum(x['counts']['trio_statistics'] for x in cached)
    ins=sum(x['descriptive_path_evaluations'] for x in proofs);inc=sum(x['cached_path_validations'] for x in proofs)
    return dict(producer_total_path_moment_calculations=ps+pc,independent_total_path_moment_calculations=ins+inc,
      total_path_moment_calculations=ps+pc+ins+inc,total_formation_return_steps_computed=252*(ps+pc+ins+inc),cached_validation_recomputes_path_and_moments=True,
      producer_original_episode_calls=sum(x['counts']['original_episode_calls'] for x in physical+cached),physical_preparations=len(physical),physical_122_reused=True,cached_decisions=len(cached),
      producer_source_price_observations=sum(x['counts']['source_price_observations'] for x in physical),producer_descriptive_path_evaluations=ps,producer_cached_path_validations=pc,
      independent_original_episode_calls=sum(x['original_episode_calls'] for x in proofs),independent_descriptive_path_evaluations=ins,independent_cached_path_validations=inc,new_model_fits=0,predictive_model_fits=0)


def preparation(e,pc,audit,features,flags,fields):
    need(all(e[k] is True for k in ('all123_order_prefixes','original_arrays_unchanged','original_eligible_unchanged')),'Original cache preservation')
    need(e['new_model_fits']==e['predictive_model_fits']==pc['new_model_fits']==pc['new_predictive_model_fits']==0 and pc['new_returns_computed'] is False,'Preparation contains no predictive fits or account returns')
    need(pc['events']==122 and e['main']['prefix']==122 and e['main']['reused_full'] is False and e['main']['source_recomputed'] is True,'Main full preparation')
    need(len(features)==122 and object_digest(features)==e['full_features_sha256'],'Feature content digest')
    need(object_digest(e['main']['gates']['source'])==e['full_source_certificate_sha256'],'Full source certificate digest')
    need([x['prefix'] for x in e['cached_prefixes']]==list(range(123)) and [x['prefix'] for x in e['physical_prefixes']]==list(PREFIXES),'123 cached and5 physical prefix inventory')
    for x in [e['main']]+e['physical_prefixes']+e['cached_prefixes']:
        n=x['prefix'];kind='selection' if 'selection' in x else 'full';need(x['exact'] is True,'Exact prefix proof')
        count_check(x['counts'],n,fields);receipt(x[kind],n,kind,flags,fields)
        receipt(x['gates']['incumbent'],n,'incumbent',flags,fields);receipt(x['gates']['source'],n,'cached_source' if kind=='selection' else 'source',flags,fields)
        need(x['counts']==x[kind]['counts'],'Complete count binding')
        c=dict(x['counts'])
        for k in ('downside_changed_holdings','symmetric_changed_holdings','candidate_control_changed_holdings'):c[k]=0
        need(c==x['gates']['source']['feature_counts'],'Source versus selection count binding')
    for x in e['physical_prefixes']:
        reuse=x['prefix']==122
        need(x['reused_full'] is reuse and x['source_recomputed'] is (not reuse),'Physical122 reused only')
        if reuse:need(x==dict(e['main'],reused_full=True,source_recomputed=False),'Exact full122 reused proof')
    need(e['main']['counts']==audit['counts']==pc['descriptive_counts'] and e['main']['gates']['incumbent']==audit['incumbent_gate'] and e['main']['gates']['source']==audit['source_gate'],'Main gates/audit/receipt exact')
    execution=execution_counts(e);need(e['execution_accounting']==execution,'All actual descriptive recomputation counters')
    return dict(full_events=122,cached_prefixes=123,physical_prefixes=list(PREFIXES),physical122_reused=True,full_arrays_hash_checked_only=True,independent_full_arrays_checked=True,execution_accounting=execution)


def stat_summary(s,keys):
    """Saved shapes/sign/objective metadata only; do not regenerate a path/moment."""
    need(set(s)==set(keys) and s['schema']=='monthly_downside_basket258_moments_v1' and s['price_count']==253 and s['return_count']==252,'Saved moment schema')
    for key,length in (('normalized_prices',253),('contributions',252)):
        need(type(s[key]) is list and len(s[key])==3 and all(type(row) is list and len(row)==length and all(finite(v) for v in row) for row in s[key]),'Saved trio matrix axis')
    need(all(row[0]==1. and all(v>0 for v in row) for row in s['normalized_prices']),'Saved normalized positive prices')
    for key,length in (('wealth',253),('prior_totals',252),('portfolio_returns',252),('direct_returns',252)):
        need(type(s[key]) is list and len(s[key])==length and all(finite(v) for v in s[key]),'Saved formation vector axis')
    for k in ('downside_square_sum','symmetric_square_sum','upside_square_sum','downside_moment','symmetric_moment','upside_moment'):need(finite(s[k]) and s[k]>=0.,'Nonnegative saved moment')
    signs=(sum(v<0 for v in s['portfolio_returns']),sum(v==0 for v in s['portfolio_returns']),sum(v>0 for v in s['portfolio_returns']))
    need(signs==(s['negative_returns'],s['zero_returns'],s['positive_returns']) and sum(signs)==252,'Saved canonical signs')
    need(type(s['zero_downside']) is bool and s['zero_downside']==(s['downside_moment']==0.)==(signs[0]==0),'Valid zero downside')
    need(type(s['zero_symmetric']) is bool and s['zero_symmetric']==(s['symmetric_moment']==0.)==(signs[1]==252),'Valid zero symmetric')


def feature_summary(features,audit,meta,orders,calendar,source,schemas):
    need(len(features)==len(audit['events'])==len(meta['dates'])==122,'Full saved features')
    need(set(audit)=={'schema','counts','events','incumbent_gate','source_gate'} and audit['schema']=='monthly_downside_basket258_decisions_v1','Five-key258 audit')
    c=dict.fromkeys(schemas['COUNT_FIELDS'],0);c.update(events=122,symbols_per_event=49,original_episode_calls=1)
    sessions=calendar['sessions'];lookup={d:i for i,d in enumerate(sessions)};branches=Counter();summaries=[]
    for i,(f,a) in enumerate(zip(features,audit['events'])):
        names=meta['symbols'][i];need(len(names)==49 and len(set(names))==49,'Full49 symbol domain')
        need(set(f)==set(schemas['EVENT_KEYS']) and set(a)==set(schemas['DECISION_EVENT_KEYS']),'Full event keys')
        need(all(a[k]==v for k,v in f.items()),'Decision preserves source feature event')
        need(f['query_index']==i and f['date']==meta['dates'][i] and f['symbols']==names,'Query identity')
        t=lookup[f['date']];dates=sessions[t-294:t-41]
        need(t>=294 and len(dates)==253 and f['session_position']==t and f['source_dates']==dates and f['clock_sessions']==sessions[t-294:t+2],'Official253 price/next-month clock')
        need(f['price_start']==dates[0] and f['price_end']==dates[-1] and source['calendar']['windows'][i]['price_start']==dates[0] and source['calendar']['windows'][i]['price_end']==dates[-1],'Pinned metadata source windows')
        partial=f['date']=='2016-08-17';need(f['initial_partial'] is partial,'Named partial')
        E=index_list(f['eligible_indices'],49,'Eligible');P=f['pre_scores'];F=f['final_scores'];C=f['weighted_core']
        need(all(type(row) is list and len(row)==49 and all(finite(v) for v in row) for row in (P,F,C)),'Saved original full49 score metadata')
        order=sorted(E,key=lambda j:(-P[j],names[j]));need(f['pre_order']==order,'Saved original pre order')
        short=[] if partial or len(E)<3 else order[:5]
        need(f['shortlist_indices']==short and len(f['source_cells'])==len(short),'Original PTop5 before source availability')
        inc=index_list(f['incumbent_choices'],49,'Incumbent',True)
        need([names[j] for j in inc]==orders[ARMS[0]][i]['symbols'],'Incumbent saved choices')
        need(set(a['choices'])==set(ARMS) and set(a['winner_trio_indices'])==set(a['tied_trio_indices'])==set(ARMS[1:]),'All choice and winner arms')
        for arm in ARMS:
            ids=index_list(a['choices'][arm],49,'Arm choices',True)
            need([names[j] for j in ids]==orders[arm][i]['symbols'] and [F[j] for j in ids]==orders[arm][i]['scores'],'Saved F/order export binding')
        for cell,j in zip(f['source_cells'],short):
            need(set(cell)==set(schemas['SOURCE_KEYS']) and cell['column_index']==j and cell['symbol']==names[j] and type(cell['source_available']) is bool,'Shortlist source cell identity')
            integer(cell['observations_checked'],'Source observations');need(cell['observations_checked']<=253,'Source observation upper bound')
            need(cell['reason'] in ('available','missing_symbol','missing_official_session','missing_nan') and cell['source_available']==(cell['reason']=='available'),'Explicit source availability reason')
            if cell['source_available']:need(cell['observations_checked']==253 and finite(cell['start_price']) and cell['start_price']>0 and finite(cell['end_price']) and cell['end_price']>0,'Full source endpoints')
            else:need(cell['start_price'] is None and cell['end_price'] is None,'No fabricated unavailable endpoints')
            bound=next((v for v in source['listing_bounds'] if v['symbol']==names[j]),None)
            listing=cell['listing']
            if bound is None:need(listing is None,'No invented listing metadata')
            else:
                need(set(listing)==set(schemas['LISTING_KEYS']) and all(listing[k]==bound[k] for k in ('issuer_key','not_before','available_from','evidence_id')),'Same listing identity/evidence')
                need(listing['visible'] is (bound['available_from']<=f['date']) and listing['start_precedes_bound'] is (f['price_start']<bound['not_before']),'Query-visible listing clock')
                need(not (cell['source_available'] and listing['visible'] and listing['start_precedes_bound']),'No falsely complete prelisting path')
        branch='initial_partial' if partial else 'fewer_than_three' if len(E)<3 else 'available' if all(x['source_available'] for x in f['source_cells']) else 'missing_source'
        need(f['branch']==branch,'Common source fallback branch');branches[branch]+=1
        H=[];seen=set()
        for k,trio in enumerate(f['trios']):
            need(set(trio)==set(schemas['TRIO_KEYS']),'Complete trio fields')
            ids=index_list(trio['indices'],49,'Trio',True)
            need(len(ids)==3 and set(ids)<=set(short) and trio['symbols']==[names[j] for j in ids] and trio['symbols']==sorted(trio['symbols']),'Canonical symbol trio')
            need(tuple(ids) not in seen,'Unique trio');seen.add(tuple(ids))
            eligible=trio['raw_core_mean']>=f['reference_raw_core_mean']-.02 and trio['final_mean']>=f['reference_final_mean']-.02
            need(trio['feasible'] is eligible,'Both exact cap predicates')
            if eligible:H.append(k)
            s=trio['statistics'];need((s is not None)==(branch=='available' and eligible),'Only common available H statistics')
            if s is not None:
                stat_summary(s,schemas['STAT_KEYS']);c['trio_statistics']+=1;c['formation_return_observations']+=252;c['zero_downside_trios']+=int(s['zero_downside']);c['zero_symmetric_trios']+=int(s['zero_symmetric'])
        need(H==f['feasible_trio_indices'] and len(f['trios'])==(len(short)*(len(short)-1)*(len(short)-2)//6 if short else 0),'All1/4/10 trios and common feasible set')
        if short:need(any(set(f['trios'][k]['indices'])==set(inc) for k in H),'Incumbent common feasible')
        for arm,key in zip(ARMS[1:],('downside_moment','symmetric_moment')):
            if branch!='available':
                need(orders[arm][i]==orders[ARMS[0]][i] and a['winner_trio_indices'][arm] is None and a['tied_trio_indices'][arm]==[],'Exact whole-event fallback')
            else:
                best=min(f['trios'][k]['statistics'][key] for k in H);ties=[k for k in H if f['trios'][k]['statistics'][key]==best]
                tiedinc=[k for k in ties if set(f['trios'][k]['indices'])==set(inc)];win=tiedinc[0] if tiedinc else min(ties,key=lambda k:tuple(f['trios'][k]['symbols']))
                need(a['winner_trio_indices'][arm]==win and a['tied_trio_indices'][arm]==ties,'Exact saved objective minimum and tie break')
                ids=f['trios'][win]['indices'];want=inc if set(ids)==set(inc) else sorted(ids,key=lambda j:(-F[j],names[j]))
                need(a['choices'][arm]==want,'Direct basket order preserves exact incumbent set')
        c['eligible_cells']+=len(E);c['initial_partial_events']+=int(partial);c['active_events']+=int(bool(short));c['shortlist_cells']+=len(short)
        c['source_available_cells']+=sum(x['source_available'] for x in f['source_cells']);c['source_price_observations']+=sum(x['observations_checked'] for x in f['source_cells'])
        c['enumerated_trios']+=len(f['trios']);c['feasible_trios']+=len(H);c['missing_source_events']+=int(branch=='missing_source');c['available_events']+=int(branch=='available')
        for arm,key in zip(ARMS[1:],('downside_changed_holdings','symmetric_changed_holdings')):c[key]+=int(set(a['choices'][arm])!=set(inc))
        c['candidate_control_changed_holdings']+=int(set(a['choices'][ARMS[1]])!=set(a['choices'][ARMS[2]]))
        summaries.append(dict(index=i,date=f['date'],branch=branch,shortlist=[names[j] for j in short],feasible_trios=len(H),trios_with_statistics=sum(t['statistics'] is not None for t in f['trios'])))
    count_check(audit['counts'],122,schemas['COUNT_FIELDS']);need(audit['counts']==c,'All22 counts from saved events')
    return c,dict(branch_counts=dict(branches),events=summaries,canonical_objective_selection_checked_from_saved_statistics=True,raw_paths_or_moments_regenerated=False,canonical_numerics_proven_by_bound_independent_receipts=True)


def saved_selection(train,changed):
    need(set(train)==set(ARMS),'Three fixed arms');invalid={a:required_metrics(train[a],a==ARMS[0]) for a in ARMS}
    guards={};changes_by_ref={};candidate=ARMS[1]
    for ref in (ARMS[0],ARMS[2]):
        ids=[r['index'] for r in changed[candidate+'_vs_'+ref]['months'] if r['index']<88];changes_by_ref[ref]=ids
        close=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['same_close_zero'],train[ref]['same_close_zero'],.005 if ref==ARMS[0] else 0.)
        opening=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['next_open_zero'],train[ref]['next_open_zero'],0.)
        support=(len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids)) if ref==ARMS[0] else len(ids)>=1
        guards[ref]=dict(close=bool(close),open=bool(opening),support=bool(support))
    admitted=all(all(v.values()) for v in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=changes_by_ref,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)


def terminals_check(record,expected_plan,hashes):
    need(record.get('plan_sha256')==expected_plan and set(record.get('phases',{}))=={'prepare','select','diagnose'},'Complete root-owned terminal records')
    for phase,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        p=record['phases'][phase]
        need(set(p)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Terminal record fields')
        need(p['session_id'] is None or (type(p['session_id']) is int and p['session_id']>0),'Terminal session')
        integer(p['exit_code'],'Terminal exit',0);need(type(p['chunk_id']) is str and bool(p['chunk_id'].strip()),'Terminal chunk')
        need(local(p['artifact_path'])==OUT/relative and p['artifact_sha256']==hashes[phase],'Terminal exact artifact')
    return record
def run(args):
    need(not REPORT.exists(),'Report write-once');need(platform.python_version()=='3.9.6','Pinned offline wrapper required')
    need(type(EXPECTED_FORMAL_TESTS) is int and EXPECTED_FORMAL_TESTS>0,'Formal terminal test count must be frozen before actual read')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h);need(ev.registered(plan,p)==h,'Registered evidence '+p)
    context=ev.read(CONTEXT);contract=ev.read(CONTRACT);source=ev.read(SOURCE)
    tests=ev.read(TESTS,ev.registered(plan,TESTS));code=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(plan['context']==context and plan['context_sha256']==digest(CONTEXT) and plan['implementation_contract_sha256']==digest(CONTRACT)==code['implementation_contract_sha256'] and contract['context_sha256']==digest(CONTEXT),'Exact context/implementation chain')
    need(context['arms']==list(ARMS) and context['promotable_arms']==[ARMS[1]] and context['registered_trials']==3 and context['new_predictive_model_fits']==0,'Fixed three arms/downside-only promotion')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==EXPECTED_FORMAL_TESTS and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Formal test/runtime binding')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['context_sha256']==digest(CONTEXT) and code['actual_returns_computed'] is False,'Prospective CODE_REVIEW')
    need(len(tests['code_sha256'])==6,'Six code files')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered code');ev.pin(p,h)
    schemas,flags=schema_flags()
    need(source['metadata_only'] is True and source['prices_loaded'] is False and source['npz_decoded'] is False and source['actual_shortlist_computed'] is False,'Metadata source not actual coverage')
    need(source['calendar']['prices']==253 and source['calendar']['return_intervals']==252 and len(source['calendar']['windows'])==122 and len(source['listing_identity_cells'])==5,'253 official clocks/known identity inventory')
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256'])
    need(len(state['completed'])==253 and sum(x['trials'] for x in state['completed'])==1270 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Registered predecessor253/1270')
    protection=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protection)==plan['protected_files']==21444,'Protected manifest count only')
    terminal=ev.read(TERMINALS);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP)
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Frozen first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exactly selected files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT)
    need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Frozen selection before full diagnostics')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed122 intervals')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','First88 split clocks')
    need(set(orders)==set(ARMS),'Three arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [x['date'] for x in orders[arm]]==meta['dates'] and [x['end'] for x in orders[arm]]==meta['ends'],'Order clocks')
        need(all(x['signal_date']==x['date'] and len(x['symbols']) in (0,2,3) and len(x['symbols'])==len(set(x['symbols'])) for x in orders[arm]),'3/2/cash fixed monthly order schema')
    need(orders[ARMS[0]]==ev.read(LOCAL/'local_linear_ORDERS.json',ev.registered(plan,LOCAL/'local_linear_ORDERS.json')),'Exact Local67 incumbent orders')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));originalplan=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'))
    parity=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    maps=[canonical_source_map(x) for x in (parity['raw_source_hashes'],source['raw_source_hashes'],originalplan['raw_source_hashes'],plan['raw_source_hashes'])]
    need(all(x==maps[0] for x in maps[1:]),'Original64 canonical source path/SHA identity')
    need(all(parity[k] is True for k in ('original_arrays_unchanged','original_orders_exact','source_metadata_not_inherited_actual_coverage','source_legacy_variance_rule_not_used')) and len(parity['original_array_fingerprints'])==27 and parity['source_review_sha256']==digest(SOURCE),'Original27 actual source receipt and support policy')
    features=ev.read(OUT/'prepared/FEATURES.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');e=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    counts,src=feature_summary(features,audit,meta,orders,calendar,source,schemas)
    prep=preparation(e,pc,audit,features,flags,schemas['COUNT_FIELDS'])
    changed={candidate+'_vs_'+ref:changes(orders,ref,candidate) for candidate,ref in ((ARMS[1],ARMS[0]),(ARMS[1],ARMS[2]),(ARMS[2],ARMS[0]))}
    selection=saved_selection(training,changed);need(selection==frozen['selection'],'Exact frozen first88 admission versus both references')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selection and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Result policy labels')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Frozen independent accounting source')
    ac,exports=accounting(ev,result,ind,training)
    incumbent=previous_reference(ev,result,exports,ind,PRIOR,PREVIOUS_REVIEW,ARMS[0],ARMS[0])
    # Inherited allowance only; its exact scope and chronological accumulation are evidence-bound.
    for path,h in {'work/basket_core251_saved_result_reader_v2.py':'35ab84ab4e6401e1858ac1e522c6e4c58483cce2a65b7d89d6da23014f97f555','work/monthly_reference_gain250_RESULT_REVIEW.json':'a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595'}.items():ev.pin(path,h)
    fee=ev.read('work/monthly_reference_gain250_RESULT_REVIEW.json')['incumbent_reproduction']['period_fee_exception']
    need(fee['absolute_bound']==PERIOD_FEE_ATOL and fee['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Inherited period fee scope')
    engine=ROOT/'outputs/monthly_factor_search_v2/validate.py';ev.pin(engine,ev.registered(plan,engine))
    ev.pin(Path(__file__));ev.pin(ROOT/'work/test_downside_basket258_saved_result_reader.py')
    for p,h in ev.pins.items():need(digest(local(p))==h,'Evidence changed during saved review '+p)
    summary={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_downside_basket258_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='258 lifecycle host author and saved-evidence reviewer; not an independent numerical oracle. Numerical/source/choice/accounting certificates come from separate hash-bound implementations.',
      scope='Saved-only manifest, source-clock/shape/sign/domain/counter, first88 guard, root terminal and exported ledger reconciliation; no model, path/moment, account or return replay.',
      plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=digest(TERMINALS),
      root_owned_terminal_receipts=terminal,formal_synthetic_tests=tests['tests'],bindings=dict(code_files=6,prepared_files=8,result_files=38,all_checked_hashes_match=True,raw64_registered_map_equal=True,raw64_path_normalization_only=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      preparation=prep,descriptive_counts=counts,source_and_moment_evidence=src,selection=selection,changed_holdings=changed,accounting_review=ac,incumbent_reproduction=incumbent,full_results=summary,
      segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},holdings_counts={a:{str(k):sum(len(o['symbols'])==k for o in orders[a]) for k in (0,2,3)} for a in ARMS},
      target_reached_by_saved_selected_result=summary[selection['primary']]['same_close_zero']['cagr']>=.7,
      cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='READY_FOR_ROOT_CLOSE_NOT_YET_CLOSED',registered_pre_state_completed=253,registered_pre_state_trials=1270,expected_post_close_completed=254,expected_post_close_trials=1273,protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,final_root_close_verified=False,post_close_central_verified=False,mutable_central_read_or_bound=False),
      source_prices_loaded=False,source_volumes_loaded=False,npz_decoded=False,actual_features_computed_by_reviewer=False,actual_orders_computed_by_reviewer=False,actual_portfolio_returns_computed_by_reviewer=False,formal_tests_reexecuted=False,phase_rerun=False,frozen_files_modified=False,observers_modified=False,observer_promotion=False,reused_sample=True,pristine_out_of_sample_claim=False,strict_pit=False,
      historical_best_reporting=dict(first88_primary=selection['primary'],first88_admitted=selection['admitted'],prior_historical_best=state['best_training_selected_candidate'],later_diagnostics_did_not_reselect=True,root_postclose_best_not_yet_verified=True),
      limitations=['No new predictive model is fit; cached path/moment verification is real descriptive recalculation and is counted separately from unique source statistics.',
       'Canonical path arithmetic and whole-source absence/identity rules are supported by bound independent certificates; this reader checks saved metadata, shapes, signs and choices, not regenerated moments or NPZ arrays.',
       'The PTop5 and dual .02 concession caps preserve a strong-leader domain. Direct basket minimization is not algebraically a 50/52-percent weighted core score.',
       'Zero moments are valid and never a variance fallback. Original P/F exports remain unchanged; basket objectives are separate saved evidence.',
       'Only downside can be admitted on first88 against incumbent and symmetric reference. The control, later34, full returns and cost outcomes cannot rescue rejection.',
       'Conditional replication only; exact131 definitions are missing and global novelty remains UNKNOWN. Reused historical sample and source vintages do not establish strict PIT or new out-of-sample success.',
       'Cross-round cost period fees retain only the prior1e-10 allowance. Aggregate fees must equal each own ordered HOLDINGS sum exactly and independent totals must match exactly; all other incumbent metrics/exports stay exact.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),input_pins=len(ev.pins),primary=selection['primary'],admitted=selection['admitted'],full_same_close_zero_cagr={a:summary[a]['same_close_zero']['cagr'] for a in ARMS},descriptive_execution=prep['execution_accounting'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())
