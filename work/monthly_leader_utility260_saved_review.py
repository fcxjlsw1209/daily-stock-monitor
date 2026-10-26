"""Extra saved-result cross-check authored by the 260 producer author.

Standard library only; no strategy imports or array decoding. Actual use needs
explicit terminal phase digests and separate root authorization. Independent
model evidence comes from the separately authored, bound numerical verifier.
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
OUT=ROOT/'outputs/monthly_leader_utility260'
PRIOR=ROOT/'outputs/monthly_joint_identity259'
LOCAL=ROOT/'outputs/monthly_local_linear70'
REBUILD=ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
REPORT=ROOT/'work/monthly_leader_utility260_RESULT_REVIEW.json'
CONTEXT=ROOT/'work/monthly_leader_utility260_context.json'
CONTRACT=ROOT/'work/monthly_leader_utility260_IMPLEMENTATION_CONTRACT.json'
TESTS=ROOT/'work/monthly_leader_utility260_TESTS.json'
CODE_REVIEW=ROOT/'work/monthly_leader_utility260_CODE_REVIEW.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PREVIOUS_REVIEW=ROOT/'work/monthly_joint_identity259_RESULT_REVIEW.json'
TERMINALS=OUT/'PHASE_TERMINALS.json'
ARMS=('incumbent_local67','unconditional_leader_utility','state_leader_utility')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
KINDS=('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')
SLICES={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
PREFIXES=(0,1,53,88,122)
PREP={'FORECASTS.npz','MARKET_INPUTS.npz','SCORES.npz','PRE_SCORES.npz','MODEL_AUDIT.json','DECISION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in KINDS}
STAT_KEYS={'start','end','years','final_equity','cagr','sharpe','max_drawdown','trading_sessions','total_return'}
ACCOUNT_ATOL=1e-7
ACCOUNT_RTOL=2e-11
PERIOD_FEE_ATOL=1e-10
FIXED={
 'work/monthly_leader_utility260_context.json':'5b0e79942aa9dbb002c5e97a33cad77d8a02726b978a8f5fd86364bfa4a89533',
 'work/monthly_leader_utility260_IMPLEMENTATION_CONTRACT.json':'88ad58fbd783a574007fe17dcc8f842b0764425d9b8d2e0260bea0f64246a381',
 'work/monthly_joint_identity259_RESULT_REVIEW.json':'0c3a92149c0b6648a114cd345e4f9bd4ceb277cfce70b7122af8c16aa7170c77',
 'work/ptr169_results_independent.py':'6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
 'work/monthly_ownsector_gate238_host.py':'94f70ab8c0d9fc1cd891bf1ce3eaa30212a54ef4f9cdb1eb4e237bd82bed30a5'}
READER_PARENT='work/monthly_joint_identity259_saved_review.py'
READER_PARENT_SHA='5daf6741694d35a50b50af5afd036533db3e1c6f42bb2821132ca0bfae83e898'
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

def saved_selection(train,changed):
    need(set(train)==set(ARMS),'Three fixed arms');invalid={a:required_metrics(train[a],a==ARMS[0]) for a in ARMS}
    candidate=ARMS[2];guards={};byref={}
    for ref in ARMS[:2]:
        ids=[r['index'] for r in changed[candidate+'_vs_'+ref]['months'] if r['index']<88];byref[ref]=ids
        close=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['same_close_zero'],train[ref]['same_close_zero'],.005 if ref==ARMS[0] else 0.)
        opening=not invalid[candidate] and not invalid[ref] and guard(train[candidate]['next_open_zero'],train[ref]['next_open_zero'],0.)
        support=(len(ids)>=3 and any(i<53 for i in ids) and any(i>=53 for i in ids)) if ref==ARMS[0] else len(ids)>=1
        guards[ref]=dict(close=bool(close),open=bool(opening),support=bool(support))
    admitted=all(all(g.values()) for g in guards.values())
    return dict(primary=candidate if admitted else ARMS[0],admitted=admitted,guards=guards,changed_months_by_reference=byref,undefined_sharpe=invalid,first88_only=True,later34_used=False,cost_used=False,diagnostic_arm_never_promoted=True)

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

def terminals_check(record,expected_plan,hashes):
    need(record.get('plan_sha256')==expected_plan and set(record.get('phases',{}))=={'prepare','select','diagnose'},'Three root-owned successful phase terminals')
    for phase,relative in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:
        p=record['phases'][phase];need(set(p)=={'session_id','exit_code','chunk_id','artifact_path','artifact_sha256'},'Terminal schema')
        need(p['session_id'] is None or (type(p['session_id']) is int and p['session_id']>0),'Terminal session identity')
        integer(p['exit_code'],'Terminal exit',0);need(type(p['chunk_id']) is str and bool(p['chunk_id'].strip()),'Terminal chunk')
        need(local(p['artifact_path'])==OUT/relative and p['artifact_sha256']==hashes[phase],'Terminal artifact exact')
    return record

def nav_metrics(days,values,initial=100000.):
    import statistics
    from datetime import date
    need(len(days)==len(values)>1 and days==sorted(set(days)),'Valid saved NAV interval')
    need(all(finite(v) and v>0 for v in values),'Positive saved NAV')
    years=(date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days/365.2425;need(years>0,'Positive interval')
    returns=[b/a-1. for a,b in zip(values,values[1:])];sd=statistics.stdev(returns) if len(returns)>1 else 0.
    peak,dd=initial,0.
    for v in values:peak=max(peak,v);dd=min(dd,v/peak-1.)
    return dict(start=days[0],end=days[-1],years=years,final_equity=values[-1],cagr=(values[-1]/initial)**(1./years)-1.,sharpe=statistics.mean(returns)/sd*math.sqrt(252.) if sd>0 else None,max_drawdown=dd,trading_sessions=len(days),total_return=values[-1]/initial-1.),sd

def saved_nav_slices(daily,meta):
    days=[r['date'] for r in daily];values=[r['equity'] for r in daily];out={};sds={}
    for name,(lo,hi) in SLICES.items():
        left,right=days.index(meta['dates'][lo]),days.index(meta['ends'][hi-1]);v=values[left:right+1]
        if name!='full122':v=[x/values[left]*100000. for x in v]
        out[name],sds[name]=nav_metrics(days[left:right+1],v)
    return out,sds

def exported_ledger_check(data,orders,axis,path,residuals):
    days=[r['date'] for r in data['DAILY']];pos=data['DAILY_POSITIONS']
    need(days==[d for d in axis if orders[0]['date']<=d<=orders[-1]['end']] and days==[r['date'] for r in pos],'Full official exported daily axis')
    mapping={d:i for i,d in enumerate(axis)}
    execution=[o['date'] if path=='same_close_zero' else axis[mapping[o['date']]+1] for o in orders]
    holdings=data['HOLDINGS'];need([h['execution_date'] for h in holdings]==execution,'Exact monthly execution calendar')
    schedule=dict(zip(execution,orders));prev=None;chosen=set();dust=[];filtered_targets=[]
    for row,p in zip(data['DAILY'],pos):
        need(finite(p['cash']) and p['cash']>-1e-5 and finite(p['equity']) and p['equity']>0,'Frozen exported cash/NAV domain')
        units,values=p['adjusted_units'],p['market_values'];need(set(units)==set(values),'Position/value security domain')
        need(all(finite(v) and abs(v)>1e-9 and v>-1e-7 for v in units.values()) and all(finite(v) for v in values.values()),'Frozen exported units/dust domain')
        need(all(values[k]*units[k]>=0 for k in units),'Saved valuation sign matches position')
        need(p['cash']+sum(values.values())==p['equity'],'Saved exported position NAV arithmetic exact')
        # Direct exported NAV pair: frozen ptr169_rebuild_evaluate.py:278, not metric or cross-round comparison.
        difference=p['equity']-row['equity'];need(abs(difference)<=1e-5+1e-10*abs(row['equity']),'Frozen DAILY/position NAV bound')
        if difference:residuals.append(dict(field='DAILY/position NAV',actual=p['equity'],reference=row['equity'],difference=difference,existing_absolute_bound=1e-5,existing_relative_bound=1e-10))
        if row['date'] in schedule:chosen=set(schedule[row['date']]['symbols'])
        elif prev is not None:need(units==prev['adjusted_units'] and p['cash']==prev['cash'],'No intra-month units/cash trading')
        need(all(s in chosen or abs(q)<=1e-7 for s,q in units.items()),'Frozen non-target residual units bound')
        need(all(q>0 for s,q in units.items() if s in chosen),'Positive retained target units')
        for symbol,q in units.items():
            if symbol not in chosen:dust.append(dict(date=row['date'],symbol=symbol,adjusted_units=q,market_value=values[symbol]))
        if chosen-set(units):filtered_targets.append(dict(date=row['date'],symbols=sorted(chosen-set(units))))
        prev=p
    for h,o in zip(holdings,orders):
        need(h['signal_date']==o['date'] and h['symbols']==['META' if s=='research_equity_FB' else s for s in o['symbols']] and h['scores']==o['scores'],'Holding/ordered choice identity')
        need(finite(h['fee']) and h['fee']>=0,'Nonnegative saved period fee')
    seen=set();trades=data['TRADES'];need([r['date'] for r in trades]==sorted(r['date'] for r in trades),'Ordered fills')
    fee=turnover=0.
    for t in trades:
        need(t['date'] in schedule and t['side'] in ('buy','sell') and (t['date'],t['symbol']) not in seen,'Unique monthly net trade')
        seen.add((t['date'],t['symbol']))
        need(all(finite(t[k]) and t[k]>0 for k in ('adjusted_units','adjusted_price','dollars')) and finite(t['cost']) and t['cost']>=0,'Finite positive saved fill')
        near(t['adjusted_units']*t['adjusted_price'],t['dollars'],'Saved fill dollar identity',residuals)
        expected=t['dollars']*(.0005 if path=='next_open_cost' else 0.)
        need(abs(t['cost']-expected)<=1e-9+ACCOUNT_RTOL*abs(expected),'Frozen ptr169 fill-fee bound')
        if path!='next_open_cost':need(t['cost']==0.,'ZERO trade fee exact')
        fee+=t['cost'];turnover+=t['dollars']
    if path!='next_open_cost':need(all(h['fee']==0. for h in holdings),'ZERO holding fee exact')
    return dict(trade_rows=len(trades),saved_fill_cost_sum=fee,saved_dollar_turnover=turnover,ordered_period_fees=ordered_period_fees(holdings),no_intra_month_units_or_cash_changes=True,export_dust_policy=dict(source='ptr169_rebuild_evaluate.py:273-280 and ptr169_results_independent.py:174-203',export_abs_units_above=1e-9,minimum_exported_units_exclusive=-1e-7,maximum_non_target_absolute_units=1e-7,actual_non_target_dust=dust,targets_absent_after_export_filter=filtered_targets,absent_tiny_target_units_verified_by_bound_independent_source_price_certificate=True))

def recompute_saved_metrics(result,exports,orders,meta,calendar,training):
    residuals=[];allstats={};newtrain={};ledgers={}
    for arm in ARMS:
        allstats[arm]={};newtrain[arm]={};ledgers[arm]={}
        for path in PATHS:
            data=exports[arm,path];stats,sds=saved_nav_slices(data['DAILY'],meta);allstats[arm][path]=stats
            ledger=exported_ledger_check(data,orders[arm],calendar['sessions'],path,residuals);ledgers[arm][path]=ledger
            saved=result['arms'][arm][path]
            compare_metrics(stats['full122'],saved['metrics'],arm+'/'+path+'/fresh_saved_nav',residuals)
            for split in SLICES:compare_metrics(stats[split],saved['slices'][split],arm+'/'+path+'/fresh/'+split,residuals)
            need(saved['metrics']['fees_paid']==saved['slices']['full122']['fees_paid']==ledger['ordered_period_fees'],'Each saved cost aggregate exact ordered HOLDINGS sum')
            near(ledger['saved_fill_cost_sum'],saved['metrics']['fees_paid'],'Saved fill versus period fee',residuals)
            near(ledger['saved_dollar_turnover'],saved['metrics']['dollar_turnover'],'Saved turnover sum',residuals)
            need(ledger['trade_rows']==saved['metrics']['trade_rows'],'Saved trade count')
            if path in PATHS[:2]:
                groups={}
                for short,split in [('training','first88'),('early','early53'),('recent','recent35')]:
                    groups[short]=dict(stats[split]);compare_metrics(groups[short],training[arm][path][short],arm+'/'+path+'/fresh_training/'+short,residuals)
                sd=sds['first88'];groups['training'].update(daily_return_sd=sd,sharpe_status='defined' if sd>0 else 'undefined_sharpe',sharpe_reason=None if sd>0 else 'zero_daily_return_sd')
                near(sd,training[arm][path]['training']['daily_return_sd'],'Saved training SD',residuals);newtrain[arm][path]=groups
    return dict(paths=9,slices=45,training_groups=18,metrics=allstats,ledger_checks=ledgers,actual_saved_metric_residuals=residuals,method='Independent stdlib statistics from saved DAILY; ptr169 scalar formula; saved fill/holding/position identities. No raw prices, model fitting or portfolio replay.'),newtrain

def schema_flags():
    schemas=constants(ROOT/'work/monthly_leader_utility260_kernel.py',('INPUT_KEYS','TARGET_RECORD_KEYS','TARGET_COUNT_FIELDS','SOURCE_COUNT_FIELDS','SOURCE_EVENT_KEYS','MODEL_COUNT_FIELDS','MODEL_AUDIT_KEYS','ANNUAL_KEYS','COUNT_FIELDS','EVENT_KEYS','AUDIT_KEYS','COLUMNS'))
    flags=constants(ROOT/'work/monthly_leader_utility260_independent.py',('INPUT_FLAGS','TARGET_FLAGS','SOURCE_FLAGS','MODEL_FLAGS','SELECTION_FLAGS'))
    flags['INCUMBENT_FLAGS']=constants(ROOT/'work/monthly_calendar_clock248_v2_independent.py',('INCUMBENT_GATE_FLAGS',))['INCUMBENT_GATE_FLAGS']
    need([len(schemas[k]) for k in ('INPUT_KEYS','TARGET_RECORD_KEYS','TARGET_COUNT_FIELDS','SOURCE_COUNT_FIELDS','MODEL_COUNT_FIELDS','ANNUAL_KEYS','COUNT_FIELDS','EVENT_KEYS')]==[15,14,9,5,10,17,15,12],'Final260 schemas')
    return schemas,flags


def model_counts(c,n):
    counts_check(c,SCHEMAS['MODEL_COUNT_FIELDS'],'Router counts')
    need(c['active_query_events']+c['cold_query_events']+c['partial_events']==n,'Router query partition')
    need(c['intercept_estimates']==c['ridge_solves']<=c['years'],'Only active paired intercept/Ridge estimates')
    need(c['valid_target_months']<=c['training_months'] and c['original_model_fits']==0,'Target subset and no original model fits')
    if n<=1:need(c['ridge_solves']==c['training_months']==0,'Empty/partial prefix performs no new training')


def decision_counts(c,n,width=49):
    counts_check(c,SCHEMAS['COUNT_FIELDS'],'Decision counts');integer(c['events'],'Decision events',n);integer(c['symbols_per_event'],'Decision width',width)
    need(c['active_events']+c['cold_events']+c['partial_events']==n,'Decision query partition')
    need(all(c[k]==1 for k in ('incumbent_episode_calls','intercept_episode_calls','state_episode_calls')) and c['new_model_fits']==0,'Three selector invocations; no new fits')


def receipt(r,n,kind):
    f={'early_incumbent':'INCUMBENT_FLAGS','incumbent':'INCUMBENT_FLAGS','inputs':'INPUT_FLAGS','targets':'TARGET_FLAGS','source':'SOURCE_FLAGS','models':'MODEL_FLAGS','selection':'SELECTION_FLAGS'}
    need(type(r) is dict and r.get('passed') is True and all(r.get(k) is True for k in FLAGS[f[kind]]),'Complete passed '+kind+' flags')
    integer(r.get('checked_events'),'Receipt query scope',n)
    need(r.get('producer_modules_imported') is False and r.get('computes_returns') is False,'Separate numerical source-only verifier')
    for k in ('old_model_solves','conditional_ridge_solves','independent_canonical_solves','independent_svd_solves','original_episode_calls'):integer(r.get(k),'Actual independent '+k)
    need(r['old_model_solves']==0,'Original model reused')
    if kind=='models':
        model_counts(r['counts'],n);fits=r['counts']['ridge_solves']
        need(all(r[k]==fits for k in ('conditional_ridge_solves','independent_canonical_solves','independent_svd_solves','intercept_estimates')),'Independent canonical and SVD fits separately counted')
        need(r.get('full_model_audit_supplied') is True,'Full model evidence')
    else:need(all(r[k]==0 for k in ('conditional_ridge_solves','independent_canonical_solves','independent_svd_solves')),'Nonmodel layer does not fit')
    need(r['original_episode_calls']==(1 if kind in ('early_incumbent','incumbent') else 3 if kind=='selection' else 0),'Actual independent selector calls')
    if kind in ('early_incumbent','incumbent','selection'):
        need(r.get('ordered_choices_checked')==(3*n if kind=='selection' else n) and r.get('full_scores_supplied') is True and r.get('full_pre_scores_supplied') is True and r.get('numerical_tolerance')==0.,'Exact complete selection evidence')
        if kind=='selection':
            decision_counts(r['counts'],n);need(r.get('full_bundle_supplied') is True and r.get('full_audit_supplied') is True,'Complete cached decision proof')
        else:need(r.get('candidate_selection_computed') is False,'Control gate precedes candidates')
    else:
        need(r.get('exact_discrete') is True,'Exact discrete proof')
        if kind in ('inputs','targets'):need(r.get('numerical_tolerance')==0.,'Exact input/teacher')
        else:need(r.get('continuous_atol')==r.get('continuous_rtol')==2e-10 and r.get('rank_tolerance')==0.,'Predeclared continuous/exact distinction')
        field={'inputs':'full_input_audit_supplied','targets':'full_target_audit_supplied','source':'full_source_audit_supplied','models':'full_model_audit_supplied'}[kind]
        need(r.get(field) is True,'Full layer supplied')
        if kind=='inputs':
            counts_check(r['counts'],('training_rows','training_months','query_rows'),'Input receipt');need(r['counts']['query_rows']==49*n,'Input receipt query rows equal original49 times prefix')
        if kind=='targets':
            counts_check(r['counts'],SCHEMAS['TARGET_COUNT_FIELDS'],'Target receipt')
            need(r['counts']['valid_months']+r['counts']['unavailable_months']==r['counts']['months'],'Target validity partition')
        if kind=='source':
            c=r['counts'];counts_check(c,SCHEMAS['SOURCE_COUNT_FIELDS'],'Source receipt')
            need(type(r.get('source_audit')) is dict and r['source_audit'].get('counts')==c,'Full source payload required and receipt counters bound')
            need(c['query_events']==n and c['state_vectors']==c['training_events']+n-c['partial_events'] and c['price_observations']==507*c['state_vectors'],'253+127+127 source scope')


def execution_accounting(proof):
    fresh=[r for r in proof['physical_prefixes'] if r['model_recomputed']]
    runs=[proof]+fresh;cached=proof['cached_prefixes'];totals={k:0 for k in ('old_model_solves','independent_canonical_solves','independent_svd_solves','original_episode_calls')}
    for run in runs:
        for kind in ('early_incumbent','inputs','targets','source','models','incumbent','selection'):
            for key in totals:totals[key]+=run[kind][key]
    for run in cached:
        for kind in ('incumbent','selection'):
            for key in totals:totals[key]+=run[kind][key]
    return dict(schema='monthly_leader_utility260_execution_v1',producer_original_model_fits=0,
      producer_intercept_estimates=sum(x['model_counts']['intercept_estimates'] for x in runs),producer_conditional_ridge_solves=sum(x['model_counts']['ridge_solves'] for x in runs),producer_original_episode_calls=4*len(runs)+3*len(cached),
      independent_intercept_estimates=sum(x['models']['intercept_estimates'] for x in runs),independent_actual_work=totals,total_linear_system_solves=sum(x['model_counts']['ridge_solves'] for x in runs)+totals['independent_canonical_solves']+totals['independent_svd_solves'],
      fresh_model_invocations=len(runs),fresh_physical_prefixes=[x['prefix'] for x in fresh],cached_decision_invocations=len(cached),physical122_reuses_main=True,
      producer_teacher_month_evaluations=sum(x['targets']['counts']['months'] for x in runs),independent_teacher_month_evaluations=sum(x['targets']['counts']['months'] for x in runs),producer_market_state_vectors=sum(x['source']['counts']['state_vectors'] for x in runs),independent_market_state_vectors=sum(x['source']['counts']['state_vectors'] for x in runs),counts_include_n0_function_invocations=True,returns_computed=False)


def target_counts_from_records(rows):
    c=dict.fromkeys(SCHEMAS['TARGET_COUNT_FIELDS'],0);c['months']=len(rows)
    for r in rows:
        c['source_rows']+=len(r['row_indices']);c['domain_rows']+=len(r['domain_indices'])
        c['cash_months']+=int(r['reason']=='cash');c['identical_set_months']+=int(r['reason']=='identical_sets')
        c['unavailable_months']+=int(not r['valid']);c['valid_months']+=int(r['valid'])
        c['selected_label_observations']+=len(r['selected_union']);c['nonzero_targets']+=int(r['valid'] and r['value']!=0.)
    return c


def decision_prefix_counts(events,n,width=49):
    integer(n,'Decision prefix');need(n<=len(events),'Available decision prefix')
    c=dict.fromkeys(SCHEMAS['COUNT_FIELDS'],0);c.update(events=n,symbols_per_event=width,incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1)
    for i,r in enumerate(events[:n]):
        need(r['query_index']==i and type(r['partial']) is type(r['active']) is bool,'Prefix event identity')
        c['partial_events']+=int(r['partial']);c['active_events']+=int(r['active']);c['cold_events']+=int(not(r['partial'] or r['active']))
        c['eligible_cells']+=0 if r['partial'] else len(r['eligible_indices'])
        for arm in ('intercept','state'):c[arm+'_changed_delta_cells']+=sum(v!=0. for v in r[arm+'_delta'])
        c['intercept_changed_holdings_events']+=int(set(r['intercept_choices'])!=set(r['incumbent_choices']))
        c['state_changed_holdings_events']+=int(set(r['state_choices'])!=set(r['incumbent_choices']))
        c['state_vs_intercept_changed_holdings_events']+=int(set(r['state_choices'])!=set(r['intercept_choices']))
    return c


def market_payload_check(market,calendar,training_dates,query_dates):
    need(type(market) is dict and set(market)=={'schema','columns','training','queries','counts'} and market['schema']=='monthly_leader_utility260_market_source_v1' and market['columns']==list(SCHEMAS['COLUMNS']),'Complete three-state source payload')
    axis=calendar['sessions'];need(axis==sorted(set(axis)) and all(type(d) is str and datetime.strptime(d,'%Y-%m-%d').strftime('%Y-%m-%d')==d for d in axis),'Official source calendar')
    positions={d:i for i,d in enumerate(axis)}
    need([r['date'] for r in market['training']]==training_dates and [r['date'] for r in market['queries']]==query_dates,'Exact source training/query scopes')
    partial=0;observations=0
    for kind in ('training','queries'):
        for r in market[kind]:
            need(set(r)==set(SCHEMAS['SOURCE_EVENT_KEYS']) and len(r['values'])==3 and all(finite(v) for v in r['values']),'Finite complete saved state evidence')
            expected_partial=kind=='queries' and r['date']=='2016-08-17'
            need(type(r['initial_partial']) is bool and r['initial_partial'] is expected_partial,'Only named query partial bypasses market source')
            for key in ('spy_price_count','xly_price_count','xlp_price_count'):integer(r[key],'Source observation count')
            if expected_partial:
                need(all(r[k] is None for k in ('spy_start','relative_start','price_end')) and all(r[k]==0 for k in ('spy_price_count','xly_price_count','xlp_price_count')) and r['values']==[0.,0.,0.],'Partial source consumes no market window')
                partial+=1
            else:
                need(r['date'] in positions and positions[r['date']]>=252,'Sufficient official source clock')
                t=positions[r['date']]
                need(r['spy_start']==axis[t-252] and r['relative_start']==axis[t-126] and r['price_end']==axis[t],'Exact official t-252/t-126 window endpoints')
                need((r['spy_price_count'],r['xly_price_count'],r['xlp_price_count'])==(253,127,127),'Exact source window lengths')
                observations+=507
    expected=dict(training_events=len(training_dates),query_events=len(query_dates),partial_events=partial,state_vectors=len(training_dates)+len(query_dates)-partial,price_observations=observations)
    counts_check(market['counts'],SCHEMAS['SOURCE_COUNT_FIELDS'],'Source payload counts');need(market['counts']==expected,'Source payload counts independently reconciled')
    return expected


def prefix_teacher_rows(model,meta,n):
    ordinary=[d for d in meta['dates'][:n] if d!='2016-08-17']
    if not ordinary:return []
    cutoff=f'{int(ordinary[-1][:4])-1:04d}-12-31'
    rows=[r for r in model['target']['records'] if r['label_end']<=cutoff]
    need(rows==model['target']['records'][:len(rows)],'Physical teacher maturity is a full-month prefix')
    return rows


def prefix_model_counts(model,meta,n,rows):
    c=dict.fromkeys(SCHEMAS['MODEL_COUNT_FIELDS'],0)
    years=sorted({int(d[:4]) for d in meta['dates'][:n] if d!='2016-08-17'})
    annual={r['year']:r for r in model['model']['annual']}
    c.update(years=len(years),training_months=len(rows),valid_target_months=sum(r['valid'] for r in rows),partial_events=int(n>0))
    for year in years:
        need(year in annual,'Physical annual proof exists');r=annual[year]
        c['annual_training_month_observations']+=len(r['training_indices'])
        fitted=r['fit_performed'];c['intercept_estimates']+=int(fitted);c['ridge_solves']+=int(fitted)
        qcount=sum(i<n for i in r['query_indices']);c['active_query_events' if fitted else 'cold_query_events']+=qcount
    return c


def preparation_evidence(p,model,decision,meta,calendar):
    need(p.get('all123_order_prefixes') is True and p.get('original_arrays_unchanged') is True and p.get('original_training_unchanged') is True,'Full original source preservation')
    need([x['prefix'] for x in p['cached_prefixes']]==list(range(123)),'Complete cached0..122')
    need([x['prefix'] for x in p['physical_prefixes']]==list(PREFIXES),'Complete physical boundaries')
    def verify(r,n):
        for k in ('early_incumbent','inputs','targets','source','models','incumbent','selection'):receipt(r[k],n,k)
        rows=prefix_teacher_rows(model,meta,n);tc=target_counts_from_records(rows)
        need(r['inputs']['counts']==dict(training_rows=tc['source_rows'],training_months=tc['months'],query_rows=49*n),'Physical input rows/months/query cross-layer scope')
        need(r['targets']['counts']==tc,'Physical teacher counters from mature saved month records')
        source=r['source']['source_audit']
        market_payload_check(source,calendar,[x['signal_date'] for x in rows],meta['dates'][:n])
        need(source['schema']==model['market']['schema'] and source['columns']==model['market']['columns'] and source['training']==model['market']['training'][:len(rows)] and source['queries']==model['market']['queries'][:n],'Every physical source payload equals its independently clock-checked saved prefix')
        model_counts(r['model_counts'],n)
        need(r['model_counts']==r['models']['counts']==prefix_model_counts(model,meta,n,rows),'Physical model counts from same saved annual/cohort scope')
        need(r['selection']['counts']==decision_prefix_counts(decision['events'],n),'Physical selection counters from saved events')
    verify(p,122)
    need(p['saved_model_audit_sha256']==digest_record(model),'Saved full input/teacher/market/model certificate')
    need(p['model_counts']==model['model']['counts'] and p['targets']['counts']==model['target']['counts'] and p['source']['source_audit']==model['market'],'Saved numerical layers bound')
    need(p['selection']['counts']==decision['counts'] and p['incumbent']==decision['incumbent_gate'],'Saved decisions bound')
    for r in p['cached_prefixes']:
        need(r['exact'] is True,'Cached exact prefix');receipt(r['incumbent'],r['prefix'],'incumbent');receipt(r['selection'],r['prefix'],'selection')
        need(r['selection']['counts']==decision_prefix_counts(decision['events'],r['prefix']),'Cached selection counters from saved events')
    for r in p['physical_prefixes']:
        n=r['prefix'];need(r['exact'] is True and r['model_recomputed'] is (n!=122),'Physical recomputation/reuse declaration')
        if n==122:
            need(r.get('reuses_main_verified_full') is True,'Explicit full reuse')
            for key in ('early_incumbent','inputs','targets','source','models','incumbent','selection','model_counts'):need(r[key]==p[key],'Main reuse exact proof')
        else:verify(r,n)
    need(p['fit_execution_accounting']==execution_accounting(p),'Actual source/model/selector execution ledger')


def source_and_model_summary(saved,meta,calendar):
    need(set(saved)=={'model','inputs','target','market'},'Complete saved four source layers')
    inp,target,market,model=(saved[k] for k in ('inputs','target','market','model'))
    need(set(inp)==set(SCHEMAS['INPUT_KEYS'])-{'plain','event_plain','labels'},'Input metadata shape')
    need(inp['n']==122 and inp['width']==49 and inp['event_dates']==meta['dates'] and inp['event_symbols']==meta['symbols'],'Original query layout')
    need(set(target)=={'schema','records','counts'},'Saved teacher audit scope');c=target['counts'];counts_check(c,SCHEMAS['TARGET_COUNT_FIELDS'],'Teacher counts')
    rows=target['records'];need(len(rows)==c['months'],'Teacher record count')
    calculated=dict.fromkeys(SCHEMAS['TARGET_COUNT_FIELDS'],0);calculated['months']=len(rows)
    for i,r in enumerate(rows):
        need(set(r)==set(SCHEMAS['TARGET_RECORD_KEYS']) and r['month_index']==i,'Complete teacher record')
        need(r['signal_date']<r['label_end'] and (i==0 or r['signal_date']>rows[i-1]['signal_date']),'Teacher chronology')
        need(type(r['valid']) is bool and r['k'] in (0,2,3),'Teacher support types')
        a,b=set(r['raw_choices']),set(r['risk_choices']);need(len(a)==len(b)==r['k'],'Paired expert size')
        need(r['selected_union']==sorted(a|b) and r['raw_only']==sorted(a-b) and r['risk_only']==sorted(b-a),'Lexical selected union/symmetric difference')
        need(r['reason'] in ('cash','selected_label_unavailable','identical_sets','available'),'Teacher availability reason')
        if r['valid']:
            need(finite(r['value']),'Valid scalar saved utility')
            if not r['k'] or a==b:need(r['value']==0.,'Explicit cash/same-set zero')
        else:need(r['value'] is None and r['reason']=='selected_label_unavailable' and r['k']>0,'No invalid label substitution')
        need(r['reason']!='cash' or r['k']==0,'Cash teacher size');need(r['reason']!='identical_sets' or (r['k']>0 and a==b),'Same-set teacher identity')
        calculated['source_rows']+=len(r['row_indices']);calculated['domain_rows']+=len(r['domain_indices']);calculated['cash_months']+=int(r['reason']=='cash');calculated['identical_set_months']+=int(r['reason']=='identical_sets');calculated['unavailable_months']+=int(not r['valid']);calculated['valid_months']+=int(r['valid']);calculated['selected_label_observations']+=len(r['selected_union']);calculated['nonzero_targets']+=int(r['valid'] and r['value']!=0.)
    need(calculated==c,'Teacher audit descriptive counters')
    need(c['source_rows']==len(inp['training_row_ids'])==len(inp['dates'])==len(inp['ends'])==len(inp['symbols']),'All training identity rows retained')
    need(inp['offsets']==[0]+[sum(len(x['row_indices']) for x in rows[:i+1]) for i in range(len(rows))],'Teacher month offsets match saved input')
    need(inp['training_row_ids']==list(range(c['source_rows'])) and len(inp['eligible'])==len(inp['common_good'])==c['source_rows'] and all(type(v) is bool for v in inp['eligible']+inp['common_good']),'Complete input row identity and Boolean source masks')
    for i,r in enumerate(rows):
        lo,hi=inp['offsets'][i:i+2];ids=inp['training_row_ids'][lo:hi]
        need(r['row_indices']==ids and inp['dates'][lo:hi]==[r['signal_date']]*(hi-lo) and inp['ends'][lo:hi]==[r['label_end']]*(hi-lo),'Teacher records bound to exact source identity rows')
        names=inp['symbols'][lo:hi];need(len(set(names))==len(names) and all(type(x) is str and x for x in names),'Teacher source identity names')
        need(all(type(j) is int and j in ids and inp['eligible'][j] for j in r['domain_indices']) and len(set(r['domain_indices']))==len(r['domain_indices']),'Teacher domain remains within saved eligible rows')
        need(set(r['raw_choices']+r['risk_choices'])<=set(inp['symbols'][j] for j in r['domain_indices']),'Chosen names belong to the saved teacher domain')
    market_payload_check(market,calendar,[r['signal_date'] for r in rows],meta['dates'])
    need(set(model)==set(SCHEMAS['MODEL_AUDIT_KEYS']) and model['original_models_refitted'] is False and model['market_columns']==list(SCHEMAS['COLUMNS']),'Router model scope')
    model_counts(model['counts'],122);annual=model['annual'];need(len(annual)==model['counts']['years'],'Annual records')
    fits=0;observations=0;active=[];cold=[];seen=[]
    for r in annual:
        need(set(r)==set(SCHEMAS['ANNUAL_KEYS']),'Complete annual sufficient-statistics audit')
        year=r['year'];integer(year,'Annual year');need(year not in seen,'One annual fit');seen.append(year)
        need(r['cutoff']==f'{year-1:04d}-12-31' and r['lower_bound']==f'{year-6:04d}-12-31','Fixed five-year prior annual cutoff')
        expected=[i for i,t in enumerate(rows) if r['lower_bound']<t['signal_date']<=r['cutoff'] and t['label_end']<=r['cutoff'] and t['valid']]
        need(r['training_indices']==expected,'Paired valid mature monthly cohort')
        qids=[i for i,d in enumerate(meta['dates']) if int(d[:4])==year and d!='2016-08-17'];need(r['query_indices']==qids,'Annual formal query partition')
        observations+=len(expected);need(r['status'] in ('insufficient_months','constant_target','fitted') and r['fit_performed'] is (r['status']=='fitted'),'Explicit fit/bypass status')
        if len(expected)<18:
            need(r['status']=='insufficient_months' and r['weights']==[] and r['weight_sum']==0.,'No insufficient-history moments or fit')
            need(all(r[k] is None for k in ('mean_x','sd_x','scale_x','mean_d','target_sd','gram','rhs','beta')),'Cold fields absent')
        else:
            need(len(r['weights'])==len(expected) and all(finite(w) and w>0 for w in r['weights']) and finite(r['weight_sum']),'Saved positive annual weights')
            need(all(isinstance(r[k],list) and len(r[k])==3 and all(finite(v) for v in r[k]) for k in ('mean_x','sd_x','scale_x')),'Saved state moments')
            need(finite(r['mean_d']) and finite(r['target_sd']) and r['target_sd']>=0.,'Saved target moments')
            need(r['scale_x']==[1. if v<=1e-12 else v for v in r['sd_x']],'Fixed source scale floor')
            if r['target_sd']<=1e-12:need(r['status']=='constant_target' and all(r[k] is None for k in ('gram','rhs','beta')),'Nearconstant whole-row bypass')
            else:
                need(r['status']=='fitted' and len(r['beta'])==len(r['rhs'])==len(r['gram'])==3 and all(finite(v) for v in r['beta']+r['rhs']) and all(len(v)==3 and all(finite(x) for x in v) for v in r['gram']),'Finite saved three-state Ridge proof');fits+=1
        (active if r['fit_performed'] else cold).extend(qids)
    c=model['counts'];need(c['ridge_solves']==c['intercept_estimates']==fits and c['active_query_events']==len(active) and c['cold_query_events']==len(cold) and c['annual_training_month_observations']==observations and c['training_months']==len(rows) and c['valid_target_months']==target['counts']['valid_months'],'Actual model counts from saved statuses')
    return dict(teacher_counts=target['counts'],market_counts=market['counts'],model_counts=c,annual_statuses=[dict(year=r['year'],status=r['status'],months=len(r['training_indices']),queries=len(r['query_indices'])) for r in annual],active_query_indices=sorted(active),cold_query_indices=sorted(cold),target_is_clipped_security_label_utility_not_portfolio_return=True,original_models_reused_not_refitted=True,model_arithmetic_not_reexecuted=True)


def decision_summary(audit,meta,orders):
    need(set(audit)==set(SCHEMAS['AUDIT_KEYS']),'Exact decision audit shape');decision_counts(audit['counts'],122)
    need(len(audit['events'])==122,'Full decision events');c=dict.fromkeys(SCHEMAS['COUNT_FIELDS'],0);c.update(events=122,symbols_per_event=49,incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1)
    for i,r in enumerate(audit['events']):
        need(set(r)==set(SCHEMAS['EVENT_KEYS']) and r['query_index']==i and r['date']==meta['dates'][i],'Decision row identity')
        need(type(r['active']) is type(r['partial']) is bool and r['partial'] is (i==0),'Partial exact flag')
        need(not(r['partial'] and r['active']),'No partial router')
        need(r['eligible_indices']==sorted(set(r['eligible_indices'])) and all(type(j) is int and 0<=j<49 for j in r['eligible_indices']),'Original E indices')
        for prefix in ('intercept','state'):
            need(finite(r[prefix+'_p']) and .25<=r[prefix+'_p']<=.75,'Mixing coefficient bounds')
            delta=r[prefix+'_delta'];need(len(delta)==49 and all(finite(v) for v in delta),'Full saved delta')
            need(all(v==0. for j,v in enumerate(delta) if j not in r['eligible_indices']),'No changed ineligible cell')
            if not r['active']:need(all(v==0. for v in delta) and r[prefix+'_p']==.5,'Whole-row incumbent bypass')
            c[prefix+'_changed_delta_cells']+=sum(v!=0. for v in delta)
        choices=[r[k] for k in ('incumbent_choices','intercept_choices','state_choices')]
        for arm,js in zip(ARMS,choices):
            need(len(js)==len(set(js)) and all(type(j) is int and 0<=j<49 for j in js),'Ordered choice indices')
            need([meta['symbols'][i][j] for j in js]==orders[arm][i]['symbols'],'Audit choices exactly exported orders')
        if not r['active']:need(orders[ARMS[1]][i]==orders[ARMS[2]][i]==orders[ARMS[0]][i],'Cold/partial complete original orders')
        c['partial_events']+=r['partial'];c['active_events']+=r['active'];c['cold_events']+=not(r['partial'] or r['active']);c['eligible_cells']+=0 if r['partial'] else len(r['eligible_indices'])
        c['intercept_changed_holdings_events']+=set(choices[1])!=set(choices[0]);c['state_changed_holdings_events']+=set(choices[2])!=set(choices[0]);c['state_vs_intercept_changed_holdings_events']+=set(choices[2])!=set(choices[1])
    need(c==audit['counts'],'All decision counts from saved evidence')
    return dict(counts=c,active_query_indices=[r['query_index'] for r in audit['events'] if r['active']],partial_complete_copy=True,cold_complete_copy=True,original_E_preserved_by_independent_certificate=True,delta_mathematics_not_recomputed=True)
def run(args):
    global SCHEMAS,FLAGS
    need(not REPORT.exists(),'Report write-once')
    need(platform.python_version()=='3.9.6','Pinned offline wrapper required')
    for name in ('plan','prepared','frozen','results','terminals'):need(sha_value(getattr(args,name+'_sha256')),'Explicit authorized '+name+' digest')
    ev=Evidence();plan=ev.read(OUT/'PLAN.json',args.plan_sha256)
    for p,h in FIXED.items():ev.pin(p,h);need(ev.registered(plan,p)==h,'Registered fixed evidence '+p)
    ev.pin(READER_PARENT,READER_PARENT_SHA)
    context=ev.read(CONTEXT);contract=ev.read(CONTRACT);tests=ev.read(TESTS,ev.registered(plan,TESTS));code=ev.read(CODE_REVIEW,ev.registered(plan,CODE_REVIEW))
    need(plan['context']==context and plan['context_sha256']==digest(CONTEXT) and plan['implementation_contract']==contract and plan['implementation_contract_sha256']==digest(CONTRACT)==code['implementation_contract_sha256'],'Prospective context and contract exact')
    need(context['arms']==list(ARMS) and context['only_promotable']==ARMS[2] and context['diagnostic_only']==ARMS[1] and context['registered_trials']==3,'Only the state arm promotable')
    need(tests['passed'] is True and tests['synthetic_only'] is True and type(tests['tests']) is int and tests['tests']>0 and tests['runtime']==plan['runtime']==dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),'Passed formal synthetic receipt and pinned runtime')
    need(code['passed'] is True and code['unresolved_blockers']==[] and code['code_sha256']==tests['code_sha256'] and code['context_sha256']==digest(CONTEXT) and code['actual_returns_computed'] is False,'Prospective code review')
    expected={ROOT/'work'/name for stem in ('host','kernel','independent') for name in ('monthly_leader_utility260_'+stem+'.py','test_monthly_leader_utility260_'+stem+'.py')}
    need({local(p) for p in tests['code_sha256']}==expected,'Exactly six reviewed final source files')
    for p,h in tests['code_sha256'].items():need(ev.registered(plan,p)==h,'Registered code hash');ev.pin(p,h)
    for p in ('work/monthly_calendar_clock248_v2_independent.py',):ev.pin(p,ev.registered(plan,p))
    SCHEMAS,FLAGS=schema_flags()
    state=ev.read(OUT/'STATE_BEFORE.json',plan['prior_central_sha256']);need(len(state['completed'])==255 and sum(r['trials'] for r in state['completed'])==1276 and state['registered_queue']==[] and state['completed'][-1]['directory']==PRIOR.name,'Fixed pre-registration255/1276 state')
    prior_close=ev.read(PRIOR/'COMPLETION.json',ev.registered(plan,PRIOR/'COMPLETION.json'))
    need(prior_close['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prior_close['protected_files']==21444,'Prior full protection closure certificate')
    protected=ev.read(PROTECTED,ev.registered(plan,PROTECTED));need(len(protected)==plan['protected_files']==21444 and all(sha_value(h) for h in protected.values()),'Registered protected inventory')
    terminal=ev.read(TERMINALS,args.terminals_sha256);terminals_check(terminal,args.plan_sha256,dict(prepare=args.prepared_sha256,select=args.frozen_sha256,diagnose=args.results_sha256))
    pc=ev.phase(OUT/'prepared',args.prepared_sha256,args.plan_sha256,PREP);need(pc['events']==122 and pc['new_returns_computed'] is False,'Prepared before returns')
    frozen=ev.read(OUT/'selected/FROZEN.json',args.frozen_sha256);training=ev.read(OUT/'selected/TRAIN_RESULTS.json',frozen['training_sha256'])
    need(frozen['plan_sha256']==args.plan_sha256 and frozen['prepared_sha256']==args.prepared_sha256 and frozen['orders_sha256']==pc['artifacts_sha256']['ORDERS.json'],'Immutable first88 chain')
    need({p.name for p in (OUT/'selected').iterdir()}=={'FROZEN.json','TRAIN_RESULTS.json'},'Exact selection files')
    rc=ev.phase(OUT/'results',args.results_sha256,args.plan_sha256,RESULT);need(rc['frozen_sha256']==args.frozen_sha256 and rc['paths']==9 and rc['slices']==45,'Nine complete paths only after frozen selection')
    meta=ev.read(OUT/'prepared/META.json');orders=ev.read(OUT/'prepared/ORDERS.json')
    need(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and meta['ends'][:-1]==meta['dates'][1:] and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Fixed122 intervals')
    need(meta['ends'][52]==meta['dates'][53]=='2020-12-31' and meta['ends'][87]==meta['dates'][88]=='2023-11-30','Frozen first88 partitions')
    need(all(len(s)==49 and len(set(s))==49 for s in meta['symbols']) and set(orders)==set(ARMS),'Original49 and three arms')
    for arm in ARMS:
        need(len(orders[arm])==122 and [o['date'] for o in orders[arm]]==meta['dates'] and [o['end'] for o in orders[arm]]==meta['ends'],'Order clocks')
        need(all(o['signal_date']==o['date'] and len(o['symbols']) in (0,2,3) and len(o['symbols'])==len(set(o['symbols'])) and len(o['scores'])==len(o['symbols']) and all(finite(x) for x in o['scores']) for o in orders[arm]),'Finite EW3/EW2/cash choices')
    for p in (LOCAL/'local_linear_ORDERS.json',REBUILD/'ORDERS.json'):
        need(orders[ARMS[0]]==ev.read(p,ev.registered(plan,p)),'Exact frozen source incumbent orders')
    calendar=ev.read(CALENDAR,ev.registered(plan,CALENDAR));axis=calendar['sessions'];need(axis==sorted(set(axis)),'Official calendar axes')
    positions={d:i for i,d in enumerate(axis)}
    for i,d in enumerate(meta['dates']):
        t=positions[d];need(i==0 or t+1<len(axis) and axis[t+1][:7]!=d[:7],'Formal monthend')
    original=ev.read(LOCAL/'PLAN.json',ev.registered(plan,LOCAL/'PLAN.json'));source=ev.read(OUT/'prepared/SOURCE_PARITY.json')
    maps=[canonical_source_map(x) for x in (source['raw_source_hashes'],original['raw_source_hashes'],plan['raw_source_hashes'])];need(all(x==maps[0] for x in maps),'All original64 source path/digest pairs')
    for key,name in [('original_training_hash','TRAIN_INPUTS.npz'),('original_identity_hash','MONTHLY_META.json'),('saved169_parity_hash','PARITY.json')]:
        p=REBUILD/name;need(source[key]==ev.registered(plan,p),'Source cache registered pin');ev.pin(p,source[key])
    for key,p in [('calendar_sha256',CALENDAR),('price_archive_sha256',ROOT/'outputs/monthly_2016_extension/PRICES.npz')]:
        need(source[key]==ev.registered(plan,p),'Source archive pin');ev.pin(p,source[key])
    for name in ('COMPLETION.json','ORDERS.json','SCORES.npz','PRE_SCORES.npz','META.json'):
        p=PRIOR/'prepared'/name;need(source['incumbent_cache_sha256'][name]==ev.registered(plan,p),'Prior prepared source pin');ev.pin(p,source['incumbent_cache_sha256'][name])
    need(source['fresh_training_array_reconstruction'] is False and source['cached_training_basis']=='Previously rebuilt pinned cache; not fresh reconstruction of raw training inputs.','Honest cached training source')
    need(all(source[k] is True for k in ('original_arrays_unchanged','original_training_unchanged','original_orders_exact','original_scores_exact','original_pre_scores_exact')) and source['original_predictive_models_refitted']==0,'Exact incumbent/source preservation certificate')
    ma=ev.read(OUT/'prepared/MODEL_AUDIT.json');audit=ev.read(OUT/'prepared/DECISION_AUDIT.json');proof=ev.read(OUT/'prepared/INDEPENDENT_PREPARE.json')
    summary=source_and_model_summary(ma,meta,calendar);selection_summary=decision_summary(audit,meta,orders);preparation_evidence(proof,ma,audit,meta,calendar)
    need(summary['active_query_indices']==selection_summary['active_query_indices'],'Deployed active years exactly model support')
    changed={a+'_vs_'+b:changes(orders,b,a) for a,b in ((ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]))}
    selected=saved_selection(training,changed);need(selected==frozen['selection'],'Exact first88 guard recomputation')
    result=ev.read(OUT/'results/RESULTS.json');ind=ev.read(OUT/'results/INDEPENDENT_OUTCOME.json')
    need(result['selection']==selected and result['reused_sample'] is True and result['strict_pit'] is False and result['zero_cost_selection'] is True and result['next_open_cost_diagnostic_only'] is True and result['observer_promotion'] is False,'Correct result labels and frozen primary')
    need(ind['accounting_source_sha256']==FIXED['work/ptr169_results_independent.py'],'Frozen independent source-price accounting')
    ac,exports=accounting(ev,result,ind,training);fresh,newtrain=recompute_saved_metrics(result,exports,orders,meta,calendar,training)
    need(saved_selection(newtrain,changed)==selected,'Saved NAV independently supports same first88 selection')
    incumbent=previous_reference(ev,result,exports,ind,PRIOR,PREVIOUS_REVIEW,ARMS[0],ARMS[0])
    ev.pin('work/monthly_reference_gain250_RESULT_REVIEW.json','a9557b13524a04a9c6e16b3a6e308676de864b40a1a3947ce927e053444e4595')
    fee=ev.read('work/monthly_reference_gain250_RESULT_REVIEW.json')['incumbent_reproduction']['period_fee_exception']
    need(fee['absolute_bound']==PERIOD_FEE_ATOL and fee['scope']=='ONLY next_open_cost individual HOLDINGS.fee; no cumulative metric exception','Previously registered cost-period tolerance only')
    for p in ('work/ptr169_rebuild_evaluate.py','outputs/monthly_factor_search_v2/validate.py'):ev.pin(p,ev.registered(plan,p))
    ev.pin(Path(__file__));ev.pin(ROOT/'work/test_monthly_leader_utility260_saved_review.py')
    for p,h in ev.pins.items():need(digest(local(p))==h,'Evidence changed during saved review')
    full={a:{p:result['arms'][a][p]['metrics'] for p in PATHS} for a in ARMS}
    report=dict(schema='monthly_leader_utility260_saved_result_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,unresolved_blockers=[],
      reviewer_role='260 producer author performing an extra saved-result cross-check. Independently authored model verification comes from the separately bound asset verifier; this is not a wholly independent second strategy implementation.',
      scope='Hash/manifest/source-scope/counter reconciliation; saved DAILY scalar statistics and exported ledger identities; first88 rules. No NPZ decoding, source-price loading, model fitting, new features, or portfolio replay.',
      plan_sha256=args.plan_sha256,context_sha256=digest(CONTEXT),implementation_contract_sha256=digest(CONTRACT),prepared_completion_sha256=args.prepared_sha256,results_completion_sha256=args.results_sha256,phase_terminals_sha256=args.terminals_sha256,
      root_owned_terminal_receipts=terminal,formal_synthetic_tests=tests['tests'],bindings=dict(code_files=6,prepared_files=len(PREP),result_files=len(RESULT),all_checked_hashes_match=True,raw64_canonical_maps_exact=True,raw64_files_rehashed=False,full_registered_input_closure_rehashed=False),
      source=source,model_summary=summary,decision_summary=selection_summary,preparation=dict(all123_cached_prefixes=True,physical_prefixes=list(PREFIXES),physical122_reuses_main=True,fit_execution_accounting=proof['fit_execution_accounting'],full_bound_independent_source_models_scores_orders=True),
      selection=selected,changed_holdings=changed,accounting_review=ac,saved_nav_recalculation=fresh,incumbent_reproduction=incumbent,full_results=full,segmented_results={a:{p:result['arms'][a][p]['slices'] for p in PATHS} for a in ARMS},
      target_reached_by_saved_selected_result=full[selected['primary']]['same_close_zero']['cagr']>=.7,cost_label_answer=dict(headline_is_same_close_zero=True,includes_transaction_costs=False,cost_path_used_for_selection=False,cost_rate_per_side=.0005),
      protection_and_close=dict(status='pending_root_close',protected_manifest_hash_and_count_checked=True,protected_members_rehashed_by_reviewer=False,prior_completed_protection_certificate_checked=True,final_root_close_verified=False,mutable_central_read_or_bound=False,registered_pre_state_completed=255,registered_pre_state_trials=1276,expected_post_close_completed=256,expected_post_close_trials=1279),
      actual_models_computed_by_reviewer=False,actual_model_solve_count=0,npz_decoded=False,source_prices_loaded=False,actual_portfolio_replay=False,saved_nav_metrics_recomputed=True,phase_rerun=False,frozen_files_modified=False,observers_modified=False,strict_pit=False,reused_sample=True,
      reader_execution=dict(script=str(Path(__file__).relative_to(ROOT)),sha256=digest(Path(__file__)),phase_digests=vars(args),status='REPORT_WRITTEN_BEFORE_PROCESS_EXIT',terminal_exit_and_chunk='Root must receive separately confirmed owning-agent terminal after this write; not claimed in advance.'),
      limitations=['NPZ files are hashed, not decoded. Model/source/core/score exactness relies on full bound separately authored verification receipts.',
       'The teacher is clipped security-label utility, not realized portfolio returns. Both arms versus incumbent also change the noncalm core horizon; only state versus intercept isolates state conditioning.',
       'No original model refits; producer conditional fits and independent canonical/SVD solves counted separately. Physical122 is reused; every cached decision invocation and n0 selector call is counted.',
       'Saved NAV/fill/position statistics are checked without fresh price inputs. Source-price replay correctness remains the independent diagnose certificate.',
       'Only the state arm can pass first88 gates. Full/later/cost diagnostics cannot rescue rejection; the unconditional arm is never promoted.',
       'Final21444 protection and central CAS remain pending_root_close. Archived sample/universe/vintage limitations remain; no observer modification or future-performance claim.'],input_sha256=ev.pins)
    with REPORT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='SAVED_EVIDENCE_REVIEW_PASSED',report_sha256=digest(REPORT),primary=selected['primary'],admitted=selected['admitted'],paths=9,slices=45,training_groups=18)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','frozen','results','terminals'):parser.add_argument('--'+name+'-sha256',required=True)
    run(parser.parse_args())
