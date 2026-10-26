"""Independent round134 scalar HMM and preparation oracle.

No strategy producer import and no real data access at import time.  Actual
artifact review is a separate, explicitly gated entry point.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'outputs/monthly_latent_state134'
LOCAL = 'outputs/monthly_local_linear70'
PRICES = 'outputs/monthly_2016_extension/PRICES.npz'
NAMES = ('incumbent_local67', 'matched_memoryless_state', 'filtered_latent_state')
MIN_OBSERVATIONS = 756
MAX_UPDATES = 300
LL_TOL = 1e-7
VAR_FLOOR = 1e-4
PROB_FLOOR = 1e-8
HELPER = 'work/monthly_ridge_support70_result_review.py'
HELPER_SHA = 'd0f053881cb94e49e7fff9eaaeaa9e260f0b3c66419f87819391c309cb7cb8ef'
LOADER = 'outputs/monthly_stability_monitor/inputs.py'
LOADER_SHA = 'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded'
CALENDAR_MODULES = ('exchange_calendars.exchange_calendar_xnys',
                    'exchange_calendars.exchange_calendar',
                    'exchange_calendars.us_holidays',
                    'exchange_calendars.calendar_helpers', 'pandas.tseries.holiday')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(x):
    require(not isinstance(x, (bool, np.bool_)), 'Boolean is not a number')
    x = float(x)
    require(math.isfinite(x), 'Nonfinite numerical input')
    return x


def lse(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Finite log weights')
    high = max(values)
    return high + math.log(math.fsum(math.exp(v-high) for v in values))


def normalize(values, floor=0.):
    values = [finite(v) for v in values]
    require(len(values) == 2 and all(v >= 0 for v in values), 'Two nonnegative weights')
    values = [max(v, floor) for v in values]
    total = math.fsum(values)
    require(total > 0, 'Positive normalization')
    return [v/total for v in values]


def validate_parameters(p):
    require(set(p) == {'pi', 'transition', 'means', 'variances'}, 'Parameter keys')
    require(len(p['pi']) == len(p['transition']) == len(p['means']) == len(p['variances']) == 2,
            'Two states')
    for row in [p['pi']] + list(p['transition']):
        require(len(row) == 2 and all(finite(v) > 0 for v in row), 'Strictly positive probabilities')
        require(abs(math.fsum(row)-1.) <= 1e-12, 'Probability normalization')
    for v in p['means']:
        finite(v)
    require(all(finite(v) > 0 for v in p['variances']), 'Positive variance')


def emissions(z, p):
    validate_parameters(p)
    return [[-.5*(math.log(2*math.pi*v)+(finite(x)-mu)**2/v)
             for mu, v in zip(p['means'], p['variances'])] for x in z]


def sufficient_statistics(z, p):
    """Independent scalar log-domain forward/backward, with normalized xi."""
    z = [finite(v) for v in z]
    require(len(z) >= 2, 'At least two observations')
    e = emissions(z, p)
    la = [[math.log(v) for v in row] for row in p['transition']]
    alpha, scales = [], []
    row = [math.log(p['pi'][k])+e[0][k] for k in range(2)]
    scale = lse(row)
    alpha.append([v-scale for v in row]); scales.append(scale)
    for t in range(1, len(z)):
        row = [lse(alpha[-1][j]+la[j][k] for j in range(2))+e[t][k] for k in range(2)]
        scale = lse(row)
        alpha.append([v-scale for v in row]); scales.append(scale)
    beta = [[0., 0.] for _ in z]
    for t in range(len(z)-2, -1, -1):
        beta[t] = [lse(la[j][k]+e[t+1][k]+beta[t+1][k] for k in range(2))-scales[t+1]
                   for j in range(2)]
    gamma = []
    for a, b in zip(alpha, beta):
        row = [a[k]+b[k] for k in range(2)]; norm = lse(row)
        gamma.append([math.exp(v-norm) for v in row])
    xi_terms = [[[] for _ in range(2)] for _ in range(2)]
    for t in range(len(z)-1):
        row = [[alpha[t][j]+la[j][k]+e[t+1][k]+beta[t+1][k] for k in range(2)] for j in range(2)]
        norm = lse(v for r in row for v in r)
        for j in range(2):
            for k in range(2): xi_terms[j][k].append(math.exp(row[j][k]-norm))
    return {'log_likelihood': math.fsum(scales),
            'filtered': [[math.exp(v) for v in row] for row in alpha],
            'gamma': gamma,
            'occupancy': [math.fsum(row[k] for row in gamma) for k in range(2)],
            'xi_sum': [[math.fsum(xi_terms[j][k]) for k in range(2)] for j in range(2)]}


def em_update(z, stats):
    counts = stats['occupancy']
    require(all(c > 0 for c in counts), 'Nonempty state occupancy')
    means = [math.fsum(row[k]*x for row, x in zip(stats['gamma'], z))/counts[k] for k in range(2)]
    variances = [max(VAR_FLOOR, math.fsum(row[k]*(x-means[k])**2
                         for row, x in zip(stats['gamma'], z))/counts[k]) for k in range(2)]
    return {'pi': normalize(stats['gamma'][0], PROB_FLOOR),
            'transition': [normalize(normalize(row), PROB_FLOOR) for row in stats['xi_sum']],
            'means': means, 'variances': variances}


def canonical(p):
    order = sorted(range(2), key=lambda k: p['variances'][k])
    return {'pi': [p['pi'][k] for k in order],
            'transition': [[p['transition'][j][k] for k in order] for j in order],
            'means': [p['means'][k] for k in order],
            'variances': [p['variances'][k] for k in order]}


def fit_oracle(raw):
    """Same declared estimator, independent scalar implementation, no market I/O."""
    values = [finite(v) for v in raw]
    if len(values) < MIN_OBSERVATIONS:
        return {'available': False, 'reason': 'insufficient_observations'}
    mean = math.fsum(values)/len(values)
    sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
    if sd <= 1e-12:
        return {'available': False, 'reason': 'scale_unavailable', 'mean': mean, 'sd': sd}
    z = [(v-mean)/sd for v in values]
    p = {'pi': [.5, .5], 'transition': [[.97, .03], [.03, .97]],
         'means': [0., 0.], 'variances': [.5, 2.]}
    stats = sufficient_statistics(z, p); trace = [stats['log_likelihood']]
    for update in range(1, MAX_UPDATES+1):
        p = em_update(z, stats)
        new_stats = sufficient_statistics(z, p)
        improvement = (new_stats['log_likelihood']-stats['log_likelihood'])/len(z)
        require(improvement >= -LL_TOL, 'Material likelihood decrease')
        stats = new_stats; trace.append(stats['log_likelihood'])
        if abs(improvement) <= LL_TOL:
            break
    else:
        raise ValueError('Nonconvergence')
    p = canonical(p); stats = sufficient_statistics(z, p)
    reason = 'available'
    if min(stats['occupancy']) < 20:
        reason = 'low_occupancy'
    elif p['variances'][1]/p['variances'][0] < 1.05:
        reason = 'unseparated_variances'
    return {'available': reason == 'available', 'reason': reason,
            'mean': mean, 'sd': sd, 'parameters': p, 'updates': update,
            'log_likelihood_trace': trace, 'occupancy': stats['occupancy'],
            'terminal_filtered': stats['filtered'][-1]}


def stationary(p):
    validate_parameters(p)
    a, b = p['transition'][0][1], p['transition'][1][0]
    return [b/(a+b), a/(a+b)]


def step_filter(value, prior, p, *, predict=True):
    validate_parameters(p)
    prior = normalize(prior)
    if predict:
        prior = [math.fsum(prior[j]*p['transition'][j][k] for j in range(2)) for k in range(2)]
    e = emissions([value], p)[0]
    row = [math.log(prior[k])+e[k] for k in range(2)]; norm = lse(row)
    return [math.exp(v-norm) for v in row]


def query_oracle(raw, fitted):
    require(fitted['available'], 'Available model required')
    p = fitted['parameters']; posterior = list(fitted['terminal_filtered']); output=[]
    for value in raw:
        z = (finite(value)-fitted['mean'])/fitted['sd']
        posterior = step_filter(z, posterior, p)
        memoryless = step_filter(z, stationary(p), p, predict=False)
        output.append({'filtered': posterior[0], 'memoryless': memoryless[0]})
    return output


def original_helper():
    path = ROOT/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA, 'Frozen independent helper')
    spec = importlib.util.spec_from_file_location('latent134_independent_baseline', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def score_oracle(a, meta, raw_rank, risk_rank, filtered, memoryless, n=None):
    h = original_helper(); x = h.original_prefix(a, meta, n); n = x['n']
    raw_rank = np.asarray(raw_rank)[:n]; risk_rank = np.asarray(risk_rank)[:n]
    require(raw_rank.shape == risk_rank.shape == x['core'].shape, 'Aligned core flavors')
    probs = {'matched_memoryless_state': memoryless, 'filtered_latent_state': filtered}
    scores, choices, deltas = {}, {}, {}
    for name in NAMES:
        pre = x['local'].copy(); delta = np.zeros_like(pre)
        if name != NAMES[0]:
            require(len(probs[name]) >= n, 'Aligned probabilities')
            for i in range(n):
                q = probs[name][i]
                if q is None or (isinstance(q, (float, np.floating)) and math.isnan(q)):
                    continue
                q = finite(q); require(0 <= q <= 1, 'Posterior range')
                for j in range(pre.shape[1]):
                    u, v = raw_rank[i,j], risk_rank[i,j]
                    require(not math.isinf(u) and not math.isinf(v), 'No infinite core rank')
                    if math.isnan(u) or math.isnan(v):
                        continue
                    require(-.5-1e-12 <= u <= .5+1e-12 and -.5-1e-12 <= v <= .5+1e-12,
                            'Centered core rank bounds')
                    desired = q*u + (1-q)*v
                    delta[i,j] = x['cw'][i,j]*desired-x['core'][i,j]
            pre[delta != 0] += delta[delta != 0]
        selected, final, _ = h.episode(pre, x)
        choices[name] = selected; scores[name] = final; deltas[name] = delta
    return scores, choices, deltas


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    files = [Path(__file__).resolve(), Path(__file__).with_name('test_monthly_latent_state134_prepare_review.py'),
             ROOT/HELPER, ROOT/LOADER]
    files += [Path(importlib.import_module(k).__file__).resolve() for k in CALENDAR_MODULES]
    files += [ROOT/'outputs/index_union/CALENDAR_EXCEPTIONS.json',
              ROOT/'outputs/index_union/sources/closure_20250109.html']
    return sorted(set(files))


def fit_record(raw):
    """Translate independently reconstructed fit into the published audit schema."""
    fit = fit_oracle(raw)
    scaler = None if 'mean' not in fit else {'mean': fit['mean'], 'sd': fit['sd']}
    if 'parameters' not in fit:
        reason = 'degenerate_training_scale' if fit['reason'] == 'scale_unavailable' else fit['reason']
        return dict(available=False, status='unavailable', reason=reason, fit_attempted=False,
            n_observations=len(raw), scaler=scaler, parameters=None, filtered_posterior=None,
            log_likelihood=None, log_likelihood_history=[], updates=0, occupancy=None,
            variance_ratio=None, certificate={'converged':False})
    p = fit['parameters']; history = fit['log_likelihood_trace']
    reasons = {'available':None, 'low_occupancy':'insufficient_effective_state_occupancy',
               'unseparated_variances':'unseparated_state_variances'}
    return dict(available=fit['available'], status='available' if fit['available'] else 'unavailable',
        reason=reasons[fit['reason']], fit_attempted=True, n_observations=len(raw), scaler=scaler,
        parameters=dict(initial=p['pi'], transition=p['transition'], means=p['means'], variances=p['variances']),
        filtered_posterior=fit['terminal_filtered'], log_likelihood=history[-1],
        log_likelihood_history=history, updates=fit['updates'], occupancy=fit['occupancy'],
        variance_ratio=p['variances'][1]/p['variances'][0],
        certificate=dict(converged=True, all_arithmetic_finite=True, low_variance_state_first=True,
            likelihood_units='standardized_returns', last_improvement_per_observation=(history[-1]-history[-2])/len(raw),
            tolerance=LL_TOL, maximum_updates=MAX_UPDATES, variance_floor=VAR_FLOOR,
            probability_floor_then_normalize=PROB_FLOOR))


def dates(values, label):
    axis = pd.DatetimeIndex(values)
    require(not axis.hasnans and axis.is_unique and axis.is_monotonic_increasing and
            axis.equals(axis.normalize()), 'Canonical dates '+label)
    return [x.date() for x in axis]


def annual_oracle(price, sessions, signals, fit_fn=fit_record):
    """Independently select complete annual windows and filter each signal prefix."""
    cal = dates(sessions, 'calendar'); sig = dates(signals, 'signals')
    idx = dates(price.index, 'price index')
    if not sig:
        return dict(filtered=np.empty(0),memoryless=np.empty(0),audit=dict(years=[],signals=[],new_model_fits=0))
    require(set(sig) <= set(cal), 'Signals are official sessions')
    cal = [d for d in cal if d <= sig[-1]]; pos={d:i for i,d in enumerate(cal)}
    # Never coerce a future price value while validating an earlier prefix.
    prices = {d:float(price.iloc[i]) for i,d in enumerate(idx) if d<=sig[-1]}
    out=[float('nan')]*len(sig); mem=[float('nan')]*len(sig); years=[]; rows=[None]*len(sig)
    for year in sorted(set(d.year for d in sig)):
        cutoff=date(year-1,12,31); lower=date(year-6,12,31)
        required=[d for d in cal if lower < d <= cutoff]
        own=[i for i,d in enumerate(sig) if d.year==year]
        row=dict(year=year,cutoff=str(cutoff),lower_bound_exclusive=str(lower),
            training_start=str(required[0]) if required else None,training_end=str(required[-1]) if required else None,
            training_observations=len(required),preceding_price_date=None,fit_attempted=False,
            status='unavailable',reason=None,model=None,first_invalid_query_date=None,query_observations=0)
        if not required or pos[required[0]]==0 or cal[0]>lower:
            row['reason']='incomplete_training_calendar'
        elif len(required)<MIN_OBSERVATIONS:
            row['reason']='insufficient_observations'
        else:
            preceding=cal[pos[required[0]]-1];row['preceding_price_date']=str(preceding)
            values=[prices.get(d,float('nan')) for d in [preceding]+required]
            if not all(math.isfinite(v) and v>0 for v in values):
                row['reason']='incomplete_training_prices'
            else:
                raw=[values[i]/values[i-1]-1 for i in range(1,len(values))]
                model=fit_fn(raw);row['model']=model;row['fit_attempted']=model['fit_attempted']
                row['status']=model['status'];row['reason']=model['reason']
        if row['status']=='available':
            model=row['model'];pp=model['parameters']
            p=dict(pi=pp['initial'],means=pp['means'],variances=pp['variances'],transition=pp['transition'])
            posterior=model['filtered_posterior'];prior=stationary(p);broken=False
            queries=[d for d in cal if cutoff<d<=sig[own[-1]]];row['query_observations']=len(queries)
            bydate={d:i for i,d in enumerate(sig)}
            for d in queries:
                before=cal[pos[d]-1];a=prices.get(before,float('nan'));b=prices.get(d,float('nan'))
                if not broken and not all(math.isfinite(v) and v>0 for v in (a,b)):
                    broken=True;row['first_invalid_query_date']=str(d)
                if broken:continue
                value=((b/a-1)-model['scaler']['mean'])/model['scaler']['sd']
                posterior=step_filter(value,posterior,p)
                memoryless=step_filter(value,prior,p,predict=False)
                if d in bydate:
                    i=bydate[d];out[i]=posterior[0];mem[i]=memoryless[0]
        for i in own:
            ok=math.isfinite(out[i]);bad=row['first_invalid_query_date']
            rows[i]=dict(signal_index=i,signal_date=str(sig[i]),year=year,available=ok,
                reason=None if ok else row['reason'] or 'invalid_query_history',
                first_invalid_query_date=bad if bad is not None and bad<=str(sig[i]) else None,
                filtered_probability=out[i] if ok else None,memoryless_probability=mem[i] if ok else None)
        years.append(row)
    return dict(filtered=np.array(out),memoryless=np.array(mem),audit=dict(years=years,signals=rows,
        new_model_fits=sum(row['fit_attempted'] for row in years)))


def centered_rank(values):
    good=[float(v) for v in values if math.isfinite(v)]
    result=[]
    for v in values:
        if not math.isfinite(v):result.append(0.);continue
        rank=sum(w<v for w in good)+(sum(w==v for w in good)+1)/2
        result.append(rank/len(good)-.5)
    return np.asarray(result)


def reconstruct_core(panel, meta):
    """Direct endpoint/sample-sigma formulas; no frozen strategy imports."""
    prices=panel['adj']; raw=[];risk=[];weights=[];eligible=[]
    for signal,names in zip(meta['dates'],meta['symbols']):
        t=prices.index.get_loc(pd.Timestamp(signal));require(isinstance(t,(int,np.integer)) and t>=294,'Core price history')
        u=[];v=[];good=[]
        for name in names:
            x=prices[name].iloc[t-294:t+1].to_numpy(float)
            a,b,c=x[-43],x[-253],x[0]
            u.append(a/b-1 if math.isfinite(a) and math.isfinite(b) and b!=0 else float('nan'))
            last=x[-64:]; rr=[last[k]/last[k-1]-1 for k in range(1,64)]
            if all(math.isfinite(r) for r in rr):
                mean=math.fsum(rr)/63;std=math.sqrt(math.fsum((r-mean)**2 for r in rr)/62)
            else:std=float('nan')
            v.append((a/c-1)/std if all(math.isfinite(z) for z in (a,c,std)) and c!=0 and std>0 else float('nan'))
            good.append(bool(np.count_nonzero(~np.isnan(x))==295 and x[-1]>0 and u[-1]>0))
        spy=prices['SPY'].iloc[t-63:t+1].to_numpy(float);rr=spy[1:]/spy[:-1]-1
        mean=math.fsum(rr)/63;sigma=math.sqrt(math.fsum((float(r)-mean)**2 for r in rr)/62)
        weights.append(.5 if sigma*math.sqrt(252)<.1 else .52)
        rraw=centered_rank(u);rrisk=centered_rank(v)
        # Published new arrays are recovered from weighted original cache.
        raw.append((.5*rraw)/.5);risk.append((.52*rrisk)/.52);eligible.append(good)
    return np.asarray(raw),np.asarray(risk),np.asarray(weights),np.asarray(eligible,dtype=bool)


class Reader:
    def __init__(self,root):self.root=Path(root).resolve();self.seen={}
    def path(self,value):
        p=Path(value);return p.resolve() if p.is_absolute() else (self.root/p).resolve()
    def sha(self,value):
        p=self.path(value);digest=hashlib.sha256(p.read_bytes()).hexdigest();key=str(p)
        require(key not in self.seen or self.seen[key]==digest,'Source changed '+key)
        self.seen[key]=digest;return digest
    def json(self,value):
        self.sha(value)
        def pairs(items):
            out={}
            for k,v in items:require(k not in out,'Duplicate JSON key');out[k]=v
            return out
        def bad(v):raise ValueError('Nonfinite JSON '+v)
        return json.loads(self.path(value).read_text(),object_pairs_hook=pairs,parse_constant=bad)
    def npz(self,value):
        self.sha(value)
        with np.load(self.path(value),allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    def verify(self,values,base=None):
        require(isinstance(values,dict) and bool(values),'Nonempty bindings');out={}
        for key,digest in values.items():
            p=Path(key)
            if base is not None and not p.is_absolute():p=Path(base)/p
            require(self.sha(p)==digest,'Hash mismatch '+str(p));out[str(self.path(p))]=digest
        return out


def official_sessions(reader,end):
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    manifest=reader.json('outputs/index_union/CALENDAR_EXCEPTIONS.json');closures=set()
    for row in manifest['closures']:
        if 'XNYS' in row['markets']:
            path=reader.path('outputs/index_union')/row['source_file']
            require(reader.sha(path)==row['source_sha256'],'Calendar evidence');closures.add(row['date'])
    cal=XNYSExchangeCalendar(start='2008-01-01',end=end)
    return pd.DatetimeIndex([str(d.date()) for d in cal.sessions if str(d.date()) not in closures])


def price_parity(panel,cache):
    idx=pd.DatetimeIndex(cache['dates']);names=list(map(str,cache['symbols']))
    require(idx.is_unique and idx.is_monotonic_increasing and len(names)==len(set(names)),'Price axes')
    require(cache['close'].shape==cache['opening'].shape==(len(idx),len(names)),'Price shapes')
    scales={};missing=[]
    for j,name in enumerate(names):
        source=panel['adj'].loc[idx,name].to_numpy(float);target=cache['close'][:,j];valid=np.isfinite(source)
        require(np.array_equal(valid,np.isfinite(target)),'Price masks')
        require(not np.isinf(source).any() and not np.isinf(target).any() and (source[valid]>0).all() and (target[valid]>0).all(),'Price validity')
        if valid.any():
            k=np.flatnonzero(valid)[0];scale=float(target[k]/source[k]);scales[name]=scale
            np.testing.assert_allclose(target[valid],source[valid]*scale,rtol=1e-10,atol=1e-10)
        else:missing.append(name)
        close=panel['close'].loc[idx,name].to_numpy(float);opening=panel['open'].loc[idx,name].to_numpy(float)
        expected=target*opening/close
        np.testing.assert_allclose(cache['opening'][:,j],expected,rtol=1e-10,atol=1e-10,equal_nan=True)
    return dict(adjusted_levels_constant_scale=True,adjusted_opening_parity=True,missing_price_masks_exact=True,
                positive_price_masks_exact=True,symbols=len(scales),scales=scales,all_missing_symbols=missing)


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'Object schema '+path)
        for k in expected:compare(actual[k],expected[k],path+'/'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'List schema '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual),'Finite float '+path)
        is_ll='log_likelihood' in path
        atol=1e-7 if is_ll else 1e-10
        rtol=0. if is_ll or '/scores' in path else 1e-9
        require(math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol),'Numeric mismatch '+path)
    else:require(type(actual) is type(expected) and actual==expected,'Discrete mismatch '+path)


CACHE_FILES=('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','PRETRAIN_ORDERS.json','PRICE_PARITY.json','STATE_AUDIT.json')
CHECKS=('original27_arrays_byte_identical','all_probabilities_and_core_ranks_independently_verified',
        'all_three_score_matrices_and_ordered_choices_independently_verified',
        'one_episode_and_prefix_isolation_verified','zero_and_missing_exact_fallback')
INPUTS=('proposal','meta','policy','original_orders','prices','transition','protected_manifest')
PROPOSAL_SHA='d752e5c823241b033a8bb4e05f49fb39224a6763eeb7de204d6075102d7fb778'


def audit(root,plan_path,plan_sha):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha) is not None,'Explicit PLAN SHA')
    r=Reader(root);p=r.path(STUDY);require(r.path(plan_path)==p/'PLAN.json','Dedicated PLAN')
    require(r.sha(plan_path)==plan_sha,'PLAN SHA');plan=r.json(plan_path)
    require(plan.get('round')==134 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Round134 policy')
    require(r.path(plan['output_directory'])==p,'Output directory')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Runtime')
    require(not any((p/x).exists() for x in ('TRAIN_TRIALS.json','FROZEN.json','RESULTS.json')),'Before outcome stages')
    pins=r.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Named inputs')
    inputs={}
    for k,ref in plan['inputs'].items():
        path=r.path(ref['path']);require(r.sha(path)==ref['sha256']==pins.get(str(path)),'Bound input '+k);inputs[k]=path
    require(r.sha(inputs['proposal'])==PROPOSAL_SHA,'Final proposal')
    require(plan.get('actual_model_fits_before_registration') == plan.get('actual_feature_calculations_before_registration') == plan.get('actual_returns_before_registration') == 0, 'Prospective registration')
    require(plan['tolerances'].get('state_atol')==1e-10 and plan['tolerances'].get('state_rtol')==1e-9 and plan['tolerances'].get('state_ll_atol')==1e-7 and plan['tolerances'].get('scores_atol')==1e-10 and plan['tolerances'].get('scores_rtol')==0, 'Frozen numeric tolerances')
    sources={str(f):r.sha(f) for f in source_files()}
    require(r.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Independent closure')
    old=r.json(LOCAL+'/PLAN.json');raw=r.verify(plan['raw_source_hashes'])
    require(raw==r.verify(old['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Raw64 source identity')
    oldcache=r.verify(r.json(LOCAL+'/CACHE_HASHES.json'),r.path(LOCAL))
    require(all(pins.get(k)==v for k,v in oldcache.items()),'All original cache artifacts bound')
    for k,file in [('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')]:
        require(inputs[k]==r.path(LOCAL+'/'+file),'Original wiring '+k)
    require(inputs['prices']==r.path(PRICES) and inputs['transition']==r.path('outputs/monthly_target70_research/TRANSITION.npz'),'Original numeric input wiring')
    h=original_helper();policy=r.json(inputs['policy']);require(all(policy.get(k)==v for k,v in h.POLICY.items()),'Frozen episode policy')
    cache=r.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Seven cached outputs');r.verify(cache,p)
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta']),'Exact metadata')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53
            and len(meta['etfs'])==17 and all(len(x)==49 for x in meta['symbols']),'Original122x49')
    original=r.npz(LOCAL+'/DATA.npz');a=r.npz(p/'DATA.npz')
    extras={'latent_filtered','latent_memoryless','latent_raw_rank','latent_risk_rank'}
    require(len(original)==27 and set(a)==set(original)|extras,'31 array schema')
    for k,v in original.items():require(a[k].shape==v.shape and a[k].dtype==v.dtype and a[k].tobytes()==v.tobytes(),'Original array bytes '+k)
    require(r.sha(LOADER)==LOADER_SHA,'Raw loader code')
    spec=importlib.util.spec_from_file_location('latent134_raw_loader',r.path(LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(LOADER))]=LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Actually consumed raw sources');r.verify(used)
    prices=r.npz(inputs['prices']);require(set(prices)=={'dates','symbols','close','opening'},'Price cache schema')
    parity=price_parity(panel,prices);compare(r.json(p/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    sessions=official_sessions(r,meta['ends'][-1])
    frame=pd.DataFrame(prices['close'],index=pd.to_datetime(prices['dates']),columns=prices['symbols'])
    raw_rank,risk_rank,cw,eligible=reconstruct_core(panel,meta);tr=r.npz(inputs['transition'])
    np.testing.assert_array_equal(eligible,original['eligible']);np.testing.assert_array_equal(eligible,tr['eligible'])
    np.testing.assert_array_equal(.5*raw_rank,tr['calm_core']);np.testing.assert_array_equal(.52*risk_rank,tr['noncalm_core'])
    np.testing.assert_array_equal(raw_rank,a['latent_raw_rank']);np.testing.assert_array_equal(risk_rank,a['latent_risk_rank'])
    expected_cw=np.broadcast_to(cw[:,None],(122,49));np.testing.assert_array_equal(expected_cw,tr['cw'])
    expected_core=np.where(cw[:,None]==.5,.5*raw_rank,.52*risk_rank)
    np.testing.assert_array_equal(expected_core,original['core']);np.testing.assert_array_equal(expected_core,tr['core'])
    fit_cache={}
    def cached_fit(values):
        key=tuple(values)
        if key not in fit_cache:fit_cache[key]=fit_record(values)
        return fit_cache[key]
    state=annual_oracle(frame['SPY'],sessions,meta['dates'],cached_fit)
    compare(r.json(p/'STATE_AUDIT.json'),state['audit'],'STATE_AUDIT')
    for key,name in [('latent_filtered','filtered'),('latent_memoryless','memoryless')]:
        np.testing.assert_allclose(a[key],state[name],rtol=1e-9,atol=1e-10,equal_nan=True)
        require(np.array_equal(np.isnan(a[key]),np.isnan(state[name])),'Probability mask')
    scores,choices,deltas=score_oracle(a,meta,raw_rank,risk_rank,state['filtered'],state['memoryless'])
    saved_scores=r.npz(p/'SCORES.npz');saved_orders=r.json(p/'PRETRAIN_ORDERS.json');require(set(saved_scores)==set(saved_orders)==set(NAMES),'Three output arms')
    wanted={}
    for name in NAMES:
        wanted[name]=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in choice],
            scores=[float(scores[name][i,j]) for j in choice]) for i,(day,choice) in enumerate(zip(meta['dates'],choices[name]))]
        compare(saved_orders[name],wanted[name],'orders/'+name)
        np.testing.assert_allclose(saved_scores[name],scores[name],rtol=0,atol=1e-10)
    h.assert_exact_tree(saved_orders[NAMES[0]],r.json(inputs['original_orders']))
    require(saved_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Baseline score bytes')
    for n in (0,1,36,53,60,76,88,122):
        cut=frame['SPY'].loc[:meta['dates'][n-1]] if n else frame['SPY'].iloc[:0]
        prefix=annual_oracle(cut,sessions,meta['dates'][:n],cached_fit)
        np.testing.assert_array_equal(prefix['filtered'],state['filtered'][:n]);np.testing.assert_array_equal(prefix['memoryless'],state['memoryless'][:n])
        compare(prefix['audit']['signals'],state['audit']['signals'][:n],'Signal prefix')
        ss,cc,_=score_oracle(a,meta,raw_rank,risk_rank,prefix['filtered'],prefix['memoryless'],n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score/choices prefix')
    missing=np.full(122,np.nan);ss,cc,dd=score_oracle(a,meta,raw_rank,risk_rank,missing,missing)
    for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Missing exact fallback')
    expected_feature=dict(original_arrays_unchanged=True,transition_cache_rebuilt_exact=True,original_months=122,original_width=49,
        new_model_fits=state['audit']['new_model_fits'],returns_or_nav_computed=False,original_auxiliary_states_and_core_coefficients_preserved=True)
    compare(r.json(p/'FEATURE_AUDIT.json'),expected_feature,'FEATURE_AUDIT')
    required={str(p/x):r.sha(p/x) for x in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for key,value in list(r.seen.items()):require(r.sha(key)==value and allowed.get(key)==value,'Unbound consumed artifact '+key)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),
        orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),**{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),
        producer_modules_imported=False,returns_or_nav_computed=False,monthly_label_models_fit=0,
        independent_daily_model_fits=len(fit_cache),orders_checked=366,raw_prices_independently_verified=True,
        calendar_overlay_independently_verified=True,price_parity=parity,all_signal_prefixes_verified=True,
        zero_posterior_is_valid_not_missing=True,missing_posterior_exact_baseline=True,
        limitations=['Unsupervised daily model fits reconstruct past data only; no strategy returns are computed.',
            'Original proxy universe, current-vintage adjusted prices and reused historical sample limitations remain.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/STUDY/'PLAN.json');parser.add_argument('--prepare-complete',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/STUDY/'INDEPENDENT_PREPARE.json');args=parser.parse_args()
    require(args.prepare_complete,'Run only after registered preparation')
    result=audit(ROOT,args.plan,args.plan_sha)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
