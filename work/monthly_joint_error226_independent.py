"""Independent joint-error covariance and boundary replacement oracle.

Only immutable caller-supplied cached forecasts and mature original labels are
consumed. No producer, strategy or predictive source fitter is imported.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'joint_prediction_error_variance', 'diagonal_prediction_error_variance')
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
MODEL_FLAGS = ('passed','independent_scalar_rebuild','exact_identity_join_checked',
    'cached_local_linear_reused','annual_prior_december_cutoffs_checked',
    'five_year_mature_window_checked','four_target_common_cohort_checked',
    'historical_original_universe_checked','fixed_top_two_checked',
    'original_boundary_domain_checked','common_complete_history_checked',
    'minimum_18_months_checked','original_targets_unchanged','column_centering_checked',
    'joint_gram_identity_checked','gram_positive_semidefinite_checked',
    'matched_diagonal_checked','negative_risk_effect_ranks_checked',
    'constant_risk_zero_effect_checked','paired_fallback_checked',
    'no_future_or_outcome_inputs','audit_checked','estimates_counted')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_support_checked',
    'original_budget_checked','fixed_top_two_checked','replacement_not_additive_checked',
    'one_episode_per_arm','paired_fallback_checked','exact_scores_and_choices_checked')
COUNT_FIELDS = ('query_months','initial_fallback_queries','domain_fallback_queries',
    'history_fallback_queries','supported_queries','boundary_candidates',
    'attempted_history_months','complete_history_months','incomplete_history_months',
    'joined_label_rows','invalid_label_rows','missing_security_rows','unsupported_source_rows',
    'skipped_initial_months','error_observations','security_mean_estimates',
    'security_variance_estimates','joint_variance_estimates','diagonal_variance_estimates',
    'parameter_estimates','cached_forecast_rows')
MODEL_COUNT_FIELDS = COUNT_FIELDS
ARRAY_KEYS = ('candidate_risk','control_risk','candidate_effect','control_effect','ready')
SOURCE_HELPER = ROOT/'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
GRAM_TOLERANCE = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return tuple(sorted(set([Path(__file__).resolve(), ACCOUNTING, *frozen_source_oracle().source_files(ROOT)])))


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('joint226_independent_accounting', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected, label, atol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaN mask')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    use = np.isfinite(expected)
    require(np.all(np.abs(actual[use] - expected[use]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, e) in enumerate(zip(actual, expected)):
            record_check(a, e, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label)
    return value


def fsum(values, label):
    try:
        return finite(math.fsum(values), label)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid ' + label) from exc


def real(value, label):
    require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
            label + ': nonboolean real numeric required')
    try:
        return float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(label + ': binary64 conversion') from exc


def days(values, label):
    raw = np.asarray(values)
    require(raw.ndim == 1, label + ': date vector')
    out = []
    for value in raw:
        if isinstance(value, np.datetime64):
            require(not np.isnat(value) and value == value.astype('datetime64[D]'),
                    label + ': normalized date required')
            value = value.astype('datetime64[D]')
        text = str(value)
        require(len(text) == 10, label + ': exact calendar-day dates')
        try:
            parsed = date.fromisoformat(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(label + ': invalid date') from exc
        require(parsed.isoformat() == text, label + ': noncanonical date')
        out.append(text)
    return out


def metadata(meta, label, width=None):
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'ends', 'symbols')),
            label + ': metadata keys')
    require(isinstance(meta['dates'], list) and isinstance(meta['ends'], list), label + ': monthly date lists')
    ds, es = days(meta['dates'], label + ' dates'), days(meta['ends'], label + ' ends')
    rows = meta['symbols']
    require(isinstance(rows, list) and len(rows) == len(ds) == len(es), label + ': metadata axes')
    require(all(a < b for a, b in zip(ds, ds[1:])), label + ': strictly ascending signal dates')
    offsets, flat = [0], []
    for i, (d, end, row) in enumerate(zip(ds, es, rows)):
        partial = label == 'query' and i == 0 and d == '2016-08-17' and end == '2016-08-31'
        month = int(d[:4])*12 + int(d[5:7])
        end_month = int(end[:4])*12 + int(end[5:7])
        require(end > d and (partial or end_month == month+1), label + ': next-natural-month label end')
        if i:
            previous_month = int(ds[i-1][:4])*12 + int(ds[i-1][5:7])
            initial_pair = label == 'query' and i == 1 and ds[0] == '2016-08-17' and d == '2016-08-31'
            require(month > previous_month or initial_pair, label + ': duplicate signal month')
        require(isinstance(row, list) and (width is None or len(row) == width) and
                all(isinstance(x, str) and x and x == x.strip() for x in row) and
                len(set(row)) == len(row), label + ': exact unique security IDs')
        flat.extend((d, s, end) for s in row)
        offsets.append(len(flat))
    require(len(set(flat)) == len(flat), label + ': duplicate identity')
    return ds, es, offsets, flat


def validated_inputs(train, training_meta, query_meta, source, historical_use):
    require(isinstance(train, dict) and isinstance(source, dict), 'Caller dictionaries required')
    eligibility = np.asarray(historical_use)
    require(eligibility.ndim == 2 and eligibility.shape[1] >= 1 and
            all(isinstance(v, (bool, np.bool_)) for v in eligibility.flat), 'Exact boolean original historical U')
    n, width = eligibility.shape
    qd, qe, qo, query_ids = metadata(query_meta, 'query', width)
    require(len(qd) == n, 'Query eligibility axis')
    td, te, to, training_ids = metadata(training_meta, 'training')
    required = ('offsets', 'dates', 'ends', 'train_plain', 'common_good') + tuple('target_'+k for k in TARGETS)
    require(all(k in train for k in required), 'Original training keys missing')
    offsets = np.asarray(train['offsets'])
    require(offsets.dtype.kind in 'iu' and np.array_equal(offsets, to), 'Original offsets differ')
    require(days(train['dates'], 'training flat dates') == [t[0] for t in training_ids] and
            days(train['ends'], 'training flat ends') == [t[2] for t in training_ids], 'Original flat date identities')
    size, count = len(training_ids), n * width
    require(np.asarray(train['train_plain']).shape == (size, 10), 'Original plain10 axes')
    for key in ('common_good',) + tuple('target_'+k for k in TARGETS):
        require(np.asarray(train[key]).shape == (size,), 'Training vector axes: '+key)
    require(all(k in source for k in ('available', 'raw_local_linear')),
            'Cached206 source keys missing')
    available = np.asarray(source['available'])
    require(available.shape == (count,) and all(isinstance(x, (bool, np.bool_)) for x in available),
            'Cached availability exact boolean axis')
    available = available.astype(bool, copy=True)
    raw = {}
    for key in ('raw_local_linear',):
        values = np.asarray(source[key])
        require(values.shape == (count,), 'Cached raw axis: '+key)
        raw[key] = np.array([real(v, key) for v in values], dtype=np.float64)
        require(np.array_equal(np.isfinite(raw[key]), available) and
                np.array_equal(np.isnan(raw[key]), ~available), 'Cached availability/raw mismatch: '+key)
    return eligibility.astype(bool), available, raw, qd, qe, qo, query_ids, training_ids


def product(a, b, label):
    value = finite(a*b, label)
    if a != 0. and b != 0. and value == 0.:
        raise FloatingPointError('Underflow '+label)
    return value


def mean(total, n, label):
    require(n > 0, 'Positive observation count required')
    value = finite(total/n, label)
    if total != 0. and value == 0.:
        raise FloatingPointError('Underflow '+label)
    return value


def frozen_source_oracle():
    require(sha(SOURCE_HELPER)==SOURCE_SHA,'Frozen independent source changed')
    spec=importlib.util.spec_from_file_location('joint226_original_independent',SOURCE_HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def negative_risk_ranks(risks):
    v=[finite(real(x,'risk'),'risk') for x in risks]
    require(len(v)>=2 and all(x>=0. for x in v),'At least two nonnegative risks')
    if all(x==v[0] for x in v):return [0.]*len(v)
    out=[]
    for x in v:
        lower=sum(y>x for y in v);equal=sum(y==x for y in v)
        rank=lower+(equal+1)/2.
        out.append((rank-1.)/(len(v)-1.)-.5)
    return out


def centered_gram(errors):
    """Independent scalar columns, common Gram, and canonical trio risks.

    Diagnostic Gram tolerance never changes canonical risks or rank ties.
    """
    values=np.asarray(errors)
    require(values.ndim==2 and values.shape[0]>=2 and values.shape[1]>=3,'Complete error matrix')
    n,k=values.shape
    rows=[[finite(real(v,'error'),'error') for v in row] for row in values]
    means=[mean(fsum([row[j] for row in rows],'error sum'),n,'error mean') for j in range(k)]
    z=[[finite(row[j]-means[j],'centered error') for j in range(k)] for row in rows]
    squares=[fsum([product(row[j],row[j],'centered error square') for row in z],'square sum') for j in range(k)]
    variances=[mean(s,n-1,'sample variance') for s in squares]
    gram=np.empty((k,k))
    for j in range(k):
        for q in range(j,k):
            value=finite(fsum([finite(row[j]*row[q],'Gram cross product') for row in z],'Gram sum')/(n-1),'Gram entry')
            gram[j,q]=gram[q,j]=value
    scale=max(1.,float(np.max(np.abs(gram))))
    bound=GRAM_TOLERANCE*scale
    minimum=float(np.linalg.eigvalsh(gram)[0])
    require(math.isfinite(minimum) and minimum>=-bound,'Common Gram is not positive semidefinite')
    candidate=[];control=[];identity_errors=[]
    for j in range(2,k):
        trio=(0,1,j)
        portfolio=[mean(fsum([row[q] for q in trio],'trio centered sum'),3,'trio centered mean') for row in z]
        full=mean(fsum([product(v,v,'trio square') for v in portfolio],'trio square sum'),n-1,'joint variance')
        diagonal=mean(fsum([squares[q] for q in trio],'trio diagonal sum'),9*(n-1),'diagonal variance')
        via_gram=fsum([float(gram[a,b]) for a in trio for b in trio],'Gram identity sum')/9.
        diag_gram=fsum([float(gram[q,q]) for q in trio],'Gram diagonal sum')/9.
        require(full>=0. and diagonal>=0.,'Negative variance')
        error=max(abs(full-via_gram),abs(diagonal-diag_gram))
        require(error<=bound,'Canonical and common-Gram risks differ')
        candidate.append(full);control.append(diagonal);identity_errors.append(error)
    return dict(means=means,square_sums=squares,variances=variances,
        candidate=candidate,control=control,gram=gram,minimum_eigenvalue=minimum,
        identity_max_error=max(identity_errors,default=0.),diagnostic_tolerance=bound)


def reconstruct(train, training_meta, query_meta, source, historical_use, pre_score, eligible):
    use, available, raw, dates, ends, offsets, query_ids, training_ids = validated_inputs(
        train, training_meta, query_meta, source, historical_use)
    n,width=use.shape
    good=np.asarray(eligible);require(good.shape==(n,width) and good.dtype==bool,'Current eligibility exact boolean axes')
    p=np.asarray(pre_score);require(p.shape==(n,width),'Current pre-score axes')
    p=np.array([[real(v,'pre-score') for v in row] for row in p],float).reshape(n,width)
    require(np.isfinite(p).all(),'Current pre-scores must be finite')
    lookup={key:i for i,key in enumerate(training_ids)}
    query_lookup=[{s:offsets[i]+j for j,s in enumerate(ss)} for i,ss in enumerate(query_meta['symbols'])]
    answer={k:np.full((n,width),np.nan) for k in ('candidate_risk','control_risk')}
    answer.update({k:np.zeros((n,width)) for k in ('candidate_effect','control_effect')})
    answer['ready']=np.zeros(n,bool)
    events=[];diagnostics=[]
    for i,(day,end,names) in enumerate(zip(dates,ends,query_meta['symbols'])):
        year=int(day[:4]);cutoff='%04d-12-31'%(year-1);start='%04d-12-31'%(year-6)
        order=sorted([j for j in range(width) if good[i,j]],key=lambda j:(-p[i,j],names[j]))
        top=order[:2]
        js=[j for j in order[2:5] if finite(float(p[i,order[2]])-float(p[i,j]),'boundary score gap')<=.02] if len(order)>=3 else []
        b=top+js
        c={key:0 for key in COUNT_FIELDS};c['query_months']=1
        c['cached_forecast_rows']=int(available[offsets[i]:offsets[i+1]].sum())
        c['boundary_candidates']=len(js)
        event=dict(query_index=i,date=day,end=end,year=year,cutoff=cutoff,
            lookback_exclusive_start=start,symbols=list(names),eligible_order=order,
            ordered_top2=top,J=js,B=b,status=None,H=[],latest_label_end=None,
            history=[],security_statistics=[],trios=[],counts=c)
        if day=='2016-08-17':
            event['status']='initial_partial';c['initial_fallback_queries']=1
        elif len(order)<3 or len(js)<2:
            event['status']='insufficient_candidates';c['domain_fallback_queries']=1
        else:
            for h,(signal,finish) in enumerate(zip(dates,ends)):
                if not (signal>start and finish<=cutoff):continue
                if signal=='2016-08-17':c['skipped_initial_months']+=1;continue
                record=dict(date=signal,end=finish,complete=False,rows=[])
                for j in b:
                    symbol=names[j];sid=query_lookup[h].get(symbol)
                    row=dict(symbol=symbol,source_row_id=sid,training_row_id=None,
                        reason=None,prediction=None,target=None,error=None)
                    if sid is None:
                        row['reason']='missing_security';c['missing_security_rows']+=1
                    elif not use[h,sid-offsets[h]]:
                        row['reason']='historical_use_false';c['unsupported_source_rows']+=1
                    elif not available[sid]:
                        row['reason']='source_unavailable';c['unsupported_source_rows']+=1
                    else:
                        key=(signal,symbol,finish)
                        require(key in lookup,'Missing mature exact training identity: '+repr(key))
                        tid=lookup[key];row['training_row_id']=tid;c['joined_label_rows']+=1
                        common=np.asarray(train['common_good'])[tid]
                        require(isinstance(common,(bool,np.bool_)),'Consumed common_good must be boolean')
                        if not common:
                            row['reason']='common_good_false';c['invalid_label_rows']+=1
                        else:
                            xs=[real(v,'historical plain10') for v in np.asarray(train['train_plain'])[tid]]
                            ys=[real(np.asarray(train['target_'+k])[tid],'historical target') for k in TARGETS]
                            if not all(math.isfinite(v) for v in xs+ys):
                                row['reason']='nonfinite_training';c['invalid_label_rows']+=1
                            else:
                                pred=float(raw['raw_local_linear'][sid]);target=ys[1]
                                row.update(reason='valid',prediction=pred,target=target,
                                    error=finite(pred-target,'historical cached prediction error'))
                    record['rows'].append(row)
                record['complete']=all(row['reason']=='valid' for row in record['rows'])
                c['attempted_history_months']+=1
                if record['complete']:
                    event['H'].append(signal);c['complete_history_months']+=1
                else:c['incomplete_history_months']+=1
                event['history'].append(record)
            full=[h for h in event['history'] if h['complete']]
            c['error_observations']=len(full)*len(b)
            event['latest_label_end']=max([h['end'] for h in full],default=None)
            if len(full)<18:
                event['status']='insufficient_history';c['history_fallback_queries']=1
            else:
                event['status']='ready';c['supported_queries']=1;answer['ready'][i]=True
                stats=centered_gram([[row['error'] for row in h['rows']] for h in full])
                ranks=[negative_risk_ranks(stats[k]) for k in ('candidate','control')]
                for t,j in enumerate(b):
                    event['security_statistics'].append(dict(index=j,symbol=names[j],
                        error_mean=stats['means'][t],error_sum_squares=stats['square_sums'][t],
                        sample_variance=stats['variances'][t]))
                for t,j in enumerate(js):
                    indices=top+[j]
                    event['trios'].append(dict(third_index=j,symbol=names[j],indices=indices,
                        symbols=[names[q] for q in indices],candidate_risk=stats['candidate'][t],
                        control_risk=stats['control'][t],candidate_effect=ranks[0][t],control_effect=ranks[1][t]))
                    for k,rank in zip(('candidate','control'),ranks):
                        answer[k+'_risk'][i,j]=stats[k][t];answer[k+'_effect'][i,j]=rank[t]
                c['security_mean_estimates']=c['security_variance_estimates']=len(b)
                c['joint_variance_estimates']=c['diagonal_variance_estimates']=len(js)
                c['parameter_estimates']=2*len(b)+2*len(js)
                diagnostics.append(dict(query_index=i,identity_max_error=stats['identity_max_error'],
                    minimum_eigenvalue=stats['minimum_eigenvalue'],tolerance=stats['diagnostic_tolerance']))
        events.append(event)
    counts={k:sum(e['counts'][k] for e in events) for k in COUNT_FIELDS}
    audit=dict(method='joint_asof_error_boundary_variance',arms=list(ARMS),query_months=n,
        symbols_per_event=width,min_months=18,lookback_calendar_years=5,ceiling=5,gap=.02,
        historical_errors='cached raw_local_linear minus unchanged target_log_excess',
        source_forecasts_refitted=False,labels_recomputed=False,local_available_filtered=False,
        historical_use_explicit=True,common_complete_history=True,counts=counts,event_history=events,
        source_model_refits=0,new_ridge_fits=0,new_tree_fits=0,
        new_statistical_estimation_queries=counts['supported_queries'],parameter_estimates=counts['parameter_estimates'])
    return dict(arrays=answer,audit=audit),diagnostics


def independent_build(*args):
    return reconstruct(*args)[0]


def check_build(train,training_meta,query_meta,source,historical_use,pre_score,eligible,built):
    expected,gram=reconstruct(train,training_meta,query_meta,source,historical_use,pre_score,eligible)
    require(isinstance(built,dict) and set(built)=={'arrays','audit'},'Build mapping')
    require(set(built['arrays'])==set(ARRAY_KEYS),'Five risk/effect arrays')
    for key,value in expected['arrays'].items():
        if key=='ready':
            observed=np.asarray(built['arrays'][key]);require(observed.dtype==bool and np.array_equal(observed,value),'Shared ready mask')
        else:close(built['arrays'][key],value,'Independent '+key,atol=0.)
    record_check(built['audit'],expected['audit'])
    a=expected['audit']
    return dict(**{key:True for key in MODEL_FLAGS},counts=a['counts'],query_rows=len(query_meta['dates']),
        source_model_refits=0,new_ridge_fits=0,new_tree_fits=0,
        new_statistical_estimation_queries=a['new_statistical_estimation_queries'],parameter_estimates=a['parameter_estimates'],
        numerical_tolerance=0.,gram_relative_tolerance=GRAM_TOLERANCE,gram_diagnostics=gram,
        producer_modules_imported=False,computes_returns=False)


verify_build=check_build


def selection_oracle(original,meta,built):
    helper=frozen_source_oracle().old.original_helper()
    x=helper.original_prefix(original,meta,None)
    pre=x['local'];n,width=pre.shape
    metadata(meta,'query',width)
    a=built['arrays'];require(set(a)==set(ARRAY_KEYS),'Selection five arrays')
    ready=np.asarray(a['ready']);require(ready.shape==(n,) and ready.dtype==bool,'Selection readiness')
    for key in ARRAY_KEYS[:-1]:require(np.asarray(a[key]).shape==(n,width),'Selection array axes')
    effects={arm:np.asarray(a[key+'_effect'],float) for arm,key in zip(ARMS[1:],('candidate','control'))}
    require(all(np.isfinite(v).all() and np.all(np.abs(v)<=.5) for v in effects.values()),'Finite bounded effects')
    # Compute each arm from the original pre exactly once. Original effect is
    # retained only for incumbent or whole-query fallback; never add two effects.
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={}
    events=built['audit']['event_history'];require(len(events)==n,'Selection event axis')
    for arm in ARMS:
        score=pre.copy();rows=[]
        for i,names in enumerate(x['names']):
            order=sorted(np.flatnonzero(x['eligible'][i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
            js=[j for j in order[2:5] if finite(float(pre[i,order[2]])-float(pre[i,j]),'boundary score gap')<=.02] if len(order)>=3 else []
            e=events[i]
            require(e['query_index']==i and e['date']==meta['dates'][i] and e['eligible_order']==order and
                e['ordered_top2']==order[:2] and e['J']==js and e['B']==order[:2]+js,'Selection domain differs')
            require(bool(ready[i])==(e['status']=='ready'),'Selection ready/status differs')
            use=arm!=ARMS[0] and bool(ready[i])
            if use:require(meta['dates'][i]!='2016-08-17' and len(js)>=2,'Ready initial or narrow domain')
            if arm!=ARMS[0]:
                outside=[j for j in range(width) if not (ready[i] and j in js)]
                require(not np.any(effects[arm][i,outside]),'Effects outside supported boundary')
                key='candidate' if arm==ARMS[1] else 'control'
                risk=np.asarray(a[key+'_risk'],float)[i]
                require(not np.isinf(risk).any() and np.array_equal(np.isfinite(risk),[ready[i] and j in js for j in range(width)]),'Risk common mask')
                if ready[i]:close(effects[arm][i,js],negative_risk_ranks(risk[js]),'Selection effect ranks')
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            for j in js:
                effect=effects[arm][i,j] if use else x['effects'][i,j]
                target=min(.5,max(-.5,x['support'][i,j]+effect))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            third=min(js,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[third],key=lambda j:(-score[i,j],names[j])))
        scores[arm]=score;choices[arm]=rows
    return scores,choices


def verify_selection(original,meta,built,scores,choices):
    require(set(scores)==set(choices)==set(ARMS),'Three selection arms')
    expected,ordered=selection_oracle(original,meta,built)
    for arm in ARMS:
        close(scores[arm],expected[arm],arm+' original-support single-episode score',atol=0.)
        require(choices[arm]==ordered[arm],arm+' ordered choices differ')
    return dict(**{key:True for key in SELECTION_FLAGS},
        ordered_choices_checked=3*len(meta['dates']),numerical_tolerance=0.,
        original_source_parity_tolerance=1e-12,producer_modules_imported=False,computes_returns=False)


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            observed = wanted['metrics']['annual_returns']
            require(isinstance(observed, list) and len(observed) == len(annual), 'Annual return count')
            for actual, expected in zip(observed, annual):
                require(set(actual) == set(expected) and actual['year'] == expected['year'] and
                        actual['partial'] == expected['partial'], 'Annual return identity')
                close(actual['return_'], expected['return_'], 'Annual return', atol=1e-12)
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
