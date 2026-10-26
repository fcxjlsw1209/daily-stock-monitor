"""Synthetic-only identification checks for a proposed local-distance metric.

No research/model imports, loaders, market inputs or portfolio evaluation.
This is a mathematical diagnostic, not a registered strategy implementation.
"""
from fractions import Fraction
from numbers import Real
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

WIDTH=10
BANDWIDTH=.5
OUT=Path(__file__).with_suffix('.json')
DEFINITION_REVIEW_SHA='f89cec265860a9bf18015b08b6a97318b4826b08332d23c9fa08886e567bfd74'


def require(ok,message):
    if not ok:raise ValueError(message)


def real_array(value,name):
    a=np.asarray(value)
    require(a.dtype.kind in 'fiuO',name+' real numeric nonboolean')
    if a.dtype.kind=='O':
        require(all(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) for x in a.flat),name+' real numeric nonboolean')
    a=np.asarray(a,dtype=float)
    require(np.isfinite(a).all(),name+' finite')
    return a.copy()


def coefficients(scale):
    s=real_array(scale,'scale')
    require(s.shape==(WIDTH,),'Ten scales')
    # Original features lie in [-.5,.5], hence weighted SD<=.5; original
    # <=1e-12 floor produces exactly1. This is the source-model domain,
    # not an assertion that every positive binary64 scale is supported.
    require(np.all(((s>1e-12)&(s<=.5))|(s==1.)),'Original legal pooled scale/floor')
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        inverse=1./(s*s)
        a=inverse/inverse.sum()
    require(np.isfinite(a).all() and (a>0.).all(),'Positive finite coefficients')
    return a


def geometry(X,query,scale,ages):
    X=real_array(X,'X');query=real_array(query,'query');ages=real_array(ages,'ages')
    require(X.ndim==2 and X.shape[1]==WIDTH and len(X)>0 and query.shape==(WIDTH,) and ages.shape==(len(X),),'Synthetic source axes')
    require((np.abs(X)<=.5).all() and (np.abs(query)<=.5).all(),'Original rank bounds')
    require(((ages>=0.)&(ages<60.)&(ages==np.floor(ages))).all(),'Past-five-year calendar-month ages')
    a=coefficients(scale)
    with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
        squared=(X-query)**2
        raw_distance=np.mean(squared,axis=1)
        diagonal_distance=np.sum(squared*a,axis=1)
        recency=np.exp2(-ages/12.)
        original=recency*np.exp(-raw_distance/(2.*BANDWIDTH**2))
        unscaled=recency*np.exp(-diagonal_distance/(2.*BANDWIDTH**2))
        original_mass=float(original.sum())
        weight=unscaled*(original_mass/float(unscaled.sum()))
    for name,v in (('distances',diagonal_distance),('original',original),('candidate',weight)):
        require(np.isfinite(v).all(),name+' finite')
    require((original>0.).all() and (weight>0.).all(),'No support removed')
    return dict(coefficients=a,raw_distance=raw_distance,diagonal_distance=diagonal_distance,
        original_weights=original,unscaled_weights=unscaled,matched_weights=weight,
        original_mass=original_mass,matched_mass=float(weight.sum()))


def fraction(value):return Fraction.from_float(float(value))


def exact_coefficients(scale):
    inv=[1/(fraction(s)**2) for s in scale];total=sum(inv)
    return [v/total for v in inv]


def past_scales(X,signals,ends,cutoff='2020-12-31',start='2015-12-31'):
    """Past-only synthetic sample selection; never cast future numeric bodies."""
    signals=np.asarray(signals,dtype='datetime64[D]');ends=np.asarray(ends,dtype='datetime64[D]')
    cutoff=np.datetime64(cutoff,'D');start=np.datetime64(start,'D')
    require(signals.shape==ends.shape and not np.isnat(signals).any() and not np.isnat(ends).any(),'Date metadata')
    keep=(signals>start)&(ends<=cutoff)
    x=real_array(np.asarray(X)[keep],'consumed past X')
    require(x.shape==(int(keep.sum()),WIDTH) and len(x)>0 and (np.abs(x)<=.5).all(),'Past rank inputs')
    months=signals[keep].astype('datetime64[M]')
    _,inv,num=np.unique(months,return_inverse=True,return_counts=True)
    age=int(cutoff.astype('datetime64[M]').astype(int))-months.astype(int)
    w=np.exp2(-age/12.)/num[inv];w*=len(w)/w.sum()
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale=np.where(scale>1e-12,scale,1.)
    return scale,np.flatnonzero(keep)


def main():
    require(not OUT.exists(),'Write-once synthetic report already exists')
    rng=np.random.default_rng(255000)
    X=rng.uniform(-.5,.5,(24,WIDTH));query=rng.uniform(-.5,.5,WIDTH)
    ages=np.arange(24,dtype=float)
    checks=[];observations={}
    def passed(name,**evidence):checks.append(dict(name=name,passed=True,**evidence));print('PASS '+name,flush=True)

    # Algebra in exact rationals, independent of binary64 summation sequence.
    for common in (.125,.5,1.):
        a=exact_coefficients(np.full(WIDTH,common))
        assert a==[Fraction(1,WIDTH)]*WIDTH
        for row in X[:4]:
            squared=[(fraction(x)-fraction(q))**2 for x,q in zip(row,query)]
            assert sum(ai*d for ai,d in zip(a,squared))==sum(squared)/WIDTH
    passed('equal_scale_exact_rational_distance_identity',scale_examples=3,rows_per_example=4)

    scale=np.array([.1,.2,.3,.4,.5,.15,.25,.35,.45,1.])
    a=exact_coefficients(scale)
    assert sum(a)==1 and all(v>0 for v in a)
    g=geometry(X,query,scale,ages)
    assert abs(float(g['coefficients'].sum())-1.)<=8*np.finfo(float).eps
    assert (g['diagonal_distance']>=0).all() and (g['diagonal_distance']<=1.+8*np.finfo(float).eps).all()
    passed('trace_normalization_and_bounded_distance',exact_trace=10,coefficient_sum_float=float(g['coefficients'].sum()))

    raw=[fraction(x) for x in g['original_weights']];u=[fraction(x) for x in g['unscaled_weights']]
    M=sum(raw);v=[x*M/sum(u) for x in u]
    assert sum(v)==M and all(x>0 for x in v)
    passed('mass_matching_exact_rational_identity',statement='For positive finite u, sum(u*M/sum(u))=M in exact arithmetic.')

    mass_error=abs(g['matched_mass']-g['original_mass'])
    assert mass_error<=16*np.finfo(float).eps*g['original_mass']
    equal=geometry(X,query,np.full(WIDTH,.25),ages)
    equal_d_error=float(np.max(np.abs(equal['diagonal_distance']-equal['raw_distance'])))
    equal_w_error=float(np.max(np.abs(equal['matched_weights']-equal['original_weights'])))
    assert equal_d_error<=16*np.finfo(float).eps
    assert equal_w_error<=16*np.finfo(float).eps
    observations.update(equal_scale_distance_max_abs_error=equal_d_error,
        equal_scale_weights_max_abs_error=equal_w_error,
        equal_scale_distance_bytes_equal=equal['diagonal_distance'].tobytes()==equal['raw_distance'].tobytes(),
        equal_scale_weight_bytes_equal=equal['matched_weights'].tobytes()==equal['original_weights'].tobytes(),
        mass_match_absolute_error=mass_error,mass_match_relative_error=mass_error/g['original_mass'])
    passed('binary64_roundoff_is_measured_not_claimed_exact',**observations)

    zero=geometry(np.tile(query,(24,1)),query,scale,ages)
    assert np.array_equal(zero['diagonal_distance'],np.zeros(24))
    assert zero['original_weights'].tobytes()==zero['matched_weights'].tobytes()
    passed('zero_distance_exact_weights',zero_distances=24)

    residual=[Fraction(1,5),Fraction(-1,10),Fraction(3,20)]
    weights=[Fraction(1,1),Fraction(1,2),Fraction(1,4)]
    intercept=sum(w*y for w,y in zip(weights,residual))/(sum(weights)+12)
    # When every history row equals the query, all ten slope columns are0;
    # the penalized normal equations are diagonal, hence slopes=0 exactly.
    assert intercept*(sum(weights)+12)==sum(w*y for w,y in zip(weights,residual))
    passed('constant_design_local_linear_reduces_to_constant',intercept_exact={'numerator':str(intercept.numerator),'denominator':str(intercept.denominator)},slopes_exact_zero=10,synthetic_regression_fits=0)

    extreme=np.array([np.nextafter(1e-12,np.inf)]+[.5]*8+[1.])
    extremeX=np.array([[-.5]*WIDTH,[.5]*WIDTH,[0.]*WIDTH]);extremeq=np.full(WIDTH,.5)
    e=geometry(extremeX,extremeq,extreme,np.array([59.,0.,23.]))
    assert (e['coefficients']>0.).all() and (e['matched_weights']>0.).all()
    assert np.isfinite(e['matched_weights']).all() and np.isfinite(e['diagonal_distance']).all()
    assert abs(e['matched_mass']-e['original_mass'])<=16*np.finfo(float).eps*e['original_mass']
    passed('extreme_original_legal_scales_preserve_support',scale_min=float(extreme.min()),scale_max=float(extreme.max()),minimum_coefficient=float(e['coefficients'].min()),minimum_sample_weight=float(e['matched_weights'].min()),sample_count=3)

    # A concrete mathematical identification witness, not a performance search.
    witness=np.zeros((2,WIDTH));witness[0,0]=.2;witness[1,1]=.3
    w=geometry(witness,np.zeros(WIDTH),np.array([.1]+[.5]*9),np.zeros(2))
    assert w['raw_distance'][0]<w['raw_distance'][1]
    assert w['diagonal_distance'][0]>w['diagonal_distance'][1]
    assert w['original_weights'][0]>w['original_weights'][1]
    assert w['matched_weights'][0]<w['matched_weights'][1]
    passed('unequal_scales_can_reorder_analogues',same_support=True,no_response_or_return_used=True)

    signals=np.array(['2018-%02d-28'%m for m in range(1,13)]+['2019-%02d-28'%m for m in range(1,13)]+['2021-01-28','2022-01-28'])
    months=np.asarray(signals,dtype='datetime64[M]');ends=(months+1).astype('datetime64[D]')
    synthetic=rng.uniform(-.5,.5,(26,WIDTH));base,ids=past_scales(synthetic,signals,ends)
    poisoned=synthetic.astype(object);poisoned[-2:]=[['future-invalid']*WIDTH,[np.inf]*WIDTH]
    got,ids2=past_scales(poisoned,signals,ends)
    assert base.tobytes()==got.tobytes() and np.array_equal(ids,ids2) and len(ids)==24
    assert np.array_equal(ids,np.arange(24))
    passed('future_numeric_poison_does_not_affect_past_scaler',consumed_rows=24,excluded_future_rows=2,scales_bytes_equal=True)

    rejected=0
    for bad in (0.,-1.,np.nan,np.inf,1e-12,1e-300,1e300):
        s=scale.copy();s[0]=bad
        try:coefficients(s)
        except ValueError:rejected+=1
        else:raise AssertionError('Malformed/out-of-source-domain scale accepted')
    for values in (np.ones(WIDTH,bool),):
        try:coefficients(values)
        except ValueError:rejected+=1
        else:raise AssertionError('Boolean scale accepted')
    passed('malformed_and_outside_original_scale_domain_rejected',cases=rejected)

    report=dict(schema='after255_distance_synthetic_v1',passed=True,unresolved_blockers=[],
        scope='Pure deterministic synthetic mathematics only; no model/market/metadata loaders, fitting, orders or returns.',
        definition_reference=dict(path='work/after255_local_distance_review.json',declared_sha256=DEFINITION_REVIEW_SHA,contents_not_reread=True),
        formula=dict(width=WIDTH,bandwidth=BANDWIDTH,raw_distance='mean((X-query)^2)',coefficients='(1/s_j^2)/sum_k(1/s_k^2)',diagonal_distance='sum_j a_j*(X_j-query_j)^2',weights='u=original_recency*exp(-distance/(2*.5^2)); w=u*sum(original_w)/sum(u)',scale_domain='original centered-rank weighted SD in (1e-12,.5], or original floor value1'),
        checks=checks,check_count=len(checks),binary64_observations=observations,
        previous_attempt=dict(command='./run_local.sh work/after255_distance_synthetic.py',exit_code=1,terminal_chunk='1b4335',passed_checks_before_failure=8,log_path='/private/tmp/after255_distance_synthetic_attempt1.log',log_sha256='e877edbd7cce076c2c39ec614249510a61ba5dd848860fc44e754bd468ec5abd',failure='Already-filtered valid numeric object array rejected by an overly narrow dtype check.',repair='Validate each consumed object element as real nonboolean before float conversion. Future rows remain excluded before validation. No metric, scale domain or parameters changed.'),
        source_construction_review=dict(
            scope='Static Python-source inspection only; no market, training, monthly metadata or cached feature values read.',
            plain10_in_order=['raw','recent_risk','vol','sector','cmf21','cmf63','signed_volume','volume_ratio','close_location21','compression'],
            assembly='monthly_session_type_research/study.py:89-94 stacks train/event_all raw,recent_risk,vol,sector plus first six aux_all columns; copied through core_expansion,earnings_factor_research,target_definition70.',
            construction={
                'raw':'rank(P[t-42]/P[t-252]-1)',
                'recent_risk':'rank(raw momentum divided by current trailing63 simple-return sample SD)',
                'vol':'rank(current trailing126 simple-return sample SD)',
                'sector':'mean of separately ranked own-sector relative skip42 momentum at nominal126 and252 endpoints; averaged ranks, not reranked after averaging',
                'cmf21':'rank(rolling21 sum(close-location*volume)/rolling21 sum(volume))',
                'cmf63':'same formula rolling63',
                'signed_volume':'rank(rolling63 sum(sign(daily simple return)*volume)/rolling63 sum(volume))',
                'volume_ratio':'rank(rolling21 mean(volume)/rolling126 mean(volume))',
                'close_location21':'rank(rolling21 mean((2*close-high-low)/(high-low)))',
                'compression':'rank(-rolling21 simple-return sample SD/rolling126 simple-return sample SD)'},
            rank_rule='architecture_restart/study.py:74-78: replace +/-Inf with NaN, pandas average rank(pct=True), fill missing with .5, subtract .5; all normalization has no E restriction before ranks.',
            equal_scale_theorem='If each column has the same complete rank multiset within each month and all rows in that multiset receive the same monthly weight, the pooled weighted marginal means and variances agree mathematically. This need not be byte-exact under different row-order summation.',
            why_not_guaranteed=['sector averages two rank columns and is not a single uniform rank grid; sector mappings and ETF neutral/missing values differ from other columns',
                'Feature-specific ties and missing-to-zero values alter marginal rank multisets',
                'Original common_good, finite features and all four finite targets select annual training rows after full-universe ranking; retained within-month ranks need not retain equal marginal distributions'],
            empirical_scales_or_correlations_computed=False,
            correlated_feature_mechanism='The raw/recent-risk pair shares the momentum numerator; cmf21/cmf63 share the close-location-times-volume construction and overlap in time. Correlated or duplicated coordinates can count the same direction repeatedly in Euclidean distance even with equal marginal scales. Diagonal rescaling cannot remove cross-coordinate correlation. This is a mathematical source-level possibility, not evidence that actual correlations are large.',
            future_separate_scope='A correlation-aware distance would require a separate prospective definition for past-only sample, estimator/regularization and mass normalization. No covariance, inverse, correlations, alternative parameters or actual model fits are implemented or tested here; missing old redundancy/kernel definitions leave exact historical duplication UNKNOWN.'),
        risks=[
            'Equal scales and mass matching are algebraic identities. Different binary64 operation order can differ by ULPs; this diagnostic does not prove byte-exact original predictions, fitted coefficients, ranks, scores or orders.',
            'A future registered implementation must choose explicit equal-scale/zero-change semantics, canonical arithmetic and validation tolerances prospectively. This diagnostic does not silently repair the final weight or alter an epsilon.',
            'Matching total own-history weight mass does not match effective sample size, relative calendar-year mass, weighted design covariance or every coefficient shrinkage direction; those differences remain part of the mechanism.',
            'Marginal scales of centered-rank features can be nearly equal, so mathematical distinguishability does not imply economically meaningful differences.',
            'The legal scale domain is inherited from bounded rank features and the original1e-12 floor. Arbitrarily extreme positive binary64 values are not certified as valid source-model scales.',
            'Historical kernel/redundancy/feature implementations are missing; global duplicate status remains UNKNOWN. This is not author reconstruction, source admission, a strategy trial or evidence of improved performance.'
        ],runtime=dict(python=sys.version,numpy=np.__version__),
        input_sha256={
            'work/after255_distance_synthetic.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'work/after255_local_distance_review.json':DEFINITION_REVIEW_SHA,
            'outputs/monthly_session_type_research/study.py':'38f5b7fb758bd53b9619e00670137dace66c83c7a413128d3b48ece9adc938f7',
            'outputs/monthly_architecture_restart/study.py':'d77e9bee28f82a9c6a20797f2d35c68f92e8a55307854d2ad8d59d080eafac25',
            'outputs/monthly_architecture_restart/residual_model.py':'ea294b9a0a1e9493799f1014dcec464effa93dce70f44fb5de53a604cd832886',
            'outputs/monthly_core_expansion/study.py':'82b535d2afb0e1f615ae2f0ce00a84987fbaec1150921ef779fdd9d0aca199f9',
            'outputs/monthly_earnings_factor_research/study.py':'a24a26a1086501ce7525174478a11a5d8e400b5fd79424cb9399a344f064fcc2',
            'outputs/monthly_target_definition70/study.py':'7a88b1489c3add0fb0c0942fd379dda3d25a4e19fa162fdb7d09ffe2a8da06c4',
            'outputs/monthly_local_linear70/model.py':'6d9075e5ce08edd0587d2b2ea0d98635620141070f1af08771b6a69435467a8a'},
        actual_scope=dict(actual_data_files_read=False,npz_decoded=False,monthly_metadata_read=False,prices_read=False,round255_results_read=False,actual_features_computed=False,actual_predictive_fits=0,actual_orders_computed=False,actual_returns_computed=False,synthetic_regression_fits=0,new_strategy_registered=False,frozen_files_modified=False,observers_modified=False))
    with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('SAVED '+str(OUT),flush=True)
    print('SHA256 '+hashlib.sha256(OUT.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':main()
