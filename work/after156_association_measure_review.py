"""Independent synthetic arithmetic review. Never opens market/source data."""
import argparse
import copy
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after156_association_measure_review.py'
OUT = 'work/after156_association_measure_review.json'
PINS = {
    'work/after156_association_measure.py': 'd7254cb3bf79cfd873afe30e1037b84fbe70708ea2db1de3eb8679b95dad62ad',
    'work/test_after156_association_measure.py': '7a2590b0b2da44e848d67964f713435abf300c9505521537a1e8713e624401d6',
    'work/after156_association_protocol.json': '9906d41f2aad9f5981030b4264540cdca0f40f8fb5c723e1060ddc32b00fa88e',
}
NAMES = ('whole_formation_beta_residual_momentum', 'filing_window_density',
         'filing_vs_nonfiling_beta_residual_contrast')
M = None


def path(changes, initial):
    result = [initial]
    for change in changes:
        result.append(result[-1] * math.exp(change))
    return result


def fixture(seed=1, beta=-0.75, labels=None):
    labels = labels or 'BB' + 'E'*19 + 'N'*50 + 'BB' + 'E'*19 + 'N'*118
    rng = random.Random(seed)
    x = [rng.uniform(-.008, .011) for _ in range(462)]
    y = [beta*r + .00083 for r in x[:252]]
    y += [8*r + {'E': .003, 'N': -.002, 'B': .006}[label]
          for r,label in zip(x[252:],labels)]
    return path(y, 57.), path(x, 123.), labels


def decimal_oracle(stock, market, labels):
    """70-digit raw-moment OLS on the supplied float prices, not intended returns."""
    with localcontext() as context:
        context.prec = 70
        def returns(values):
            exact = [Decimal.from_float(float(value)) for value in values]
            return [(b/a).ln() for a,b in zip(exact,exact[1:])]
        x,y = returns(market),returns(stock)
        n = Decimal(252)
        sx,sy = sum(x[:252]),sum(y[:252])
        beta = (n*sum(a*b for a,b in zip(x[:252],y[:252]))-sx*sy)/(n*sum(a*a for a in x[:252])-sx*sx)
        z = [b-beta*a for a,b in zip(x[252:],y[252:])]
        e = [value for value,label in zip(z,labels) if label=='E']
        ordinary = [value for value,label in zip(z,labels) if label=='N']
        whole = Decimal(21)*sum(z)/Decimal(210)
        contrast = Decimal(21)*(sum(e)/len(e)-sum(ordinary)/len(ordinary))
        density = Decimal(len(e))/Decimal(len(e)+len(ordinary))
        def h(value):
            value /= Decimal('.05')
            return value/(1+abs(value))
        return float(beta), [float(whole),float(density),float(contrast)], [float(h(whole)),float(density),float(h(contrast))]


class IndependentTests(unittest.TestCase):
    def close(self,a,b,tol=3e-11):
        self.assertTrue(math.isclose(a,b,rel_tol=tol,abs_tol=tol), (a,b))

    def assert_common_missing(self,result,attempts=0):
        self.assertFalse(result['available'])
        self.assertEqual(tuple(result['hands']),NAMES)
        self.assertTrue(all(math.isnan(value) for value in result['hands'].values()))
        self.assertTrue(all(value is None for value in result['raw'].values()))
        self.assertIsNone(result['beta'])
        self.assertEqual(result['nuisance_beta_estimates_attempted'],attempts)
        self.assertEqual(result['nuisance_beta_estimates_completed'],0)
        self.assertEqual(result['new_predictive_model_fits'],0)

    def test_01_decimal_raw_moment_oracle(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                s,p,l=fixture(seed,beta=-2+seed*.47)
                actual=M.measure(s,p,l)
                beta,raw,hands=decimal_oracle(s,p,l)
                self.close(actual['beta'],beta)
                for name,a,b in zip(NAMES,raw,hands):
                    self.close(actual['raw'][name],a)
                    self.close(actual['hands'][name],b)

    def test_02_formation_does_not_fit_beta(self):
        s,p,l=fixture(beta=-.75)
        a=M.measure(s,p,l)
        self.close(a['beta'],-.75)
        # Alter every formation price while keeping the full 253-price prefix.
        for k in range(253,463):
            s[k]*=math.exp(.007*(k-252))
            p[k]*=math.exp(-.003*(k-252))
        b=M.measure(s,p,l)
        self.assertEqual(a['beta'],b['beta'])
        self.assertNotEqual(a['raw'][NAMES[0]],b['raw'][NAMES[0]])

    def test_03_last_beta_boundary_uses_index252(self):
        s,p,l=fixture()
        before=M.measure(s,p,l)['beta']
        s[252]*=1.13
        after=M.measure(s,p,l)
        expected=decimal_oracle(s,p,l)
        self.assertGreater(abs(before-after['beta']),.001)
        self.close(after['beta'],expected[0])
        self.close(after['raw'][NAMES[2]],expected[1][2])

    def test_04_first_formation_price_does_not_change_beta(self):
        s,p,l=fixture()
        before=M.measure(s,p,l)
        s[253]*=.91
        after=M.measure(s,p,l)
        self.assertEqual(before['beta'],after['beta'])
        self.close(after['raw'][NAMES[2]],decimal_oracle(s,p,l)[1][2])

    def test_05_last_required_price_is_used(self):
        s,p,l=fixture()
        before=M.measure(s,p,l)
        s[-1]*=math.exp(.009)
        after=M.measure(s,p,l)
        self.assertEqual(before['beta'],after['beta'])
        self.close(after['raw'][NAMES[0]]-before['raw'][NAMES[0]],21*.009/210)
        self.close(after['raw'][NAMES[2]]-before['raw'][NAMES[2]],-21*.009/l.count('N'))

    def test_06_no_intercept_subtracted_from_formation(self):
        _,p,l=fixture()
        x=[math.log(b/a) for a,b in zip(p,p[1:])]
        s=path([1.23*r+.0007 for r in x],77.)
        result=M.measure(s,p,l)
        self.close(result['beta'],1.23)
        self.close(result['raw'][NAMES[0]],.0147)
        self.close(result['raw'][NAMES[2]],0.)

    def test_07_blackout_affects_whole_but_not_contrast(self):
        _,p,l=fixture()
        x=[math.log(b/a) for a,b in zip(p,p[1:])]
        y=[1.1*r+.0003 for r in x]
        before=M.measure(path(y,39.),p,l)
        for i,label in enumerate(l):
            if label=='B': y[252+i]+=.019
        after=M.measure(path(y,39.),p,l)
        self.close(before['raw'][NAMES[2]],after['raw'][NAMES[2]])
        self.assertEqual(before['hands'][NAMES[1]],after['hands'][NAMES[1]])
        self.close(after['raw'][NAMES[0]]-before['raw'][NAMES[0]],21*.019*l.count('B')/210)

    def test_08_label_swap_reverses_contrast_only(self):
        l='E'*80+'N'*80+'B'*50
        s,p,_=fixture(labels=l)
        a=M.measure(s,p,l)
        b=M.measure(s,p,l.translate(str.maketrans({'E':'N','N':'E'})))
        self.assertEqual(a['beta'],b['beta'])
        self.assertEqual(a['hands'][NAMES[0]],b['hands'][NAMES[0]])
        self.assertEqual(a['hands'][NAMES[1]],b['hands'][NAMES[1]])
        self.close(a['hands'][NAMES[2]],-b['hands'][NAMES[2]])

    def test_09_independent_common_scale_invariance(self):
        s,p,l=fixture()
        a=M.measure(s,p,l)
        for scale_s,scale_p in ((1e-200,1e180),(1e180,1e-200),(19.,.031)):
            b=M.measure([v*scale_s for v in s],[v*scale_p for v in p],l)
            self.close(a['beta'],b['beta'])
            for name in NAMES:
                self.close(a['raw'][name],b['raw'][name])
                self.close(a['hands'][name],b['hands'][name])

    def test_10_all_required_price_positions_share_missing_mask(self):
        s,p,l=fixture()
        for axis in (0,1):
            for position in range(463):
                values=[s.copy(),p.copy()]
                values[axis][position]=float('nan')
                self.assert_common_missing(M.measure(*values,l))
        for value in (0.,-1.,float('inf'),float('-inf')):
            values=s.copy();values[462]=value
            self.assert_common_missing(M.measure(values,p,l))

    def test_11_nuisance_counters_success_and_degeneracy(self):
        s,p,l=fixture()
        a=M.measure(s,p,l)
        self.assertEqual((a['nuisance_beta_estimates_attempted'],a['nuisance_beta_estimates_completed'],a['new_predictive_model_fits']),(1,1,0))
        self.assert_common_missing(M.measure(s,[99.]*463,l),attempts=1)

    def test_12_fixed_shape_and_label_contract(self):
        s,p,l=fixture()
        for values in ((s+[1.],p+[1.],l),(s[1:],p[1:],l),(s,p,l[:-1]),(s,p,'N'*210),(s,p,'E'*131+'N'*79),(s,p,'X'+l[1:])):
            with self.assertRaises(ValueError):M.measure(*values)
        for value in (True,'77',None,1+2j):
            bad=s.copy();bad[253]=value
            with self.assertRaises(ValueError):M.measure(bad,p,l)

    def test_13_input_objects_are_not_mutated(self):
        s,p,l=fixture()
        values=[s,p,list(l)]; saved=copy.deepcopy(values)
        M.measure(*values)
        self.assertEqual(values,saved)

    def test_14_extreme_finite_price_ratio_fallback(self):
        self.close(M.log_return(1e-300,1e300),600*math.log(10),1e-13)
        self.close(M.log_return(1e300,1e-300),-600*math.log(10),1e-13)
        s,p,l=fixture();s[0]=1e-300;s[1]=1e300
        result=M.measure(s,p,l)
        expected=decimal_oracle(s,p,l)
        self.close(result['beta'],expected[0],1e-9)
        self.assertTrue(all(math.isfinite(v) and abs(v)<=1 for v in result['hands'].values()))

    def test_15_full_bundle_proof_is_external_not_fabricated(self):
        s,p,_=fixture()
        # No event identities are supplied here. Counts alone can be processed,
        # but cannot prove two admitted complete event bundles.
        result=M.measure(s,p,'E'*38+'N'*172)
        self.assertTrue(result['available'])
        self.assertNotIn('source_gate_passed',result)
        self.assertNotIn('full19_bundles_verified',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    if args.write and args.test_only: raise ValueError('Choose one output mode')
    hashes={}
    for name,expected in PINS.items():
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if hashes[name]!=expected: raise ValueError('Reviewed input changed: '+name)
    global M
    M=importlib.import_module('after156_association_measure')
    if M.NAMES != NAMES: raise ValueError('Arm names changed')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    for name in (SELF,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name,value in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:
            raise ValueError('Input changed during review: '+name)
    report=dict(schema_version=1,stage='INDEPENDENT_SYNTHETIC_ASSOCIATION_MEASURE_REVIEW',
        status='PURE_ARITHMETIC_PASSED_SOURCE_AND_FINAL_INTERSECTION_NOT_ADMITTED',
        created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        independent_validation=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            decimal_precision=70,decimal_random_fixtures=12,all_required_positions_missing_cases=926,
            tests=unittest.defaultTestLoader.getTestCaseNames(IndependentTests)),
        findings=[],input_hashes=hashes,
        verified=['Centered preformation slope uses only returns0..251 from prices0..252; formation starts at return252 and ends at return461.',
            'Independent70-digit raw-moment OLS and float-price log oracle reproduce beta, whole-window raw signal, E/N contrast, density and bounded hands.',
            'Formation price changes leave beta fixed; shared boundary price252 participates in the last beta and first formation return; last required price462 is included.',
            'No intercept is subtracted from formation residuals. Blackouts affect whole-window momentum, but not E/N means; density excludes blackouts.',
            'Stock and market may be rescaled independently without changing the three hands within numerical tolerance. Inputs are not mutated.',
            'Invalid required values yield all-three missing, no completed estimate and no predictive fit. Zero preformation variance counts one attempted/zero completed estimate. A valid cell counts one attempted/one completed estimate and zero predictive fits.'],
        limits=['This API has no dates, issuer identities, source receipts or event IDs. Exactly463 prices and210 labels cannot prove calendar alignment, cutoff compliance, source eligibility or two full original event bundles.',
            'The actual caller must enforce the independently approved source/issuer/exception/warmup intersection and full19 bundle rule before using this result. No strict-clock Admission is asserted by this function.',
            'Only synthetic prices were generated or loaded. The original author test file was hash-bound and read, but this review record counts only the15 independently executed tests.',
            'A passing pure mathematical review is not permission for real features, returns, registration or strategy promotion.'],
        activity=dict(real_price_files_opened=0,real_factor_calculations=0,real_returns=0,network_requests=0,registrations=0,frozen_files_modified=0),
        admission=dict(actual_source=False,actual_features=False,strategy=False))
    if args.write:
        with (ROOT/OUT).open('x') as handle:
            json.dump(report,handle,sort_keys=True,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(passed=True,independent_tests=result.testsRun,output=OUT if args.write else None,
        output_sha256=hashlib.sha256((ROOT/OUT).read_bytes()).hexdigest() if args.write else None)))


if __name__=='__main__':main()
