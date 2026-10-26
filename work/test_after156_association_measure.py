import math
import random
import unittest

import after156_association_measure as m


def path(returns):
    values = [100.0]
    for r in returns:
        values.append(values[-1]*math.exp(r))
    return values


class AssociationTests(unittest.TestCase):
    def fixture(self):
        labels = 'BB'+'E'*19+'N'*50+'BB'+'E'*19+'N'*118
        market = [.003*math.sin(i)+.0002 for i in range(462)]
        stock = [2*r+.0004 for r in market[:252]]
        stock += [2*r+({'E': .002, 'N': .001, 'B': -.004}[l])
                  for r, l in zip(market[252:], labels)]
        return path(stock), path(market), labels

    def test_known_beta_and_distinct_three_hands(self):
        s, p, labels = self.fixture()
        result=m.measure(s, p, labels)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['beta'], 2., places=11)
        self.assertAlmostEqual(result['raw'][m.NAMES[2]], .021, places=11)
        self.assertAlmostEqual(result['raw'][m.NAMES[0]], 21*(38*.002+168*.001-4*.004)/210, places=11)
        self.assertEqual(result['raw'][m.NAMES[1]], 38/206)
        self.assertEqual(result['nuisance_beta_estimates_completed'], 1)
        self.assertEqual(result['new_predictive_model_fits'], 0)

    def test_constant_residual_has_zero_contrast(self):
        _, p, labels = self.fixture()
        mr=[m.log_return(a,b) for a,b in zip(p,p[1:])]
        s=path([1.5*r+.0007 for r in mr])
        z=m.measure(s,p,labels)
        self.assertAlmostEqual(z['raw'][m.NAMES[2]], 0., places=11)
        self.assertAlmostEqual(z['raw'][m.NAMES[0]], .0147, places=11)

    def test_scale_invariant(self):
        s,p,l=self.fixture()
        a,b=m.measure(s,p,l),m.measure([v*17 for v in s],[v*.03 for v in p],l)
        for name in m.NAMES:
            self.assertAlmostEqual(a['hands'][name],b['hands'][name],places=11)

    def test_bad_price_shared_unavailable_mask(self):
        s,p,l=self.fixture()
        for offset in (0,252,253,462):
            for invalid in (0.,-1.,float('nan'),float('inf')):
                a=s.copy();a[offset]=invalid
                z=m.measure(a,p,l)
                self.assertFalse(z['available'])
                self.assertTrue(all(math.isnan(v) for v in z['hands'].values()))
                self.assertEqual(z['nuisance_beta_estimates_completed'],0)

    def test_fixed_shapes_reject_future_or_shortened_history(self):
        s,p,l=self.fixture()
        for a,b,c in ((s+[200],p+[100],l),(s[1:],p[1:],l),(s,p,l[1:]),(s,p,'X'+l[1:])):
            with self.assertRaises(ValueError):m.measure(a,b,c)

    def test_zero_market_variance_counted_attempt(self):
        s,_,l=self.fixture()
        r=m.measure(s,[100.]*463,l)
        self.assertFalse(r['available'])
        self.assertEqual(r['nuisance_beta_estimates_attempted'],1)
        self.assertEqual(r['nuisance_beta_estimates_completed'],0)

    def test_masks_require_sufficient_days(self):
        s,p,l=self.fixture()
        for invalid in ('N'*210,'E'*131+'N'*79):
            with self.assertRaises(ValueError):m.measure(s,p,invalid)

    def test_price_types_are_not_silently_coerced(self):
        s,p,l=self.fixture()
        for bad in (True,'123',None):
            a=s.copy();a[0]=bad
            with self.assertRaises(ValueError):m.measure(a,p,l)

    def test_random_reference_regression(self):
        # Direct centered regression and arithmetic formula, not producer sums.
        rng=random.Random(156)
        _,_,l=self.fixture()
        for _ in range(60):
            x=[rng.gauss(.0003,.012) for i in range(462)]
            y=[rng.uniform(-2,2)*v+rng.gauss(.001,.007) for v in x]
            a,b=path(y),path(x)
            actual=m.measure(a,b,l)
            observed_x=[math.log(b[i+1]/b[i]) for i in range(462)]
            observed_y=[math.log(a[i+1]/a[i]) for i in range(462)]
            mx,my=sum(observed_x[:252])/252,sum(observed_y[:252])/252
            beta=sum((v-mx)*(w-my) for v,w in zip(observed_x[:252],observed_y[:252]))/sum((v-mx)**2 for v in observed_x[:252])
            residual=[w-beta*v for v,w in zip(observed_x[252:],observed_y[252:])]
            e=[v for v,k in zip(residual,l) if k=='E'];n=[v for v,k in zip(residual,l) if k=='N']
            contrast=21*(sum(e)/len(e)-sum(n)/len(n))
            self.assertAlmostEqual(actual['beta'],beta,places=11)
            self.assertAlmostEqual(actual['raw'][m.NAMES[2]],contrast,places=11)


if __name__=='__main__':unittest.main(verbosity=2)
