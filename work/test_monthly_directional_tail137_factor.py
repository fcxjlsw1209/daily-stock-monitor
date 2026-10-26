"""Pure synthetic state tests; no original market data or factor runs."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_directional_tail137_factor as f


def prices(returns):
    x=np.full((22,32),100.);x[-1]=100.*(1+np.asarray(returns));return x


def fixture():
    sessions=pd.bdate_range('2020-01-01',periods=60)
    stocks=['S'+str(i) for i in range(32)];etfs=['E'+str(i) for i in range(17)]
    px=pd.DataFrame(100.,index=sessions,columns=stocks+etfs)
    px.iloc[21,:32]=prices([.31]+[-.01]*31)[-1]
    meta=dict(dates=[str(sessions[21].date())],symbols=[stocks+etfs],etfs=etfs)
    return px,meta,sessions


class ScalarTests(unittest.TestCase):
    def test_symmetric_and_direction(self):
        self.assertAlmostEqual(f.tail_balance(prices([.1,-.1]*16))[0],.5)
        self.assertAlmostEqual(f.tail_balance(prices([.31]+[-.01]*31))[0],31/32)
        self.assertAlmostEqual(f.tail_balance(prices([-.31]+[.01]*31))[0],1/32)
    def test_common_return_shift_and_reflection(self):
        r=np.linspace(-.2,.3,32)**3
        q=f.tail_balance(prices(r))[0]
        self.assertAlmostEqual(f.tail_balance(prices(r+.1))[0],q,places=12)
        self.assertAlmostEqual(f.tail_balance(prices(-r))[0],1-q,places=12)
    def test_return_scale_and_per_stock_price_scale(self):
        r=np.linspace(-.2,.3,32)**3;x=prices(r);q=f.tail_balance(x)[0]
        self.assertAlmostEqual(f.tail_balance(prices(r*2))[0],q,places=12)
        self.assertAlmostEqual(f.tail_balance(x*np.arange(1,33))[0],q,places=12)
    def test_equal_returns_and_flat_are_unavailable(self):
        for r in ([0.]*32,[.2]*32):
            q,d=f.tail_balance(prices(r));self.assertTrue(np.isnan(q));self.assertEqual(d['reason'],'zero_dispersion')
            self.assertEqual(d['zero_count'],32);self.assertIsNone(d['q']);json.dumps(d,allow_nan=False)
    def test_nearly_equal_not_thresholded(self):
        x=np.ones((22,32));x[-1]=1.25;x[-1,0]=np.nextafter(1.25,2)
        q,d=f.tail_balance(x);self.assertTrue(d['available']);self.assertAlmostEqual(q,31/32)
    def test_interior_missing_inf_nonpositive_common_fallback(self):
        for bad in (np.nan,np.inf,0.,-1.):
            x=prices(np.linspace(-.2,.3,32));x[10,7]=bad
            q,d=f.tail_balance(x);self.assertTrue(np.isnan(q));self.assertEqual(d['reason'],'invalid_price')
    def test_derived_overflow_is_error(self):
        x=prices(np.zeros(32));x[0,0]=1e-308;x[-1,0]=1e308
        with self.assertRaises(FloatingPointError):f.tail_balance(x)
    def test_scaling_avoids_squared_overflow(self):
        x=np.ones((22,32));x[-1,0]=1e200
        q,d=f.tail_balance(x);self.assertAlmostEqual(q,31/32);self.assertTrue(np.isfinite(d['centered_scale']))
    def test_wrong_shape_bool_string_rejected(self):
        for x in (np.ones((21,32)),np.ones((22,31)),np.ones((22,32),bool),np.full((22,32),'100')):
            with self.assertRaises(ValueError):f.tail_balance(x)
    def test_permutation_and_input_nonmutation(self):
        x=prices(np.linspace(-.2,.3,32)**3);old=x.copy()
        self.assertAlmostEqual(f.tail_balance(x)[0],f.tail_balance(x[:,::-1])[0],places=14)
        np.testing.assert_array_equal(x,old)


class CalendarTests(unittest.TestCase):
    def test_full22_window_and_exact32_population(self):
        px,m,s=fixture();z=f.build_features(px,m,s);d=z['audit']['months'][0]
        self.assertEqual(len(d['stock_symbols']),32);self.assertEqual(d['window_dates'],[str(x.date())for x in s[:22]])
        self.assertEqual(z['q'][0],d['q']);self.assertTrue(z['audit']['no_eligibility_filter'])
        json.dumps(z['audit'],allow_nan=False)
    def test_etf_values_never_enter(self):
        px,m,s=fixture();before=f.build_features(px,m,s)
        px.loc[:,m['etfs']]=np.nan
        self.assertEqual(before['audit'],f.build_features(px,m,s)['audit'])
    def test_missing_official_row_not_compressed(self):
        px,m,s=fixture();px=px.drop(s[7]);z=f.build_features(px,m,s)
        self.assertEqual(z['audit']['months'][0]['reason'],'invalid_price');self.assertTrue(np.isnan(z['q'][0]))
    def test_missing_stock_column_not_substituted(self):
        px,m,s=fixture();z=f.build_features(px.drop(columns=['S3']),m,s)
        self.assertEqual(z['audit']['months'][0]['reason'],'missing_stock_columns')
    def test_insufficient_history(self):
        px,m,s=fixture();m['dates']=[str(s[20].date())];z=f.build_features(px,m,s)
        self.assertEqual(z['audit']['months'][0]['window_dates'],[])
        self.assertEqual(z['audit']['months'][0]['reason'],'insufficient_history')
    def test_future_poison_prefix_and_no_io(self):
        px,m,s=fixture();expected=f.build_features(px,m,s)
        px=px.astype(object);px.iloc[22:,:]='future'
        with patch('builtins.open',side_effect=AssertionError('I/O')):
            actual=f.build_features(px,m,s)
        self.assertEqual(actual['audit'],expected['audit']);np.testing.assert_array_equal(actual['q'],expected['q'])
    def test_duplicate_nonofficial_and_invalid_universe(self):
        px,m,s=fixture()
        with self.assertRaises(ValueError):f.build_features(px,m,s.insert(2,s[2]))
        bad=deepcopy(m);bad['symbols'][0][-1]='S0'
        with self.assertRaises(ValueError):f.build_features(px,bad,s)
        other=px.copy();other.loc[pd.Timestamp('2020-01-04')]=100.;other=other.sort_index()
        with self.assertRaises(ValueError):f.build_features(other,m,s)
    def test_eligibility_field_not_read(self):
        px,m,s=fixture();original=f.build_features(px,m,s);m['eligible']='not a state input'
        self.assertEqual(original['audit'],f.build_features(px,m,s)['audit'])

if __name__=='__main__':unittest.main()
