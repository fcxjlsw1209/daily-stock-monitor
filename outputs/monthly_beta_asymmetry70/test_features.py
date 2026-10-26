"""Synthetic grouped-OLS, fixed formation clock and ranking verification."""
from pathlib import Path
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('beta_asymmetry70_synthetic', Path(__file__).with_name('features.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
ETFS = ['SPY']+['ETF%02d' % i for i in range(16)]
SYMBOLS = ['STOCK%02d' % i for i in range(32)]+ETFS


def fixture(rows=345):
    rng = np.random.default_rng(37419)
    index = pd.bdate_range('2000-01-03', periods=rows)
    market = rng.normal(.0001,.008,rows)
    returns = np.where(market[:,None]>0, market[:,None]*np.linspace(.5,2.5,49),
                       market[:,None]*np.linspace(1.5,.5,49))
    returns += rng.normal(0,.003,(rows,49)); returns[:,32] = market
    panel = {'adj':pd.DataFrame(100.*np.exp(np.cumsum(returns,axis=0)),index=index,columns=SYMBOLS)}
    return panel,[(index[294],SYMBOLS.copy()),(index[320],SYMBOLS.copy())]


def returns_fixture():
    m=np.r_[np.linspace(.001,.02,100),-np.linspace(.001,.03,100),np.zeros(52)]
    s=np.where(m>0,.013+2.4*m,-.021+.8*m)
    s[m==0]=100.
    return s,m


class Poison:
    def __float__(self):raise AssertionError('unused price converted')
    def __array__(self,*a,**kw):raise AssertionError('unused price inspected')


class FeatureTests(unittest.TestCase):
    def test_independent_grouped_ols_with_intercepts_and_full49_rank(self):
        panel,events=fixture();got,audit=f.prepare(panel,events,ETFS)
        for i,(day,symbols) in enumerate(events):
            t=panel['adj'].index.get_loc(day)
            market=np.diff(np.log(panel['adj']['SPY'].iloc[t-294:t-41].to_numpy()))
            for j,symbol in enumerate(symbols):
                own=np.diff(np.log(panel['adj'][symbol].iloc[t-294:t-41].to_numpy()))
                betas=[]
                for mask in (market>0,market<0):
                    design=np.column_stack([np.ones(mask.sum()),market[mask]])
                    beta=np.linalg.lstsq(design,own[mask],rcond=None)[0][1];betas.append(beta)
                self.assertAlmostEqual(got['raw_beta_gap'][i,j],betas[0]-betas[1],places=12)
                self.assertAlmostEqual(audit['months'][i]['beta_up'][j],betas[0],places=12)
                self.assertAlmostEqual(audit['months'][i]['beta_down'][j],betas[1],places=12)
            np.testing.assert_array_equal(got['rank_beta_gap'][i],pd.Series(got['raw_beta_gap'][i]).rank(pct=True)-.5)
        self.assertTrue(got['available'].all()); self.assertEqual(audit['available_cells'],98)
        self.assertEqual(set(got),{'raw_beta_gap','rank_beta_gap','available'})
        json.dumps(audit,allow_nan=False)

    def test_separate_group_intercepts_and_zero_returns_excluded(self):
        own,market=returns_fixture();gap,up,down,counts,status=f._beta_gap(own,market)
        self.assertEqual(counts,(100,100,52));self.assertEqual(status,'available')
        self.assertAlmostEqual(up,2.4,places=14);self.assertAlmostEqual(down,.8,places=14)
        self.assertAlmostEqual(gap,1.6,places=14)
        wrong=[]
        for mask in (market>0,market<0):wrong.append(np.dot(market[mask],own[mask])/np.dot(market[mask],market[mask]))
        self.assertGreater(abs((wrong[0]-wrong[1])-gap),.01)
        shifted=own.copy();shifted[market>0]+=50.;shifted[market<0]-=70.;shifted[market==0]=-1e100
        self.assertAlmostEqual(f._beta_gap(shifted,market)[0],gap,places=10)

    def test_exact40_group_boundary_and_zero_exclusion_from_count(self):
        for nu,nd,valid in ((40,40,True),(39,40,False),(40,39,False),(0,100,False)):
            m=np.r_[np.linspace(.001,.02,nu),-np.linspace(.002,.03,nd),np.zeros(252-nu-nd)]
            y=2*m;result=f._beta_gap(y,m)
            self.assertEqual(result[3],(nu,nd,252-nu-nd))
            self.assertEqual(np.isfinite(result[0]),valid)
            if valid:self.assertAlmostEqual(result[0],0.,places=14)
            else:self.assertEqual(result[4],'insufficient_group')

    def test_zero_market_group_variance_unavailable_and_self_constant_valid_zero(self):
        own,market=returns_fixture()
        for mask in (market>0,market<0):
            bad=market.copy();bad[mask]=.01 if mask[0] else -.01
            result=f._beta_gap(own,bad);self.assertTrue(np.isnan(result[0]));self.assertEqual(result[4],'zero_market_variance')
        for y in (market,np.zeros(252),np.full(252,.0123)):
            result=f._beta_gap(y,market);self.assertEqual(result[0],0.);self.assertEqual(result[4],'available')
        panel,events=fixture();panel['adj'].iloc[:,0]=100.
        got,audit=f.prepare(panel,events,ETFS)
        np.testing.assert_array_equal(got['raw_beta_gap'][:,[0,32]],np.zeros((2,2)))
        self.assertTrue(got['available'][:,[0,32]].all())

    def test_valid_zero_ties_are_ranked_and_missing_does_not_enter_population(self):
        panel,events=fixture();panel['adj'].loc[:,:]=100.
        r=np.tile([.01,.02,-.01,-.02],86)[:len(panel['adj'])-1]
        panel['adj']['SPY']=100.*np.exp(np.r_[0,np.cumsum(r)])
        panel['adj'].iloc[30,0]=np.nan
        got,audit=f.prepare(panel,events[:1],ETFS)
        self.assertFalse(got['available'][0,0]);self.assertTrue(got['available'][0,1:].all())
        np.testing.assert_array_equal(got['raw_beta_gap'][0,1:],np.zeros(48))
        np.testing.assert_array_equal(got['rank_beta_gap'][0,1:],np.full(48,24.5/48-.5))
        self.assertEqual(audit['invalid_security_path_cells'],1)

    def test_price_units_and_stable_log_returns_extreme_scales(self):
        panel,events=fixture();before,_=f.prepare(panel,events,ETFS)
        binary=panel['adj']*np.exp2(np.arange(49)-24)
        exact,_=f.prepare({'adj':binary},events,ETFS)
        for key in before:np.testing.assert_array_equal(exact[key],before[key])
        arbitrary=panel['adj']*np.exp(np.linspace(-600.,600.,49))
        got,_=f.prepare({'adj':arbitrary},events,ETFS)
        np.testing.assert_allclose(got['raw_beta_gap'],before['raw_beta_gap'],rtol=3e-11,atol=3e-12)
        np.testing.assert_array_equal(got['rank_beta_gap'],before['rank_beta_gap'])
        p=np.resize(np.array([1e-300,1e300]),253)
        np.testing.assert_allclose(f._log_returns(p),np.diff(np.log(p)),rtol=2e-16,atol=1e-12)

    def test_tiny_market_variance_not_epsilon_rejected_and_unbounded_beta(self):
        own,market=returns_fixture();base=f._beta_gap(own,market)[0]
        tiny=f._beta_gap(own,market*1e-200)
        self.assertEqual(tiny[4],'available');self.assertAlmostEqual(tiny[0]*1e-200,base,places=13)
        large=f._beta_gap(own*1e200,market)
        self.assertEqual(large[4],'available');self.assertAlmostEqual(large[0]/1e200,base,places=13)

    def test_exact253_window_boundary_and_dates(self):
        panel,_=fixture();days=panel['adj'].index
        got,audit=f.prepare(panel,[(days[293],SYMBOLS),(days[294],SYMBOLS)],ETFS)
        self.assertFalse(got['available'][0].any());self.assertTrue(got['available'][1].all())
        row=audit['months'][1]
        self.assertEqual(row['first_price_date'],str(days[0].date()));self.assertEqual(row['first_return_date'],str(days[1].date()))
        self.assertEqual(row['last_price_date'],str(days[252].date()))
        for pos in (0,1,125,251,252):
            px=panel['adj'].copy();px.iloc[pos,0]=np.nan
            bad,_=f.prepare({'adj':px},[(days[294],SYMBOLS)],ETFS)
            self.assertFalse(bad['available'][0,0]);self.assertTrue(bad['available'][0,1:].all())

    def test_missing_market_affects_event_and_bad_security_only_its_cell(self):
        panel,events=fixture()
        for j,val in enumerate((np.nan,0.,-1.,np.inf,-np.inf)):panel['adj'].iloc[100,j]=val
        got,audit=f.prepare(panel,events[:1],ETFS)
        self.assertFalse(got['available'][0,:5].any());self.assertTrue(got['available'][0,5:].all())
        self.assertEqual(audit['invalid_security_path_cells'],5)
        panel['adj'].iloc[100,32]=np.nan
        got,audit=f.prepare(panel,events[:1],ETFS)
        self.assertFalse(got['available'].any());self.assertEqual(audit['invalid_market_path_cells'],49)

    def test_recent42_future_and_unused_prefix_poison_never_converted(self):
        panel,events=fixture();events=events[1:];baseline,_=f.prepare(panel,events,ETFS)
        px=panel['adj'].astype(object);t=320
        for row in list(range(t-294))+list(range(t-41,len(px))):
            for col in range(49):px.iat[row,col]=Poison()
        got,audit=f.prepare({'adj':px},events,ETFS)
        for key in baseline:np.testing.assert_array_equal(got[key],baseline[key])
        self.assertFalse(audit['recent_42_prices_read'])

    def test_prefix_order_no_mutation_and_no_other_fields_read(self):
        panel,events=fixture();original=panel['adj'].copy();old_events=[(d,s.copy()) for d,s in events]
        full,_=f.prepare(panel,events,ETFS)
        for n in range(len(events)+1):
            p=panel if n==0 else {'adj':panel['adj'].loc[:events[n-1][0]]}
            short,_=f.prepare(p,events[:n],ETFS)
            for key in full:np.testing.assert_array_equal(short[key],full[key][:n])
        perm=np.random.default_rng(44).permutation(49)
        class Panel(dict):
            def __getitem__(self,key):
                if key!='adj':raise AssertionError('unneeded field '+key)
                return super().__getitem__(key)
        got,_=f.prepare(Panel(adj=original.iloc[:,::-1],eligible=Poison(),state=Poison()),
            [(d,[s[k] for k in perm]) for d,s in events],ETFS[::-1])
        for key in full:np.testing.assert_array_equal(got[key],full[key][:,perm])
        pd.testing.assert_frame_equal(panel['adj'],original);self.assertEqual(events,old_events)

    def test_strict_axes_dates_ids_and_internal_shapes(self):
        panel,events=fixture();px=panel['adj']
        for bad in ({},{'adj':np.zeros((4,49))},{'adj':px.iloc[::-1]},
                    {'adj':pd.concat([px.iloc[:1],px])},{'adj':px.set_axis(px.index.tz_localize('UTC'))},
                    {'adj':px.set_axis(px.index+pd.Timedelta(hours=1))},
                    {'adj':px.set_axis([SYMBOLS[0]]*49,axis=1)}):
            with self.assertRaises(ValueError):f.prepare(bad,events,ETFS)
        for e in (None,['x'],events[::-1],[events[0],events[0]],[(pd.NaT,SYMBOLS)],
                  [(events[0][0],SYMBOLS[:-1])],[(events[0][0],SYMBOLS[:-1]+['UNKNOWN'])],
                  [(events[0][0],SYMBOLS[:-1]+[SYMBOLS[0]])]):
            with self.assertRaises(ValueError):f.prepare(panel,e,ETFS)
        for etfs in (ETFS[:-1],ETFS[:-1]+[ETFS[0]],ETFS[:-1]+['OTHER'],'SPY'):
            with self.assertRaises(ValueError):f.prepare(panel,events,etfs)
        with self.assertRaises(ValueError):f._beta_gap(np.zeros(251),np.zeros(252))
        with self.assertRaises(ValueError):f._log_returns(np.ones(252))
        with self.assertRaises(ValueError):f._slope(np.ones(1),np.ones(1))

    def test_nonfinite_calculation_propagates_without_clipping(self):
        own,market=returns_fixture()
        with self.assertRaises(FloatingPointError):f._beta_gap(own,np.full(252,np.nan))
        with self.assertRaises(FloatingPointError):f._slope(market,np.full(252,np.inf))
        with self.assertRaises(FloatingPointError):f._slope(np.array([1e-300,2e-300]),np.array([1e300,2e300]))
        with patch.object(f,'_slope',side_effect=[1e308,-1e308]):
            with self.assertRaises(FloatingPointError):f._beta_gap(own,market)


if __name__=='__main__':unittest.main()
