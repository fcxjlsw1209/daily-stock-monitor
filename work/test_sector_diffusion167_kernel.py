"""Synthetic-only sector diffusion tests; no market input or outcome I/O."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import sector_diffusion167_kernel as k


def fixture():
    sessions = pd.bdate_range('2015-01-01', '2017-04-05')
    names = ['A', 'B', 'C', 'ETF', 'UNKNOWN', 'XLK', 'XLV']
    rng = np.random.default_rng(167)
    shape = (len(sessions), len(names))
    panel = {key: pd.DataFrame(base*np.exp(np.cumsum(rng.normal(0., scale, shape), axis=0)),
                              index=sessions, columns=names)
             for key, base, scale in (('adj', 80., .017), ('volume', 1e6, .05))}
    events = [('2016-12-30', names[:5])]
    panel['benchmark_map'] = {str(day.date()): {'A':'XLK','B':'XLV','C':'XLK','ETF':None,'UNKNOWN':None}
                              for day in sessions}
    return panel, events, [str(day.date()) for day in sessions]


def run(panel, events, calendar, limit=None):
    return k.build_features(panel, object(), events, calendar, limit)


def volume_for_activity(activity):
    logs = [0.]*21
    for value in activity:
        logs.append(float(value)+sum(logs[-21:])/21.)
    return np.exp(logs)


def prices_for_returns(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def source_arrays():
    rng = np.random.default_rng(12)
    return (prices_for_returns(rng.normal(0., .02, 211)),
            prices_for_returns(rng.normal(0., .017, 211)),
            volume_for_activity(rng.normal(0., .04, 210)))


def independent_lstsq(stock, sector, volume):
    """Direct centered least squares; no production helpers or SVD basis."""
    r, b = np.diff(np.log(stock)), np.diff(np.log(sector))
    logs = np.log(volume)
    a = logs[21:] - np.array([sum(logs[i:i+21])/21 for i in range(210)])
    def center(x): return x-x.mean()
    y, rr, bb, dd, aa = [center(x) for x in (r[1:], r[:-1], b[:-1], b[1:], a)]
    q = center(bb*aa)
    out = []
    for x, controls in ((q, (bb, aa, rr, dd)), (bb, (aa, rr, dd))):
        design = np.column_stack(controls)
        ex = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ey = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        out.append(float(ex@ey / np.sqrt((ex@ex)*(ey@ey))))
    return out


class SectorDiffusion167(unittest.TestCase):
    def test_manual_lstsq_oracle_windows_mapping_and_json(self):
        panel, events, calendar = fixture(); t=calendar.index(events[0][0])
        delta, audit=run(panel,events,calendar); row=audit['event_history'][0]
        for j, s in enumerate(events[0][1][:3]):
            sec=panel['benchmark_map'][events[0][0]][s]
            expected=independent_lstsq(panel['adj'][s].iloc[t-253:t-41].to_numpy(),
                                      panel['adj'][sec].iloc[t-253:t-41].to_numpy(),
                                      panel['volume'][sec].iloc[t-273:t-42].to_numpy())
            np.testing.assert_allclose([row['raw_followthrough'][j],row['raw_control'][j]], expected, atol=2e-12,rtol=0)
        self.assertEqual(row['sector_symbols'],['XLK','XLV','XLK',None,None])
        self.assertEqual(row['eligible_direct_stock'],[True,True,True,False,False])
        self.assertEqual(row['common_support'],[True,True,True,False,False])
        self.assertEqual(audit['eligible_direct_stock_cells'],3)
        self.assertEqual(row['window_start'],calendar[t-253]);self.assertEqual(row['window_end'],calendar[t-42])
        self.assertEqual(row['volume_window_start'],calendar[t-273]);self.assertEqual(row['volume_window_end'],calendar[t-43])
        self.assertEqual(row['pair_start'],calendar[t-252]);self.assertEqual(row['pair_end'],calendar[t-43])
        for suffix in ('followthrough','control'):
            ranks=pd.Series(row['raw_'+suffix]).rank(method='average',pct=True)-.5
            np.testing.assert_array_equal(delta['delta_'+suffix][0,:3],.04*ranks[:3])
            np.testing.assert_array_equal(delta['delta_'+suffix][0,3:],[0.,0.])
        json.dumps(audit,allow_nan=False)

    def test_each_consumed_boundary_and_interior_missing_neutralizes_both(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0])
        for field,symbol,start,end in (('adj','A',-253,-42),('adj','XLK',-253,-42),('volume','XLK',-273,-43)):
            for offset in (start,start+1,-100,end):
                for bad in (np.nan,np.inf,-np.inf,0.,-1.):
                    changed=deepcopy(panel);changed[field].loc[pd.Timestamp(calendar[t+offset]),symbol]=bad
                    delta,audit=run(changed,events,calendar);row=audit['event_history'][0]
                    with self.subTest(field=field,symbol=symbol,offset=offset,bad=bad):
                        self.assertIsNone(row['raw_followthrough'][0]);self.assertIsNone(row['raw_control'][0])
                        self.assertFalse(row['common_support'][0])
                        for a in delta.values():self.assertEqual(a[0,0],0.)
                        self.assertTrue(row['common_support'][1])

    def test_missing_sector_shared_only_by_its_own_stocks(self):
        panel,events,calendar=fixture();panel['volume']['XLK']=np.nan
        _,audit=run(panel,events,calendar)
        self.assertEqual(audit['event_history'][0]['common_support'],[False,True,False,False,False])

    def test_stock_volume_and_unmapped_prices_never_converted(self):
        panel,events,calendar=fixture();expected,first=run(panel,events,calendar)
        changed=deepcopy(panel)
        for key in ('adj','volume'):changed[key]=changed[key].astype(object)
        changed['volume'].loc[:,['A','B','C','ETF','UNKNOWN']]='unused stock volume'
        changed['adj'].loc[:,['ETF','UNKNOWN']]='unused unmapped price'
        actual,second=run(changed,events,calendar)
        self.assertEqual(first,second)
        for key in actual:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())

    def test_outside_window_and_current_future_sources_never_converted(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0]);expected,first=run(panel,events,calendar)
        changed=deepcopy(panel)
        for key in ('adj','volume'):
            changed[key]=changed[key].astype(object)
            changed[key].iloc[:t-(253 if key=='adj' else 273)]='too early'
            changed[key].iloc[t-(41 if key=='adj' else 42):]='current or future'
        actual,second=run(changed,events,calendar)
        self.assertEqual(first,second)
        for key in actual:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())
        for key in ('adj','volume'):panel[key]=panel[key].iloc[:t-41]
        actual,second=run(panel,events,calendar);self.assertEqual(first,second)

    def test_each_consumed_endpoint_changes_measurement(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0]);_,base=run(panel,events,calendar)
        for key,symbol,offset in (('adj','A',-253),('adj','A',-42),('adj','XLK',-253),('adj','XLK',-42),('volume','XLK',-273),('volume','XLK',-43)):
            changed=deepcopy(panel);changed[key].loc[pd.Timestamp(calendar[t+offset]),symbol]*=1.3
            _,audit=run(changed,events,calendar)
            self.assertNotEqual(audit['event_history'][0]['raw_followthrough'][0],base['event_history'][0]['raw_followthrough'][0])

    def test_shape_and_positive_finite_contract(self):
        stock,sector,volume=source_arrays()
        for sizes in ((211,212,231),(212,211,231),(212,212,230),(212,212,232)):
            with self.assertRaisesRegex(ValueError,'212 adjusted'):
                k._paired_correlations(*(np.ones(s) for s in sizes))
        for i in range(3):
            arrays=[stock.copy(),sector.copy(),volume.copy()];arrays[i][50]=np.nan
            self.assertTrue(np.isnan(k._paired_correlations(*arrays)).all())

    def test_synthetic_signed_diffusion_residual_direction(self):
        rng=np.random.default_rng(2);b=rng.normal(0,.03,211);a=rng.normal(0,.05,210)
        for direction in (-1,1):
            r=[.01]
            for i in range(210):
                r.append(.2*r[-1]+.1*b[i]+.1*b[i+1]+.01*a[i]+direction*12*b[i]*a[i]+rng.normal(0,1e-6))
            sources=(prices_for_returns(r),prices_for_returns(b),volume_for_activity(a))
            out=k._paired_correlations(*sources)
            self.assertGreater(direction*out[0],.999)
            np.testing.assert_allclose(out,independent_lstsq(*sources),atol=2e-12,rtol=0)

    def test_additive_control_is_partial_not_plain_sector_lag(self):
        stock,sector,volume=source_arrays();out=k._paired_correlations(stock,sector,volume)
        np.testing.assert_allclose(out,independent_lstsq(stock,sector,volume),atol=2e-12,rtol=0)
        plain=np.corrcoef(np.diff(np.log(sector))[:-1],np.diff(np.log(stock))[1:])[0,1]
        self.assertGreater(abs(out[1]-plain),1e-5)

    def test_svd_rank_threshold_is_relative_to_largest_singular(self):
        rng=np.random.default_rng(5);q,_=np.linalg.qr(rng.normal(size=(210,4)))
        self.assertIsNotNone(k._basis([q[:,0],q[:,1],q[:,2],q[:,3]]))
        self.assertIsNone(k._basis([q[:,0],q[:,1],q[:,2],q[:,0]+1e-14*q[:,3]]))
        self.assertIsNotNone(k._basis([q[:,0],q[:,1],q[:,2],q[:,0]+1e-8*q[:,3]]))
        with patch.object(k.np.linalg,'svd',side_effect=np.linalg.LinAlgError):self.assertIsNone(k._basis([q[:,0],q[:,1]]))

    def test_constant_sources_or_collinear_full_and_additive_controls_share_mask(self):
        stock,sector,volume=source_arrays();b=np.diff(np.log(sector));r=np.diff(np.log(stock))
        cases=[(np.ones(212),sector,volume),(stock,np.ones(212),volume),(stock,sector,np.ones(231)),
               (sector,sector,volume), # r=b: full4 rank failure, additive3 may retain rank.
               (stock,sector,volume_for_activity(r[:-1]))] # a=r: additive3 also fails.
        for source in cases:
            self.assertTrue(np.isnan(k._paired_correlations(*source)).all())

    def test_response_or_interaction_residual_degeneracy_shared(self):
        rng=np.random.default_rng(71);b=rng.uniform(-.2,.2,211);a=rng.normal(0,.05,210)
        r=[.01]
        for i in range(210):r.append(.3*r[-1]+.2*b[i]+.1*b[i+1]+.02*a[i])
        self.assertTrue(np.isnan(k._paired_correlations(prices_for_returns(r),prices_for_returns(b),volume_for_activity(a))).all())
        stock,_,_=source_arrays();a=1./(b[:-1]-1.) # b*a=1+a, hence interaction in nuisance span.
        self.assertTrue(np.isnan(k._paired_correlations(stock,prices_for_returns(b),volume_for_activity(a))).all())

    def test_partial_residual_threshold_and_correlation_clip(self):
        rng=np.random.default_rng(52);q,_=np.linalg.qr(rng.normal(size=(210,3)))
        self.assertTrue(np.isnan(k._partial(q[:,0]+1e-14*q[:,2],q[:,2],q[:,:2])))
        self.assertAlmostEqual(k._partial(q[:,0]+1e-8*q[:,2],q[:,2],q[:,:2]),1.,places=10)
        self.assertEqual(k._roundoff_clip(1.+k.ROUNDING_EPS),1.)
        self.assertTrue(np.isnan(k._roundoff_clip(np.nextafter(1.+k.ROUNDING_EPS,np.inf))))
        self.assertIsNone(k._center_unit(np.ones(210)+1e-13*np.linspace(-1,1,210)))
        self.assertIsNotNone(k._center_unit(np.ones(210)+1e-9*np.linspace(-1,1,210)))

    def test_extreme_finite_positive_sources_and_unit_scaling(self):
        rng=np.random.default_rng(62)
        sources=[np.exp(rng.uniform(-700,700,n)) for n in (212,212,231)]
        result=k._paired_correlations(*sources);self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result,independent_lstsq(*sources),atol=2e-12,rtol=0)
        panel,events,calendar=fixture();expected,first=run(panel,events,calendar)
        panel['adj']*=np.array([2.,.5,8.,4.,.25,16.,.125]);panel['volume']*=np.array([1.,1.,1.,1.,1.,1e-100,1e100])
        actual,second=run(panel,events,calendar)
        for suffix in ('followthrough','control'):
            np.testing.assert_allclose(second['event_history'][0]['raw_'+suffix][:3],first['event_history'][0]['raw_'+suffix][:3],atol=3e-12,rtol=0)
            np.testing.assert_array_equal(actual['delta_'+suffix],expected['delta_'+suffix])

    def test_ties_singleton_and_neutral_etf_ranks(self):
        panel,events,calendar=fixture();panel['adj']['C']=panel['adj']['A'];panel['benchmark_map'][events[0][0]]['B']=None
        delta,audit=run(panel,events,calendar);row=audit['event_history'][0]
        for suffix in ('followthrough','control'):
            self.assertEqual(row['ranks_'+suffix],[.25,None,.25,None,None])
            np.testing.assert_array_equal(delta['delta_'+suffix],[[.01,0.,.01,0.,0.]])
        panel['benchmark_map'][events[0][0]]['C']=None
        delta,audit=run(panel,events,calendar)
        for values in delta.values():np.testing.assert_array_equal(values,[[.02,0.,0.,0.,0.]])

    def test_initial_partial_measurements_kept_but_denominator_excluded(self):
        panel,events,calendar=fixture();events=[('2016-08-17',events[0][1])]
        delta,audit=run(panel,events,calendar);row=audit['event_history'][0]
        self.assertTrue(all(x is not None for x in row['raw_followthrough'][:3]))
        self.assertEqual(row['common_support'],[False]*5);self.assertEqual(audit['eligible_direct_stock_cells'],0)
        for values in delta.values():self.assertEqual(values.tobytes(),np.zeros((1,5)).tobytes())

    def test_unknown_mapping_entire_panel_neutral_without_conversion(self):
        panel,events,calendar=fixture();panel['benchmark_map'][events[0][0]]={s:None for s in events[0][1]}
        for key in ('adj','volume'):panel[key]=panel[key].astype(object);panel[key].iloc[:]='unconsumed'
        delta,audit=run(panel,events,calendar)
        self.assertEqual(audit['eligible_direct_stock_cells'],0);self.assertEqual(audit['common_support_cells'],0)
        for values in delta.values():np.testing.assert_array_equal(values,np.zeros((1,5)))

    def test_mapping_changes_are_signal_specific_not_future_filled(self):
        panel,events,calendar=fixture();symbols=events[0][1]
        events=[('2016-08-31',symbols),('2016-12-30',symbols)]
        expected,first=run(panel,events,calendar,1)
        panel['benchmark_map']['2016-12-30']={'malformed':'future mapping not read'}
        actual,second=run(panel,events,calendar,1);self.assertEqual(first,second)
        for key in actual:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())
        with self.assertRaises(ValueError):run(panel,events,calendar)

    def test_all123_prefixes_exact_with_synthetic_sources(self):
        sessions=pd.bdate_range('2013-01-01','2027-01-31');rng=np.random.default_rng(92)
        names=['A','ETF','XLK'];dates=['2016-08-17']+[str(x.date()) for x in pd.date_range('2016-08-31',periods=121,freq='BME')]
        panel={key:pd.DataFrame(base*np.exp(np.cumsum(rng.normal(0,scale,(len(sessions),3)),axis=0)),index=sessions,columns=names)
               for key,base,scale in (('adj',100.,.01),('volume',1e6,.03))}
        panel['benchmark_map']={day:{'A':'XLK','ETF':None} for day in dates}
        events=[(day,names[:2]) for day in dates];calendar=[str(x.date()) for x in sessions]
        full,audit=run(panel,events,calendar)
        for n in range(123):
            part,prefix=run(panel,events,calendar,n)
            self.assertEqual(prefix['event_history'],audit['event_history'][:n])
            for key in full:self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())

    def test_missing_session_or_sector_column_neutral_without_fill(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0])
        for kind in ('session','sector'):
            changed=deepcopy(panel)
            for key in ('adj','volume'):
                changed[key]=changed[key].drop(index=pd.Timestamp(calendar[t-100])) if kind=='session' else changed[key].drop(columns='XLK')
            _,audit=run(changed,events,calendar)
            self.assertFalse(audit['event_history'][0]['common_support'][0])

    def test_unofficial_rows_ignored_and_exact_history_boundary(self):
        panel,events,calendar=fixture();expected,first=run(panel,events,calendar);t=calendar.index(events[0][0])
        changed=deepcopy(panel)
        for key in ('adj','volume'):
            changed[key]=changed[key].astype(object);changed[key].loc[pd.Timestamp('2016-04-09')]='closed session'
            changed[key]=changed[key].sort_index()
        actual,second=run(changed,events,calendar);self.assertEqual(first,second)
        _,audit=run(panel,events,calendar[t-273:]);self.assertEqual(audit['common_support_cells'],3)
        _,audit=run(panel,events,calendar[t-272:]);self.assertEqual(audit['common_support_cells'],0)

    def test_malformed_mapping_and_panel_schema(self):
        panel,events,calendar=fixture()
        for mapping in ({},{'A':'XLK'},{s:'SPY' for s in events[0][1]},'wrong'):
            changed=deepcopy(panel);changed['benchmark_map'][events[0][0]]=mapping
            with self.assertRaises((ValueError,TypeError)):run(changed,events,calendar)
        changed=deepcopy(panel);changed['benchmark_map'][events[0][0]]['A']='A'
        with self.assertRaises(ValueError):run(changed,events,calendar)
        for changed in (dict(panel,extra=1),{'adj':panel['adj'],'volume':panel['volume']},dict(panel,volume=panel['volume'].iloc[:-1])):
            with self.assertRaises(ValueError):run(changed,events,calendar)
        for limit in (-1,2,True,1.,'1'):
            with self.assertRaises(ValueError):run(panel,events,calendar,limit)
        for changed in ([('2016-12-29',events[0][1])],events*2,[('2016-12-30',['A','A'])]):
            with self.assertRaises(ValueError):run(panel,changed,calendar,0)
        with self.assertRaisesRegex(ValueError,'next session'):run(panel,events,calendar[:calendar.index(events[0][0])+1])

    def test_empty_prefix_no_source_conversion_or_mapping_read_and_no_mutation(self):
        panel,events,calendar=fixture();saved=deepcopy(panel);run(panel,events,calendar)
        for key in ('adj','volume'):pd.testing.assert_frame_equal(panel[key],saved[key])
        self.assertEqual(panel['benchmark_map'],saved['benchmark_map'])
        for key in ('adj','volume'):panel[key]=pd.DataFrame('future',index=panel[key].index,columns=panel[key].columns)
        panel['benchmark_map']={}
        delta,audit=run(panel,events,calendar,0);self.assertEqual(audit['event_history'],[])
        for values in delta.values():self.assertEqual(values.shape,(0,5))


if __name__=='__main__':unittest.main()
