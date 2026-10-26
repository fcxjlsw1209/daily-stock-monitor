"""Synthetic-only followthrough/persistence tests; no market files or outcomes."""
from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd
import followthrough166_kernel as k


def fixture(names=('A', 'B', 'C', 'ETF')):
    sessions = pd.bdate_range('2015-01-01', '2017-04-05')
    rng = np.random.default_rng(166)
    shape = (len(sessions), len(names))
    returns = rng.normal(.0003, .017, shape)
    adj = 80.*np.exp(np.cumsum(returns, axis=0))
    volume = 1e6*np.exp(np.cumsum(rng.normal(0., .05, shape), axis=0))
    panel = {name: pd.DataFrame(data, index=sessions, columns=names)
             for name, data in (('adj', adj), ('volume', volume))}
    return panel, [('2016-12-30', list(names))], [str(day.date()) for day in sessions]


def run(panel, events, calendar, limit=None, metadata=None):
    return k.build_features(panel, metadata, events, calendar, limit)


def scalar_partial(adj, volume):
    """Independent scalar normal-equation oracle, without Gram-Schmidt."""
    import math
    returns = [math.log(adj[i+1])-math.log(adj[i]) for i in range(211)]
    r, y = returns[:-1], returns[1:]
    logs = [math.log(v) for v in volume]
    a = [logs[i+21]-math.fsum(logs[i:i+21])/21. for i in range(210)]
    def center(v):
        mean=math.fsum(v)/len(v)
        return [x-mean for x in v]
    def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
    r,a,y=map(center,(r,a,y))
    q=center([x*z for x,z in zip(r,a)])
    rr,aa,ra=dot(r,r),dot(a,a),dot(r,a)
    determinant=rr*aa-ra*ra
    def residual(v):
        vr,va=dot(v,r),dot(v,a)
        br,ba=(vr*aa-va*ra)/determinant,(va*rr-vr*ra)/determinant
        return [x-br*z-ba*w for x,z,w in zip(v,r,a)]
    eq,ey=map(residual,(q,y))
    return dot(eq,ey)/math.sqrt(dot(eq,eq)*dot(ey,ey)),dot(r,y)/math.sqrt(rr*dot(y,y))


def volume_for_activity(activity):
    logs=[0.]*21
    for value in activity:logs.append(float(value)+sum(logs[-21:])/21.)
    return np.exp(logs)


class Followthrough166(unittest.TestCase):
    def test_manual_scalar_partial_correlation_exact_sources_and_json(self):
        panel, events, calendar = fixture(); t = calendar.index(events[0][0])
        delta, audit = run(panel, events, calendar); row = audit['event_history'][0]
        for col in range(4):
            expected = scalar_partial(panel['adj'].iloc[t-253:t-41,col].tolist(),
                                      panel['volume'].iloc[t-273:t-42,col].tolist())
            self.assertAlmostEqual(row['raw_followthrough'][col], expected[0], places=12)
            self.assertAlmostEqual(row['raw_control'][col], expected[1], places=12)
        for suffix in ('followthrough','control'):
            ranks = pd.Series(row['raw_'+suffix]).rank(method='average', pct=True).to_numpy()-.5
            np.testing.assert_array_equal(delta['delta_'+suffix][0], .04*ranks)
        self.assertEqual(row['window_start'], calendar[t-253])
        self.assertEqual(row['pair_start'], calendar[t-252])
        self.assertEqual(row['pair_end'], calendar[t-43])
        self.assertEqual(row['volume_window_start'], calendar[t-273])
        self.assertEqual(row['volume_window_end'], calendar[t-43])
        self.assertEqual(row['window_end'], calendar[t-42])
        self.assertEqual(row['common_support'], [True]*4)
        self.assertEqual(audit['window_pairs'],210)
        self.assertEqual(audit['adjusted_window_prices'],212)
        self.assertEqual(audit['volume_window_values'],231)
        self.assertEqual(audit['activity_baseline_sessions'],21)
        json.dumps(audit, allow_nan=False)

    def test_all_consumed_boundaries_and_interiors_missing_neutralize_both(self):
        panel, events, calendar = fixture(); t = calendar.index(events[0][0])
        for field in panel:
            final = -42 if field == 'adj' else -43
            start = -253 if field == 'adj' else -273
            for offset in (start,start+1,-253,-252,-100,final):
                for bad in (np.nan,np.inf,-np.inf,0.,-1.):
                    changed = {key:value.copy() for key,value in panel.items()}
                    changed[field].iloc[t+offset,0] = bad
                    delta,audit = run(changed,events,calendar); row=audit['event_history'][0]
                    with self.subTest(field=field,offset=offset,bad=bad):
                        self.assertIsNone(row['raw_followthrough'][0]); self.assertIsNone(row['raw_control'][0])
                        self.assertIsNone(row['ranks_followthrough'][0]); self.assertIsNone(row['ranks_control'][0])
                        self.assertFalse(row['common_support'][0])
                        for values in delta.values():
                            self.assertEqual(values[0,0],0.); self.assertFalse(np.signbit(values[0,0]))

    def test_volume_t_minus42_current_future_and_preformation_poison_ignored(self):
        panel,events,calendar=fixture(); t=calendar.index(events[0][0])
        expected,audit=run(panel,events,calendar)
        changed={key:value.astype(object) for key,value in panel.items()}
        for field,frame in changed.items():
            frame.iloc[:t-(253 if field=='adj' else 273)] = 'unconsumed early'
            frame.iloc[t-41 if field=='adj' else t-42:] = 'unconsumed current/future'
        actual,second=run(changed,events,calendar)
        self.assertEqual(audit,second)
        for key in actual: self.assertEqual(actual[key].tobytes(),expected[key].tobytes())
        # A common aligned panel may end at the last consumed adjusted close.
        short={key:value.iloc[:t-41] for key,value in panel.items()}
        actual,second=run(short,events,calendar)
        self.assertEqual(audit,second)

    def test_each_consumed_boundary_changes_the_measurement(self):
        panel,events,calendar=fixture(); t=calendar.index(events[0][0])
        _,baseline=run(panel,events,calendar)
        for field,offset in (('adj',-253),('adj',-42),('volume',-273),('volume',-43)):
            changed={key:value.copy() for key,value in panel.items()}
            changed[field].iloc[t+offset,0] *= 1.25
            _,audit=run(changed,events,calendar)
            self.assertNotEqual(audit['event_history'][0]['raw_followthrough'][0],baseline['event_history'][0]['raw_followthrough'][0])

    def test_centered_controls_and_residual_degeneracy_share_missing_support(self):
        rng=np.random.default_rng(11)
        returns=rng.normal(.001,.02,211)
        adj=np.exp(np.r_[0.,np.cumsum(returns)])
        independent=volume_for_activity(rng.normal(0.,.03,210))
        cases=[(np.ones(212),independent),(adj,np.ones(231)),
               (adj,volume_for_activity(returns[:-1]))]
        # Return persistence is perfectly explained by the return control.
        linear=np.asarray([.01*(.99**i) for i in range(211)])
        cases.append((np.exp(np.r_[0.,np.cumsum(linear)]),independent))
        # Centered interaction is constant when both controls share binary levels.
        binary=np.resize([-.02,.02],211)
        cases.append((np.exp(np.r_[0.,np.cumsum(binary)]),volume_for_activity(binary[:-1])))
        for prices,volume in cases:
            with self.subTest(prices=prices[:3]):
                self.assertTrue(all(np.isnan(v) for v in k._paired_correlations(prices,volume)))

    def test_interaction_degenerate_after_independent_controls(self):
        rng=np.random.default_rng(81)
        returns=rng.uniform(-.2,.2,211)
        # r*a=1+a exactly; q therefore lies in the span of the two controls
        # even after centering r and a individually. The broad return range
        # keeps those controls distinct after the source log/exp round trip;
        # near-collinear controls amplify that roundoff beyond the fixed
        # 1e-12 residual threshold and do not isolate interaction degeneracy.
        activity=1./(returns[:-1]-1.)
        prices=np.exp(np.r_[0.,np.cumsum(returns)])
        volume=volume_for_activity(activity)
        self.assertLess(abs(np.corrcoef(returns[:-1],activity)[0,1]),.999)
        self.assertTrue(all(np.isnan(v) for v in k._paired_correlations(prices,volume)))

    def test_signed_interaction_has_expected_direction(self):
        rng=np.random.default_rng(82)
        activity=rng.uniform(-1.,1.,210)
        noise=rng.normal(0.,.0001,210)
        for direction in (-1.,1.):
            returns=[.01]
            for a,error in zip(activity,noise):
                returns.append(.25*returns[-1]+.03*a+direction*.8*returns[-1]*a+error)
            prices=np.exp(np.r_[0.,np.cumsum(returns)])
            result=k._paired_correlations(prices,volume_for_activity(activity))
            self.assertGreater(direction*result[0],.99)
            np.testing.assert_allclose(result,scalar_partial(prices,volume_for_activity(activity)),atol=1e-12,rtol=0)

    def test_prefix_does_not_convert_later_event_only_sources(self):
        panel,_,calendar=fixture()
        events=[(day,['A','B','C','ETF']) for day in ('2016-08-31','2016-12-30')]
        expected,audit=run(panel,events,calendar,1)
        cutoff=calendar.index(events[0][0])
        poison={key:value.astype(object) for key,value in panel.items()}
        for key,frame in poison.items():
            frame.iloc[cutoff-(41 if key=='adj' else 42):]='later event only'
        actual,second=run(poison,events,calendar,1)
        self.assertEqual(second,audit)
        for key in actual:self.assertEqual(actual[key].tobytes(),expected[key].tobytes())

    def test_extreme_positive_finite_sources_do_not_overflow(self):
        rng=np.random.default_rng(661)
        adj=np.exp(rng.uniform(-700.,700.,212))
        volume=np.exp(rng.uniform(-700.,700.,231))
        result=k._paired_correlations(adj,volume)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result,scalar_partial(adj,volume),atol=1e-12,rtol=0)

    def test_fixed_dimensionless_degeneracy_threshold(self):
        self.assertIsNone(k._center_unit(np.ones(210)))
        self.assertIsNone(k._center_unit(np.ones(210)+1e-13*np.linspace(-1,1,210)))
        self.assertIsNotNone(k._center_unit(np.ones(210)+1e-9*np.linspace(-1,1,210)))
        for scale in (1e-120,1.,1e120):
            u=k._center_unit(scale*np.arange(1.,211.))
            np.testing.assert_allclose(u,k._center_unit(np.arange(1.,211.)),atol=1e-15,rtol=0)
        self.assertEqual(k.RELATIVE_EPS,1e-12)
        self.assertEqual(k._roundoff_clip(1.+np.finfo(float).eps),1.)
        self.assertTrue(np.isnan(k._roundoff_clip(1.+1e-10)))
        self.assertEqual(k._roundoff_clip(-1.-k.ROUNDING_EPS),-1.)
        self.assertTrue(np.isnan(k._roundoff_clip(np.nextafter(1.+k.ROUNDING_EPS,np.inf))))

    def test_adjusted_price_volume_units_preserve_correlations_and_ranks(self):
        panel,events,calendar=fixture(); expected,first=run(panel,events,calendar)
        changed={key:value.copy() for key,value in panel.items()}
        changed['volume'] *= [1e-100,1e70,.03125,8.]
        changed['adj'] *= [8.,.5,16.,.25]
        actual,second=run(changed,events,calendar)
        for suffix in ('followthrough','control'):
            np.testing.assert_allclose(second['event_history'][0]['raw_'+suffix],first['event_history'][0]['raw_'+suffix],atol=1e-11,rtol=0)
            np.testing.assert_array_equal(actual['delta_'+suffix],expected['delta_'+suffix])

    def test_shared_support_and_average_ties_use_original_pool(self):
        panel,events,calendar=fixture()
        for frame in panel.values(): frame['B']=frame['A']
        panel['volume']['C']=np.nan
        delta,audit=run(panel,events,calendar); row=audit['event_history'][0]
        self.assertEqual(row['common_support'],[True,True,False,True])
        for suffix in ('followthrough','control'):
            self.assertEqual(row['raw_'+suffix][0],row['raw_'+suffix][1])
            ranks=pd.Series(row['raw_'+suffix]).rank(method='average',pct=True)-.5
            self.assertEqual(row['ranks_'+suffix][:2],ranks.tolist()[:2])
            self.assertIsNone(row['ranks_'+suffix][2])
            np.testing.assert_array_equal(delta['delta_'+suffix][0,[0,1,3]],.04*ranks.iloc[[0,1,3]].to_numpy())

    def test_singleton_rank_half_maximum_positive_dose(self):
        panel,events,calendar=fixture()
        for frame in panel.values(): frame.loc[:,['B','C','ETF']]=np.nan
        delta,audit=run(panel,events,calendar)
        for suffix in ('followthrough','control'):
            self.assertEqual(audit['event_history'][0]['ranks_'+suffix],[.5,None,None,None])
            np.testing.assert_array_equal(delta['delta_'+suffix],[[.02,0.,0.,0.]])

    def test_initial_forced_neutral_preserves_measurements(self):
        panel,_,calendar=fixture(); events=[('2016-08-17',['A','B','C','ETF'])]
        delta,audit=run(panel,events,calendar); row=audit['event_history'][0]
        self.assertTrue(all(v is not None for v in row['raw_followthrough']))
        self.assertEqual(row['common_support'],[False]*4)
        self.assertTrue(row['initial_forced_neutral'])
        for values in delta.values(): self.assertEqual(values.tobytes(),np.zeros((1,4)).tobytes())

    def test_all_prefixes_exact_and_unused_training_metadata(self):
        panel,_,calendar=fixture()
        events=[(day,['A','B','C','ETF']) for day in ('2016-08-17','2016-08-31','2016-09-30','2016-10-31','2016-11-30','2016-12-30')]
        full,audit=run(panel,events,calendar,metadata=object())
        for n in range(len(events)+1):
            part,prefix=run(panel,events,calendar,n,{'ignored':float('nan')})
            self.assertEqual(prefix['event_history'],audit['event_history'][:n])
            for key in full: self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())

    def test_all123_prefixes_match_full122_features(self):
        sessions=pd.bdate_range('2013-01-01','2027-01-31')
        dates=['2016-08-17']+[str(day.date()) for day in pd.date_range('2016-08-31',periods=121,freq='BME')]
        rng=np.random.default_rng(22)
        panel={key:pd.DataFrame(base*np.exp(np.cumsum(rng.normal(0,scale,(len(sessions),2)),axis=0)),
                                 index=sessions,columns=['A','ETF'])
               for key,base,scale in (('adj',100.,.01),('volume',1e6,.03))}
        calendar=[str(day.date()) for day in sessions];events=[(day,['A','ETF']) for day in dates]
        full,audit=run(panel,events,calendar)
        for n in range(123):
            part,prefix=run(panel,events,calendar,n)
            self.assertEqual(prefix['event_history'],audit['event_history'][:n])
            for key in full:self.assertEqual(part[key].tobytes(),full[key][:n].tobytes())

    def test_missing_official_session_and_missing_symbol_are_neutral(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0])
        for change in ('session','symbol'):
            changed={key:value.drop(index=pd.Timestamp(calendar[t-100])) if change=='session' else value.drop(columns='A')
                     for key,value in panel.items()}
            delta,audit=run(changed,events,calendar)
            self.assertIsNone(audit['event_history'][0]['raw_followthrough'][0])
            self.assertTrue(all(values[0,0]==0 for values in delta.values()))

    def test_unofficial_extra_rows_not_used(self):
        panel,events,calendar=fixture();expected,audit=run(panel,events,calendar)
        changed={}
        for key,frame in panel.items():
            frame=frame.astype(object);frame.loc[pd.Timestamp('2016-04-09')]='closed session poison'
            changed[key]=frame.sort_index()
        actual,second=run(changed,events,calendar)
        self.assertEqual(audit,second)
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key])

    def test_exact_history_boundary_and_unavailable_insufficient_history(self):
        panel,events,calendar=fixture();t=calendar.index(events[0][0])
        _,audit=run(panel,events,calendar[t-273:]);self.assertEqual(audit['common_support_cells'],4)
        delta,audit=run(panel,events,calendar[t-272:]);self.assertEqual(audit['common_support_cells'],0)
        self.assertIsNone(audit['event_history'][0]['volume_window_start'])
        for values in delta.values():np.testing.assert_array_equal(values,np.zeros((1,4)))

    def test_panel_alignment_and_metadata_schema(self):
        panel,events,calendar=fixture()
        invalid=[{'adj':panel['adj']},dict(panel,extra=panel['adj']),dict(panel,close=panel['adj']),
                 dict(panel,volume=panel['volume'].iloc[:-1]),dict(panel,volume=panel['volume'][['B','A','C','ETF']]),
                 {key:value.iloc[::-1] for key,value in panel.items()},
                 {key:value.rename(columns={'B':'A'}) for key,value in panel.items()}]
        for changed in invalid:
            with self.assertRaises((ValueError,TypeError)):run(changed,events,calendar)
        for limit in (-1,2,True,1.,'1'):
            with self.assertRaises(ValueError):run(panel,events,calendar,limit)
        for changed in ([('2016-12-29',events[0][1])],events*2,[('2016-12-30',['A','A'])]):
            with self.assertRaises(ValueError):run(panel,changed,calendar,0)
        with self.assertRaisesRegex(ValueError,'next session'):run(panel,events,calendar[:calendar.index(events[0][0])+1])

    def test_empty_prefix_never_converts_source_values_or_mutates_inputs(self):
        panel,events,calendar=fixture();saved=deepcopy(panel);saved_events=deepcopy(events)
        run(panel,events,calendar)
        for key in panel:pd.testing.assert_frame_equal(panel[key],saved[key])
        self.assertEqual(events,saved_events)
        poison={key:pd.DataFrame('future',index=value.index,columns=value.columns) for key,value in panel.items()}
        delta,audit=run(poison,events,calendar,0)
        self.assertEqual(audit['event_history'],[])
        self.assertTrue(all(values.shape==(0,4) for values in delta.values()))

    def test_scalar_shape_contract(self):
        for shapes in ((211,231),(212,230),(212,232)):
            with self.assertRaisesRegex(ValueError,'212 adjusted'):k._paired_correlations(*(np.ones(n) for n in shapes))


if __name__=='__main__':unittest.main()
