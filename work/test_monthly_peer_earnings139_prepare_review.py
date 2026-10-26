"""Synthetic-only tests for the independent139 prepare oracle; no cache reads."""
from copy import deepcopy
from datetime import date, timedelta
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Use the same pinned dependency directories as the local research wrapper.
ROOT = Path('/Users/shiweili/code/monthly-stock-research')
for directory in (ROOT/'work/monthly-decision-deps', ROOT/'work/research-plot-deps'):
    sys.path.insert(0, str(directory))
import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name('monthly_peer_earnings139_prepare_review.py')
spec = importlib.util.spec_from_file_location('independent139_fixture_oracle', MODULE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def feature_fixture(months=2):
    calendar = [(date(2020, 1, 1) + timedelta(days=2*i)).isoformat() for i in range(110)]
    signal_ids = [80, 90][:months]
    shape = (months, 5)
    meta = dict(dates=[calendar[i] for i in signal_ids], symbols=[list('ABCDE') for _ in signal_ids], etfs=['E'])
    return dict(meta=meta, event10=np.tile([.06, .12, -.03, .2, np.nan], (months, 1)),
        matched_recent10=np.tile([.02, .04, .10, .3, np.nan], (months, 1)),
        valid10=np.tile([True, True, True, True, False], (months, 1)),
        anchor_sessions=np.array([[calendar[i-20]]*4+[None] for i in signal_ids], object).reshape(shape),
        event_ids=np.array([['eventA'+str(i), 'eventB'+str(i), 'eventC'+str(i), 'eventD'+str(i), None]
                            for i in signal_ids], object).reshape(shape),
        official_sessions=calendar, sector_benchmarks=np.tile(['XLK', 'XLK', 'XLK', 'XLY', 'XLK'], (months, 1)),
        issuer_keys=np.tile(['issuerA', 'issuerB', 'issuerC', 'issuerD', 'fundE'], (months, 1)))


def run_features(data):
    return r.build_peer_features(**data)


def score_fixture(months=2):
    baseline=np.array([[.16,.12,.08,.07,-.10],[.15,.13,.10,.09,-.10]])
    core=np.array([[.10,.08,.05,.04,-.10],[.10,.08,.06,.05,-.10]])
    base=dict(baseline=baseline,core=core,eligible=np.ones((2,5),bool),core_weight=np.array([.5,.52]),
       state_weak_narrow=np.array([False,True]),state_calm10=np.array([True,False]),
       pre_score=baseline.copy(),effects=np.array([[0.,0.,.03,-.01,0.],[0.,0.,-.02,.01,0.]]),
       forecast_available=np.zeros((2,5),bool),rank_log_excess=np.zeros((2,5)),old_ml4=np.zeros((2,5)),
       extension_available=np.zeros((2,5),bool),model_available=np.zeros((2,5),bool),
       raw_local_constant=np.zeros((2,5)),raw_local_linear=np.zeros((2,5)),raw_source_log=np.zeros((2,5)))
    if months == 122:
        names=['S%02d'%j for j in range(32)]+['E%02d'%j for j in range(17)]
        arrays={key:np.tile(value,(61,10))[:,:49] if value.ndim==2 else np.tile(value,61) for key,value in base.items()}
        core_ranks=np.tile(np.linspace(-.48,.48,49),(122,1))
        arrays['core']=arrays['core_weight'][:,None]*core_ranks
        arrays['baseline']=arrays['core']+np.tile(np.sin(np.arange(49))*.015,(122,1))
        arrays['pre_score']=arrays['baseline'].copy()
        meta=dict(symbols=[names[:] for _ in range(122)],etfs=names[32:])
    else:
        arrays={key:value[:months].copy() for key,value in base.items()}
        meta=dict(symbols=[list('ABCDE') for _ in range(months)],etfs=['E'])
    meta['dates']=[(date(2020,1,1)+timedelta(days=30*i)).isoformat() for i in range(months)]
    meta['ends']=[(date(2020,1,1)+timedelta(days=30*(i+1))).isoformat() for i in range(months)]
    return arrays, meta


def hands_for(arrays, meta, value=.2):
    result={name:np.full(arrays['baseline'].shape,value,float) for name in r.ARMS[1:]}
    for hand in result.values():
        for i,pool in enumerate(meta['symbols']):
            for j,symbol in enumerate(pool):
                if symbol in meta['etfs']:hand[i,j]=np.nan
    return result


class FeatureOracleTests(unittest.TestCase):
    def setUp(self): self.data=feature_fixture()

    def test_declared_raw_formulas_and_tanh(self):
        result=run_features(self.data); a=result['arrays']; cell=result['audit']['cells'][0]
        for key,value in zip(r.RAW_KEYS,(.07,-.03,.045)):
            self.assertAlmostEqual(a[key][0,0],value,places=15)
        for raw,hand in zip(r.RAW_KEYS,r.HAND_KEYS):
            np.testing.assert_allclose(a[hand],.5*np.tanh(a[raw]/.05),atol=1e-15,rtol=0,equal_nan=True)
        self.assertEqual(cell['peer_symbols'],['B','C'])
        self.assertEqual(cell['peer_issuer_keys'],['issuerB','issuerC'])
        self.assertEqual(cell['peer_count'],2)
        self.assertEqual(set(a),set(r.EXTRA_KEYS))

    def test_own_event_control_is_divided_by_actual_peer_count(self):
        self.data['sector_benchmarks'][:,3]='XLK'
        a=run_features(self.data)['arrays']
        self.assertEqual(a['peer_count'][0,0],3)
        self.assertAlmostEqual(a['raw_negative_own_event10'][0,0],-.06/3)

    def test_official_session_age_10_and_63_inclusive(self):
        for age in (10,63):
            data=deepcopy(self.data);data['anchor_sessions'][0,:3]=data['official_sessions'][80-age]
            result=run_features(data)
            self.assertTrue(result['arrays']['common_available'][0,:3].all())
            self.assertEqual(result['audit']['cells'][0]['age_sessions'],age)

    def test_age9_and64_excluded_without_calendar_day_substitution(self):
        for age,reason in ((9,'immature_event'),(64,'stale_event')):
            data=deepcopy(self.data);data['anchor_sessions'][0,0]=data['official_sessions'][80-age]
            result=run_features(data);a=result['arrays']
            self.assertEqual(result['audit']['cells'][0]['reason'],reason)
            self.assertFalse(a['source_available'][0,0])
            self.assertEqual(a['peer_count'][0,0],0)
            self.assertEqual(a['peer_count'][0,1],1)
            self.assertFalse(a['common_available'][0,:3].any())

    def test_unknown_issuer_and_mapping_have_nan_fallback(self):
        for field,reason in (('issuer_keys','unknown_issuer'),('sector_benchmarks','unknown_sector')):
            data=deepcopy(self.data);data[field]=data[field].astype(object);data[field][0,0]=None
            result=run_features(data)
            self.assertEqual(result['audit']['cells'][0]['reason'],reason)
            for key in r.RAW_KEYS+r.HAND_KEYS:self.assertTrue(np.isnan(result['arrays'][key][0,0]))

    def test_same_issuer_alias_fails_even_when_its_event_missing(self):
        self.data['issuer_keys'][0,1]='issuerA'
        self.data['valid10'][0,1]=False
        self.data['event10'][0,1]=self.data['matched_recent10'][0,1]=np.nan
        with self.assertRaisesRegex(ValueError,'Duplicate known'):run_features(self.data)

    def test_etf_never_donor_even_with_valid_source_and_duplicate_issuer(self):
        self.data['issuer_keys'][0,4]='issuerA';self.data['valid10'][0,4]=True
        self.data['event10'][0,4]=500.;self.data['matched_recent10'][0,4]=600.
        self.data['anchor_sessions'][0,4]=self.data['official_sessions'][60];self.data['event_ids'][0,4]='fundEvent'
        result=run_features(self.data)
        self.assertEqual(result['audit']['cells'][4]['reason'],'etf')
        self.assertEqual(result['audit']['cells'][0]['peer_symbols'],['B','C'])
        self.assertTrue(np.isnan(result['arrays']['hand_peer_event10'][0,4]))

    def test_focal_requires_own_valid_mature_event(self):
        self.data['valid10'][0,0]=False
        self.data['event10'][0,0]=self.data['matched_recent10'][0,0]=np.nan
        result=run_features(self.data)
        self.assertEqual(result['audit']['cells'][0]['reason'],'missing_event')
        self.assertEqual(result['arrays']['peer_count'][0,0],0)

    def test_common_mask_and_peer_lists_when_insufficient(self):
        result=run_features(self.data);a=result['arrays']
        for key in r.RAW_KEYS+r.HAND_KEYS:
            np.testing.assert_array_equal(np.isfinite(a[key]),a['common_available'])
        self.assertTrue(a['source_available'][0,3]);self.assertFalse(a['common_available'][0,3])
        self.assertEqual(result['audit']['cells'][3]['reason'],'insufficient_peer_issuers')

    def test_real_zero_remains_zero_without_cross_sectional_ranking(self):
        self.data['event10'][:,:3]=0.;self.data['matched_recent10'][:,:3]=0.
        result=run_features(self.data)
        for key in r.RAW_KEYS+r.HAND_KEYS:
            values=result['arrays'][key][:,:3]
            self.assertTrue((values==0).all());self.assertFalse(np.signbit(values).any())

    def test_peer_order_is_deterministic_not_original_pool_order(self):
        data=deepcopy(self.data);perm=[2,0,1,3,4]
        data['meta']['symbols']=[[row[j] for j in perm] for row in data['meta']['symbols']]
        for key in ('event10','matched_recent10','valid10','anchor_sessions','event_ids','sector_benchmarks','issuer_keys'):
            data[key]=data[key][:,perm]
        result=run_features(data)
        self.assertEqual(result['audit']['cells'][1]['peer_symbols'],['B','C'])
        self.assertAlmostEqual(result['arrays']['raw_peer_event10'][0,1],.045)

    def test_future_anchor_and_unofficial_anchor_fail(self):
        for anchor in (self.data['official_sessions'][81],'2020-01-02'):
            data=deepcopy(self.data);data['anchor_sessions'][0,0]=anchor
            with self.assertRaises(ValueError):run_features(data)

    def test_calendar_duplicates_and_signal_not_in_calendar_fail(self):
        data=deepcopy(self.data);data['official_sessions'].insert(0,data['official_sessions'][0])
        with self.assertRaises(ValueError):run_features(data)
        data=deepcopy(self.data);data['meta']['dates'][0]='2020-01-02'
        with self.assertRaises(ValueError):run_features(data)

    def test_event_ids_are_provenance_not_feature_values(self):
        expected=run_features(self.data)['arrays'];self.data['event_ids'][0,0]='new-name-same-validated-event'
        actual=run_features(self.data)['arrays']
        for key in r.EXTRA_KEYS:np.testing.assert_array_equal(actual[key],expected[key])
        self.data['event_ids'][0,0]=None
        with self.assertRaises(ValueError):run_features(self.data)

    def test_input_masks_types_and_infinity_fail(self):
        mutations=[lambda x:x.update(valid10=x['valid10'].astype(int)),
                   lambda x:x['event10'].__setitem__((0,0),np.nan),
                   lambda x:x['event10'].__setitem__((0,4),0.),
                   lambda x:x['matched_recent10'].__setitem__((0,0),np.inf),
                   lambda x:x.update(event10=x['event10'].astype(str))]
        for change in mutations:
            data=deepcopy(self.data);change(data)
            with self.assertRaises(ValueError):run_features(data)

    def test_input_bytes_unchanged_and_prefix_causal(self):
        before={key:value.copy() for key,value in self.data.items() if isinstance(value,np.ndarray)}
        full=run_features(self.data);cut=deepcopy(self.data)
        cut['meta']['dates']=cut['meta']['dates'][:1];cut['meta']['symbols']=cut['meta']['symbols'][:1]
        for key in before:cut[key]=cut[key][:1]
        cut['official_sessions']=cut['official_sessions'][:81]
        prefix=run_features(cut)
        for key in r.EXTRA_KEYS:np.testing.assert_array_equal(prefix['arrays'][key],full['arrays'][key][:1])
        self.assertEqual(prefix['audit']['cells'],full['audit']['cells'][:5])
        for key,value in before.items():np.testing.assert_array_equal(self.data[key],value)

    def test_empty_prefix_is_valid(self):
        data=feature_fixture(0);result=run_features(data)
        self.assertEqual(result['audit']['cells'],[])
        for array in result['arrays'].values():self.assertEqual(array.shape,(0,5))

    def test_no_eligibility_argument(self):
        with self.assertRaises(TypeError):r.build_peer_features(**self.data,eligible=np.ones((2,5),bool))


class ScoreOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.helper=r.load_baseline_helper()

    def setUp(self):self.arrays,self.meta=score_fixture()

    def baseline(self,n=None):
        x=self.helper.original_prefix(self.arrays,self.meta,n)
        c,s,_=self.helper.episode(x['local'],x)
        return s,c

    def test_four_arms_exactly_one_episode_each(self):
        with patch.object(self.helper,'episode',wraps=self.helper.episode) as spy:
            scores,choices,_=r.score_oracle(self.arrays,self.meta,hands_for(self.arrays,self.meta),helper=self.helper)
        self.assertEqual(spy.call_count,4)
        self.assertEqual(set(scores),set(r.ARMS));self.assertEqual(set(choices),set(r.ARMS))

    def test_zero_and_missing_exact_baseline_and_no_input_mutation(self):
        expected,c=self.baseline();before={k:v.copy() for k,v in self.arrays.items()}
        for fill in (0.,-0.,np.nan):
            scores,choices,deltas=r.score_oracle(self.arrays,self.meta,hands_for(self.arrays,self.meta,fill),helper=self.helper)
            for name in r.ARMS:
                self.assertEqual(scores[name].tobytes(),expected.tobytes());self.assertEqual(choices[name],c)
                self.assertTrue((deltas[name]==0).all())
        for k,v in before.items():self.assertEqual(self.arrays[k].tobytes(),v.tobytes())

    def test_three_distinct_hands_have_independent_scalar_corrections(self):
        hands=hands_for(self.arrays,self.meta)
        for name,value in zip(r.ARMS[1:],(.5,-.5,.1)):
            hands[name][:,:4]=value
        scores,choices,deltas=r.score_oracle(self.arrays,self.meta,hands,helper=self.helper)
        x=self.helper.original_prefix(self.arrays,self.meta)
        for name in r.ARMS[1:]:
            expected=np.zeros_like(x['local'])
            for i in range(2):
                for j in range(4):
                    s=x['support'][i,j];h=hands[name][i,j]
                    expected[i,j]=.025*(max(-.5,min(.5,s+h))-s)
            np.testing.assert_array_equal(deltas[name],expected)
            pre=x['local'].copy();pre[expected!=0]+=expected[expected!=0]
            c,s,_=self.helper.episode(pre,x)
            self.assertEqual(scores[name].tobytes(),s.tobytes());self.assertEqual(choices[name],c)

    def test_ineligible_hand_does_not_change_score_delta(self):
        self.arrays['eligible'][0,0]=False
        _,_,deltas=r.score_oracle(self.arrays,self.meta,hands_for(self.arrays,self.meta),helper=self.helper)
        for name in r.ARMS:self.assertEqual(deltas[name][0,0],0.)

    def test_bad_hand_masks_bounds_etf_or_arm_fail(self):
        edits=[lambda h:h[r.ARMS[1]].__setitem__((0,0),np.nan),
               lambda h:h[r.ARMS[1]].__setitem__((0,0),.50001),
               lambda h:h[r.ARMS[1]].__setitem__((0,0),np.inf),
               lambda h:h[r.ARMS[1]].__setitem__((0,4),0.),
               lambda h:h.pop(r.ARMS[3])]
        for edit in edits:
            hands=hands_for(self.arrays,self.meta);edit(hands)
            with self.assertRaises(ValueError):r.score_oracle(self.arrays,self.meta,hands,helper=self.helper)

    def test_future_poison_not_interpreted_by_prefix(self):
        hands=hands_for(self.arrays,self.meta)
        wanted=r.score_oracle(self.arrays,self.meta,hands,n=1,helper=self.helper)
        for hand in hands.values():hand[1,:]=np.inf
        self.arrays['pre_score'][1,:]=np.nan
        actual=r.score_oracle(self.arrays,self.meta,hands,n=1,helper=self.helper)
        for name in r.ARMS:
            self.assertEqual(actual[0][name].tobytes(),wanted[0][name].tobytes())
            self.assertEqual(actual[1][name],wanted[1][name])

    def test_488_orders_prefixes_and_incumbent_identity(self):
        arrays,meta=score_fixture(122);hands=hands_for(arrays,meta,.1)
        scores,choices,_=r.score_oracle(arrays,meta,hands,helper=self.helper)
        orders=r.orders_oracle(meta,scores,choices)
        self.assertEqual(sum(map(len,orders.values())),488)
        x=self.helper.original_prefix(arrays,meta);bc,bs,_=self.helper.episode(x['local'],x)
        expected_orders=[dict(date=day,signal_date=day,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in c],scores=[float(bs[i,j]) for j in c])
                         for i,(day,c) in enumerate(zip(meta['dates'],bc))]
        r.assert_incumbent_unchanged(bs,scores[r.ARMS[0]],expected_orders,orders[r.ARMS[0]],helper=self.helper)
        for n in (0,1,36,53,60,76,88,122):
            ss,cc,_=r.score_oracle(arrays,meta,hands,n=n,helper=self.helper)
            for name in r.ARMS:
                self.assertEqual(ss[name].tobytes(),scores[name][:n].tobytes());self.assertEqual(cc[name],choices[name][:n])

    def test_original27_arrays_byte_identity_detects_signed_zero_and_dtype(self):
        original={str(i):np.array([[0.,1.]]) for i in range(27)}
        prepared={k:v.copy() for k,v in original.items()}
        prepared.update({key:np.zeros((1,2)) for key in r.EXTRA_KEYS})
        r.assert_original_arrays_unchanged(original,prepared)
        prepared['0'][0,0]=-0.
        with self.assertRaises(ValueError):r.assert_original_arrays_unchanged(original,prepared)
        prepared['0']=original['0'].astype(np.float32)
        with self.assertRaises(ValueError):r.assert_original_arrays_unchanged(original,prepared)

    def test_incumbent_order_score_tampering_fails(self):
        score=np.array([[0.,1.]])
        with self.assertRaises(ValueError):r.assert_incumbent_unchanged(score,-score,[],[],helper=self.helper)
        with self.assertRaises(AssertionError):r.assert_incumbent_unchanged(score,score,[{'scores':[0.]}],[{'scores':[-0.]}],helper=self.helper)

    def test_no_producer_or_actual_data_imports(self):
        text=MODULE.read_text()
        self.assertNotIn('import monthly_peer_earnings139_factor',text)
        self.assertNotIn('import monthly_peer_earnings139_inputs',text)
        self.assertNotIn('import monthly_peer_earnings139_runner',text)
        self.assertRegex(r.PROPOSAL_SHA, r'^[0-9a-f]{64}$')
        with patch.object(r, 'PROPOSAL_SHA', 'UNREGISTERED_DRAFT'), \
             patch.object(r.Reader, 'sha', side_effect=AssertionError('unexpected input read')):
            with self.assertRaisesRegex(ValueError,'Finalize proposal'):
                r.audit('/nonexistent/synthetic/root','/nonexistent/PLAN.json','0'*64)


class LatestEventTests(unittest.TestCase):
    def fixture(self):
        meta=dict(dates=['2020-05-01'],symbols=[['A','E']],etfs=['E'])
        calendar=SimpleNamespace(session_close=lambda day:pd.Timestamp('2020-05-01T20:00:00Z'))
        events=[dict(event_id='old',symbol='A',status='ready',known_by_at='2020-03-01T12:00:00Z'),
                dict(event_id='new',symbol='A',status='ready',known_by_at='2020-04-30T12:00:00Z')]
        rows=[dict(signal_date='2020-05-01',symbol='A',signal_asof='2020-05-01T20:00:00Z',
                   status='verified_event_available',event_id='new',reactions={}),
              dict(signal_date='2020-05-01',symbol='E',signal_asof='2020-05-01T20:00:00Z',
                   status='not_applicable_etf',event_id=None,reactions={})]
        return meta,deepcopy(rows),events,dict(rows=rows),calendar

    def test_latest_immature_event_kept_instead_of_older_mature(self):
        args=self.fixture();rows=r.latest_event_rows(*args)
        self.assertEqual(rows['2020-05-01','A']['event_id'],'new')
        args[1][0]['event_id']=args[3]['rows'][0]['event_id']='old'
        with self.assertRaisesRegex(ValueError,'latest known'):r.latest_event_rows(*args)

    def test_future_event_does_not_replace_known_event(self):
        args=self.fixture();args[2].append(dict(event_id='future',symbol='A',status='ready',known_by_at='2020-05-02T12:00:00Z'))
        self.assertEqual(r.latest_event_rows(*args)['2020-05-01','A']['event_id'],'new')

    def test_duplicate_or_equal_latest_timestamp_fails(self):
        for duplicate in ('id','clock'):
            args=self.fixture();event=deepcopy(args[2][-1])
            if duplicate=='clock':event['event_id']='new-alias'
            args[2].append(event)
            with self.assertRaises(ValueError):r.latest_event_rows(*args)

    def test_bad_candidate_clock_or_reaction_provenance_fails(self):
        args=self.fixture();args[3]['rows'][0]['signal_asof']='2020-05-01T19:59:00Z'
        with self.assertRaises(ValueError):r.latest_event_rows(*args)
        args=self.fixture();args[3]['rows'][0]['reactions']={'10':{'known_by_at':'2020-03-01T12:00:00Z'}}
        with self.assertRaises(ValueError):r.latest_event_rows(*args)

    def test_issuer_aliases_and_conflicting_identity(self):
        data=[dict(symbol='META',url='issuer/meta'),dict(symbol='BRK-B',url='issuer/berkshire')]
        self.assertEqual(r.issuer_map(data),{'research_equity_FB':'issuer/meta','BRK.B':'issuer/berkshire'})
        with self.assertRaises(ValueError):r.issuer_map(data+[dict(symbol='research_equity_FB',url='different')])


class ReaderTests(unittest.TestCase):
    def test_duplicate_nonfinite_and_changed_file_fail(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            path=Path(folder)/'item.json'
            for payload in ('{"a":1,"a":2}','{"a":NaN}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):r.Reader(folder).json(path)
            path.write_text('{"a":1}');reader=r.Reader(folder);reader.json(path)
            path.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError,'Artifact changed'):reader.json(path)

    def test_manifest_digest_and_cache_escape_fail(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            path=Path(folder)/'item';path.write_bytes(b'payload')
            reader=r.Reader(folder)
            with self.assertRaises(ValueError):reader.verify({'item':'0'*64})
            with self.assertRaises(ValueError):reader.verify({'../item':'0'*64},Path(folder))


class NormalizedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='peer139-normalized-fixture-',dir='/private/tmp')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.fixture=feature_fixture();self.meta=self.fixture['meta'];self.calendar_days=self.fixture['official_sessions']
        rows=[];events=[];anchor=self.calendar_days[60];known=anchor+'T12:00:00Z'
        for symbol in 'ABCD':events.append(dict(event_id='event'+symbol,symbol=symbol,status='ready',known_by_at=known))
        for day in self.meta['dates']:
            for symbol in 'ABCDE':
                etf=symbol=='E'
                rows.append(dict(signal_date=day,symbol=symbol,signal_asof=day+'T20:00:00Z',
                    status='not_applicable_etf' if etf else 'verified_event_available',event_id=None if etf else 'event'+symbol,
                    reactions={} if etf else {'10':dict(anchor_session=anchor,known_by_at=known)}))
        self.document=dict(rows=rows)
        self.write(r.REPAIR+'/MONTHLY_FEATURES.json',self.document)
        self.write(r.REPAIR+'/MONTHLY_AVAILABILITY.json',deepcopy(rows))
        self.write(r.REPAIR+'/EVENTS.json',events)
        raw_names=['MONTHLY_FEATURES.json','MONTHLY_AVAILABILITY.json','EVENTS.json']
        raw={name:self.digest(self.root/r.REPAIR/name) for name in raw_names}
        self.write(r.REPAIR+'/MONTHLY_PANEL_AUDIT.json',dict(input_sha256=raw))
        raw['MONTHLY_PANEL_AUDIT.json']=self.digest(self.root/r.REPAIR/'MONTHLY_PANEL_AUDIT.json')
        self.readiness=dict(scoped_post_known_drift_data_ready=True,unresolved_count=0,unresolved=[],invalid_candidate_price_windows=[],protected_files_changed=0,inputs=raw)
        self.write(r.REPAIR+'/SCOPED_READINESS.json',self.readiness)
        self.write(r.ISSUER_SOURCE,dict(records=[dict(symbol=symbol,url='issuer'+symbol) for symbol in 'ABCD']))
        for source in (r.EVENT_VALIDATOR,r.REPAIR+'/research_calendar.py','outputs/index_union/CALENDAR_EXCEPTIONS.json','outputs/index_union/sources/closure_20250109.html'):
            self.write(source,{'synthetic_source':source})
        self.pins={str(path):self.digest(path) for path in self.root.rglob('*') if path.is_file()}
        self.event={key:self.fixture[key].copy() for key in ('event10','matched_recent10','valid10')}
        self.calendar=SimpleNamespace(sessions=pd.to_datetime(self.calendar_days),session_close=lambda day:pd.Timestamp(day).tz_localize('UTC')+pd.Timedelta(hours=20))
        self.validator=SimpleNamespace(_calendar=lambda:self.calendar,build_features=lambda document,meta,prices,etfs:(self.event,{'synthetic_frozen_validator':True}))
        sector=pd.DataFrame([['Technology']*3+['Consumer Cyclical',None]]*2,index=pd.to_datetime(self.meta['dates']),columns=list('ABCDE'))
        self.panel={'adj':pd.DataFrame(), 'sectors':sector}

    def digest(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def load(self):
        with patch.object(r,'frozen_module',return_value=self.validator):
            return r.normalized_event_inputs(r.Reader(self.root),self.meta,self.panel,self.calendar_days,self.pins)

    def test_bound_source_normalization_keeps_original_event_arrays(self):
        normalized,provenance,consumed=self.load()
        for key in ('event10','matched_recent10','valid10'):
            self.assertEqual(normalized[key].tobytes(),self.event[key].tobytes())
        self.assertEqual(normalized['event_ids'][0,0],'eventA')
        self.assertEqual(normalized['anchor_sessions'][0,0],self.calendar_days[60])
        self.assertEqual(normalized['sector_benchmarks'][0,0],'XLK')
        self.assertEqual(normalized['issuer_keys'][0,0],'issuerA')
        self.assertIsNone(normalized['sector_benchmarks'][0,4])
        self.assertTrue(provenance['original_event10_values_preserved'])
        self.assertFalse(provenance['raw_event_prices_reconstructed'])
        self.assertEqual(consumed,set(self.pins))

    def test_missing_event_closure_entry_fails(self):
        del self.pins[str(self.root/'outputs/index_union/sources/closure_20250109.html')]
        with self.assertRaisesRegex(ValueError,'Event source closure'):self.load()

    def test_readiness_false_fails_even_when_its_new_hash_is_bound(self):
        self.readiness['unresolved_count']=1
        self.write(r.REPAIR+'/SCOPED_READINESS.json',self.readiness)
        path=self.root/r.REPAIR/'SCOPED_READINESS.json';self.pins[str(path)]=self.digest(path)
        with self.assertRaisesRegex(ValueError,'Frozen event readiness'):self.load()

    def test_validator_calendar_mismatch_fails(self):
        self.calendar.sessions=self.calendar.sessions.delete(2)
        with self.assertRaisesRegex(ValueError,'official calendar'):self.load()

    def test_changed_issuer_source_is_not_accepted(self):
        self.write(r.ISSUER_SOURCE,dict(records=[]))
        with self.assertRaisesRegex(ValueError,'Event source closure'):self.load()


class AuditBindingTests(unittest.TestCase):
    """Synthetic files test I/O gates; no production data/139 output is loaded."""
    @classmethod
    def setUpClass(cls):cls.helper=r.load_baseline_helper()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='peer139-audit-fixture-',dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/r.STUDY
        self.directory.mkdir(parents=True)
        self.original,self.meta=score_fixture(122)
        for i in range(27-len(self.original)):
            self.original['frozen_auxiliary_'+str(i)]=np.zeros((122,49))
        calendar=pd.date_range('2015-01-01','2027-12-31',freq='B')
        self.sessions=[str(day.date()) for day in calendar]
        self.meta.update(dates=[self.sessions[410+i*21] for i in range(122)],
                         ends=[self.sessions[410+(i+1)*21] for i in range(122)],train_months=88,early_months=53)
        shape=(122,49);valid=np.zeros(shape,bool);valid[:,:32]=True
        event=np.full(shape,np.nan);recent=event.copy()
        event[:,:32]=np.sin(np.arange(32))*.03;recent[:,:32]=np.cos(np.arange(32))*.02
        anchors=np.full(shape,None,object);ids=anchors.copy();sectors=anchors.copy();issuers=anchors.copy()
        for i in range(122):
            for j in range(32):
                anchors[i,j]=self.sessions[410+i*21-20];ids[i,j]='event%d-%d'%(i,j)
                sectors[i,j]='XLK';issuers[i,j]='issuer%02d'%j
        self.normalized=dict(meta=self.meta,event10=event,matched_recent10=recent,valid10=valid,
            anchor_sessions=anchors,event_ids=ids,official_sessions=self.sessions,sector_benchmarks=sectors,issuer_keys=issuers)
        self.provenance=dict(synthetic_fixture=True)
        features=r.build_peer_features(**self.normalized)
        self.prepared={**self.original,**features['arrays']}
        hands={name:features['arrays'][key] for name,key in zip(r.ARMS[1:],r.HAND_KEYS)}
        scores,choices,_=r.score_oracle(self.original,self.meta,hands,helper=self.helper)
        orders=r.orders_oracle(self.meta,scores,choices)
        self.j(r.LOCAL+'/META.json',self.meta)
        self.j(r.LOCAL+'/INCUMBENT_POLICY.json',self.helper.POLICY)
        self.j(r.LOCAL+'/local_linear_ORDERS.json',orders[r.ARMS[0]])
        self.z(r.LOCAL+'/DATA.npz',self.original)
        self.raw={}
        self.raw[r.RAW_LOADER]=self.b(r.RAW_LOADER,b'# synthetic raw loader\n')
        for i in range(63):
            rel='raw/%02d.txt'%i;self.raw[rel]=self.b(rel,('raw fixture %d'%i).encode())
        self.raw_abs={str(self.root/rel):digest for rel,digest in self.raw.items()}
        self.j(r.LOCAL+'/PLAN.json',dict(raw_source_hashes=self.raw_abs))
        old_names=['DATA.npz','META.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json']
        self.j(r.LOCAL+'/CACHE_HASHES.json',{name:self.sha(self.root/r.LOCAL/name) for name in old_names})
        self.z(r.PRICES,dict(dates=np.array(self.meta['dates'][:2]),symbols=np.array(['S00']),close=np.ones((2,1)),opening=np.ones((2,1))))
        self.proposal=self.b('proposal.json',b'{"synthetic":true}\n')
        self.j('protected.json',{'synthetic':True})
        calendar_binding=self.b('calendar/synthetic-source.py',b'# synthetic calendar\n')
        calendar_plan=dict(start='2015-01-01',end='2027-12-31',bindings={str(self.root/'calendar/synthetic-source.py'):calendar_binding})
        self.calendar_sha=self.j('calendar/CALENDAR_PLAN.json',calendar_plan)
        self.j('calendar/SESSIONS.json',dict(plan_sha256=self.calendar_sha,sessions=self.sessions))
        self.event_path=self.root/'event/normalized-source.json';self.j('event/normalized-source.json',{'synthetic':True})
        self.source_path=self.root/'work/independent-fixture.py';self.b('work/independent-fixture.py',b'# synthetic bound reviewer source\n')
        self.parity=dict(synthetic_price_parity=True)
        self.j(r.STUDY+'/META.json',self.meta);self.z(r.STUDY+'/DATA.npz',self.prepared)
        self.z(r.STUDY+'/SCORES.npz',scores);self.j(r.STUDY+'/PRETRAIN_ORDERS.json',orders)
        self.j(r.STUDY+'/FEATURE_AUDIT.json',dict(features['audit'],input_provenance=self.provenance,new_model_fits=0))
        self.j(r.STUDY+'/PRICE_PARITY.json',self.parity)
        self.refresh_cache()
        pins={str(path):self.sha(path) for path in self.root.rglob('*') if path.is_file() and self.directory not in path.parents}
        refs={'proposal':'proposal.json','meta':r.LOCAL+'/META.json','policy':r.LOCAL+'/INCUMBENT_POLICY.json',
              'original_orders':r.LOCAL+'/local_linear_ORDERS.json','prices':r.PRICES,
              'sessions':'calendar/SESSIONS.json','protected_manifest':'protected.json'}
        self.plan=dict(round=139,status='REGISTERED',arms=list(r.ARMS),primary_challenger=r.ARMS[-1],output_directory=r.STUDY,
            runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),hashes=pins,
            inputs={name:dict(path=path,sha256=self.sha(self.root/path)) for name,path in refs.items()},
            actual_model_fits_before_registration=0,actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,
            tolerances=dict(feature_atol=r.ATOL,feature_rtol=0,scores_atol=r.ATOL,scores_rtol=0),
            independent_prepare_source_hashes={str(self.source_path):self.sha(self.source_path)},
            raw_source_hashes=self.raw_abs,event_input_hashes={str(self.event_path):self.sha(self.event_path)})
        self.save_plan()
        tools=SimpleNamespace(price_parity=lambda panel,prices:self.parity,compare=r.compare,
                              official_sessions=lambda reader,end:calendar)
        loader=SimpleNamespace(load_panel=lambda etfs:({'synthetic_panel':True},dict(source_hashes={key:value for key,value in self.raw_abs.items() if key!=str(self.root/r.RAW_LOADER)})))
        patches=[patch.object(r,'PROPOSAL_SHA',self.proposal),patch.object(r,'RAW_LOADER_SHA',self.raw[r.RAW_LOADER]),
                 patch.object(r,'CALENDAR_SHA',self.calendar_sha),patch.object(r,'source_files',return_value=[self.source_path]),
                 patch.object(r,'load_baseline_helper',return_value=self.helper),patch.object(r,'verify_mapping_source'),
                 patch.object(r,'original_tools',return_value=tools),patch.object(r,'frozen_module',return_value=loader),
                 patch.object(r,'normalized_event_inputs',side_effect=self.normalized_load)]
        for context in patches:context.start();self.addCleanup(context.stop)

    def b(self,relative,payload):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        return self.sha(path)

    def j(self,relative,value):
        return self.b(relative,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())

    def z(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**value)

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh_cache(self):
        self.j(r.STUDY+'/CACHE.json',{name:self.sha(self.directory/name) for name in r.CACHE_FILES})

    def save_plan(self):self.plan_sha=self.j(r.STUDY+'/PLAN.json',self.plan)

    def normalized_load(self,reader,meta,panel,sessions,event_pins):
        reader.json(self.event_path)
        return self.normalized,self.provenance,{str(self.event_path)}

    def audit(self):return r.audit(self.root,self.directory/'PLAN.json',self.plan_sha)

    def test_complete_synthetic_io_closure_passes(self):
        report=self.audit()
        self.assertTrue(report['passed']);self.assertEqual(report['orders_checked'],488)
        self.assertFalse(report['returns_or_nav_computed']);self.assertEqual(report['event_sources_checked'],1)
        for key in r.CHECKS:self.assertTrue(report[key])
        self.assertEqual(report['source_cells_checked'],5978)

    def test_actual_phase_and_plan_hash_guards(self):
        with self.assertRaisesRegex(ValueError,'Exact PLAN'):r.audit(self.root,self.directory/'PLAN.json','0'*64)
        (self.directory/'RESULTS.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'precedes'):self.audit()

    def test_event_pin_must_also_belong_to_registered_all_pins(self):
        del self.plan['hashes'][str(self.event_path)];self.save_plan()
        with self.assertRaisesRegex(ValueError,'event input closure'):self.audit()

    def test_unbound_extra_read_is_rejected(self):
        extra=self.root/'event/undeclared.json';extra.write_text('{}')
        previous=self.normalized_load
        def load(*args):
            args[0].json(extra)
            return previous(*args)
        with patch.object(r,'normalized_event_inputs',side_effect=load):
            with self.assertRaisesRegex(ValueError,'Bound consumed artifact'):self.audit()

    def test_changed_original_array_fails_even_if_prepared_cache_rehashed(self):
        self.prepared['core']=self.prepared['core'].copy();self.prepared['core'][0,0]+=.001
        self.z(r.STUDY+'/DATA.npz',self.prepared);self.refresh_cache()
        with self.assertRaisesRegex(ValueError,'Original array byte identity'):self.audit()

    def test_three_arm_cache_is_not_accepted(self):
        with np.load(self.directory/'SCORES.npz') as data:
            scores={key:data[key].copy() for key in data.files if key!=r.ARMS[-1]}
        self.z(r.STUDY+'/SCORES.npz',scores);self.refresh_cache()
        with self.assertRaisesRegex(ValueError,'Four prepared arms'):self.audit()

    def test_corrupted_feature_audit_is_rejected(self):
        path=self.directory/'FEATURE_AUDIT.json';data=json.loads(path.read_text());data['cells'][0]['peer_count']-=1
        self.j(r.STUDY+'/FEATURE_AUDIT.json',data);self.refresh_cache()
        with self.assertRaisesRegex(ValueError,'comparison'):self.audit()

    def test_raw64_identity_and_source_closure_are_not_relaxed(self):
        self.plan['raw_source_hashes'].pop(next(iter(self.plan['raw_source_hashes'])));self.save_plan()
        with self.assertRaisesRegex(ValueError,'Original64'):self.audit()

    def test_independent_source_digest_must_be_in_plan(self):
        del self.plan['hashes'][str(self.source_path)];self.save_plan()
        with self.assertRaisesRegex(ValueError,'reviewer source closure'):self.audit()

    def test_feature_mask_or_score_tamper_fails(self):
        self.prepared['hand_peer_event10']=self.prepared['hand_peer_event10'].copy()
        self.prepared['hand_peer_event10'][0,0]=np.nan
        self.z(r.STUDY+'/DATA.npz',self.prepared);self.refresh_cache()
        with self.assertRaisesRegex(ValueError,'Feature availability'):self.audit()

    def test_score_matrix_dtype_is_part_of_contract(self):
        with np.load(self.directory/'SCORES.npz') as data:
            scores={key:data[key].copy() for key in data.files}
        scores[r.ARMS[1]]=scores[r.ARMS[1]].astype(np.float32)
        self.z(r.STUDY+'/SCORES.npz',scores);self.refresh_cache()
        with self.assertRaisesRegex(ValueError,'Score matrix schema'):self.audit()


if __name__ == '__main__':unittest.main(verbosity=2)
