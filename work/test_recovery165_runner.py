"""Synthetic only: frozen selector integration, support and fixed accounting."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import numpy as np
import pandas as pd
import recovery165_runner as r

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.base.module('recovery165_synthetic_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
EPISODE = r.base.module('recovery165_synthetic_episode', ROOT/'outputs/monthly_daily_episodes/episodes.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)
ORIGINAL_KEYS = ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow', 'eligible',
                 'effects', 'old_ml4', 'pre_score', 'forecast_available', 'rank_simple_excess',
                 'rank_log_excess', 'rank_vol_scaled', 'rank_loss_utility', 'raw_simple_excess',
                 'raw_log_excess', 'raw_vol_scaled', 'raw_loss_utility', 'extension_available',
                 'raw_source_log', 'model_available', 'local_available', 'raw_local_constant',
                 'raw_local_linear', 'rank_local_constant', 'rank_local_linear', 'paired_month_ready')


def fixture(n=4):
    names = ['S'+str(i) for i in range(32)] + ['ETF'+str(i) for i in range(17)]
    dates = [(date(2017, 1, 31)+timedelta(days=31*i)).isoformat() for i in range(n)]
    meta = dict(dates=dates, ends=dates, symbols=[names[:] for _ in dates], etfs=names[-17:], train_months=88)
    shape = (n, 49)
    a = {key: np.zeros(shape) for key in ORIGINAL_KEYS}
    a.update(baseline=np.tile(np.linspace(.2, .1, 49), (n, 1)),
             core_weight=np.full(n, .5), eligible=np.ones(shape, bool),
             state_calm10=np.ones(n, bool), state_weak_narrow=np.zeros(n, bool))
    a['core'] = .5*a['baseline']
    a['delta_recovery'] = np.tile(np.linspace(-.02, .02, 49), (n, 1))
    a['delta_control'] = -a['delta_recovery'].copy()
    return a, meta


class SyntheticBackend:
    adapter = ADAPTER

    def __init__(self):
        self.calls = []

    def pre(self, arrays, n):
        pre = np.zeros_like(arrays['baseline'])
        pre[:n] = arrays['baseline'][:n]
        return pre

    def selector(self, scores, good, symbols, effects, original, policy):
        self.calls.append(deepcopy(dict(scores=scores, good=good, symbols=symbols,
                                       effects=effects, original=original, policy=policy)))
        return EPISODE.select_boundary(scores, good, symbols, effects, original, policy)

    def baseline(self, a, m, n, policy):
        return self.adapter.score_and_choose('incumbent66', 0., np.zeros_like(a['baseline']), a, m, n,
                                             pre_score=self.pre(a, n), effects=a['effects'],
                                             episode_policy=policy, boundary_selector=self.selector)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in choice],
                     scores=[float(scores[i, j]) for j in choice]) for i, choice in enumerate(choices)]


class Scoring(unittest.TestCase):
    def test_actual_delta_applied_once_then_one_frozen_episode(self):
        a, m = fixture(); b = SyntheticBackend()
        choices, scores, _ = r.choices_all(a, m, 4, b, POLICY)
        self.assertEqual(len(b.calls), 3)
        for index, key in enumerate(r.HANDS, 1):
            call = b.calls[index]
            np.testing.assert_array_equal(call['scores'], b.pre(a, 4)+a[key])
            np.testing.assert_array_equal(call['original']['baseline'], a['baseline'])
            np.testing.assert_array_equal(call['effects'], a['effects'])
            expected = EPISODE.select_boundary(call['scores'], a['eligible'], m['symbols'],
                                                a['effects'], call['original'], POLICY)
            self.assertEqual(choices[r.NAMES[index]], expected[0])
            np.testing.assert_array_equal(scores[r.NAMES[index]], expected[1])

    def test_zero_deltas_exact_bytes_and_single_baseline_call(self):
        a, m = fixture(); b = SyntheticBackend()
        for key in r.HANDS: a[key][:] = 0
        choices, scores, _ = r.choices_all(a, m, 4, b, POLICY)
        self.assertEqual(len(b.calls), 1)
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name], choices[r.NAMES[0]])
            self.assertEqual(scores[name].tobytes(), scores[r.NAMES[0]].tobytes())

    def test_initial_and_zero_rows_exact_in_otherwise_active_arm(self):
        a, m = fixture(); m['dates'][0] = r.INITIAL_PARTIAL
        for key in r.HANDS: a[key][:2] = 0
        choices, scores, _ = r.choices_all(a, m, 4, SyntheticBackend(), POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name][:2], choices[r.NAMES[0]][:2])
            self.assertEqual(scores[name][:2].tobytes(), scores[r.NAMES[0]][:2].tobytes())

    def test_nonzero_initial_rejected(self):
        a, m = fixture(); m['dates'][0] = r.INITIAL_PARTIAL
        with self.assertRaisesRegex(ValueError, 'Initial partial'):
            r.choices_all(a, m, 4, SyntheticBackend(), POLICY)

    def test_full49_including_etfs_and_ineligible_are_not_reranked(self):
        a, m = fixture(); a['eligible'][:, 0] = False; b = SyntheticBackend()
        a['delta_recovery'][:] = .003  # Deliberately uncentered, including ETFs.
        choices, _, _ = r.choices_all(a, m, 4, b, POLICY)
        np.testing.assert_array_equal(b.calls[1]['scores'], b.pre(a, 4)+.003)
        self.assertTrue(all(0 not in row for row in choices[r.NAMES[1]]))

    def test_invalid_delta_values_and_bounds(self):
        for bad in (np.nan, np.inf, -.020001, .020001, True, 'bad'):
            a, m = fixture(); a['delta_recovery'] = a['delta_recovery'].astype(object); a['delta_recovery'][0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.choices_all(a, m, 4, SyntheticBackend(), POLICY)

    def test_wrong_shape_axes_boolean_mask_or_policy(self):
        for fault in ('shape', 'axes', 'mask', 'duplicates', 'policy'):
            a, m = fixture(); policy = POLICY
            if fault == 'shape': a['delta_recovery'] = a['delta_recovery'][:, :-1]
            if fault == 'axes': m['dates'] = m['dates'][:-1]
            if fault == 'mask': a['eligible'] = a['eligible'].astype(int)
            if fault == 'duplicates': m['symbols'][0][0] = m['symbols'][0][1]
            if fault == 'policy': policy = dict(POLICY, budget=.06)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                r.choices_all(a, m, 4, SyntheticBackend(), policy)

    def test_all123_prefixes_and_original27_are_unchanged(self):
        a, m = fixture(122); before = {k: v.tobytes() for k, v in a.items()}
        self.assertEqual(len(set(a)-set(r.EXTRA)), 27)
        with patch.object(r, 'choices_all', wraps=r.choices_all) as choose:
            orders, scores, audit = r.panels(a, m, SyntheticBackend(), POLICY)
        self.assertEqual([x.args[2] for x in choose.call_args_list], [122]+list(range(123)))
        self.assertEqual(sum(map(len, orders.values())), 366)
        self.assertEqual(len(audit), 122)
        self.assertEqual(before, {k: v.tobytes() for k, v in a.items()})

    def test_poison_future_deltas_and_metadata_ignored(self):
        a, m = fixture(); expected = r.choices_all(a, m, 1, SyntheticBackend(), POLICY)
        for key in r.HANDS:
            a[key] = a[key].astype(object); a[key][1:] = 'future'
        m['symbols'][1:] = [None]*3
        actual = r.choices_all(a, m, 1, SyntheticBackend(), POLICY)
        self.assertEqual(expected[0], actual[0])
        for name in r.NAMES: self.assertEqual(expected[1][name].tobytes(), actual[1][name].tobytes())

    def test_empty_prefix_ignores_all_future_values(self):
        a, m = fixture()
        for key in r.HANDS: a[key] = np.full(a[key].shape, 'future', dtype=object)
        choices, scores, audit = r.choices_all(a, m, 0, SyntheticBackend(), POLICY)
        self.assertEqual(audit, [])
        self.assertTrue(all(choices[name] == [] and scores[name].shape == (0, 49) for name in r.NAMES))

    def test_original_two_half_and_cash_choice_counts(self):
        for count in (1, 2):
            a, m = fixture(); a['eligible'][:] = False; a['eligible'][:, :count] = True
            choices, _, _ = r.choices_all(a, m, 4, SyntheticBackend(), POLICY)
            self.assertTrue(all(len(row) == (0 if count == 1 else 2) for name in r.NAMES for row in choices[name]))

    def test_backend_inherits_original_config4_and_price_parity(self):
        b = object.__new__(r.Backend)
        specs = [{'id': i} for i in range(5)]
        b.local = NS(configs=Mock(return_value=specs), modified_pre_score=Mock(return_value='pre'),
                     choices=Mock(return_value='choices'))
        b.selector = Mock()
        self.assertEqual(b.pre('arrays', 3), 'pre')
        b.local.modified_pre_score.assert_called_once_with(specs[4], 'arrays', 3)
        self.assertEqual(b.baseline('arrays', 'meta', 3, POLICY), 'choices')
        self.assertEqual(b.local.choices.call_args.args[:4], (specs[4], 'arrays', 'meta', 3))
        self.assertIs(r.Backend.price_parity, r.previous.Backend.price_parity)


def source_fixture(supported=True):
    _,meta=fixture(122)
    dates=[r.INITIAL_PARTIAL]+[str(x.date()) for x in pd.date_range('2016-08-31',periods=121,freq='BME')]
    meta.update(dates=dates,ends=dates)
    rows=[];deltas={key:np.zeros((122,49)) for key in r.EXTRA}
    for i,day in enumerate(dates):
        active=bool(i and supported);date=pd.Timestamp(day)
        raw=[((j%21)-10)/21. for j in range(32)]+[None]*17 if supported else [None]*49
        ranks=(pd.Series(raw).rank(method='average',pct=True)-.5).tolist()
        ranks=[None if pd.isna(x) else x for x in ranks]
        row=dict(date=day,symbols=meta['symbols'][i],
                 window_start=str((date-pd.Timedelta(days=360)).date()),
                 window_end=str((date-pd.Timedelta(days=60)).date()),
                 sector_etfs=['XLK']*32+[None]*17,
                 joint_down_counts=([38]*32 if supported else [19]*32)+[None]*17,
                 source_status=(['available']*32 if supported else ['insufficient_joint_down']*32)+['etf_not_applicable']*17,
                 common_support=[active]*32+[False]*17,initial_forced_neutral=i==0,
                 raw_recovery=raw.copy(),raw_control=raw.copy(),ranks_recovery=ranks.copy(),ranks_control=ranks.copy())
        if active:
            for key in deltas:deltas[key][i,:32]=.04*np.asarray(ranks[:32])
        rows.append(row)
    count=121*32 if supported else 0
    audit=dict(events=122,symbols_per_event=49,window_returns=210,window_prices=211,skip_sessions=42,
               start_count=189,recovery_horizon=21,censor_value=22,minimum_joint_down=20,
               trading_eligibility_modified=False,outcomes_used=False,event_history=rows,
               common_support_cells=count,neutral_cells=122*49-count)
    return audit,meta,deltas


class SourceSupport(unittest.TestCase):
    def test_real_kernel_synthetic_sources_with_original32_stocks17_etfs(self):
        import recovery165_kernel as kernel
        sessions=pd.bdate_range('2013-01-01','2027-01-31')
        _,meta,_=source_fixture();names=meta['symbols'][0]
        pattern=[100.,80.,100.,100.,100.]
        prices=pd.DataFrame({name:np.resize(pattern,len(sessions)) for name in names},index=sessions)
        prices['XLK']=np.resize([100.,70.,80.,90.,100.],len(sessions))
        mapping=dict(etfs=meta['etfs'],by_date={day:{name:None if name in meta['etfs'] else 'XLK' for name in names} for day in meta['dates']})
        delta,audit=kernel.build_features(prices,mapping,list(zip(meta['dates'],meta['symbols'])),[str(x.date()) for x in sessions])
        support=r.source_support(audit,meta,delta)
        self.assertTrue(support['passed']);self.assertEqual(support['common_support_cells'],121*32)
        arrays,_=fixture(122);arrays.update(delta);r.choices_all(arrays,meta,122,SyntheticBackend(),POLICY)

    def test_exact_era_counts_include_neutral_etfs_and_no_threshold(self):
        audit,meta,deltas=source_fixture();result=r.source_support(audit,meta,deltas)
        self.assertTrue(result['passed']);self.assertIsNone(result['coverage_threshold'])
        eras=('full122','first88','early53','recent35','later34')
        self.assertEqual([result['coverage'][key]['candidate_cells'] for key in eras],[5978,4312,2597,1715,1666])
        self.assertEqual([result['coverage'][key]['common_support_cells'] for key in eras],[3872,2784,1664,1120,1088])
        audit,meta,deltas=source_fixture(False)
        result=r.source_support(audit,meta,deltas)
        self.assertTrue(result['passed']);self.assertEqual(result['common_support_cells'],0)

    def test_initial_keeps_measurements_but_zero_deltas(self):
        audit,meta,deltas=source_fixture()
        self.assertTrue(all(v is not None for v in audit['event_history'][0]['raw_recovery'][:32]))
        self.assertTrue(r.source_support(audit,meta,deltas)['passed'])
        audit['event_history'][0]['common_support'][0]=True
        with self.assertRaises(ValueError):r.source_support(audit,meta,deltas)

    def test_shared_stock_ties_singleton_and_zero_rank_are_still_supported(self):
        for raw,ranks in (([.2],[.5]),([.1,.2],[0.,.5]),([.1,.1,.2],[0.,0.,.5]),([0.],[.5])):
            audit,meta,deltas=source_fixture(False);row=audit['event_history'][1];n=len(raw)
            for suffix,key in zip(('recovery','control'),r.EXTRA):
                row['raw_'+suffix][:n]=raw;row['ranks_'+suffix][:n]=ranks;deltas[key][1,:n]=.04*np.asarray(ranks)
            row['common_support'][:n]=[True]*n;row['joint_down_counts'][:n]=[20]*n;row['source_status'][:n]=['available']*n
            audit['common_support_cells']=n;audit['neutral_cells']-=n
            self.assertEqual(r.source_support(audit,meta,deltas)['common_support_cells'],n)

    def test_unshared_raw_availability_rejected_even_initial(self):
        for index in (0,1):
            audit,meta,deltas=source_fixture();row=audit['event_history'][index]
            row['raw_control'][0]=None;row['ranks_control'][0]=None
            with self.assertRaises(ValueError):r.source_support(audit,meta,deltas)

    def test_invalid_or_unknown_sources_can_be_neutral(self):
        for reason in ('invalid_stock_path','invalid_sector_path','insufficient_history','missing_sector_mapping'):
            audit,meta,deltas=source_fixture(False);row=audit['event_history'][1]
            row['source_status'][0]=reason;row['joint_down_counts'][0]=None
            if reason=='missing_sector_mapping':row['sector_etfs'][0]=None
            self.assertTrue(r.source_support(audit,meta,deltas)['passed'])

    def test_missing_initial_wrong_sign_double_dose_dtype_rejected(self):
        for fault in ('missing','initial','sign','dose','dtype','etf'):
            audit,meta,deltas=source_fixture(fault!='missing')
            if fault=='missing':deltas[r.EXTRA[0]][5,0]=.001
            if fault=='initial':deltas[r.EXTRA[0]][0,0]=.001
            if fault=='sign':deltas[r.EXTRA[0]]*=-1
            if fault=='dose':deltas[r.EXTRA[1]]*=.5
            if fault=='dtype':deltas[r.EXTRA[1]]=deltas[r.EXTRA[1]].astype(np.float32)
            if fault=='etf':deltas[r.EXTRA[1]][5,40]=.001
            with self.subTest(fault=fault),self.assertRaises(ValueError):r.source_support(audit,meta,deltas)

    def test_audit_count_mapping_status_axes_and_rank_tamper_rejected(self):
        for fault in ('short','date','window','countlow','counthigh','countbool','status','mapping','etfmap',
                      'mask','rawnan','rawoutside','rankmissing','rankwrong','aggregate','outcomes','eligibility'):
            audit,meta,deltas=source_fixture();row=audit['event_history'][1]
            if fault=='short':audit['event_history'].pop()
            if fault=='date':row['date']='wrong'
            if fault=='window':row['window_end']=None
            if fault=='countlow':row['joint_down_counts'][0]=19
            if fault=='counthigh':row['joint_down_counts'][0]=190
            if fault=='countbool':row['joint_down_counts'][0]=True
            if fault=='status':row['source_status'][0]='mystery'
            if fault=='mapping':row['sector_etfs'][0]='SPY'
            if fault=='etfmap':row['sector_etfs'][40]='XLK'
            if fault=='mask':row['common_support'][0]=1
            if fault=='rawnan':row['raw_recovery'][0]=float('nan')
            if fault=='rawoutside':row['raw_control'][0]=1.1
            if fault=='rankmissing':row['ranks_control'][0]=None
            if fault=='rankwrong':row['ranks_recovery']=np.linspace(-.5,.5,49).tolist()
            if fault=='aggregate':audit['common_support_cells']-=1
            if fault=='outcomes':audit['outcomes_used']=True
            if fault=='eligibility':audit['trading_eligibility_modified']=True
            with self.subTest(fault=fault),self.assertRaises(ValueError):r.source_support(audit,meta,deltas)


class SectorMapping(unittest.TestCase):
    def backend(self):
        b=object.__new__(r.Backend);b.root=ROOT
        _,meta=fixture(4)
        sectors=pd.DataFrame('Technology',index=pd.to_datetime(meta['dates']),columns=meta['symbols'][0][:32])
        def mapping(day,names,sectors,etfs):return {name:None if name in etfs else 'XLK' for name in names}
        strict=NS(__file__=str(ROOT/'outputs/monthly_strict_relative/strict_study.py'),
                  SECTOR={str(i):name for i,name in enumerate(r.SECTOR_ETFS)},FOREIGN={},benchmark_map=Mock(side_effect=mapping))
        b.local=NS(ex=NS(t=NS(strict=strict)));b.panel={'sectors':sectors}
        return b,meta,strict,sectors

    def test_frozen_mapper_receives_original_metadata_and_sector_frame(self):
        b,meta,strict,sectors=self.backend();result=b.sector_mapping(meta)
        self.assertEqual(result['etfs'],meta['etfs']);self.assertIsNot(result['etfs'],meta['etfs'])
        self.assertEqual(len(result['by_date']),4);self.assertEqual(strict.benchmark_map.call_count,4)
        for call in strict.benchmark_map.call_args_list:self.assertIs(call.args[2],sectors)
        for row in result['by_date'].values():
            self.assertTrue(all(row[name] is None for name in meta['etfs']))

    def test_mapper_hash_set_neutrality_and_axes_are_checked(self):
        for fault in ('hash','file','allowed','foreign','etf','shape'):
            b,meta,strict,sectors=self.backend()
            if fault=='file':strict.__file__='/tmp/wrong.py'
            if fault=='allowed':strict.SECTOR['wrong']='SPY'
            if fault=='foreign':strict.FOREIGN['wrong']='SPY'
            if fault=='etf':strict.benchmark_map.side_effect=lambda day,names,sectors,etfs:{name:'XLK' for name in names}
            if fault=='shape':meta['symbols'][0]=meta['symbols'][0][:-1]
            with self.subTest(fault=fault),patch.object(r.base,'sha',return_value='bad' if fault=='hash' else r.MAPPING_SHA):
                with self.assertRaises(ValueError):b.sector_mapping(meta)


class Selection(unittest.TestCase):
    def rows(self):
        return [dict(name=name,training=dict(cagr=.2,sharpe=1.,max_drawdown=-.2),
                     early=dict(cagr=.2),recent=dict(cagr=.2)) for name in r.NAMES]

    def select(self,rows,opened=None,support=None,changed=None):
        return r.select_first88(rows,self.rows() if opened is None else opened,
                                {'passed':True} if support is None else support,
                                {name:{'passed':True} for name in r.NAMES[1:]} if changed is None else changed)

    def test_recovery_close_gain_and_open_noninferiority(self):
        rows=self.rows();rows[1]['training']['cagr']=.2049
        self.assertFalse(self.select(rows)['admitted'])
        rows[1]['training']['cagr']=.205
        self.assertEqual(self.select(rows)['primary'],r.NAMES[1])
        opened=self.rows();opened[1]['training']['cagr']=.199
        self.assertFalse(self.select(rows,opened)['admitted'])

    def test_every_guard_against_incumbent_and_matched_control(self):
        for phase in ('close','open'):
            for reference in (0,2):
                for field,improvement in (('sharpe',.051),('max_drawdown',.031),('early',.026),('recent',.026)):
                    rows,opened=self.rows(),self.rows();rows[1]['training']['cagr']=.21
                    target=rows if phase=='close' else opened
                    if field in ('early','recent'):target[reference][field]['cagr']+=improvement
                    else:target[reference]['training'][field]+=improvement
                    with self.subTest(phase=phase,reference=reference,field=field):self.assertFalse(self.select(rows,opened)['admitted'])
        for phase in ('close','open'):
            rows,opened=self.rows(),self.rows();rows[1]['training']['cagr']=.21
            target=rows if phase=='close' else opened;target[2]['training']['cagr']=.22
            self.assertEqual(self.select(rows,opened)['primary'],r.NAMES[0])

    def test_control_never_selected_even_best_result_and_activity_rejected(self):
        rows=self.rows();rows[2]['training']['cagr']=.9
        result=self.select(rows)
        self.assertEqual(result['primary'],r.NAMES[0]);self.assertFalse(result['admitted'])
        self.assertEqual(result['guards'][r.NAMES[2]],{'promotable':False})
        rows[1]['training']['cagr']=.9
        self.assertEqual(self.select(rows)['primary'],r.NAMES[1])

    def test_missing_support_changes_and_nonfinite_metrics_reject(self):
        rows=self.rows();rows[1]['training']['cagr']=.21
        self.assertFalse(self.select(rows,support={'passed':False})['admitted'])
        self.assertFalse(self.select(rows,changed={name:{'passed':False} for name in r.NAMES[1:]})['admitted'])
        for index in (0,1,2):
            changed=deepcopy(rows);changed[index]['training']['cagr']=np.inf
            self.assertFalse(self.select(changed)['admitted'])
        # A control need not change holdings to serve as a nonpromotable comparator.
        changed={r.NAMES[1]:{'passed':True},r.NAMES[2]:{'passed':False}}
        self.assertTrue(self.select(rows,changed=changed)['admitted'])

    def test_later34_ignored_and_fixed_order(self):
        rows=self.rows();rows[1]['training']['cagr']=.21
        rows[1]['later34']={'cagr':-1.};rows[2]['later34']={'cagr':999.}
        result=self.select(rows)
        self.assertEqual(result['primary'],r.NAMES[1]);self.assertEqual(result['selection_count'],1)
        self.assertFalse(result['later34_used_for_selection'])
        with self.assertRaises(ValueError):self.select(list(reversed(rows)))

    def test_changes_require_three_heldsets_and_both_training_eras(self):
        orders={name:[dict(symbols=['S0','S1','S2']) for _ in range(122)] for name in r.NAMES}
        name=r.NAMES[1];orders[name][0]['symbols'].reverse();orders[name][100]['symbols']=['S0','S1','S3']
        self.assertEqual(r.changed_support(orders,name)['months'],[])
        for index in (1,2,3):orders[name][index]['symbols']=['S0','S1','S3']
        self.assertFalse(r.changed_support(orders,name)['passed'])
        orders[name][53]['symbols']=['S0','S1','S3']
        self.assertTrue(r.changed_support(orders,name)['passed'])
        self.assertEqual(r.changed_support(orders,name)['months'],[1,2,3,53])
        orders[name][0]['symbols']=['S0','S1','S1']
        with self.assertRaises(ValueError):r.changed_support(orders,name)


class AccountingWiring(unittest.TestCase):
    def backend(self):
        _, m = fixture(122)
        orders = {name: [dict(date=d, symbols=['S0', 'S1']) for d in m['dates']] for name in r.NAMES}
        index = pd.date_range(m['dates'][0], m['ends'][-1])
        eq = pd.Series(np.linspace(100., 120., len(index)), index=index)
        metrics = dict(training={'cagr': .1}, early={'cagr': .1}, recent={'cagr': .1})
        rp = NS(paths=Mock(return_value=(None, None, None, None)), fast_returns=Mock(return_value=np.zeros(10)),
                stats_rr=Mock(return_value={'cagr': .1}))
        replay = Mock(side_effect=lambda *a, **kw: ({'annual_returns': [{'year': 2016}, {'year': 2026}]}, eq.copy(), [], []))
        b = object.__new__(r.Backend)
        b.local = NS(rp=rp, old=NS(session=NS(measure=Mock(return_value=metrics))),
                     ex=NS(e=NS(engine=NS(replay=replay))),
                     arch=NS(old=NS(prior=NS(old=NS(previous=NS(cuts=Mock(return_value={})))))))
        return b, m, orders, rp, replay

    def test_training_only_first88_no_fold_reselection(self):
        b, m, orders, rp, replay = self.backend(); rows, opened = b.training(orders, m, None, None)
        self.assertEqual(rp.paths.call_count, 1); self.assertEqual(rp.paths.call_args.args[2], 88)
        self.assertEqual(replay.call_count, 3)
        self.assertEqual([x['name'] for x in rows], list(r.NAMES))
        self.assertEqual([x['name'] for x in opened], list(r.NAMES))
        for call in replay.call_args_list:
            self.assertEqual(len(call.args[2]), 88); self.assertEqual(call.kwargs['cost'], .0005)
            self.assertEqual(call.kwargs['execution'], 'open')
            self.assertEqual(call.kwargs['end'], pd.Timestamp(m['ends'][87]))

    def test_diagnostics_exact_nine_paths_and27_artifacts(self):
        b, m, orders, _, replay = self.backend(); written = []
        result = b.diagnostics(orders, m, None, None, lambda name, value: written.append(name))
        self.assertEqual(replay.call_count, 9); self.assertEqual(len(set(written)), 27)
        self.assertEqual(set(result), set(r.NAMES))
        for name in r.NAMES: self.assertEqual(set(result[name]), {'same_close', 'next_open_zero', 'next_open_cost'})
        self.assertTrue(all(len(call.args[2]) == 122 for call in replay.call_args_list))

    def test_contracts_match_fixed_interface(self):
        contract=r.measurement_contract()
        self.assertEqual(r.EXTRA, ('delta_recovery','delta_control'))
        self.assertEqual(contract['max_abs_delta'],.02)
        self.assertEqual(contract['adjusted_price_offsets'],[-252,-42])
        self.assertEqual(contract['window_prices'],211)
        self.assertEqual(contract['start_indices'],[1,189])
        self.assertEqual(contract['recovery_horizon'],21)
        self.assertEqual(contract['censor_value'],22)
        self.assertEqual(contract['minimum_joint_down'],20)
        self.assertEqual(contract['prepared_arrays'],29)
        self.assertEqual(r.NAMES,('incumbent_local67','recovery_race','impact_control'))
        self.assertEqual(r.selection_contract()['nonpromotable_control'],r.NAMES[2])
        self.assertIsNone(r.selection_contract()['minimum_common_support_cells'])

    def test_shared_import_implementation_closure_is_pinned(self):
        # Hash implementation files only; source_files never loads market arrays.
        pins = r.source_files(ROOT)
        for name in ('recovery165_kernel.py', 'test_recovery165_kernel.py', 'recovery165_runner.py',
                     'test_recovery165_runner.py', 'basic_shares159_host.py',
                     'basic_shares159_runner_draft.py', 'after158_basic_shares_kernel.py',
                     'monthly_annual_lag158_runner.py', 'monthly_annual_lag158_model.py',
                     'monthly_formation142_runner.py', 'monthly_formation142_features.py',
                     'monthly_formation142_selector.py'):
            self.assertEqual(pins[str(ROOT/'work'/name)], r.base.sha(ROOT/'work'/name))


if __name__ == '__main__':
    unittest.main()
