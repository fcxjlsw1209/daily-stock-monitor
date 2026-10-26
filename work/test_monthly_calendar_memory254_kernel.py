"""Synthetic-only calendar memory source/selection contracts."""
from copy import deepcopy
from functools import lru_cache
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_calendar_memory254_kernel as kernel


@lru_cache(maxsize=1)
def study():
    return kernel._module('memory254_synthetic_local67',
        kernel.ROOT/'outputs/monthly_local_linear70/study.py')


def fixture():
    s = study(); n = 3; w = 49; z = np.zeros((n,w)); f = np.zeros((n,w),bool)
    names = ['S%02d'%i for i in range(w)]; calm = np.array([True, False, True])
    core = np.tile(np.linspace(-.2,.2,w),(n,1)); cw = np.where(calm,.5,.52)[:,None]
    a = dict(baseline=core+(1-cw)*.12, core=core, core_weight=cw,
        eligible=np.ones((n,w),bool), effects=np.tile([0.,0.,0.,0.,.4]+[0.]*44,(n,1)),
        state_calm10=calm, state_weak_narrow=np.array([False,True,False]),
        pre_score=np.tile(.31-.005*np.arange(w),(n,1)), old_ml4=z.copy(),
        rank_log_excess=z.copy(), extension_available=f.copy(), forecast_available=f.copy(),
        local_available=f.copy(), model_available=f.copy(), raw_local_constant=z.copy(),
        raw_local_linear=z.copy(), raw_source_log=z.copy())
    m = dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
        ends=['2016-08-31','2016-09-30','2016-10-31'],
        symbols=[names.copy() for _ in range(n)], etfs=names[-17:])
    dates = pd.bdate_range('2010-01-04','2016-11-03'); t = np.arange(len(dates),dtype=float)
    px = pd.DataFrame({name:100*np.exp((.0007+j*.000002)*t + .006*np.sin(t/(9+j%5)))
                      for j,name in enumerate(names)},index=dates)
    px['SPY'] = 100*np.exp(.0002*t+.002*np.sin(t/17))
    px['QQQ'] = 100*np.exp(.0003*t+.003*np.sin(t/19))
    high = px*1.06; low = px*.94
    high.loc[:,names[:4]] = px[names[:4]]*1.02
    low.loc[:,names[:4]] = px[names[:4]]*.999
    panel = dict(adj=px, open=px*1.001, close=px.copy(), high=high, low=low,
                 volume=pd.DataFrame(100000.,index=dates,columns=px.columns))
    calendar = dates.strftime('%Y-%m-%d').tolist()
    return s,a,m,panel,calendar


def physical(args, n):
    s,a,m,p,c = args; mm = deepcopy(m)
    for k in ('dates','ends','symbols'): mm[k] = mm[k][:n]
    aa = {k:v[:n].copy() for k,v in a.items()}
    pp = {k:(v.loc[:mm['dates'][-1]].copy() if n else v.iloc[:0].copy()) for k,v in p.items()}
    return s,aa,mm,pp,c.copy()


def build(args,n=None):
    return kernel.build_features(args[3],args[4],args[2],n)


def run(args,n=None,data=None,**kw):
    n = len(args[2]['dates']) if n is None else n
    data = build(args,n)[0] if data is None else data
    return kernel.decisions(args[0],args[1],args[2],data,n,**kw)


def same(case,left,right):
    case.assertEqual(tuple(left),tuple(right))
    for k in left:
        case.assertEqual(left[k].dtype,right[k].dtype)
        np.testing.assert_array_equal(left[k],right[k],err_msg=k)


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = fixture(); cls.data,cls.source_audit = build(cls.args)
        cls.result = run(cls.args,data=cls.data)

    def copied(self):
        s,a,m,p,c = self.args
        return s,deepcopy(a),deepcopy(m),{k:v.copy() for k,v in p.items()},c.copy()

    def test_source_schema_no_object_and_counts(self):
        d,a=self.data,self.source_audit
        self.assertEqual(tuple(d),kernel.DATA_KEYS)
        self.assertTrue(all(v.dtype.kind!='O' for v in d.values()))
        self.assertEqual(tuple(a['counts']),kernel.SOURCE_COUNT_FIELDS)
        self.assertEqual(a['counts']['query_cells'],147)
        self.assertFalse(a['candidate_selection_computed']);self.assertFalse(a['portfolio_returns_computed'])
        self.assertGreater(a['counts']['anchors'],20);json.dumps(a,allow_nan=False)

    def test_official_monthends_no_lockout_and_variable_paths(self):
        d=self.data;c=self.args[4]
        for i in range(len(d['event_dates'])):
            t=int(d['event_anchor_index'][i]);day=str(d['event_dates'][i])
            self.assertEqual(day,c[t]);self.assertNotEqual(day[:7],c[t+1][:7])
            self.assertEqual(str(d['event_entry'][i]),c[t+1]);self.assertEqual(str(d['event_exit21'][i]),c[t+21])
            want=str(pd.Timestamp(day).to_period('M')+1)
            self.assertEqual(str(d['event_exit_calendar'][i]),[x for x in c if x[:7]==want][-1])
        ids=np.flatnonzero(d['event_symbol']=='S00')
        self.assertTrue((np.diff(d['event_anchor_index'][ids])<21).any())
        self.assertTrue((d['event_exit21']!=d['event_exit_calendar']).any())

    def test_calendar_holiday_monthend_and_complete_witness(self):
        c=pd.bdate_range('2024-03-01','2024-05-02').strftime('%Y-%m-%d').tolist();c.remove('2024-03-29')
        lookup={d:i for i,d in enumerate(c)};last={d[:7]:d for d in c}
        p=kernel._plan(lookup['2024-03-28'],c,last,lookup)
        self.assertEqual(p[0],'2024-04-01');self.assertEqual(p[2],'2024-04-30')
        short=c[:c.index('2024-04-30')+1];look={d:i for i,d in enumerate(short)}
        with self.assertRaisesRegex(ValueError,'witness'):
            kernel._plan(look['2024-03-28'],short,{d[:7]:d for d in short},look)

    def test_fixed21_missing_future_calendar_hardfail(self):
        args=self.copied();args[4][:]=args[4][:args[4].index('2016-09-30')+2]
        with self.assertRaisesRegex(ValueError,'Calendar'):build(args)

    def test_original_query_features_exact_layout(self):
        p={k:v.loc[:'2016-09-30'] for k,v in self.args[3].items()}
        sh,mk,sig,good,vol=kernel.ep.features(p)
        for i,day in enumerate(self.args[2]['dates']):
            ss=self.args[2]['symbols'][i]
            np.testing.assert_array_equal(self.data['query_shape'][i],np.column_stack([x.loc[day,ss] for x in sh]))
            np.testing.assert_array_equal(self.data['query_market'][i],mk.loc[day])
            np.testing.assert_array_equal(self.data['query_signal'][i],sig.loc[day,ss])
            np.testing.assert_array_equal(self.data['query_vol'][i],vol.loc[day,ss])

    def test_both_path_utility_exact_original_helper(self):
        d=self.data;px=self.args[3]['adj'];opening=px*self.args[3]['open']/self.args[3]['close']
        for i in np.flatnonzero(d['event_common_valid'])[::31]:
            sym=str(d['event_symbol'][i]);start=px.index.get_loc(pd.Timestamp(d['event_dates'][i]))
            for kind in ('21','_calendar'):
                end=px.index.get_loc(pd.Timestamp(d['event_exit'+kind][i]))
                y,u=kernel.ep.label_path(px[sym].to_numpy(),opening[sym].to_numpy(),px.SPY.to_numpy(),opening.SPY.to_numpy(),start,end-start)
                self.assertEqual(y,d['event_excess'+kind][i]);self.assertEqual(u,d['event_utility'+kind][i])

    def test_future_label_missing_preserves_anchor_and_exit(self):
        d=self.data;rows=d['event_dates']==np.datetime64('2016-09-30')
        self.assertTrue(rows.any());self.assertFalse(d['event_common_valid'][rows].any())
        self.assertTrue((d['event_exit_calendar'][rows]==np.datetime64('2016-10-31')).all())
        self.assertTrue(np.isnan(d['event_utility_calendar'][rows]).all())

    def test_label_nonpositive_nonfinite_missing_unavailable(self):
        args=self.copied();p=args[3];c=args[4];t=c.index('2016-01-29')
        for v in (np.nan,0.,-1.,np.inf):
            pp={k:f.copy() for k,f in p.items()};pp['adj'].at[pd.Timestamp(c[t+2]),'S00']=v
            op=pp['adj']*pp['open']/pp['close']
            y,u,valid=kernel._label(pp,op,'S00',t,t+21,c)
            self.assertFalse(valid);self.assertTrue(np.isnan(y) and np.isnan(u))

    def test_absent_row_equals_explicit_missing_without_clock_compression(self):
        args=self.copied();missing=pd.Timestamp('2016-07-05')
        dropped=args[:3]+({k:v.drop(index=missing) for k,v in args[3].items()},args[4])
        nanargs=self.copied()
        for v in nanargs[3].values():v.loc[missing,:]=np.nan
        left=build(dropped)[0];right=build(nanargs)[0];same(self,left,right)
        before=self.data['event_dates']<np.datetime64(missing,'D')
        self.assertEqual(set(zip(left['event_symbol'],left['event_dates'])) & set(zip(self.data['event_symbol'][before],self.data['event_dates'][before])),set(zip(self.data['event_symbol'][before],self.data['event_dates'][before])))

    def test_missing_query_row_hardfail(self):
        args=self.copied();day=pd.Timestamp(args[2]['dates'][1])
        for k in args[3]:args[3][k]=args[3][k].drop(index=day)
        with self.assertRaisesRegex(ValueError,'Missing query'):build(args)

    def test_structural_type_and_axis_errors(self):
        for bad in (True,'100'):
            args=self.copied();args[3]['adj']=args[3]['adj'].astype(object)
            args[3]['adj'].iat[0,0]=bad
            with self.assertRaisesRegex(ValueError,'nonboolean'):build(args)
        args=self.copied();args[3]['open']=args[3]['open'].iloc[1:]
        with self.assertRaisesRegex(ValueError,'axes'):build(args)

    def test_missing_benchmark_column_hardfail(self):
        args=self.copied()
        for k in args[3]:args[3][k]=args[3][k].drop(columns='SPY')
        with self.assertRaisesRegex(ValueError,'column'):build(args)

    def test_zero_volume_keeps_original_feature_semantics(self):
        args=self.copied()
        for f in args[3].values():self.assertTrue(isinstance(f,pd.DataFrame))
        args[3]['volume'].iloc[10:20,0]=0.
        d,_=build(args);self.assertEqual(d['query_dates'].tolist(),self.data['query_dates'].tolist())

    def test_source_does_not_read_future_prices_or_metadata(self):
        args=self.copied();base=build(args,2)
        args[2]['dates'][2]='bad';args[2]['symbols'][2]=object()
        for k,v in args[3].items():
            v=v.astype(object);v.loc['2016-09-01':,:]='future_bad';args[3][k]=v
        got=build(args,2);same(self,base[0],got[0]);self.assertEqual(base[1],got[1])

    def test_n0_empty_sources_calendar_and_exact_shapes(self):
        args=physical(self.args,0);data,a=kernel.build_features({},[],args[2],0)
        self.assertEqual(data['query_shape'].shape,(0,49,8));self.assertEqual(len(data['event_dates']),0)
        out=run(args,data=data);self.assertTrue(all(x==[] for x in out[0].values()))
        self.assertEqual(out[4]['counts']['events'],0)

    def test_original_incumbent_exact_and_pre_unchanged_all_arms(self):
        s,a,m,_,_=self.args;out=self.result
        ch,sc=s.choices(s.configs()[4],a,m,3,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(out[0][kernel.ARMS[0]],s.rp.make_orders(ch,sc,m))
        self.assertEqual(out[1][kernel.ARMS[0]].tobytes(),sc.tobytes())
        pre=s.modified_pre_score(s.configs()[4],a,3)
        for k in kernel.ARMS:self.assertEqual(out[2][k].tobytes(),pre.tobytes())

    def test_partial_exact_all_effects_orders_scores(self):
        o,s,p,e,a=self.result
        for arm in kernel.ARMS:
            self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0])
            self.assertEqual(s[arm][0].tobytes(),s[kernel.ARMS[0]][0].tobytes())
            self.assertEqual(e[arm][0].tobytes(),self.args[1]['effects'][0].tobytes())

    def test_each_arm_one_selector_no_new_fit_counts(self):
        with patch.object(self.args[0].ep,'select_boundary',wraps=self.args[0].ep.select_boundary) as ep:
            out=run(self.args,data=self.data)
        self.assertEqual(ep.call_count,3);self.assertEqual(tuple(out[4]['counts']),kernel.COUNT_FIELDS)
        self.assertTrue(all(tuple(e)==kernel.AUDIT_EVENT_KEYS for e in out[4]['events']))
        self.assertEqual(out[4]['counts']['new_model_fits'],0);json.dumps(out[4],allow_nan=False)

    def test_gate_sequence_source_then_inc_then_effects(self):
        marks=[]
        def source(d):
            self.assertEqual(ep.call_count,0);self.assertEqual(est.call_count,0)
            marks.append('source');return {'passed':True}
        def inc(o,s,p):
            self.assertEqual(ep.call_count,1);self.assertEqual(est.call_count,0)
            self.assertEqual(o,self.result[0][kernel.ARMS[0]])
            marks.append('inc');return {'passed':True}
        with patch.object(self.args[0].ep,'select_boundary',wraps=self.args[0].ep.select_boundary) as ep,patch.object(kernel,'_forecast',wraps=kernel._forecast) as est:
            run(self.args,data=self.data,source_validator=source,incumbent_validator=inc,expected_incumbent_orders=self.result[0][kernel.ARMS[0]])
        self.assertEqual(marks,['source','inc'])

    def test_source_gate_failure_precedes_all_selectors(self):
        with patch.object(self.args[0].ep,'select_boundary',side_effect=AssertionError('selector')),self.assertRaisesRegex(ValueError,'source gate'):
            run(self.args,data=self.data,source_validator=lambda d:{'passed':False})

    def test_inc_gate_and_expected_orders_failure_before_effects(self):
        with patch.object(kernel,'_forecast',side_effect=AssertionError('effects')):
            with self.assertRaisesRegex(ValueError,'incumbent gate'):
                run(self.args,data=self.data,incumbent_validator=lambda *a:{'passed':False})
            with self.assertRaisesRegex(ValueError,'ordered choices'):
                run(self.args,data=self.data,expected_incumbent_orders=[])

    def test_no_signal_no_history_and_invalid_queries_zero_not_excluded(self):
        for kind in ('signal','history','shape'):
            d=deepcopy(self.data)
            if kind=='signal':d['query_signal'][1:]=0
            elif kind=='history':
                for key in kernel.EVENT_DATA_KEYS:d[key]=d[key][:0]
            else:d['query_shape'][1:]=np.nan
            out=run(self.args,data=d)
            for arm in kernel.ARMS[1:]:
                self.assertFalse(out[3][arm][1:].any())
                self.assertTrue(all(len(row['symbols'])==3 for row in out[0][arm]))

    def test_two_and_less_than_two_cash_unchanged(self):
        args=self.copied();args[1]['eligible'][1:]=False;args[1]['eligible'][1,:2]=True;args[1]['eligible'][2,0]=True
        out=run(args,data=self.data)
        for arm in kernel.ARMS:
            self.assertEqual(len(out[0][arm][1]['symbols']),2);self.assertEqual(out[0][arm][2]['symbols'],[])

    def test_original_top_two_lock_and_ceiling_preserved(self):
        o,s,p,e,a=self.result
        for i,event in enumerate(a['events']):
            locked=set(event['pre_order'][:2])
            for k in ('incumbent_choices','candidate_choices','control_choices'):
                self.assertTrue(locked <= set(event[k]))
                self.assertTrue(set(event[k]) <= set(event['pre_order'][:5]))

    def test_common_mask_withholds_one_valid_arm(self):
        d=deepcopy(self.data);mature=d['event_common_maturity']<=d['query_dates'][-1]
        d['event_valid_calendar'][mature]=False;d['event_common_valid'][mature]=False
        d['event_excess_calendar'][mature]=np.nan;d['event_utility_calendar'][mature]=np.nan
        out=run(self.args,data=d)
        for arm in kernel.ARMS[1:]:self.assertFalse(out[3][arm][1:].any())

    def test_common_maturity_withholds_earlier_exit(self):
        d=deepcopy(self.data);mask=d['event_dates']<d['query_dates'][1]
        d['event_exit21'][mask]=np.datetime64('2016-09-01');d['event_exit_calendar'][mask]=np.datetime64('2016-08-31');d['event_common_maturity'][mask]=np.datetime64('2016-09-01')
        out=run(self.args,2,data=d)
        for arm in kernel.ARMS[1:]:self.assertFalse(out[3][arm][1].any())

    def test_unmatured_future_label_poison_no_effect(self):
        n=2;base=run(self.args,n,data=self.data);d=deepcopy(self.data)
        future=d['event_common_maturity']>d['query_dates'][n-1]
        for key in kernel.LABEL_KEYS:d[key][future]=np.inf
        got=run(self.args,n,data=d)
        self.assertEqual(base[0],got[0]);self.assertEqual(base[4],got[4])
        for i in (1,2,3):same(self,base[i],got[i])

    def test_mature_mask_and_label_tamper_rejected(self):
        for kind in ('mask','label','maturity'):
            d=deepcopy(self.data);i=int(np.flatnonzero(d['event_common_valid'])[0])
            if kind=='mask':d['event_common_valid'][i]=False
            elif kind=='label':d['event_utility21'][i]=2.
            else:d['event_common_maturity'][i]+=np.timedelta64(1,'D')
            with self.assertRaises(ValueError):run(self.args,data=d)

    def test_cached_logical_and_physical_prefix_predictions_exact(self):
        for n in range(4):
            physical_args=physical(self.args,n);d,_=build(physical_args)
            left=run(self.args,n,data=self.data);right=run(physical_args,data=d)
            self.assertEqual(left[0],right[0]);self.assertEqual(left[4],right[4])
            for i in (1,2,3):same(self,left[i],right[i])
            for arm in kernel.ARMS:self.assertEqual(left[0][arm],self.result[0][arm][:n])

    def test_dtype_and_query_identity_tamper_rejected(self):
        for kind in ('signal_dtype','symbol','date'):
            d=deepcopy(self.data)
            if kind=='signal_dtype':d['query_signal']=d['query_signal'].astype(float)
            elif kind=='symbol':d['query_symbols'][1,0]='WRONG'
            else:d['query_dates'][1]-=np.timedelta64(1,'D')
            with self.assertRaises(ValueError):run(self.args,data=d)

    def test_input_cache_panel_and_data_not_mutated(self):
        args=self.copied();a=deepcopy(args[1]);m=deepcopy(args[2]);p={k:v.copy() for k,v in args[3].items()};d=deepcopy(self.data)
        run(args,data=d);build(args)
        same(self,args[1],a);self.assertEqual(args[2],m);same(self,d,self.data)
        for k in p:pd.testing.assert_frame_equal(p[k],args[3][k])


if __name__=='__main__':unittest.main()
