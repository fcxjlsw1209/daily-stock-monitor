"""Only synthetic peer source declarations, prices, decisions and ledgers."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_peer239_independent as audit


def make_report(meta,calendar,sectors=None,issuers=None):
    events=[];records=[]
    for i,(d,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        partial=d=='2016-08-17';t=calendar.index(d)
        positions=[] if partial else [t-252,t-42];dates=[calendar[k] for k in positions]
        stock=[j for j,s in enumerate(names) if s not in meta['etfs']];group=[]
        for j in stock:
            s=names[j];issuer=(issuers or {}).get(s,'issuer:'+s)
            sector=(sectors or {}).get(s,'XLK');bound=copy.deepcopy(audit.BOUNDS.get(s))
            bound=bound if not partial and bound is not None and issuer==bound['issuer_key'] and d>=bound['available_from'] else None
            status='initial_partial' if partial else ('official_pre_inception' if bound and dates[0]<bound['not_before'] else 'observed_full_window')
            points=[];gaps=[]
            if sector is None:gaps.append('missing_sector')
            for role,day in zip(('security_start','security_end'),dates):
                absent=status=='official_pre_inception' and day<bound['not_before']
                points.append(dict(role=role,symbol=s,date=day,status='missing' if absent else 'valid',
                                   reason='missing_value' if absent else None))
                if absent:gaps.append('endpoint:'+role+':missing_value')
            group.append(dict(query_index=i,column_index=j,date=d,symbol=s,issuer_key=issuer,
                original_sector=sector,sector=sector,mapping_origin='query_date_sector_panel',
                mapping_repair='unchanged' if sector is not None else 'unresolved',mapping_evidence=None,
                peer_indices=[],peer_issuer_count=0,structural_peer_support=False,initial_partial=partial,
                source_positions=positions.copy(),source_dates=dates.copy(),endpoints=points,gap_reasons=[],
                original_gap_reasons=sorted(gaps),history_status=status,history_bound=bound,
                deferred_gap_reasons=sorted(gaps) if status=='official_pre_inception' else []))
        mature=[r for r in group if r['history_status']=='observed_full_window']
        for r in group:
            peers=sorted([p for p in mature if r in mature and p['sector']==r['sector'] and p['issuer_key']!=r['issuer_key']],
                         key=lambda p:(p['issuer_key'],p['symbol'],p['column_index']))
            r['peer_indices']=[p['column_index'] for p in peers]
            r['peer_issuer_count']=len({p['issuer_key'] for p in peers});r['structural_peer_support']=r['peer_issuer_count']>=2
        events.append(dict(query_index=i,date=d,initial_partial=partial,stock_indices=stock,
            source_positions=positions,source_dates=dates,mature_indices=[r['column_index'] for r in mature],
            supported_indices=[r['column_index'] for r in group if r['structural_peer_support']]))
        records.extend(group)
    return dict(schema='peer239_mature_source_report_v1',status='ADMITTED',admitted=True,events=events,
        records=records,gaps=[],counts=dict(events=len(events),stock_cells=len(records)),source_only=True,
        original_inventory_preserved=True,strict_pit=False,momentum_computed=False,rankings_computed=False,
        orders_computed=False,returns_computed=False,model_fits=0,original_failed_audit_preserved=True)


def fixture():
    axis=pd.bdate_range('2015-01-01','2017-02-01');calendar=axis.strftime('%Y-%m-%d').tolist()
    names=list('ABCDEFGH')+['FUND'];dates=['2016-08-17','2016-08-31','2016-09-30']
    meta=dict(dates=dates,ends=['2016-08-31','2016-09-30','2016-10-31'],
              symbols=[names.copy() for _ in dates],etfs=['FUND'])
    e=np.ones((3,9),bool);e[1:,1]=False;e[1:,5]=False
    arrays=dict(eligible=e)
    frame=pd.DataFrame(128.,index=axis,columns=names).astype(object)
    sectors={s:'XLK' for s in 'ABCD'};sectors.update({s:'XLP' for s in 'EFG'});sectors['H']='XLI'
    for d in dates[1:]:frame.loc[calendar[calendar.index(d)-42],names]=[160.,192.,224.,256.,64.,144.,160.,96.,'unused']
    return {'adj':frame},meta,calendar,arrays,make_report(meta,calendar,sectors)


def warm_fixture():
    axis=pd.bdate_range('2024-01-01','2026-10-01');calendar=axis.strftime('%Y-%m-%d').tolist()
    names=['A','B','C','SPCX','SKHY','ETF'];dates=['2026-06-30','2026-07-31','2026-08-31']
    meta=dict(dates=dates,ends=['2026-07-31','2026-08-31','2026-09-30'],symbols=[names.copy() for _ in dates],etfs=['ETF'])
    arrays=dict(eligible=np.zeros((3,6),bool));frame=pd.DataFrame(128.,index=axis,columns=names).astype(object)
    # Before SKHY's evidence is visible the source is mature and must be valid;
    # omit that name from the first query without changing the common width.
    meta['symbols'][0][4]='D';frame['D']=128.
    for d in dates:frame.loc[calendar[calendar.index(d)-42],list('ABC')]=[160.,192.,224.]
    frame['SPCX']='unread';frame['SKHY']='unread';frame['ETF']='unread'
    issuers={s:audit.BOUNDS[s]['issuer_key'] for s in ('SPCX','SKHY')}
    return {'adj':frame},meta,calendar,arrays,make_report(meta,calendar,{'SPCX':None},issuers)


def physical(args,n):
    panel,meta,calendar,arrays,report=copy.deepcopy(args)
    mm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in meta.items()}
    aa={k:v[:n].copy() for k,v in arrays.items()}
    report['events']=report['events'][:n];report['records']=[r for r in report['records'] if r['query_index']<n]
    if n:
        d=meta['dates'][n-1];panel={'adj':panel['adj'].loc[:d].copy()};calendar=calendar[:calendar.index(d)+2]
    else:panel={'adj':panel['adj'].iloc[:0].copy()};calendar=[]
    return panel,mm,calendar,aa,report


def selection_fixture():
    p,m,c,_,r=fixture();n,w=3,9;shape=(n,w);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True]);weak=np.array([True,True,False]);cw=np.where(calm,.5,.52)[:,None]
    support=np.tile(np.linspace(.40,-.40,w),(n,1));baseline=(1-cw)*support
    a=dict(baseline=baseline,core=z.copy(),core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,-.4,.5,.3,-.1,.2,0.,0.],(n,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.20,.19,.10,.095,.09,.08,.07,.06,.05],(n,1)),old_ml4=z.copy(),
        rank_log_excess=z.copy(),extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),
        model_available=f.copy(),raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    return p,m,c,a,r


def orders(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,audit.ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FeatureTests(unittest.TestCase):
    def test_known_negative_momentum_medians_cross_sector_ranks(self):
        args=fixture();v,r=audit.independent_features(*args)
        self.assertEqual(v['momentum'][1,:8].tolist(),[.25,.5,.75,1.,-.5,.125,.25,-.25])
        self.assertEqual(v['peer_median'][1,:7].tolist(),[.75,.75,.5,.5,.1875,-.125,-.1875])
        want=[2/7-.5,3/7-.5,4.5/7-.5,1.-.5,1/7-.5,4.5/7-.5,6/7-.5]
        audit.exact_array(v['peer_target'][1,:7],np.array(want),'Hand-computed peer ties')
        self.assertNotEqual(v['peer_target'][1,2],v['absolute_target'][1,2])
        self.assertEqual(r['counts'],dict(events=3,stock_cells=24,mature_cells=16,supported_cells=14,endpoint_observations=32,new_model_fits=0))
        self.assertTrue(np.isnan(v['peer_target'][:,7:]).all())
        receipt=audit.verify_features(*args,v,r);self.assertTrue(all(receipt[k] for k in audit.FEATURE_FLAGS))

    def test_average_percentile_not_endpoint_scaled_rank(self):
        self.assertEqual(audit.average_percentile([1.,2.,3.,4.]),[-.25,0.,.25,.5])
        self.assertEqual(audit.average_percentile([7.,7.,7.]),[2/3-.5]*3)
        self.assertEqual(audit.average_percentile([-0.,0.]),[.25,.25])
        self.assertEqual(audit.average_percentile([]),[])
        self.assertEqual(audit.median([-.5,.125]),-.1875)

    def test_before_E_domain_and_no_input_mutation(self):
        args=fixture();before=copy.deepcopy(args);full=audit.independent_features(*args)
        self.assertTrue(full[0]['supported'][1,1]);self.assertTrue(full[0]['supported'][1,5])
        args[3]['eligible'][:]=False;other=audit.independent_features(*args)
        for k in audit.VALUE_KEYS:audit.exact_array(other[0][k],full[0][k],k)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj'])
        self.assertEqual(args[4],before[4])

    def test_partial_and_ETFs_read_no_numeric_sources(self):
        args=physical(fixture(),1);args[0]['adj']=args[0]['adj'].iloc[:0]
        v,r=audit.independent_features(*args);self.assertEqual(r['counts']['endpoint_observations'],0)
        self.assertFalse(v['mature'].any());self.assertTrue(np.isnan(v['momentum']).all())
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='FUND')
        audit.independent_features(*args)

    def test_warmup_bounds_no_price_reads_and_inventory_retained(self):
        args=warm_fixture();v,r=audit.independent_features(*args)
        self.assertFalse(v['mature'][:,3].any());self.assertFalse(v['mature'][1:,4].any())
        self.assertTrue(v['mature'][0,4]);self.assertEqual(r['counts']['stock_cells'],15)
        self.assertEqual(r['events'][2]['cells'][3]['history_status'],'official_pre_inception')
        for field,value in [('available_from','2027-01-01'),('not_before','2020-01-01'),('issuer_key','other')]:
            bad=copy.deepcopy(args);bad[4]['records'][3]['history_bound'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.independent_features(*bad)

    def test_warmup_source_certificate_contamination_and_post_bound_gap(self):
        for mutation in ('pre_valid','post_missing','undeferred','sector_filled'):
            args=warm_fixture();rows=args[4]['records'];row=next(r for r in rows if r['query_index']==2 and r['symbol']=='SPCX')
            if mutation=='pre_valid':row['endpoints'][0].update(status='valid',reason=None)
            elif mutation=='post_missing':row['endpoints'][1].update(status='missing',reason='missing_value')
            elif mutation=='undeferred':row['deferred_gap_reasons']=[]
            else:row['sector']='XLI'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_peer_inventory_missing_duplicate_self_wrong_sector_rejected(self):
        for mutation in ('missing','self','wrongsector','duplicate','bool_index','reorder'):
            args=fixture();rows=args[4]['records'];row=rows[8]
            if mutation=='missing':row['peer_indices']=row['peer_indices'][:-1]
            elif mutation=='self':row['peer_indices'][0]=row['column_index']
            elif mutation=='wrongsector':row['peer_indices'].append(4)
            elif mutation=='duplicate':rows[9]['issuer_key']=row['issuer_key']
            elif mutation=='bool_index':row['column_index']=False
            else:row['peer_indices'].reverse()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_mature_endpoint_certificate_and_data_errors(self):
        for value in (np.nan,0.,-1.,math.inf,True,'128',None):
            args=fixture();d=args[4]['events'][1]['source_dates'][0];args[0]['adj'].at[pd.Timestamp(d),'H']=value
            with self.subTest(value=value),self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='H')
        with self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();args[4]['records'][8]['endpoints'][0]['status']='missing'
        with self.assertRaises(ValueError):audit.independent_features(*args)

    def test_only_exact_two_endpoints_and_derived_overflow(self):
        args=fixture();ev=args[4]['events'][1];middle=args[2][ev['source_positions'][0]+10]
        args[0]['adj'].at[pd.Timestamp(middle),'A']='ignored interior'
        audit.independent_features(*args)
        args=fixture();s,e=args[4]['events'][1]['source_dates'];args[0]['adj'].at[pd.Timestamp(s),'A']=np.nextafter(0.,1.)
        args[0]['adj'].at[pd.Timestamp(e),'A']=np.finfo(float).max
        with self.assertRaises(FloatingPointError):audit.independent_features(*args)
        with self.assertRaises(FloatingPointError):audit.median([np.finfo(float).max]*2)

    def test_metadata_clock_and_initial_name_not_first_arbitrary_row(self):
        args=fixture();args[4]['events'][1]['source_positions'][1]+=1
        with self.assertRaises(ValueError):audit.independent_features(*args)
        args=list(fixture());args[2]=args[2][:args[2].index(args[1]['dates'][-1])+1]
        with self.assertRaises(ValueError):audit.independent_features(*args)
        p,m,c,a,_=fixture();m={k:(v[1:2] if k in ('dates','ends','symbols') else v) for k,v in m.items()};a={'eligible':a['eligible'][1:2]}
        r=make_report(m,c);_,out=audit.independent_features(p,m,c,a,r)
        self.assertEqual(out['counts']['mature_cells'],8)

    def test_future_poison_logical_and_physical_prefixes(self):
        args=fixture();base=audit.independent_features(*args,limit=2)
        poisoned=copy.deepcopy(args);poisoned[1]['dates'][2]='bad';poisoned[1]['symbols'][2]=None
        poisoned[3]['eligible']=poisoned[3]['eligible'].astype(object);poisoned[3]['eligible'][2]='bad'
        poisoned[0]['adj'].loc[pd.Timestamp(args[1]['dates'][1])+pd.Timedelta(days=1):]='poison'
        poisoned[4]['events'][2]=None
        poisoned[4]['records']=[r if r['query_index']<2 else {'query_index':2,'poison':object()} for r in poisoned[4]['records']]
        got=audit.independent_features(*poisoned,limit=2)
        for k in audit.VALUE_KEYS:audit.exact_array(got[0][k],base[0][k],k)
        audit.record_check(got[1],base[1])
        full=audit.independent_features(*args)
        for n in range(4):
            small=physical(args,n);v,r=audit.independent_features(*small)
            logical=audit.independent_features(*args,limit=n);audit.record_check(r,logical[1])
            for k in audit.VALUE_KEYS:audit.exact_array(v[k],full[0][k][:n],k)

    def test_feature_and_audit_tamper_rejected(self):
        args=fixture();v,r=audit.independent_features(*args)
        for key in audit.VALUE_KEYS:
            wrong=copy.deepcopy(v);wrong[key][1,0]=not wrong[key][1,0] if wrong[key].dtype==bool else wrong[key][1,0]+.01
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_features(*args,wrong,r)
        wrong=copy.deepcopy(r);wrong['counts']['supported_cells']-=1
        with self.assertRaises(ValueError):audit.verify_features(*args,v,wrong)


class SelectionTests(unittest.TestCase):
    def test_real_original_adapter_one_episode_exact_scores_pre_and_choices(self):
        args=selection_fixture();_,m,_,a,_=args;v,_=audit.independent_features(*args)
        expected,chosen,pres=audit.selection_oracle(a,m,v)
        study=load('outputs/monthly_local_linear70/study.py','peer239_test_original')
        adapter=load('outputs/monthly_path70_research/incumbent_adapter.py','peer239_test_adapter')
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        calls=[];actual={};oo={};policy=dict(budget=.05,ceiling=5,gap=.02)
        for arm,key in zip(audit.ARMS,(None,'peer_target','absolute_target')):
            factor=np.full(a['eligible'].shape,np.nan) if key is None else v[key]
            def boundary(pre,eligible,names,effects,original,rule):
                calls.append(arm);audit.exact_array(pre,pres[arm],'Independent funded pre')
                audit.exact_array(eligible,a['eligible'],'Original E retained')
                return study.ep.select_boundary(pre,eligible,names,effects,original,rule)
            cc,ss=adapter.score_and_choose('incumbent66',.05,factor,a,m,3,pre_score=x['local'],
                effects=a['effects'],episode_policy=policy,boundary_selector=boundary)
            self.assertEqual(cc,chosen[arm]);audit.exact_array(ss,expected[arm],arm)
            actual[arm]=ss;oo[arm]=study.rp.make_orders(cc,ss,m)
        self.assertEqual(calls,list(audit.ARMS))
        result=audit.verify_selection(a,m,v,oo,scores=actual,pre_scores=pres)
        self.assertTrue(all(result[k] for k in audit.SELECTION_FLAGS));self.assertEqual(result['ordered_choices_checked'],9)

    def test_finite_zero_target_funds_replacement_and_missing_preserves(self):
        p,m,c,a,_=selection_fixture();r=make_report(m,c)
        for d in m['dates'][1:]:p['adj'].loc[c[c.index(d)-42],list('ABCDEFGH')]=[128.*(i+1) for i in range(8)]
        v,_=audit.independent_features(p,m,c,a,r);ss,ch,pres=audit.selection_oracle(a,m,v)
        self.assertEqual(v['absolute_target'][1,3],0.)
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        self.assertNotEqual(pres[audit.ARMS[2]][1,3],x['local'][1,3])
        self.assertEqual(pres[audit.ARMS[2]][1,3],x['local'][1,3]+.05*(0.-x['support'][1,3]))
        for arm in audit.ARMS:audit.exact_array(pres[arm][0],x['local'][0],'Initial unchanged')
        self.assertEqual(pres[audit.ARMS[1]][1,8],x['local'][1,8])

    def test_complete_budget_includes_calm_nonweak_extension(self):
        args=selection_fixture();_,m,_,a,_=args
        a['extension_available'][:]=True;a['forecast_available'][:]=True;a['rank_log_excess'][:]=.2
        a['eligible'][2,1]=False
        v,_=audit.independent_features(*args);ss,ch,pres=audit.selection_oracle(a,m,v)
        out=audit.verify_selection(a,m,v,orders(ss,ch,m),scores=ss,pre_scores=pres)
        self.assertEqual(out['calm_nonweak_extension_count'],8)
        self.assertGreaterEqual(out['minimum_remaining_support'],.33-1e-12)
        self.assertLessEqual(out['maximum_budget_sum_error'],1e-12)
        self.assertLessEqual(out['maximum_absolute_delta'],.05+1e-12)

    def test_original_model_domain_and_all_prefix_cash_two_three(self):
        for count in (0,1,2,9):
            args=selection_fixture();_,m,_,a,_=args;a['eligible'][:,count:]=False
            a['forecast_available'][:]=True;a['model_available'][:]=True
            a['raw_local_constant'][:]=np.arange(9);a['raw_local_linear'][:]=np.arange(9)[::-1]
            v,_=audit.independent_features(*args);full,cc,pp=audit.selection_oracle(a,m,v)
            for arm in audit.ARMS:self.assertEqual(len(cc[arm][1]),3 if count==9 else 2 if count==2 else 0)
            for n in range(4):
                ss,ch,pres=audit.selection_oracle(a,m,v,n);audit.verify_selection(a,m,v,orders(ss,ch,m),scores=ss,pre_scores=pres,n=n)
                for arm in audit.ARMS:audit.exact_array(ss[arm],full[arm][:n],arm)

    def test_scores_pres_choices_and_rank_mutations_rejected(self):
        args=selection_fixture();_,m,_,a,_=args;v,_=audit.independent_features(*args)
        ss,ch,pres=audit.selection_oracle(a,m,v);oo=orders(ss,ch,m)
        for field in ('scores','pre_scores'):
            data=dict(scores=copy.deepcopy(ss),pre_scores=copy.deepcopy(pres));data[field][audit.ARMS[1]][1,0]+=.001
            with self.subTest(field=field),self.assertRaises(ValueError):audit.verify_selection(a,m,v,oo,**data)
        bad=copy.deepcopy(oo);bad[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,bad,scores=ss,pre_scores=pres)
        bad=copy.deepcopy(v);bad['peer_target'][1,0]=0.
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,bad)

    def test_no_new_producer_imports_and_dependency_pin(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names]
        self.assertNotIn('monthly_peer239_kernel',imports);self.assertNotIn('monthly_peer239_host',imports)
        deps=audit.dependency_files()
        self.assertIn(audit.ROOT/'work/test_monthly_upside_retention181_independent.py',deps)
        with patch.object(audit,'ACCOUNTING_SHA','bad'),self.assertRaises(ValueError):audit.accounting()


class OutcomeTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_cost_tamper(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        result=audit.audit_outcomes(*args);self.assertEqual((result['checked_paths'],result['checked_slices']),(9,45))
        for path,key in [('same_close_zero','fees_paid'),('next_open_cost','cagr')]:
            wrong=copy.deepcopy(args);wrong[4][audit.ARMS[1]][path]['metrics'][key]+=1.
            with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_pre_inception_missing_column_and_row_taxonomy(self):
        import monthly_peer239_kernel as producer
        for reason in ('missing_value','missing_column','missing_row','missing_symbol','missing_date'):
            args=warm_fixture()
            row=next(r for r in args[4]['records'] if r['query_index']==2 and r['symbol']=='SPCX')
            row['endpoints'][0]['reason']=reason
            for key in ('original_gap_reasons','deferred_gap_reasons'):
                row[key]=sorted('endpoint:security_start:'+reason if x=='endpoint:security_start:missing_value' else x
                                for x in row[key])
            with self.subTest(reason=reason):
                if reason in ('missing_symbol','missing_date'):
                    with self.assertRaises(ValueError):audit.independent_features(*args)
                    with self.assertRaises(ValueError):producer.build_features(*args)
                else:
                    result=producer.build_features(*args)
                    self.assertTrue(audit.verify_features(*args,*result)['passed'])

    def test_unique_both_fixtures_all_prefixes_values_and_full_audit(self):
        import monthly_peer239_kernel as producer
        from test_monthly_peer239_kernel import fixture as other_fixture, warm_fixture as other_warm
        for args in (fixture(),warm_fixture(),other_fixture(),other_warm()):
            full=producer.build_features(*args);audit.verify_features(*args,*full)
            for n in range(len(args[1]['dates'])+1):
                logical=producer.build_features(*args,limit=n);audit.verify_features(*args,*logical,limit=n)
                small=physical(args,n);got=producer.build_features(*small);audit.verify_features(*small,*got)
                audit.record_check(logical[1],got[1])
                for k in audit.VALUE_KEYS:audit.exact_array(full[0][k][:n],got[0][k],k)


if __name__=='__main__':unittest.main(verbosity=2)
