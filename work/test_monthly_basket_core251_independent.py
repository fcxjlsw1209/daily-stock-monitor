"""251 synthetic-only source, exact basket selector and accounting tests."""
import ast
import copy
from fractions import Fraction as F
from functools import lru_cache
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_basket_core251_independent as audit


def helper():return audit.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'basket251_old_fixture')

def review_for(p,m,c,a):
    base=audit.original_base(a,m,len(m['dates']));sessions=[pd.Timestamp(d) for d in c]
    return dict(schema='monthly_basket_core251_metadata_source_review_v1',panel_first_session=str(p['adj'].index[0].date()),
      listing_bounds=[],identity_cells=[],clock_events=[audit.clock_row(i,base,m,sessions) for i in range(len(m['dates']))])

@lru_cache(None)
def _fixture():
    p,m,c,a,_=helper().fixture();p={'adj':p['adj']};names=m['symbols'][0];t=np.arange(len(p['adj']),dtype=float)
    p['adj']=pd.DataFrame({s:30*np.exp((.0005+j*.000005)*t+.025*np.sin((.013+j*.0002)*t+j)) for j,s in enumerate(names)},index=p['adj'].index)
    for i,day in enumerate(m['dates']):
        values,_=audit.old.original_values(p['adj'],pd.Timestamp(day),names,str(p['adj'].index[0].date()))
        ranks,_=audit.old.independent_rank([x[0 if a['state_calm10'][i] else 1] for x in values]);a['core'][i]=float(a['core_weight'][i,0])*ranks
    a['baseline']=a['core']+(1.-a['core_weight'])*np.linspace(-.18,.18,49)
    a['eligible'][1:]=False;a['eligible'][1:,:8]=True
    return p,m,c,a,review_for(p,m,c,a)

def fixture():return copy.deepcopy(_fixture())

def physical(args,n):
    p,m,c,a,r=copy.deepcopy(args);a={k:v[:n].copy() for k,v in a.items()}
    m={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    if n:c=c[:c.index(m['dates'][-1])+2];p={'adj':p['adj'].loc[:m['dates'][-1]].copy()}
    else:c=[];p={'adj':p['adj'].iloc[:0].copy()}
    r['clock_events']=r['clock_events'][:n];r['identity_cells']=[v for v in r['identity_cells'] if v['query_index']<n]
    return p,m,c,a,r

def run(args,n=None):
    p,m,c,a,r=args;f=audit.independent_features(p,m,c,a,r,n,include_selection=True)
    s,ch,pre,d=audit.selection_oracle(a,m,f,n);return f,audit.orders_from_choices(s,ch,m),s,pre,d

def verify(args,out,n=None):
    p,m,c,a,r=args;f,o,s,pre,d=out
    return audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n)

@lru_cache(None)
def full():return run(_fixture())

def source_only(f):
    out=copy.deepcopy(f)
    for e in out:
        for k in audit.CANDIDATE_KEYS:e[k]=None
        if not e['initial_partial']:e['branch']='source_verified'
    return out


def set_blocks(e,j,rs):
    gross=F(1)
    for r in rs:gross*=1+r
    e['cells'][j]['measurement']=dict(schema=audit.exactmath.MEASURE_SCHEMA,price_count=253,block_count=12,
      momentum=audit.exactmath.pair(gross-1),block_returns=[audit.exactmath.pair(x) for x in rs])


def refresh_core(args):
    p,m,c,a,r=args
    for i,day in enumerate(m['dates']):
        values,_=audit.old.original_values(p['adj'],pd.Timestamp(day),m['symbols'][i],r['panel_first_session'])
        ranks,_=audit.old.independent_rank([x[0 if a['state_calm10'][i] else 1] for x in values]);a['core'][i]=float(a['core_weight'][i,0])*ranks
    a['baseline']=a['core']+(1.-a['core_weight'])*np.linspace(-.18,.18,49)

class SourceTests(unittest.TestCase):
    def test_full_receipt_input_immutability_and_count_scope(self):
        args=fixture();before=copy.deepcopy(args);out=copy.deepcopy(full());r=verify(args,out)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj'],check_exact=True)
        for k in args[3]:self.assertEqual(args[3][k].tobytes(),before[3][k].tobytes())
        self.assertEqual((args[1],args[2],args[4]),(before[1],before[2],before[4]))
        self.assertTrue(all(r[k] for k in audit.PREPARE_FLAGS));self.assertEqual(r['counts']['original_episode_calls'],1)
        self.assertEqual(r['counts']['experimental_episode_calls'],0)
        self.assertEqual(r['counts']['source_price_observations'],253*r['counts']['source_required_cells'])
        self.assertEqual(r['counts']['block_return_estimates'],12*r['counts']['source_required_cells'])

    def test_full_clock_endpoints_and_blocks(self):
        args=fixture();f=full()[0]
        for e in f:
            t=args[2].index(e['date']);self.assertEqual(e['price_start'],args[2][t-294]);self.assertEqual(e['price_end'],args[2][t-42])
            self.assertEqual(e['block_boundary_dates'],[args[2][t-294+21*b] for b in range(13)])

    def test_source_gate_cannot_run_episode_or_candidate_enumeration(self):
        p,m,c,a,r=fixture();f=source_only(full()[0])
        with patch.object(audit.old,'episode',side_effect=AssertionError('early episode')),patch.object(audit,'candidate_event',side_effect=AssertionError('early basket')):
            receipt=audit.verify_source_features(p,m,c,a,r,f)
        self.assertEqual(receipt['episode_calls'],0);self.assertFalse(receipt['new_basket_selection_computed'])
        with self.assertRaises(ValueError):audit.verify_source_features(p,m,c,a,r,copy.deepcopy(full()[0]))

    def test_incumbent_gate_exact_single_arm(self):
        p,m,c,a,r=fixture();f,o,s,pre,d=full();arm=audit.ARMS[0]
        with patch.object(audit,'candidate_event',side_effect=AssertionError('no basket')):
            rec=audit.verify_incumbent(a,m,o[arm],s[arm],pre[arm])
        self.assertEqual(rec['ordered_choices_checked'],len(m['dates']))
        bad=copy.deepcopy(o[arm]);bad[0]['scores'][0]+=1e-9
        with self.assertRaises(ValueError):audit.verify_incumbent(a,m,bad,s[arm],pre[arm])

    def test_zero_logical_and_physical_prefix_width(self):
        for args,n in ((fixture(),0),(physical(fixture(),0),None)):
            out=run(args,n);r=verify(args,out,n);self.assertEqual(out[0],[])
            self.assertEqual(r['counts']['symbols_per_event'],49);self.assertEqual(r['counts']['original_episode_calls'],1)

    def test_all_logical_physical_prefixes_and_future_poison(self):
        args=fixture();ref=full()
        for n in range(len(args[1]['dates'])+1):
            cut=physical(args,n);out=run(cut);self.assertEqual(out[0],ref[0][:n]);verify(cut,out)
            logical=run(args,n);self.assertEqual(logical[0],out[0]);self.assertEqual(logical[1],out[1])
        p,m,c,a,r=fixture();n=2;p['adj']=p['adj'].astype(object);p['adj'].loc[p['adj'].index>m['dates'][n-1]]='future poison'
        m['symbols'][n]=['future bad'];r['clock_events'][n]={'query_index':n,'bad':'future'};r['identity_cells'].append({'query_index':n,'bad':'future'})
        f=audit.independent_features(p,m,c,a,r,n,include_selection=True);self.assertEqual(f,ref[0][:n])

    def test_partial_no_numeric_reads_and_exact_all_arm_copy(self):
        args=physical(fixture(),1);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'][:]='poison'
        out=run(args);verify(args,out);self.assertEqual(out[0][0]['cells'],[])
        for arm in audit.ARMS[1:]:
            self.assertEqual(out[1][arm],out[1][audit.ARMS[0]])
            self.assertEqual(out[2][arm].tobytes(),out[2][audit.ARMS[0]].tobytes())
            self.assertEqual(out[3][arm].tobytes(),out[3][audit.ARMS[0]].tobytes())

    def test_required_nan_outside_shortlist_still_hard_fails(self):
        args=physical(fixture(),2);p,m,c,a,r=args;d=r['clock_events'][1]['price_start']
        p['adj'].at[pd.Timestamp(d),m['symbols'][1][7]]=np.nan;refresh_core(args)
        with self.assertRaisesRegex(ValueError,'Required Eold'):run(args)

    def test_Efalse_new_numeric_not_required_but_original_core_still_checked(self):
        args=physical(fixture(),2);p,m,c,a,r=args;d=r['clock_events'][1]['price_start'];j=20
        self.assertFalse(a['eligible'][1,j]);p['adj'].at[pd.Timestamp(d),m['symbols'][1][j]]=np.nan;refresh_core(args)
        out=run(args);verify(args,out);self.assertIsNone(out[0][1]['cells'][j]['measurement'])
        self.assertFalse(out[0][1]['cells'][j]['source_available'])

    def test_invalid_required_source_and_missing_axes_hard(self):
        for value in (0.,-1.,float('inf'),True,'bad'):
            args=physical(fixture(),2);p,m,c,a,r=args;d=pd.Timestamp(r['clock_events'][1]['price_start'])
            p['adj']=p['adj'].astype(object);p['adj'].at[d,m['symbols'][1][0]]=value
            with self.assertRaises(ValueError):run(args)
        args=physical(fixture(),2);args[0]['adj']=args[0]['adj'].drop(pd.Timestamp(args[4]['clock_events'][1]['price_start']))
        with self.assertRaisesRegex(ValueError,'last295'):run(args)

    def test_original_core_tolerance_and_wrong_origin(self):
        args=physical(fixture(),2);args[3]['core'][1,0]+=5e-13;verify(args,run(args))
        args=physical(fixture(),2);args[3]['core'][1,0]+=1e-8
        with self.assertRaisesRegex(ValueError,'core source'):run(args)
        args=physical(fixture(),2);args[4]['panel_first_session']='2008-01-03'
        with self.assertRaisesRegex(ValueError,'origin'):run(args)

    def test_visible_listing_and_future_evidence(self):
        args=physical(fixture(),2);p,m,c,a,r=args;symbol=m['symbols'][0][0]
        b=dict(symbol=symbol,issuer_key='invented',not_before=c[c.index(m['dates'][1])-251],available_from='2016-01-01',kind='synthetic',evidence_id='fake')
        r['listing_bounds']=[b]
        def identities():
            r['identity_cells']=[]
            for e in r['clock_events']:
                t=c.index(e['date']);r['identity_cells'].append(dict(query_index=e['query_index'],column_index=0,date=e['date'],symbol=symbol,issuer_key='invented',not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=e['date'],price_start=e['price_start'],price_end=e['price_end'],price_window_dates_before_bound=sum(x<b['not_before'] for x in c[t-294:t-41])))
        identities()
        with self.assertRaisesRegex(ValueError,'listing'):run(args)
        a['eligible'][1,0]=False
        excluded=run(args);verify(args,excluded)
        self.assertTrue(excluded[0][1]['cells'][0]['original_finite'])
        self.assertIsNone(excluded[0][1]['cells'][0]['measurement'])
        self.assertFalse(excluded[0][1]['cells'][0]['source_available'])
        a['eligible'][1,0]=True
        b['available_from']='2017-01-01';identities();verify(args,run(args))

    def test_source_metadata_tamper_rejected(self):
        args=physical(fixture(),2);args[4]['clock_events'][1]['block_boundary_dates'][1]='2000-01-01'
        with self.assertRaises(ValueError):run(args)

class SelectorTests(unittest.TestCase):
    def test_direct_selection_no_experimental_episode(self):
        p,m,c,a,r=fixture();f=full()[0];orig=audit.old.episode
        with patch.object(audit.old,'episode',wraps=orig) as call:
            s,ch,pre,d=audit.selection_oracle(a,m,f)
        self.assertEqual(call.call_count,1)
        for i,e in enumerate(f):
            if e['initial_partial']:continue
            for arm in audit.ARMS[1:]:
                self.assertEqual(ch[arm][i],e['winners'][arm]['selected_indices']);self.assertEqual(s[arm][i].tobytes(),pre[arm][i].tobytes())
                selected=set(ch[arm][i]);S=len(e['shortlist_indices'])
                for j,v in enumerate(s[arm][i]):self.assertEqual(v,float(S-e['shortlist_indices'].index(j)) if j in selected else 0.)

    def test_experimental_choices_do_not_depend_on_original_P_or_effects(self):
        p,m,c,a,r=fixture();f=copy.deepcopy(full()[0]);s,ch,pre,d=audit.selection_oracle(a,m,f)
        changed=copy.deepcopy(a);changed['baseline']+=.05
        other=audit.selection_oracle(changed,m,f)
        for arm in audit.ARMS[1:]:self.assertEqual(other[1][arm][1:],ch[arm][1:])

    def test_new_domain_and_enumeration_all_sizes(self):
        src=source_only(full()[0])[1]
        for n in range(9):
            e=copy.deepcopy(src)
            for c in e['cells']:
                if c['original_eligible']:
                    rs=[F(1,16)]*12 if c['column_index']<n else [F(0)]*12;set_blocks(e,c['column_index'],rs)
            got=audit.candidate_event(e);S=min(n,6)
            self.assertEqual(len(got['shortlist_indices']),S);self.assertEqual(len(got['trios']),0 if S<3 else {3:1,4:4,5:10,6:20}[S])
            self.assertEqual(got['branch'],'cash' if n<2 else 'two_names' if n==2 else 'zero_variance')
            for arm in audit.ARMS[1:]:self.assertEqual(len(got['winners'][arm]['selected_indices']),0 if n<2 else 2 if n==2 else 3)

    def test_exact_M_ties_shortlist_lex_and_zero_fallback(self):
        e=copy.deepcopy(source_only(full()[0])[1])
        for j in e['required_indices']:set_blocks(e,j,[F(1,16)]*12)
        out=audit.candidate_event(e);self.assertEqual(out['shortlist_indices'],list(range(6)))
        for arm in audit.ARMS[1:]:self.assertEqual(out['winners'][arm],dict(branch='zero_variance',winning_trio_index=None,tied_trio_indices=[],selected_indices=[0,1,2],selected_symbols=e['symbols'][:3]))

    def test_positive_variance_ties_choose_lex(self):
        e=copy.deepcopy(source_only(full()[0])[1]);rs=[F(1,16)+(F(1,64) if b%2 else -F(1,64)) for b in range(12)]
        for j in e['required_indices']:set_blocks(e,j,rs)
        out=audit.candidate_event(e);self.assertEqual(out['branch'],'optimized')
        for arm in audit.ARMS[1:]:self.assertEqual(out['winners'][arm]['tied_trio_indices'],list(range(20)));self.assertEqual(out['winners'][arm]['selected_indices'],[0,1,2])

    def test_zero_joint_positive_diagonal_uses_shared_exclusion(self):
        e=copy.deepcopy(source_only(full()[0])[1])
        for j in e['required_indices']:
            rs=([F(1,16)+(F(1,64) if b%2 else -F(1,64)) for b in range(12)] if j==0 else [F(1,16)+(-F(1,64) if b%2 else F(1,64)) for b in range(12)] if j==1 else [F(1,16)]*12 if j==2 else [F(0)]*12)
            set_blocks(e,j,rs)
        out=audit.candidate_event(e);self.assertEqual(out['trios'][0]['statistics']['reason'],'zero_joint_variance');self.assertEqual(out['common_trio_indices'],[])

    def test_cached_all_fields_and_score_tamper_rejected(self):
        p,m,c,a,r=fixture();base=copy.deepcopy(full())
        for change in ('mean','choice','count','display','pres','E'):
            f,o,s,pre,d=copy.deepcopy(base)
            if change=='mean':f[1]['trios'][0]['statistics']['mean']['numerator']='999999'
            elif change=='choice':f[1]['winners'][audit.ARMS[1]]['selected_indices'].reverse()
            elif change=='count':d['counts']['enumerated_trios']+=1
            elif change=='display':s[audit.ARMS[1]][1,0]+=1
            elif change=='pres':pre[audit.ARMS[1]][1,0]+=1
            else:f[1]['original_eligible_indices'].pop()
            with self.assertRaises((ValueError,AssertionError)):audit.verify_selection(a,m,f,o,scores=s,pre_scores=pre,audit=d)

    def test_no_new_producer_import_and_accounting_exact_copy(self):
        tree=ast.parse(audit.Path(audit.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any('basket_core251' in (x or '') for x in names))
        prior=ast.parse(audit.ACCOUNT_BASE.read_text())
        f=lambda t:next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='audit_outcomes')
        self.assertEqual(ast.dump(f(tree)),ast.dump(f(prior)))
        dependencies=set(audit.dependency_files())
        bridge=audit.load_pinned(audit.ACCOUNT_BASE,audit.ACCOUNT_BASE_SHA,'basket251_dependency_test')
        self.assertTrue(set(bridge.dependency_files()).issubset(dependencies))
        self.assertIn(bridge.MATH,dependencies)
        self.assertIn(bridge.PRIOR_TEST,dependencies)

class AccountingTests(unittest.TestCase):
    def test_full9paths45slices18groups_and_fee_corruption(self):
        h=audit.load_pinned(audit.old.ACCOUNT_TEST,audit.old.ACCOUNT_TEST_SHA,'basket251_account_fixture')
        p,o,orders,exports,results,m=h.account_fixture();first=next(iter(orders))
        args=(p,o,{a:copy.deepcopy(orders[first]) for a in audit.ARMS},{a:copy.deepcopy(exports[first]) for a in audit.ARMS},{a:copy.deepcopy(results[first]) for a in audit.ARMS},m)
        r=audit.audit_outcomes(*args);self.assertEqual((r['checked_paths'],r['checked_slices'],r['checked_training_groups']),(9,45,18))
        args[3][audit.ARMS[1]]['same_close_zero']['HOLDINGS'][0]['fee']=1e-16
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)

class BilateralCross(unittest.TestCase):
    def test_both_fixtures_full_artifacts_gates_and_every_prefix(self):
        import monthly_basket_core251_kernel as producer
        import test_monthly_basket_core251_kernel as fixtures
        own=fixture();study=helper().helper().helper().real_study();ours=(study,own[3],own[1],own[0],own[2],own[4])
        excluded=fixtures.scope(fixtures.fixture(),[0,1])
        bound=dict(symbol='S48',issuer_key='issuer48',not_before='2016-01-01',available_from='2016-01-02',evidence_id='listing',kind='new_issue')
        excluded[5].update(fixtures.make_review(excluded[2],excluded[4],excluded[4][0],[bound]))
        for args in (fixtures.fixture(),ours,excluded):
            study,a,m,p,c,r=args
            def check(args,n):
                study,a,m,p,c,r=args
                out=producer.prepare(*args,n,source_validator=lambda f:audit.verify_source_features(p,m,c,a,r,f,n),incumbent_validator=lambda o,s,pr:audit.verify_incumbent(a,m,o,s,pr,n))
                f,o,s,pre,d=out;audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pre,audit=d,n=n);return out
            whole=check(args,len(m['dates']))
            for n in range(len(m['dates'])+1):
                logical=check(args,n);self.assertEqual(logical[0],whole[0][:n])
                o,s,pre,d=producer.decisions(study,a,m,whole[0],n,incumbent_validator=lambda o,s,pr:audit.verify_incumbent(a,m,o,s,pr,n))
                audit.verify_selection(a,m,whole[0],o,scores=s,pre_scores=pre,audit=d,n=n)
                cut=fixtures.physical(args,n);out=check(cut,n);self.assertEqual(out[0],whole[0][:n])

if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only');suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (SourceTests,SelectorTests,AccountingTests));result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
