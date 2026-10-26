"""253 independent synthetic checks; no actual source/cache or result reads."""
import ast
import copy
from fractions import Fraction as F
from functools import lru_cache
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_leader_bridge253_independent as audit


@lru_cache(None)
def helper():return audit.prior.load_pinned(audit.PRIOR_TEST,audit.PRIOR_TEST_SHA,'leader253_frozen_fixture')
def fixture():return helper().fixture()


@lru_cache(None)
def complete():
    args=fixture();p,m,c,a,r=args
    features=audit.independent_features(p,m,c,a,r,include_selection=True)
    return args,features


def run(a,m,f,n=None):
    s,ch,p,d=audit.selection_oracle(a,m,f,n)
    return audit.orders_from_choices(s,ch,m),s,p,d


def checked(a,m,f,n=None):
    out=run(a,m,f,n);o,s,p,d=out
    audit.verify_selection(a,m,f,o,scores=s,pre_scores=p,audit=d,n=n)
    return out


def endpoint(a,m,f,n=None):
    n=len(m['dates']) if n is None else n
    s,ch,p,_=audit.prior.selection_oracle(a,m,f,n);o=audit.prior.orders_from_choices(s,ch,m)
    names=audit.prior.ARMS[:2]
    return dict(orders={k:o[k] for k in names},scores={k:s[k] for k in names},pre_scores={k:p[k] for k in names})


def scalar_base(pre=None):
    if pre is None:pre=[.25,.125,.0625,.0546875,.046875,.0390625]
    p=np.array([pre],dtype=float);n=p.shape[1]
    b=dict(names=[[chr(65+j) for j in range(n)]],support=np.zeros_like(p),effects=np.zeros_like(p),budget=np.array([.05]))
    return p,b,np.ones_like(p,dtype=bool)


class EpisodeTests(unittest.TestCase):
    def test_keep_old_three_does_not_imply_same_winner(self):
        p,b,e=scalar_base();b['effects'][0,2]=-.1;b['effects'][0,3]=.15;b['effects'][0,5]=.5
        original=audit.restricted_episode(p,b,e)
        e[0,2]=False;candidate=audit.restricted_episode(p,b,e)
        self.assertEqual(original[1][0],[0,1,3]);self.assertTrue(all(e[0,j] for j in original[1][0]))
        self.assertEqual(candidate[1][0],[0,1,5]);self.assertNotEqual(original[0].tobytes(),candidate[0].tobytes())

    def test_sixth_P_rank_cannot_win_even_with_extreme_effect(self):
        p,b,e=scalar_base();b['effects'][0,5]=1e6
        f,ch,order,j=audit.restricted_episode(p,b,e)
        self.assertEqual(order[0],list(range(6)));self.assertNotIn(5,j[0]);self.assertNotIn(5,ch[0]);self.assertEqual(f[0,5],p[0,5])

    def test_top_two_members_locked_without_final_global_rerank(self):
        p,b,e=scalar_base([.03,.02,.019,.018,.017,.016]);b['effects'][0,3]=.5
        f,ch,_,_=audit.restricted_episode(p,b,e)
        self.assertEqual(set(ch[0]),{0,1,3});self.assertEqual(ch[0][0],3)
        self.assertGreater(f[0,3],f[0,0]);self.assertEqual(p[0,3],.018)

    def test_gap_exact_boundary_and_lexicographic_ties(self):
        p,b,e=scalar_base([.4,.3,.02,0.,np.nextafter(0.,-np.inf),-.1])
        # Subnormal difference can round back; use the adjacent representable threshold.
        p[0,4]=np.nextafter(-.02,-np.inf)+.02
        out=audit.restricted_episode(p,b,e);self.assertIn(3,out[3][0]);self.assertNotIn(4,out[3][0])
        p,b,e=scalar_base([.2]*6);b['names'][0]=list('FEDCBA')
        out=audit.restricted_episode(p,b,e);self.assertEqual(out[2][0],[5,4,3,2,1,0]);self.assertEqual(out[1][0],[5,4,3])

    def test_cash_two_three_and_no_mutation(self):
        for count in range(4):
            p,b,e=scalar_base();e[:]=False;e[0,:count]=True;before=copy.deepcopy((p,b,e))
            out=audit.restricted_episode(p,b,e)
            self.assertEqual(len(out[1][0]),0 if count<2 else count)
            self.assertEqual(p.tobytes(),before[0].tobytes());self.assertEqual(e.tobytes(),before[2].tobytes())
            if count<3:self.assertEqual(out[0].tobytes(),p.tobytes())

    def test_preserving_original_pre_top5_restores_all_scores(self):
        p,b,e=scalar_base();b['effects'][0]=np.linspace(-.3,.3,6);orig=audit.restricted_episode(p,b,e)
        e[0,5]=False;other=audit.restricted_episode(p,b,e)
        self.assertEqual(orig[0].tobytes(),other[0].tobytes());self.assertEqual(orig[1],other[1])


class IntegrationTests(unittest.TestCase):
    def test_complete_artifacts_and_original_P_unchanged(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args;before=copy.deepcopy(a)
        o,s,pr,d=checked(a,m,f);proof=audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pr,audit=d)
        self.assertTrue(all(proof[k] for k in audit.PREPARE_FLAGS));self.assertEqual(proof['ordered_choices_checked'],3*len(m['dates']))
        self.assertEqual(pr[audit.ARMS[0]].tobytes(),pr[audit.ARMS[1]].tobytes())
        self.assertEqual(tuple(d['counts']),audit.COUNT_FIELDS);self.assertEqual((d['counts']['original_episode_calls'],d['counts']['candidate_episode_calls'],d['counts']['diagnostic_episode_calls']),(1,1,0))
        self.assertTrue(all(tuple(e)==audit.EVENT_KEYS for e in d['events']))
        for k in a:self.assertEqual(a[k].tobytes(),before[k].tobytes())

    def test_source_inc_and_endpoint_gates_never_call_candidate(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args;ep=endpoint(a,m,f)
        with patch.object(audit,'restricted_episode',side_effect=AssertionError('candidate before gate')):
            source=audit.verify_source_features(p,m,c,a,r,audit.source_view(f,len(m['dates'])))
            inc=audit.verify_incumbent(a,m,ep['orders'][audit.ARMS[0]],ep['scores'][audit.ARMS[0]],ep['pre_scores'][audit.ARMS[0]])
            control=audit.verify_endpoints(a,m,f,ep['orders'],scores=ep['scores'],pre_scores=ep['pre_scores'])
        self.assertEqual(source['episode_calls'],0);self.assertFalse(inc['candidate_selection_computed']);self.assertFalse(control['candidate_selection_computed'])
        for k in audit.ENDPOINT_FLAGS:self.assertTrue(control[k])

    def test_endpoint_tampering_rejected(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args
        for kind in ('orders','scores','pre_scores'):
            ep=endpoint(a,m,f);arm=audit.prior.ARMS[1]
            if kind=='orders':ep[kind][arm][1]['symbols'].reverse()
            else:ep[kind][arm][1,0]+=1.
            with self.assertRaises(ValueError):audit.verify_endpoints(a,m,f,ep['orders'],scores=ep['scores'],pre_scores=ep['pre_scores'])

    def test_zero_logical_physical_and_future_features_unconsumed(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args
        for n in range(len(m['dates'])+1):
            logical=checked(a,m,f,n);pa,ma,ca,aa,ra=helper().physical(args,n)
            physical=checked(aa,ma,f[:n]);self.assertEqual(logical[0],physical[0]);self.assertEqual(logical[3],physical[3])
            for arm in audit.ARMS:self.assertEqual(logical[1][arm].tobytes(),physical[1][arm].tobytes())
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('future copied')
        f[2]=Poison();m['symbols'][2]=Poison();a['effects']=a['effects'].astype(object);a['effects'][2:]=Poison()
        checked(a,m,f,2)

    def test_partial_exact_all_arms_and_no_numeric_sources(self):
        args,_=copy.deepcopy(complete());p,m,c,a,r=helper().physical(args,1);p['adj']=p['adj'].astype(object);p['adj'][:]='unread'
        f=audit.independent_features(p,m,c,a,r,include_selection=True);o,s,pr,d=checked(a,m,f)
        for arm in audit.ARMS[1:]:
            self.assertEqual(o[arm],o[audit.ARMS[0]]);self.assertEqual(s[arm].tobytes(),s[audit.ARMS[0]].tobytes());self.assertEqual(pr[arm].tobytes(),pr[audit.ARMS[0]].tobytes())
        self.assertIsNone(d['events'][0]['shortlist_indices']);self.assertEqual(d['counts']['removed_eligible_cells'],0)

    def test_exact_new_domain_all_sizes_and_shortlist_ties(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args
        for count in range(9):
            fs=copy.deepcopy(f)
            for e in fs[1:]:
                for j in e['required_indices']:helper().set_blocks(e,j,[F(1,16) if j<count else F(0)]*12)
                e.update(audit.prior.candidate_event(e))
            out=checked(a,m,fs)
            for e in out[3]['events'][1:]:
                self.assertEqual(e['shortlist_indices'],list(range(min(count,6))))
                self.assertEqual(len(e['candidate_choices']),0 if count<2 else 2 if count==2 else 3)

    def test_required_source_invalid_does_not_fallback(self):
        args,_=copy.deepcopy(complete());args=helper().physical(args,2);p,m,c,a,r=args
        p['adj'].at[pd.Timestamp(r['clock_events'][1]['price_start']),m['symbols'][1][7]]=np.nan;helper().refresh_core(args)
        with self.assertRaisesRegex(ValueError,'Required Eold'):audit.independent_features(p,m,c,a,r)

    def test_original_model_eligibility_is_not_recomputed_on_S(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args;out=checked(a,m,f);base=audit.original_base(a,m,len(m['dates']))
        self.assertEqual(out[2][audit.ARMS[1]].tobytes(),base['P'].tobytes())
        for i,e in enumerate(out[3]['events']):
            self.assertEqual(e['original_eligible_indices'],np.flatnonzero(a['eligible'][i]).tolist())
            self.assertTrue(set(e['selector_eligible_indices']).issubset(e['original_eligible_indices']))

    def test_all_full_audit_and_scores_tampering_rejected(self):
        args,f=copy.deepcopy(complete());p,m,c,a,r=args;baseline=run(a,m,f)
        for field in ('count','boundary','E','shortlist','F','P','order','control'):
            fs=copy.deepcopy(f);o,s,pr,d=copy.deepcopy(baseline)
            if field=='count':d['counts']['restricted_eligible_cells']+=1
            elif field=='boundary':d['events'][1]['boundary_candidates']=[]
            elif field=='E':fs[1]['original_eligible_indices'].pop()
            elif field=='shortlist':fs[1]['shortlist_indices'].reverse()
            elif field=='F':s[audit.ARMS[1]][1,48]=np.nextafter(s[audit.ARMS[1]][1,48],np.inf)
            elif field=='P':pr[audit.ARMS[1]][1,48]+=1.
            elif field=='order':o[audit.ARMS[1]][1]['symbols'].reverse()
            else:s[audit.ARMS[2]][1,48]+=1.
            with self.subTest(field=field),self.assertRaises(ValueError):audit.verify_selection(a,m,fs,o,scores=s,pre_scores=pr,audit=d)

    def test_no_new_producer_import_and_dependency_pins(self):
        source=audit.Path(audit.__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any('253_kernel' in (x or '') or '253_host' in (x or '') for x in names))
        self.assertEqual(audit.hashlib.sha256(audit.PRIOR.read_bytes()).hexdigest(),audit.PRIOR_SHA)
        self.assertIn(audit.PRIOR,set(audit.dependency_files()))


class AccountingTests(unittest.TestCase):
    def test_full9paths45slices18groups_and_exact_zero_aggregate(self):
        old=audit.prior.old
        h=audit.prior.load_pinned(old.ACCOUNT_TEST,old.ACCOUNT_TEST_SHA,'leader253_account_fixture')
        px,op,orders,exports,results,m=h.account_fixture();key=next(iter(orders))
        args=(px,op,{a:copy.deepcopy(orders[key]) for a in audit.ARMS},{a:copy.deepcopy(exports[key]) for a in audit.ARMS},{a:copy.deepcopy(results[key]) for a in audit.ARMS},m)
        # The older account fixture predates the saved full122 fee display.
        for arm in audit.ARMS:
            for path in ('same_close_zero','next_open_zero','next_open_cost'):
                args[4][arm][path]['slices']['full122']['fees_paid']=args[4][arm][path]['metrics']['fees_paid']
        proof=audit.audit_outcomes(*args);self.assertEqual((proof['checked_paths'],proof['checked_slices'],proof['checked_training_groups']),(9,45,18));self.assertEqual(set(proof['paths']),set(audit.ARMS))
        for location in ('aggregate','full122','trade','period'):
            changed=copy.deepcopy(args);arm=audit.ARMS[1];path='same_close_zero'
            if location=='aggregate':changed[4][arm][path]['metrics']['fees_paid']=1e-16
            elif location=='full122':changed[4][arm][path]['slices']['full122']['fees_paid']=1e-16
            elif location=='trade':changed[3][arm][path]['TRADES'][0]['cost']=1e-16
            else:changed[3][arm][path]['HOLDINGS'][0]['fee']=1e-16
            with self.subTest(location=location),self.assertRaisesRegex(ValueError,'Every zero fee'):audit.audit_outcomes(*changed)


class BilateralCross(unittest.TestCase):
    def test_public_prepare_and_decisions_do_not_copy_or_compare_future_bodies(self):
        import monthly_leader_bridge253_kernel as producer
        import test_monthly_leader_bridge253_kernel as fixtures
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('future deepcopy')
            def __eq__(self,other):raise AssertionError('future equality')
        for n in (0,2):
            study,a,m,p,c,r,f=fixtures.fixture();ep=endpoint(a,m,f,n)
            wanted=run(a,m,f,n)
            for i in range(n,len(m['dates'])):
                f[i]=Poison()
                for key in ('dates','ends','symbols'):m[key][i]=Poison()
            kwargs=dict(incumbent_validator=lambda o,s,pr:audit.verify_incumbent(a,m,o,s,pr,n),
              endpoint_validator=lambda o,s,pr:audit.verify_endpoints(a,m,f,o,scores=s,pre_scores=pr,n=n),expected_endpoints=ep)
            outputs=[producer.decisions(study,a,m,f,n,**kwargs),
              producer.prepare(study,a,m,p,c,r,f,n,
                source_validator=lambda fs:audit.verify_source_features(p,m,c,a,r,fs,n),**kwargs)]
            for o,s,pr,d in outputs:
                self.assertEqual(o,wanted[0]);self.assertEqual(d,wanted[3])
                audit.verify_selection(a,m,f,o,scores=s,pre_scores=pr,audit=d,n=n)

    def test_both_fixtures_all_artifacts_gates_and_every_prefix(self):
        import monthly_leader_bridge253_kernel as producer
        import test_monthly_leader_bridge253_kernel as fixtures
        own,f=copy.deepcopy(complete());p,m,c,a,r=own;study=helper().helper().helper().helper().real_study()
        for args in (fixtures.fixture(),(study,a,m,p,c,r,f)):
            # The public producer fixture contract is seven values, final251 features last.
            study,a,m,p,c,r,f=args
            def check(n):
                ep=endpoint(a,m,f,n)
                out=producer.prepare(study,a,m,p,c,r,f,n,
                  source_validator=lambda fs:audit.verify_source_features(p,m,c,a,r,fs,n),
                  incumbent_validator=lambda o,s,pr:audit.verify_incumbent(a,m,o,s,pr,n),
                  endpoint_validator=lambda o,s,pr:audit.verify_endpoints(a,m,f,o,scores=s,pre_scores=pr,n=n),expected_endpoints=ep)
                o,s,pr,d=out;audit.verify_prepared(p,m,c,a,r,f,o,scores=s,pre_scores=pr,audit=d,n=n);return out
            full=check(len(m['dates']))
            for n in range(len(m['dates'])+1):
                out=check(n);self.assertEqual(out[3]['events'],full[3]['events'][:n])
                ep=endpoint(a,m,f,n)
                o,s,pr,d=producer.decisions(study,a,m,f,n,
                  incumbent_validator=lambda o,s,pr:audit.verify_incumbent(a,m,o,s,pr,n),
                  endpoint_validator=lambda o,s,pr:audit.verify_endpoints(a,m,f,o,scores=s,pre_scores=pr,n=n),expected_endpoints=ep)
                audit.verify_selection(a,m,f,o,scores=s,pre_scores=pr,audit=d,n=n)
                sp,ap,mp,pp,cp,rp,fp=fixtures.physical(args,n)
                ep=endpoint(ap,mp,fp,n)
                op,sc,pre,da=producer.prepare(sp,ap,mp,pp,cp,rp,fp,n,
                  source_validator=lambda fs:audit.verify_source_features(pp,mp,cp,ap,rp,fs,n),
                  incumbent_validator=lambda o,s,pr:audit.verify_incumbent(ap,mp,o,s,pr,n),
                  endpoint_validator=lambda o,s,pr:audit.verify_endpoints(ap,mp,fp,o,scores=s,pre_scores=pr,n=n),expected_endpoints=ep)
                audit.verify_prepared(pp,mp,cp,ap,rp,fp,op,scores=sc,pre_scores=pre,audit=da,n=n)
                self.assertEqual(op,out[0])


if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only');suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (EpisodeTests,IntegrationTests,AccountingTests))
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
