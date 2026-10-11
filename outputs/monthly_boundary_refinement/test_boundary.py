import unittest
import boundary_research as s

class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m,cls.d,cls.px,cls.op,cls.a,cls.pred=s.load()

    def selected(self,sc,rule,eligible=None,symbols=None):
        sc=s.np.array(sc,float);eligible=s.np.ones_like(sc,bool) if eligible is None else s.np.array(eligible,bool)
        symbols=[list('ABCDE')[:sc.shape[1]] for _ in sc] if symbols is None else symbols
        return [[ss[j] for j in ids] for ids,ss in zip(s.choose(sc,eligible,symbols,rule),symbols)]

    def test_previous122_orders_and_scores_exact(self):
        for row,name in [(s.grid()[0],'baseline'),(s.grid()[1],'primary')]:
            ch,sc=s.choices(row,self.a,self.pred,self.m,self.d,122);ref=s.read(s.old.P/(name+'_ORDERS.json'))
            for i,ids in enumerate(ch):
                self.assertEqual([self.m['symbols'][i][j] for j in ids],ref[i]['symbols']);s.np.testing.assert_allclose(sc[i,ids],ref[i]['scores'],atol=1e-12)

    def test_gap_zero_matches_top3_even_ties(self):
        gen=s.np.random.default_rng(1);sc=gen.integers(0,5,(40,5)).astype(float);el=gen.uniform(size=(40,5))>.2;ss=[list('ABCDE') for _ in sc]
        for c in [3,4,5]:self.assertEqual(s.choose(sc,el,ss,dict(kind='gap',ceiling=c,gap=0)),s.rp.choose(sc,el,ss,3))

    def test_hurdle_retains_near_tie_then_replaces(self):
        z=self.selected([[.4,.3,.2,.1],[.4,.3,.19,.20],[.4,.3,.18,.21]],dict(kind='gap',ceiling=4,gap=.02))
        self.assertEqual(z,[['A','B','C'],['A','B','C'],['A','B','D']])

    def test_buffer_does_not_ignore_rank_ceiling(self):
        z=self.selected([[5,4,3,2,1],[5,4,1,3,2]],dict(kind='buffer',ceiling=4))
        self.assertEqual(z,[['A','B','C'],['A','B','D']])

    def test_noneligible_incumbent_must_exit_despite_hurdle(self):
        z=self.selected([[5,4,3,2],[5,4,3,2]],dict(kind='gap',ceiling=5,gap=100),[[1,1,1,1],[1,1,0,1]])
        self.assertEqual(z[1],['A','B','D'])

    def test_removed_from_current_pool_must_exit(self):
        z=self.selected([[5,4,3,2],[5,4,3,2]],dict(kind='gap',ceiling=5,gap=100),symbols=[list('ABCD'),list('ABDE')])
        self.assertEqual(z[1],['A','B','D'])

    def test_two_names_equal_and_one_name_cash_choices(self):
        z=self.selected([[5,4,3],[5,4,3],[5,4,3]],dict(kind='gap',ceiling=4,gap=.02),[[1,1,1],[1,1,0],[1,0,0]])
        self.assertEqual(z,[['A','B','C'],['A','B'],[]])

    def test_disagreement_has_correct_units_and_combines_max(self):
        p={'W_utility_10_rank':s.np.array([[.3,.1]]),'W_utility_100_rank':s.np.array([[.1,.05]]),'M_utility_10_rank':s.np.array([[.25,-.3]])}
        s.np.testing.assert_allclose(s.disagreement(p,{'comparator':'both'},1),[[.2,.4]])

    def test_no_disagreement_preserves_full_score(self):
        pred={k:v.copy() for k,v in self.pred.items()}
        pred['W_utility_100_rank']=pred['W_utility_10_rank'].copy();pred['M_utility_10_rank']=pred['W_utility_10_rank'].copy()
        row=dict(s.grid()[1],shrink=dict(comparator='both',threshold=.15,floor=0))
        s.np.testing.assert_allclose(s.score(row,self.a,pred,122),s.score(s.grid()[1],self.a,pred,122))

    def test_complete_shrink_returns_budget_to_original_support(self):
        pred={k:v.copy() for k,v in self.pred.items()};pred['W_utility_100_rank']=pred['W_utility_10_rank']+1
        spec=s.read(s.old.P/'FROZEN.json')['selected']['primary']['spec'];residual=[z for z in spec if z['key']!='ml_4']
        full=s.score(s.grid()[1],self.a,pred,122)
        for floor in [0.,.5]:
            row=dict(s.grid()[1],shrink=dict(comparator='alpha',threshold=.15,floor=floor));got=s.score(row,self.a,pred,122);res=s.old.score(residual,self.a,122)
            s.np.testing.assert_allclose(got,res+floor*(full-res),atol=1e-12)

    def test_later_data_cannot_change_prefix_decisions(self):
        a={k:v.copy() for k,v in self.a.items()};p={k:v.copy() for k,v in self.pred.items()}
        for arr in list(a.values())+list(p.values()):arr[88:]=0
        for row in s.grid()[::7]:
            original,_=s.choices(row,self.a,self.pred,self.m,self.d,122);changed,_=s.choices(row,a,p,self.m,self.d,88)
            self.assertEqual(original[:88],changed)

    def test_grid_fixed_unique_and_all_long_only(self):
        rows=s.grid();self.assertEqual(len(rows),118)
        keys=[s.json.dumps({k:v for k,v in r.items() if k not in ['id','family']},sort_keys=True) for r in rows];self.assertEqual(len(keys),len(set(keys)))
        for r in rows:
            ch,sc=s.choices(r,self.a,self.pred,self.m,self.d,88)
            for i,ids in enumerate(ch):
                self.assertIn(len(ids),[0,2,3]);self.assertEqual(len(ids),len(set(ids)));self.assertTrue(self.d['eligible'][i,ids].all())

if __name__=='__main__':unittest.main()
