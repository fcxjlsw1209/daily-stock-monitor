from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
old=(ROOT/'work/test_monthly_sector_recurrence192_host.py').read_text()
tree=ast.parse(old)
def node(name):
 n=next(x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==name)
 return ''.join(old.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
head=old[:old.index('def fixture():')].replace('sectorrecurrence192','volumecluster195').replace('monthly_sector_recurrence192','monthly_volume_cluster195').replace('Synthetic-only tests of192 direct signed-block transforms, frozen-score wiring and governance.','Synthetic-only195 rank-slot wiring, selection and source gate tests.')
fixture="""def add_ranks(values):
    for key in ('candidate','control'):
        result=np.zeros_like(values[key],dtype=float)
        for i,valid in enumerate(values['available']):
            ids=np.flatnonzero(valid)
            if len(ids)>1:
                for j in ids:
                    x=values[key][i,ids]
                    result[i,j]=(np.count_nonzero(x<values[key][i,j])+(np.count_nonzero(x==values[key][i,j])-1)/2)/(len(ids)-1)-.5
        values[key+'_rank']=result
    return values

def fixture():
    shape=(6,5)
    values=dict(candidate=np.tile([-.2,-.1,0.,.1,.2],(6,1)),control=np.tile([.9,.7,.5,.3,.1],(6,1)),available=np.ones(shape,bool))
    values['available'][2,:4]=False;values['available'][3,:]=False
    for key in ('candidate','control'):values[key][~values['available']]=np.nan
    add_ranks(values)
    a=dict(baseline=np.tile([.2,.199,.198,.197,.196],(6,1)),eligible=np.ones(shape,bool),effects=np.zeros(shape),core=np.zeros(shape),core_weight=np.full((6,1),.52),state_calm10=np.zeros(6,bool),state_weak_narrow=np.zeros(6,bool))
    a['eligible'][:,-1]=False
    return a,dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE') for _ in range(6)]),values
"""
score="""class ScoreTests(unittest.TestCase):
    def test_higher_rank_exact_bound_ineligible_receives_original_slot(self):
        a,m,v=fixture();r,d=h.scale_and_delta(v,6)
        np.testing.assert_array_equal(r[h.NAMES[1]][1],[-.5,-.25,0.,.25,.5])
        np.testing.assert_array_equal(d[h.NAMES[1]][1],[-.005,-.0025,0.,.0025,.005])
        self.assertFalse(a['eligible'][1,-1]);self.assertEqual(d[h.NAMES[1]][1,-1],.005)
        self.assertTrue(np.all(d[h.NAMES[1]][0]==0))
    def test_missing_singleton_and_ties_neutral(self):
        _,_,v=fixture();v['candidate'][1]=.1;add_ranks(v);r,d=h.scale_and_delta(v,6)
        for i in (1,2,3):np.testing.assert_array_equal(d[h.NAMES[1]][i],np.zeros(5))
    def test_average_ties(self):
        _,_,v=fixture();v['candidate'][1]=[-.2,0.,0.,0.,.2];add_ranks(v)
        r,d=h.scale_and_delta(v,6);np.testing.assert_array_equal(r[h.NAMES[1]][1],[-.5,0.,0.,0.,.5])
    def test_corrupt_rank_bounds_missing_and_raw_rejected(self):
        for kind in ('rankwrong','rankinf','rankbound','missingrank','rawnan','rawinf','rawbound','mask','shape'):
            _,_,v=fixture()
            if kind=='rankwrong':v['candidate_rank'][1,0]=0.
            if kind=='rankinf':v['candidate_rank'][1,0]=np.inf
            if kind=='rankbound':v['candidate_rank'][1,0]=.5000001
            if kind=='missingrank':v['candidate_rank'][3,0]=.1
            if kind=='rawnan':v['candidate'][1,0]=np.nan
            if kind=='rawinf':v['control'][1,0]=np.inf
            if kind=='rawbound':v['candidate'][1,0]=.251
            if kind=='mask':v['available']=v['available'].astype(int)
            if kind=='shape':v['control_rank']=v['control_rank'][:,:-1]
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,6)
    def test_future_feature_values_not_inspected(self):
        _,_,v=fixture();rr,dd=h.scale_and_delta(v,6)
        for n in range(7):
            vv=copy.deepcopy(v)
            for key in ('candidate','control','candidate_rank','control_rank'):
                vv[key]=vv[key].astype(object);vv[key][n:]='future poison'
            r,d=h.scale_and_delta(vv,n)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(r[arm],rr[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
    def test_invalid_prefix(self):
        _,_,v=fixture()
        for n in (-1,7,True,1.5,np.int64(2)):
            with self.assertRaises(ValueError):h.scale_and_delta(v,n)
    def test_full_score_exactly_one_episode_no_input_mutation(self):
        a,m,v=fixture();aa,vv=copy.deepcopy(a),copy.deepcopy(v);study=SyntheticStudy();policy={'synthetic':True}
        original=study.modified_pre_score(dict(id=4),a,6)
        with patch.object(h,'read',return_value=policy):orders,r,d=h.decisions(study,a,m,v,6)
        self.assertEqual(study.adapter.score_and_choose.call_count,2);self.assertEqual(study.ep.select_boundary.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            expected=original.copy();mask=d[arm]!=0;expected[mask]+=d[arm][mask]
            np.testing.assert_array_equal(call.kwargs['pre_score'],expected);self.assertIs(call.kwargs['episode_policy'],policy)
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
        for k in a:np.testing.assert_array_equal(a[k],aa[k])
        for k in v:np.testing.assert_array_equal(v[k],vv[k])
    def test_all_decision_prefixes_and_zero_missing_fallback(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):
            oo,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                small,_,_=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(small[arm],oo[arm][:n])
            for key in ('candidate','control'):v[key][:]=np.nan
            v['available'][:]=False;add_ranks(v)
            neutral,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            self.assertTrue(all(neutral[arm]==neutral[h.NAMES[0]] for arm in h.NAMES))
"""
sel=node('SelectionAndBindingTests');parsed=ast.parse(sel)
for name in ('test_raw_bounds_enforced_and_equal_arms_stay_exact','test_completed_predecessor_and_counters_required'):
 n=next(x for x in ast.parse(sel).body[0].body if isinstance(x,ast.FunctionDef) and x.name==name);lines=sel.splitlines(keepends=True);sel=''.join(lines[:n.lineno-1]+lines[n.end_lineno:])
source="""class SourceAndLifecycleTests(unittest.TestCase):
    def test_metadata_source_binding_pins(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.volume_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64):
            with self.assertRaises(ValueError):h.volume_source_binding(raw)
        missing=dict(raw);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):h.volume_source_binding(missing)
    def test_independent_review_gate_rejects_stale_or_failed(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        with patch.object(h,'read',return_value={'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with self.assertRaises(ValueError):h.code_review_binding()
    def test_original_volume_and_prefix_zero(self):
        volume=pd.DataFrame({'A':[1.,2.,3.]},index=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06']))
        source={'volume':volume};m={'etfs':[str(i) for i in range(17)],'dates':['2020-01-03','2020-01-06']}
        p=h.volume_inputs(source,m);self.assertIs(p['volume'],volume)
        self.assertEqual(len(h.feature_prefix_panel(p,m,0)['volume']),0)
        self.assertEqual(len(h.feature_prefix_panel(p,m,1)['volume']),2)
        with self.assertRaises(ValueError):h.volume_inputs({},m)
    def test_counts_source_period_and_contract_fixed(self):
        c=h.contract();self.assertEqual(c['dose'],.01);self.assertEqual(c['max_absolute_delta'],.005)
        self.assertEqual(c['source_window']['volume_count'],273);self.assertEqual(c['source_window']['latest_offset'],42)
        self.assertIn('UNKNOWN',c['context']['novelty_scope']);self.assertIn(h.CONTEXT,h.code_files())
        ok=dict(available_cells=3,event_observations=630,adjacent_pair_observations=627,rolling_medians_computed=630)
        self.assertEqual(h.estimate_counts(ok)['rolling_medians_computed'],630)
        for key in h.ESTIMATE_FIELDS:
            bad=dict(ok);bad[key]+=1
            with self.assertRaises(ValueError):h.estimate_counts(bad)
    def test_completed_predecessor_state(self):
        rows=[dict(trials=1,directory='old') for _ in range(192)];rows[0]['trials']=902;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state);state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
    def test_gate_before_real_loader_and_no_missing_review(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaises(ValueError):h.prepare('bad')
        loader.assert_not_called()
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        register=ast.get_source_segment(code,next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='register'))
        self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))
"""
s=head+fixture+'\n\n'+node('SyntheticStudy')+'\n\n'+score+'\n\n'+node('selection_fixture')+'\n\n'+sel+'\n\n'+source+"\nif __name__=='__main__':unittest.main()\n"
ast.parse(s);p=ROOT/'work/test_monthly_volume_cluster195_host.py';assert not p.exists();p.write_text(s);print(p)
