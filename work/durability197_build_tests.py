from pathlib import Path
p=Path('work/test_monthly_recovery_durability197_host.py');assert not p.exists()
s=Path('work/test_monthly_pairwise_majority196_host.py').read_text().replace('Synthetic-only196','Synthetic-only197').replace('monthly_pairwise_majority196','monthly_recovery_durability197').replace('majority196','durability197')
s=s.replace("candidate=np.tile([-.2,-.1,0.,.1,.2],(6,1)),control=np.tile([.9,.7,.5,.3,.1],(6,1))","candidate=np.tile([0.,.2,.4,.6,.8],(6,1)),control=np.tile([.99,.98,.97,.96,.95],(6,1))")
s=s.replace("v['candidate'][1]=[-.2,0.,0.,0.,.2]","v['candidate'][1]=[0.,.4,.4,.4,.8]")
s=s.replace('test_balance_control_cannot_promote_and_must_be_beaten','test_frequency_control_cannot_promote_and_must_be_beaten')
s=s.replace("ok=dict(source_available_by_event=[3]+[0]*121,source_available_cells=3,available_cells=3,valid_price_observations=759,block_comparisons=36,pair_majority_decisions=3)","ok=counts_fixture()")
s=s.replace("self.assertEqual(h.estimate_counts(ok)['block_comparisons'],36)","self.assertEqual(h.estimate_counts(ok)['negative_anchors'],4)")
a=s.index('    def test_single_source_domain_counts_without_pair_support(self):');b=s.index('    def test_prepared_model_receipt_and_pair_counts_are_required(self):',a)
s=s[:a]+'''    def test_valid_source_without_negative_anchors(self):
        counts=counts_fixture(source=1,available=0,negative=0,recovered=0,durable=0)
        self.assertEqual(h.estimate_counts(counts)['negative_anchors'],0)
        for key,value in [('source_available_cells',2),('available_cells',1),('valid_price_observations',0),('negative_anchors',1)]:
            bad=dict(counts);bad[key]=value
            with self.assertRaises(ValueError):h.estimate_counts(bad)
    def test_singleton_valid_recovery_counts_and_bounds(self):
        h.estimate_counts(counts_fixture(source=1,available=1,negative=1,recovered=0,durable=0))
        for kw in [dict(source=1,available=2),dict(negative=0),dict(negative=631),dict(recovered=5),dict(durable=4)]:
            with self.assertRaises(ValueError):h.estimate_counts(counts_fixture(**kw))
    def test_durability_above_recovery_is_rejected(self):
        _,_,v=fixture();v['candidate'][1,4]=.96;add_ranks(v)
        with self.assertRaisesRegex(ValueError,'Durability cannot exceed'):h.scale_and_delta(v,6)
'''+s[b:]
s=s.replace('test_prepared_model_receipt_and_pair_counts_are_required','test_prepared_model_receipt_and_recovery_counts_are_required')
s=s.replace("counts=dict(source_available_by_event=[3]+[0]*121,source_available_cells=3,available_cells=3,unavailable_cells=5975,valid_price_observations=759,block_comparisons=36,pair_majority_decisions=3)","counts=dict(counts_fixture(),unavailable_cells=5975)")
s=s.replace("('block_comparisons',35)","('negative_anchors',3)")
s=s.replace("for _ in range(193)];rows[0]['trials']=904","for _ in range(194)];rows[0]['trials']=906")
a=s.index('\n\nclass SyntheticStudy:')
s=s[:a]+'''\n\ndef counts_fixture(source=3,available=3,negative=4,recovered=3,durable=2):
    event=dict(source_available=source,available=available,negative_anchors=negative,recovered_anchors=recovered,durable_anchors=durable)
    return dict(source_available_by_event=[source]+[0]*121,available_by_event=[available]+[0]*121,negative_anchors_by_event=[negative]+[0]*121,recovered_anchors_by_event=[recovered]+[0]*121,durable_anchors_by_event=[durable]+[0]*121,source_available_cells=source,available_cells=available,valid_price_observations=253*source,negative_anchors=negative,recovered_anchors=recovered,durable_anchors=durable)
'''+s[a:]
p.write_text(s)
p=Path('work/monthly_recovery_durability197_host.py');s=p.read_text();assert s.count('PENDING_SOURCE197')==1;p.write_text(s.replace('PENDING_SOURCE197','2cc71fdf86d63bbb68c63149d078cbd6d686423b0f234e7f3c2e8b08fb6156cb'))
print('New197 host tests and source binding, not registered')
