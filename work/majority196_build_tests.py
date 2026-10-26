from pathlib import Path
p=Path('work/test_monthly_pairwise_majority196_host.py');assert not p.exists()
s=Path('work/test_monthly_volume_cluster195_host.py').read_text().replace('Synthetic-only195','Synthetic-only196').replace('monthly_volume_cluster195','monthly_pairwise_majority196').replace('volumecluster195','majority196')
s=s.replace('h.volume_source_binding','h.price_source_binding').replace('h.volume_inputs','h.price_inputs')
s=s.replace("v['candidate'][1,0]=.251", "v['candidate'][1,0]=1.001")
s=s.replace("source={'volume':volume}", "source={'adj':volume}").replace("p['volume']", "p['adj']").replace(")[\'volume\']", ")[\'adj\']")
s=s.replace("['volume_count'],273", "['price_count'],253")
s=s.replace("ok=dict(available_cells=3,event_observations=630,adjacent_pair_observations=627,rolling_medians_computed=630)", "ok=dict(source_available_by_event=[3]+[0]*121,source_available_cells=3,available_cells=3,valid_price_observations=759,block_comparisons=36,pair_majority_decisions=3)")
s=s.replace("self.assertEqual(h.estimate_counts(ok)['rolling_medians_computed'],630)", "self.assertEqual(h.estimate_counts(ok)['block_comparisons'],36)")
s=s.replace('for _ in range(192)];rows[0][\'trials\']=902','for _ in range(193)];rows[0][\'trials\']=904')
s=s.replace('test_absolute_control_cannot_promote_and_must_be_beaten','test_balance_control_cannot_promote_and_must_be_beaten')
idx=s.index("    def test_completed_predecessor_state(self):")
s=s[:idx]+'''    def test_single_source_domain_counts_without_pair_support(self):
        counts=dict(source_available_by_event=[1]+[0]*121,source_available_cells=1,available_cells=0,valid_price_observations=253,block_comparisons=0,pair_majority_decisions=0)
        self.assertEqual(h.estimate_counts(counts)['pair_majority_decisions'],0)
        for key,value in [('source_available_cells',2),('available_cells',1),('valid_price_observations',0),('block_comparisons',1)]:
            bad=dict(counts);bad[key]=value
            with self.assertRaises(ValueError):h.estimate_counts(bad)
    def test_prepared_model_receipt_and_pair_counts_are_required(self):
        mask=np.zeros((122,49),bool);mask[0,:3]=True
        counts=dict(source_available_by_event=[3]+[0]*121,source_available_cells=3,available_cells=3,unavailable_cells=5975,valid_price_observations=759,block_comparisons=36,pair_majority_decisions=3)
        feature=dict({k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,computes_returns=False,predictive_model_fits=0,**counts)
        deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
        delta=dict({k:True for k in ['passed','full_candidate_universe','initial_delta_zero','common_available_only','no_eligibility_or_model_filter','centered_cross_sectional_ranking','direct_additive_pre_score','fixed_linear_mapping_checked']},dose=.01,max_absolute_delta=.005,changed_cells={k:0 for k in deltas})
        base=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**{k:counts[k] for k in h.ESTIMATE_FIELDS})
        audit=dict(base,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),feature_prefix_sources_truncated=True,original_orders_exact=True,original_arrays_unchanged=True)
        receipt=dict(base,new_returns_computed=False,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES})
        h.verify_prepared_evidence(receipt,{'available':mask},deltas,audit)
        for target in ('receipt','audit'):
            for key,value in [('new_predictive_model_fits',None),('block_comparisons',35)]:
                c,a=copy.deepcopy(receipt),copy.deepcopy(audit);(c if target=='receipt' else a)[key]=value
                with self.assertRaises(ValueError):h.verify_prepared_evidence(c,{'available':mask},deltas,a)
'''+s[idx:]
p.write_text(s);print('New host synthetic tests written; not run pending modules and source binding.')
