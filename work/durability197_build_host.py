from pathlib import Path
p=Path('work/monthly_recovery_durability197_host.py');assert not p.exists()
s=Path('work/monthly_pairwise_majority196_host.py').read_text()
s=s.replace('monthly_pairwise_majority196','monthly_recovery_durability197').replace('majority196','durability197').replace('PAIRWISE_MAJORITY196','RECOVERY_DURABILITY197')
s=s.replace('Round196: frozen Local67 plus pairwise block-return majority.','Round197: frozen Local67 plus post-recovery durability.').replace('pairwise historical comparisons are descriptive.','historical recovery paths are descriptive.')
s=s.replace("PRIOR = ROOT / 'outputs/monthly_volume_cluster195'","PRIOR = ROOT / 'outputs/monthly_pairwise_majority196'")
s=s.replace("'tournament_majority', 'tournament_balance_control'","'recovery_durability', 'recovery_frequency_control'")
s=s.replace("ESTIMATE_FIELDS = ('valid_price_observations', 'block_comparisons', 'pair_majority_decisions')","ESTIMATE_FIELDS = ('valid_price_observations', 'negative_anchors', 'recovered_anchors', 'durable_anchors')")
a=s.index('MODEL_FIT_ACCOUNTING = ');b=s.index('\nCONTEXT =',a)
s=s[:a]+"MODEL_FIT_ACCOUNTING = 'Zero new predictive model fits; fixed historical loss/recovery/durability counts. Main pass only; overlapping observations are not independent samples.'"+s[b:]
s=s.replace('9d95fc3b4fba608a2751aa0ecaeab0eacaee1d5b9bdfdad6f51024b7af628d3b','89d00c63a1281700e1b55c490011d04c7ced178b29c9bac1f3d430f4389d1923')
s=s.replace('5e1af9637afba453a181597a1de684dee841444825fe453ef13c2a99a84e4c32','PENDING_SOURCE197')
s=s.replace('170af7dd7ac97eb2aaa61e239a1937dd574f159d80040d5ab5d0be9f388a9374','650f8b4ba00c5c8ecc0db16b27cdb611f42bbf0d8337d9ef91e4c2ddc2af6893')
s=s.replace('a5192918944fd2b6d92ab4d88805b2223de188e46e953de6cd2dd1e8fe0e502c','99132f0b7be990b95b8dbaef175e19818ddf45b69f683a26f5a9f5fd6294ac8d')
s=s.replace('fbb68f08a4e65616f73276b6d5f042e66ef443d5873a6bc0d871f3b0efe6a5dc','c7aca0f43f69d864a16714cd0a13a117ccfb66caafc04f034e38ba61d78ed316')
a=s.index('FEATURE_FLAGS = ');b=s.index('\nTEST_RECEIPT',a)
s=s[:a]+"FEATURE_FLAGS = ('passed','independent_scalar_rebuild','negative_anchor_comparisons_checked','first_recovery_checked','post_recovery_survival_checked','matched_frequency_checked','full253_price_support_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs','centered_high_tie_ranks_checked')"+s[b:]
s=s.replace('Original stored adjusted close; exact binary64 rational comparisons; no new adjustment or imputation','Original stored adjusted close; strict losses and exact reclaim/rebreach comparisons; no new adjustment or imputation')
a=s.index('def estimate_counts(check):');b=s.index('\n\ndef feature_prefix_panel',a)
s=s[:a]+'''def estimate_counts(check):
    keys=('source_available_by_event','available_by_event','negative_anchors_by_event','recovered_anchors_by_event','durable_anchors_by_event')
    series={k:check.get(k) for k in keys}
    require(all(isinstance(v,list) and len(v)==122 and all(type(n) is int for n in v) for v in series.values()),'Invalid descriptive event counts')
    for source,available,negative,recovered,durable in zip(*(series[k] for k in keys)):
        require(0<=available<=source<=49 and available<=negative<=210*available and 0<=durable<=recovered<=negative,'Inconsistent recovery support/counts')
    require(type(check.get('source_available_cells')) is int and check['source_available_cells']==sum(series[keys[0]]),'Source coverage differs')
    require(type(check.get('available_cells')) is int and check['available_cells']==sum(series[keys[1]]),'Shared anchor support differs')
    expected=dict(valid_price_observations=253*sum(series[keys[0]]),**{k:sum(series[k+'_by_event']) for k in ESTIMATE_FIELDS[1:]})
    counts={key:check.get(key) for key in ESTIMATE_FIELDS}
    require(all(type(value) is int and value==expected[key] for key,value in counts.items()),'Descriptive recovery/price counts differ')
    return counts
'''+s[b:]
s=s.replace("len(state['completed']) == 194 and sum(r['trials'] for r in state['completed']) == 1099, 'Wrong final counters'","len(state['completed']) == 195 and sum(r['trials'] for r in state['completed']) == 1102, 'Wrong final counters'")
s=s.replace("len(state['completed']) == 193 and sum(r['trials'] for r in state['completed']) == 1096","len(state['completed']) == 194 and sum(r['trials'] for r in state['completed']) == 1099").replace('Expected193 completed entries /1096 trials','Expected194 completed entries /1099 trials').replace('Round195 must','Round196 must')
s=s.replace('actual_block_comparisons_before_registration=0, actual_pair_majority_decisions_before_registration=0','actual_negative_anchors_before_registration=0, actual_recovery_paths_before_registration=0')
s=s.replace('registered_before_new_pairwise_majority_features','registered_before_new_recovery_durability_features').replace('Prepare fixed pairwise majority and matched pooled balance','Prepare fixed post-recovery durability and matched first-recovery frequency').replace('Review pairwise monthly majority versus pooled win balance','Review post-recovery durability versus first-recovery frequency')
s=s.replace('lo,hi=-1.,1.','lo,hi=0.,1.')
s=s.replace('    return transforms,deltas\n','    require((np.asarray(values[\'candidate\'][:n],float)[available[:n]]<=np.asarray(values[\'control\'][:n],float)[available[:n]]).all(),\'Durability cannot exceed recovery frequency\')\n    return transforms,deltas\n')
s=s.replace('original stored adjusted close with exact endpoint comparisons','original stored adjusted close with ordered reclaim and post-recovery price comparisons')
s=s.replace('close_vs_balance','close_vs_frequency').replace('open_vs_balance','open_vs_frequency').replace('pairwise_monthly_majority=True','post_recovery_durability=True')
p.write_text(s)
print('197 host written; pending source/test/review registration')
