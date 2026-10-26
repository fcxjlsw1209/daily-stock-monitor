"""Metadata-only249 host/contract binding check; no source bars or NPZ."""
import monthly_volume_breadth249_host as h
h.source_review_check(h.read(h.SOURCE_REVIEW))
s=h.read(h.CENTRAL);h.initial_state(s);h.predecessor_binding(s)
assert not s['registered_queue']
c=h.read(h.CONTEXT)
assert c['arms']==list(h.ARMS) and c['registered_trials']==3 and c['new_predictive_model_fits']==0
assert c['selection']['training_cagr_comparison_atol']==1e-12
assert all(c['selection'][k]==0 for k in ('sharpe_comparison_atol','max_drawdown_comparison_atol','era_cagr_comparison_atol'))
print('PASS: source/context/predecessor metadata binding; no feature/order/return computation')
