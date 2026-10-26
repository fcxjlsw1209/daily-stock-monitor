"""Source-only prospective scope for annual trade working-capital cash components."""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import hashlib, json, os, tempfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pin(rel,digest):
    p=ROOT/rel
    assert sha(p)==digest,rel
    return read(p)
now=datetime.now(timezone.utc).isoformat()
prior=pin('outputs/monthly_cash_ppe193/COMPLETION.json','586821986e6dfc10d7070215c1379f22ddefaaadc42e816b325f972938afa2ef')
assert prior['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and not prior['target_reached']
base=pin('work/capex193_source_scope.json','71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08')
central=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
before=central.read_bytes();state=json.loads(before)
assert state['current']=='monthly_cash_ppe193' and not state['registered_queue']
assert len(state['completed'])==191 and sum(x['trials'] for x in state['completed'])==1090
assert len(base['prospective_targets'])==44 and sum(x['inherited_admitted'] for x in base['prospective_targets'])==31
pins=dict(base['input_sha256'])
for rel,digest in pins.items():assert sha(ROOT/rel)==digest,rel
for rel,digest in {
 'work/capex193_source_scope.json':'71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08',
 'work/capex193_pdf_runtime/RUNTIME.json':'ad437674d06761ef278da476110c1d3aedf386cd5a0276161304627d2140c98a',
 'work/capex193_pdf_runtime/ACQUISITION.json':'477fe83ca171825d760284c9f1480702271c22e0446f179049d672b4e2570d34',
 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json':'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
}.items():assert sha(ROOT/rel)==digest,rel;pins[rel]=digest
scope=dict(schema='trade_working_capital194_source_scope_v1',created_utc=now,
 scope='All44 originally pinned annual reports, before measuring new financial components; preserve original13 holds. New193 cash-PPE availability is not the eligibility gate for194.',
 prospective_targets=deepcopy(base['prospective_targets']),expected_event_count=44,inherited_admitted_count=31,
 input_sha256=pins,clock=base['clock'],use_comparison_flows=False,
 required_components={
  'accounts_receivable':'Signed current annual operating-cashflow reconciliation contribution from trade accounts receivable, net if explicitly reported. Not a balance-sheet amount; exclude vendor/nontrade or combined other-receivable rows.',
  'inventories':'Signed current annual operating-cashflow reconciliation contribution from inventories. Not a balance-sheet inventory amount.',
  'accounts_payable':'Signed current annual operating-cashflow reconciliation contribution from trade accounts payable. Reject rows combined with accrued expenses, other liabilities, taxes or deferred revenue.'},
 evidence_required=['Original unchanged source/receipt hash and URL','Exact operating cashflow section, label, full row and row hash','Current annual column and period matching original current CFO','Signed displayed literal and parsed signed decimal without summing components','Same statement scale and currency evidence; missing row currency remains missing with linked statement evidence','Release date and inherited first subsequent exchange session','Separate raw extraction success and retained original admission status','PDF current annual column geometry plus visual review and full isolated runtime pins'],
 holds=['Any inherited original146 source/period/currency/identity hold remains held.', 'All three distinct components must be explicitly disclosed in same current annual report and same consolidated scope as CFO.', 'Missing, combined or ambiguous row/annual column/sign/unit/scope stays held; never infer a missing component as zero.', 'A dash can be parsed as zero only if it occupies the confirmed numeric current annual cell; preserve the raw dash and evidence.', 'Do not use changes in balance sheet balances, comparison-year flows, total working capital, adjusted marketing CFO or future filings as substitutes.', 'Do not merge continuing/discontinued scopes or select one opportunistically.'],
 independent_validation='Separate source-cell reconstruction using independent extraction logic. Preserve source currency and accounting caveats. All candidate rows reviewed before any factor or portfolio run.',
 source_limitations=base['source_limitations'][:4]+['Three trade operating-cash contributions are not full working capital; residual CFO is not GAAP CFO or organic cash generation.', 'Only four issuers; different accounting scope, acquisitions and reclassification can affect comparability.'],
 new_financial_ratios_permitted=False,factor_computation_permitted=False,return_computation_permitted=False,observer_changes_permitted=False,
 component_sum_permitted=False,research_hypothesis_only='Test whether removing trade collection, inventory and supplier-payment cash timing contributions improves a small auxiliary ranking correction; no current amounts or outcomes used to fit its definition.')
scopepath=ROOT/'work/working_capital194_source_scope.json'
conceptpath=ROOT/'work/working_capital194_experiment_concept.json'
assert not scopepath.exists() and not conceptpath.exists()
write(scopepath,scope)
concept=dict(schema_version=1,round=194,created_utc=now,status='PROSPECTIVE_CONCEPT_PENDING_SOURCE_ADMISSION_AND_FROZEN_PLAN',registered=False,
 source_scope=dict(path=str(scopepath.relative_to(ROOT)),sha256=sha(scopepath)),
 arms=['incumbent_local67','same_support_cashflow_control','cash_before_trade_working_capital'],
 purpose='Independent auxiliary financial confirmation within unchanged long-momentum month-end EW3; not claimed author reconstruction.',
 measurement=dict(W='Current signed cashflow contributions of accounts receivable + inventories + accounts payable, all three required from same current annual report.',
 candidate='(current annual CFO - W) / ((current assets + comparison assets from same report) / 2)',control='current annual CFO / same average assets',
 bounded='h(x)=x/(1+abs(x)) using exact scaled rational arithmetic before final bounded float',dose=0.01,
 placement='On original eligible and common-available cells add dose*h to original Local67 pre-score then exactly one unchanged original episode selector; no new first-row suppression.',
 annual_duration_days=[330,400],max_age_days_from_fiscal_end=550,annualize_flow=False,use_comparison_flows=False,zero_missing_values=False,
 latest_report_before_validity=True,held_latest_blocks_older=True,original146_net_income_gate_retained=True,
 same_support='Both cashflow arms require all three component rows plus original annual-source admission; otherwise both exactly fall back to incumbent.'),
 validation=dict(first_training_months=88,early_training_months=53,recent_training_months=35,later_months=34,later34_selection=False,
 walkforward_prefixes=[[36,60],[60,76],[76,88]],parameter_search=False,new_model_fits=0,
 guard_source=dict(path='work/monthly_same_report145_runner.py',sha256='f400350c20a6d95f1d7ed60753ff529b5544b2b8901abbd0e801999843500939',function='guard'),
 same_close_admission='Strictly beat both incumbent and same-support CFO in first88 CAGR; pass original145 risk and early/recent guards.',
 next_open_cost_admission='CAGR no worse than incumbent and strictly better than same-support CFO; same original145 risk and early/recent guards.',
 paths=[dict(execution='same_close',side_cost=0.0),dict(execution='next_open',side_cost=0.0),dict(execution='next_open',side_cost=0.0005)],
 target='2016-08-17 to2026-09-24 full-period CAGR>=70%; full-period diagnostic gain cannot override failed first88 selection.'),
 invariants=['Long momentum remains dominant and skips42 recent trading days','Original27 arrays, eligibility and models unchanged','Month-end EW3, two each half, less than two cash; fixed intramonth holdings','No observer changes or downloads inside replay'],
 data_limitations=scope['source_limitations']+['Historical periods already reused extensively; later34 and walk-forward are not pristine independent OOS.'],
 actual_new_factors=0,actual_new_orders=0,actual_new_returns=0,
 execution_gate='Source admission, independent source review, implementation tests and hash-bound numerical PLAN are required before new ratios/features/orders/returns.')
write(conceptpath,concept)
receipt=dict(created_utc=now,source_scope=dict(path=str(scopepath.relative_to(ROOT)),sha256=sha(scopepath)),concept=dict(path=str(conceptpath.relative_to(ROOT)),sha256=sha(conceptpath)),script_sha256=sha(Path(__file__)),new_actual_ratios=0,new_factors=0,new_orders=0,new_returns=0,observer_changes=0)
write(ROOT/'work/WORKING_CAPITAL194_SOURCE_START.json',receipt)
assert central.read_bytes()==before
backup=central.parent/('PRE_WORKING_CAPITAL194_SOURCE_'+hashlib.sha256(before).hexdigest()+'.json')
with backup.open('xb') as f:f.write(before)
state.update(next_source_preflight=receipt,next_step='Audit all44 original current annual trade receivables/inventory/payables cashflow rows, preserving13 inherited holds, before194 numerical registration.',updated_utc=now,last_goal_turn_classification='progress')
with tempfile.NamedTemporaryFile(dir=central.parent,delete=False) as f:
 tmp=Path(f.name);f.write((json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
assert central.read_bytes()==before
os.replace(tmp,central)
print(json.dumps(receipt))
