"""Prespecified MU/PEP source batch; never rank, score or compute returns."""
import argparse
import copy
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import importlib
import io
import re
import unittest

import after149_source_batch as shared

ROOT = shared.ROOT
OUT = ROOT / 'outputs/monthly_financial_coverage_batch4_v1'
CONTEXT = 'work/after152_next_source_context.json'
CONTEXT_SHA = 'f48acb7e7e57d1212b03bcf435b510dd75670f334a3bc228ceacb22ce65bfea7'
FEASIBILITY = 'work/after152_next_source_feasibility.json'
FEASIBILITY_SHA = 'c79d523697ecbb570ac99a2ecddeebc998db6ab77c92cd23697cf70596c5b468'
POLICY = 'work/after152_batch4_admission_policy.json'
PREDECESSOR = 'outputs/monthly_financial_coverage152/COMPLETION.json'
READY = 'standard_HTML_annual_header_context_located_not_admitted'
MODULES = {'MU': 'after152_mu_extract', 'PEP': 'after152_pep_extract'}
FIELDS = shared.FIELDS
check = shared.check
BASE_INPUTS = (CONTEXT, FEASIBILITY, 'work/after149_coverage_extension_concept.json',
    'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json',
    'outputs/monthly_financial_coverage_admission_batch3_v1/DATASET.json',
    'outputs/monthly_financial_coverage_admission_batch3_v1/ADMISSION.json',
    'outputs/monthly_financial_coverage_admission_batch3_v1/MANIFEST.json')


def records(context):
    docs = context['events']
    check(len(docs) == len({x['event_id'] for x in docs}) == 23, 'Complete distinct23-event catalog required')
    check([(x['symbol'], x['fiscal_year']) for x in docs] ==
          [('MU', y) for y in range(2019, 2026)] + [('PEP', y) for y in range(2010, 2026)],
          'Changed full catalog order or scope')
    chosen = [x for x in docs if x.get('context_status') == READY]
    check([(x['symbol'], x['fiscal_year']) for x in chosen] ==
          [('MU', 2024), ('MU', 2025)] + [('PEP', y) for y in range(2010, 2017)],
          'Changed prespecified nine-report scope')
    check(all(x.get('source_kind') == 'HTML_original' and x.get('original_report_bytes_verified') is True
              for x in chosen), 'Only verified original HTML may enter extraction')
    return chosen


def catalog_metadata(context):
    keys = ('event_id', 'symbol', 'fiscal_year', 'publication_date', 'available_session',
            'available_session_verified', 'official_actual_period_end', 'source', 'source_kind',
            'original_report_bytes_verified', 'context_status', 'context_holds')
    return [{k: copy.deepcopy(x[k]) for k in keys} for x in context['events']]


def make_policy(context, inputs, created_utc=None):
    """Pure prospective policy construction; receives only metadata/hash pins."""
    records(context)
    check(set(inputs) == set(BASE_INPUTS), 'Incomplete policy input closure')
    check(all(re.fullmatch('[0-9a-f]{64}', h) for h in inputs.values()), 'Invalid policy digest')
    check(inputs[CONTEXT] == CONTEXT_SHA and inputs[FEASIBILITY] == FEASIBILITY_SHA, 'Changed prepared metadata')
    boundaries = {s: dict(enabled=True, not_a_quoted_fact=True,
        assumption_id=s+'_consecutive_annual_boundaries_v1',
        formula='derived start = same-report prior annual assets date + one calendar day',
        days_inclusive=[364, 371],
        required_checks=['Explicit annual NI header with full fiscal end date.',
            'NI end, current assets end and catalog end agree; prior assets are the preceding annual end.',
            'Independent physical header evidence for both ends; no quarter or stub substitution.',
            'Independent review explicitly verifies the derived boundary assumption.']) for s in MODULES}
    return dict(schema_version=1, policy_id='mu_pep_annual_ni_same_report_assets_batch4_v1',
        stage='prospective_source_policy_not_strategy_registration', created_utc=created_utc or shared.now(),
        definition='current_annual_net_income_average_assets_v1',
        purpose='Prepare annual consolidated NI and same-report asset endpoints; do not compute their ratio.',
        source_scope=dict(symbols=list(MODULES), catalog_events=23, html_context_reviewed_events=9,
            source_context=dict(path=CONTEXT, sha256=CONTEXT_SHA),
            source_feasibility=dict(path=FEASIBILITY, sha256=FEASIBILITY_SHA),
            event_selection='Next fixed alphabetic issuer pair; no returns, positions or rankings selection.',
            extraction_scope='MU FY2024–25 and PEP FY2010–16 only; remaining14 catalog entries held unprocessed.',
            catalog_requirement='All23 entries retained in source order; latest held or unprocessed blocks older fallback.',
            no_automatic_admission=True),
        required_fields=list(FIELDS), fixed_catalog_metadata=catalog_metadata(context),
        net_income_scope=dict(required='Annual consolidated GAAP total net income before parent/noncontrolling attribution.',
            MU='Exact Net income or Net income (loss) in annual formal consolidated operations column.',
            PEP='Exact total Net income in annual formal consolidated income statement; not attributable-to-PepsiCo NI.',
            prohibited='No quarter/YTD summing, EPS, operating income, adjusted/non-GAAP substitution, annualization or TTM.',
            negative_and_zero='Preserve signed annual loss and exact zero; no profitable-company filter.'),
        periods=dict(assets='Same current report: annual closing and preceding annual closing Total assets.',
            income='Complete annual period only; preserve52/53-week364/371 days, never assume365.',
            boundary_assumptions=boundaries,
            representation='Starts remain explicit proposed assumptions, not quoted or certified facts.'),
        numeric=dict(income='Finite exact signed decimal with displayed literal and physical sign geometry.',
            assets='Finite strictly positive total consolidated assets required for later intended use.',
            scale='Verify printed field/table scale independently; fixed candidates use millions.',
            currency='Only shared report dollar-symbol unit assumption; currency_iso remains null.',
            source_only='No ratios, growth, ranks, models, scores, portfolios, orders or returns.'),
        known_time=dict(available_from='First pinned exchange session strictly after publication; never the period end.',
            unverified_early_calendar='Keep outside-calendar publication as a time hold; do not infer weekdays or sessions.',
            publication='Mapping a supplied publication date is not independent first-public-time certification.',
            catalog_known_rule='Verified available_session<=signal; unverified publication<signal conservatively blocks fallback.',
            monthly_selection='Select latest known catalog event before validity or age checks; unresolved ties hold.',
            maximum_age_days=550, age_from='current annual period end',
            identity='Own report-local issuer/ticker anchors only, from availability through period end+550days; no certified security master.'),
        accounting=dict(review='Retain all context holds; require field-specific independent evidence for NI and both assets.',
            unresolved='Missing, unresolved or scope-changing accounting evidence keeps the event held.',
            no_keyword_clearance='Keywords neither automatically clear nor automatically reject accounting comparability.',
            PEP='Reclassification comments require field-scope review; consolidated NI differs from parent-attributable NI.',
            MU='Unit-attribution tool gaps in earlier HTML are not evidence that reports lack units.',
            missing='Index-only or absent-table pages do not establish that an annual report itself is absent.'),
        independent_review_contract=dict(top_level=['passed','plan_sha256','extractions_sha256','input_hashes','event_reviews'],
            event_flags=['passed','metadata_verified','identity_verified','periods_verified','currency_verified'],
            boundary='Both issuers require boundary_assumption_verified with source header/date evidence.',
            field_flags='Nonnull: value, sign, scale, period and locator independently verified. Null: precise cause independently verified.',
            population='All9 extracted events plus all23 catalog identities; no successful-row subset.',
            source_binding='PLAN,policy,context,raws,receipts,pinned calendar and all code/tests.',
            scope='Extraction/review alone never confers admission; preserve early-time and unresolved-field holds.'),
        preservation=dict(previous_batches='Reference old44 plus batch1/2/3 records, decisions and full catalogs unchanged; preserve3904 stock-month axis.',
            all_context_holds=True, no_observer_changes=True, no_new_strategy_trials=True,
            first_public_version_verified=False, first_public_time_verified=False,
            complete_correction_history_verified=False, certified_daily_security_master=False),
        input_hashes=copy.deepcopy(inputs))


def validate_policy(policy, context, context_sha):
    check(context_sha == CONTEXT_SHA, 'Changed context digest')
    expected = make_policy(context, policy['input_hashes'], policy['created_utc'])
    check(policy == expected, 'Changed prospective policy or catalog contract')


def validate_record(value, context):
    for key in ('symbol','event_id','fiscal_year','source','context_holds','publication_date'):
        check(value.get(key) == context[key], 'Changed event '+key)
    check(value.get('accounting_review_flags') == context.get('accounting_terms', []) and
          value.get('accounting_context_numbers_redacted') == context.get('accounting_context_numbers_redacted', []), 'Changed accounting context')
    for key in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified',
                'first_public_time_verified','complete_correction_history_verified','certified_daily_security_master'):
        check(value.get(key) is False, 'Premature admission or certification: '+key)
    check(type(value.get('financial_calculations')) is int and value['financial_calculations'] == 0, 'Unexpected financial calculations')
    check(value.get('available_from') == context['available_session'], 'Changed availability')
    check(set(value['fields']) == set(FIELDS), 'Changed fields')
    p, c = value['periods'], context['periods']
    start, end, prior = (date.fromisoformat(c[k]) for k in ('proposed_current_start','current_end','prior_end'))
    days = (end-start).days+1
    check(prior+timedelta(days=1) == start and days in (364,371), 'Invalid52/53-week boundaries')
    check(end.isoformat() == context['official_actual_period_end'] and end.year == context['fiscal_year'] and
          prior.year == end.year-1, 'Changed annual year-end')
    current = dict(period_start=str(start),period_end=str(end),days_inclusive=days)
    check(p['current'] == current and p['comparison']['period_end'] == str(prior) and
          p['start_basis'] == 'derived_same_report_prior_annual_end_plus_one_not_directly_quoted' and
          p['start_assumption_certified'] is False and p['assumption_id'] == context['symbol']+'_consecutive_annual_boundaries_v1',
          'Changed extracted periods or assumption')
    gaps = value.get('gaps', [])
    check(all(isinstance(g,dict) and g.get('field') in FIELDS and isinstance(g.get('reason'),str) and g['reason'] for g in gaps), 'Invalid gap record')
    check(len({g['field'] for g in gaps}) == len(gaps), 'Duplicate field gap')
    for name, field in value['fields'].items():
        if field is None:
            check(any(g['field']==name for g in gaps), 'Null field needs explicit gap')
            continue
        check(not any(g['field']==name for g in gaps), 'Nonnull field has contradictory gap')
        check(field.get('fact_admitted') is False and field.get('source') == context['source'], 'Changed field source/admission')
        check(field.get('available_from') == context['available_session'], 'Changed field clock')
        check(isinstance(field.get('value_decimal'), str) and isinstance(field.get('scale'), str), 'Exact decimal strings required')
        number, scale = Decimal(field['value_decimal']), Decimal(field['scale'])
        check(number.is_finite() and scale.is_finite() and scale>0, 'Invalid amount or scale')
        check(field['scale'] == context['fields'][name]['scale'], 'Changed source scale')
        check(field.get('currency_literal')=='$' and 'currency_iso' in field and field['currency_iso'] is None, 'Changed currency assumption')
        check(isinstance(field.get('displayed_amount'),str) and field['displayed_amount'] and bool(field.get('locator')), 'Missing literal/physical evidence')
        check(field.get('as_of') == str(prior if name=='prior_total_assets' else end), 'Changed endpoint')
        check(field.get('period') == (current if name=='annual_net_income' else None), 'Changed period')
        if name != 'annual_net_income':check(number>0, 'Nonpositive assets held, never synthesized')
    check(value.get('complete_required_fields') is (not gaps), 'Inconsistent completeness')


def synthetic_tests():
    results=[]
    for name in ['test_after152_source_batch']+['test_'+m for m in MODULES.values()]:
        stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=0).run(unittest.defaultTestLoader.loadTestsFromName(name))
        check(r.wasSuccessful() and r.testsRun>0,name+'\n'+stream.getvalue())
        results.append(dict(module=name,tests=r.testsRun,passed=True,output=stream.getvalue()))
    return results


def _pins(mapping, target):
    for raw, digest in mapping.items():
        path=shared.relative(raw)
        check(re.fullmatch('[0-9a-f]{64}',digest) and shared.sha(path)==digest,'Changed dependency: '+path)
        check(path not in target or target[path]==digest,'Conflicting dependency: '+path)
        target[path]=digest


def _record_pins(context):
    return [dict(symbol=x['symbol'],event_id=x['event_id'],fiscal_year=x['fiscal_year'],
                 source_path=x['source']['path'],source_sha256=x['source']['sha256']) for x in records(context)]


def _code_files():
    return [__file__,'work/test_after152_source_batch.py']+[p for module in MODULES.values() for p in ('work/'+module+'.py','work/test_'+module+'.py')]


def freeze(predecessor_sha, parser_review_path, parser_review_sha):
    check(not (OUT/'PLAN.json').exists(), 'PLAN already exists')
    check(shared.sha(CONTEXT)==CONTEXT_SHA and shared.sha(FEASIBILITY)==FEASIBILITY_SHA,'Changed prepared metadata')
    context=shared.load(CONTEXT);records(context);policy=shared.load(POLICY);validate_policy(policy,context,CONTEXT_SHA)
    bindings={CONTEXT:CONTEXT_SHA,POLICY:shared.sha(POLICY)}
    _pins(context['input_hashes'],bindings);_pins(policy['input_hashes'],bindings)
    for event in context['events']:
        s=event['source'];_pins({s['path']:s['sha256']},bindings)
        check((s['receipt_path'] is None)==(s['receipt_sha256'] is None),'Partial receipt reference')
        if s['receipt_path'] is not None:_pins({s['receipt_path']:s['receipt_sha256']},bindings)
        else:check(event['source_kind']!='HTML_original','Original report missing receipt')
    state=shared.load('outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    check(len(state['completed'])==152 and not state['registered_queue'],'Changed research stage')
    check(state['completed'][-1]['directory']=='monthly_financial_coverage152' and
          state['completed'][-1]['completion']['sha256']==predecessor_sha,'Wrong predecessor')
    _pins({PREDECESSOR:predecessor_sha,parser_review_path:parser_review_sha},bindings)
    review=shared.load(parser_review_path)
    check(review.get('passed') is True,'Parser review not passed')
    review_pins=review.get('input_hashes',review.get('source_hashes',{}))
    check(bool(review_pins),'Parser review lacks code pins');_pins(review_pins,bindings)
    parser_files=['work/'+module+'.py' for module in MODULES.values()]+['work/test_'+module+'.py' for module in MODULES.values()]
    for path,digest in shared.code_closure(parser_files).items():
        check(shared.relative(path) in {shared.relative(p) for p in review_pins} and bindings[path]==digest,'Parser review missing code dependency: '+path)
    code=shared.code_closure(_code_files());tests=synthetic_tests()
    check(code==shared.code_closure(_code_files()),'Code changed during tests');_pins(bindings,{})
    plan=dict(schema_version=1,created_utc=shared.now(),runtime=shared.runtime(),
        purpose='Source-only MU/PEP annual NI and same-report annual total assets',context=CONTEXT,policy=POLICY,
        modules=MODULES,fields=list(FIELDS),catalog_ref=dict(path=CONTEXT,sha256=CONTEXT_SHA),catalog_events=23,
        catalog_ids=[x['event_id'] for x in context['events']],catalog_metadata=catalog_metadata(context),
        records=_record_pins(context),inputs=bindings,code=code,synthetic_tests=tests,
        predecessor=dict(path=PREDECESSOR,sha256=predecessor_sha),parser_review=dict(path=shared.relative(parser_review_path),sha256=parser_review_sha),
        no_admission_by_extraction=True,new_strategy_trials=0,preserve_all_prior_source_holds=True,
        preserve_unprocessed_catalog_events=True,no_observer_changes=True,no_network=True,no_scores_or_returns=True)
    shared.write_new(OUT/'PLAN.json',plan)
    print(shared.json.dumps(dict(plan=str((OUT/'PLAN.json').relative_to(ROOT)),sha256=shared.sha(OUT/'PLAN.json'),records=9)))


def run(plan_sha):
    check(shared.sha(OUT/'PLAN.json')==plan_sha,'Changed PLAN');plan=shared.load(OUT/'PLAN.json')
    check(plan['runtime']==shared.runtime(),'Changed runtime')
    bindings={};_pins(plan['inputs'],bindings);_pins(plan['code'],bindings)
    check(not (OUT/'EXTRACTIONS.json').exists() and not (OUT/'EXTRACTION_ERRORS.json').exists(),'Completed/failed extraction immutable')
    full=shared.load(CONTEXT);validate_policy(shared.load(POLICY),full,CONTEXT_SHA)
    check(plan['records']==_record_pins(full) and plan['catalog_metadata']==catalog_metadata(full) and
          plan['catalog_ids']==[x['event_id'] for x in full['events']] and plan['catalog_events']==23 and
          plan['modules']==MODULES and plan['fields']==list(FIELDS),'Changed frozen census or parser scope')
    contexts={x['event_id']:x for x in records(full)};modules={s:importlib.import_module(m) for s,m in MODULES.items()}
    values=[];errors=[]
    for record in plan['records']:
        try:
            ctx=contexts[record['event_id']]
            value=modules[record['symbol']].extract_document(shared.path(record['source_path']).read_bytes(),ctx)
            validate_record(value,ctx);values.append(value)
        except Exception as exc:
            errors.append(dict(event_id=record['event_id'],error_type=type(exc).__name__,error=str(exc)))
    if errors:
        shared.write_new(OUT/'EXTRACTION_ERRORS.json',dict(created_utc=shared.now(),plan_sha256=plan_sha,errors=errors,
            successful_events_not_published=len(values),new_financial_admissions=0,new_strategy_trials=0))
        raise ValueError('Extraction errors retained; no partial dataset published')
    check(len(values)==9,'Incomplete batch');_pins(bindings,{})
    result=dict(schema_version=1,created_utc=shared.now(),plan_sha256=plan_sha,status='source_values_extracted_pending_independent_review',
        catalog_ref=plan['catalog_ref'],catalog_ids=plan['catalog_ids'],catalog_metadata=plan['catalog_metadata'],
        counts=dict(events=9,by_symbol=dict(Counter(x['symbol'] for x in values)),nonnull_fields=sum(x['fields'][f] is not None for x in values for f in FIELDS),unprocessed_catalog_events=14),
        events=values,new_financial_admissions=0,new_strategy_trials=0,ratios_computed=0,returns_computed=0,
        limitations=['Extraction is not independent audit, admission or strategy testing.',
            'All23 catalog entries, source holds and proposed364/371-day boundaries retained.',
            'Original first-public time/version, ISO currency and continuous security master remain uncertified.'])
    shared.write_new(OUT/'EXTRACTIONS.json',result)
    print(shared.json.dumps(dict(result=str((OUT/'EXTRACTIONS.json').relative_to(ROOT)),sha256=shared.sha(OUT/'EXTRACTIONS.json'),counts=result['counts'])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['policy','tests','freeze','extract'])
    p.add_argument('--plan-sha');p.add_argument('--predecessor-sha');p.add_argument('--parser-review');p.add_argument('--parser-review-sha');args=p.parse_args()
    if args.phase=='policy':
        inputs={path:shared.sha(path) for path in BASE_INPUTS};ctx=shared.load(CONTEXT)
        shared.write_new(POLICY,make_policy(ctx,inputs));print(shared.json.dumps(dict(policy=POLICY,sha256=shared.sha(POLICY))))
    elif args.phase=='tests':print(shared.json.dumps(synthetic_tests()))
    elif args.phase=='freeze':
        check(all((args.predecessor_sha,args.parser_review,args.parser_review_sha)),'Explicit predecessor and parser review hashes required')
        freeze(args.predecessor_sha,args.parser_review,args.parser_review_sha)
    else:
        check(args.plan_sha is not None,'Exact PLAN digest required');run(args.plan_sha)


if __name__=='__main__':main()
