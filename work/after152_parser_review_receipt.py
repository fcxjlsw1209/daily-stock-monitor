"""Publish the authorized static/synthetic MU/PEP parser review, without extraction."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/after152_parser_independent_review.json'
INPUTS = {
    'work/after152_mu_extract.py': 'b17125ee9e33566dbbe7ad163b983c87980368942b0154d20f89a24621533081',
    'work/after152_pep_extract.py': '9341c0bb06c7010702d478eb7e2104dde92a59910de534eac9c564a8f4eadca7',
    'work/test_after152_mu_extract.py': '3a33ee77970574229308be9b73ac11b76cda2abfe6026ae5cff3279f6cccced4',
    'work/test_after152_pep_extract.py': 'c7342b3ff2e74ae2a860f8f89d01334e6af60bd8100cdbdfd2c85de3475e002f',
    'work/after149_amzn_extract.py': 'f59473e6e5703fa5829100679c0887f5bf3853b52433bdf3b5598e4458378a06',
    'work/after141_amd_source_extract.py': '868cae81055318b9df934f4bfe2849c4971f861c4b480942139c8c7606ae4766',
    'work/after152_next_source_feasibility.json': 'c79d523697ecbb570ac99a2ecddeebc998db6ab77c92cd23697cf70596c5b468',
    'work/after152_next_source_context.json': 'f48acb7e7e57d1212b03bcf435b510dd75670f334a3bc228ceacb22ce65bfea7',
}
LOGS = (
    ('independent', 'MU', 32, '/tmp/after152_parser_independent_mu.log', '3c131b82952a43c906b8aca5819ca245fab5c012aec083dc0dbd21d5c4366582'),
    ('independent', 'PEP', 34, '/tmp/after152_parser_independent_pep.log', '70b7de54cf155d2d0c8f0722231d306f03d93a5b4287c438a81ae99b3f8864db'),
    ('producer', 'MU', 32, '/tmp/after152_mu_extract_tests.log', '3789c8e750e29a661296df7f39920fbe27550ad32d5f3530177c062106dc1044'),
    ('producer', 'PEP', 34, '/tmp/after152_pep_extract_tests.log', 'ef5a52397381eb8f3368a778811db3df77ca287545a7d48252d1c222241ae3c8'),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    for rel, expected in INPUTS.items():
        if digest((ROOT / rel).read_bytes()) != expected:
            raise ValueError('Review input changed: ' + rel)
    runs = []
    for owner, issuer, count, path, expected in LOGS:
        raw = Path(path).read_bytes()
        text = raw.decode('utf-8')
        if digest(raw) != expected or not re.search(r'Ran ' + str(count) + r' tests in [0-9.]+s\s+OK\s*$', text):
            raise ValueError('Synthetic log changed or failed: ' + path)
        runs.append(dict(owner=owner, issuer=issuer, tests=count, passed=True,
                         exit_code=0, path=path, sha256=expected))
    context = json.loads((ROOT / 'work/after152_next_source_context.json').read_bytes())
    candidates = [e['event_id'] for e in context['events'] if e['context_status'] ==
                  'standard_HTML_annual_header_context_located_not_admitted']
    expected_candidates = ['MU_2024-09-25', 'MU_2025-09-23', 'PEP_2011-02-10',
        'PEP_2012-02-09', 'PEP_2013-02-14', 'PEP_2014-02-13', 'PEP_2015-02-11',
        'PEP_2016-02-11', 'PEP_2017-02-15']
    if candidates != expected_candidates or len(context['events']) != 23:
        raise ValueError('Frozen context scope mismatch')
    inputs = dict(INPUTS)
    inputs['work/after152_parser_review_receipt.py'] = digest(Path(__file__).read_bytes())
    report = dict(
        schema_version=1, stage='static_and_synthetic_parser_review', scope='static+synthetic_only',
        created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        input_hashes=inputs, source_code_hashes={k: v for k, v in INPUTS.items() if k.endswith('.py')},
        candidates=expected_candidates, candidate_events=9, retained_metadata_catalog_events=23,
        synthetic_tests=dict(independent_tests=66, producer_tests=66, independent_suites=2,
                             independent_runs_passed=True, runs=runs),
        checks=dict(
            fixed_MU_FY2024_2025_and_PEP_FY2010_2016_profiles=True,
            frozen_context_candidate_interface_compatible=True,
            raw_digest_source_receipt_reference_and_identity_checks=True,
            whole_date_cells_exclude_caption_audit_dates=True,
            annual_parent_domain_excludes_quarter_same_date=True,
            current_and_prior_assets_resolved_by_date_not_position=True,
            MU_intermediate_quarter_assets_excluded=True,
            exact_consolidated_NI_excludes_attributable_rows=True,
            derived_annual_boundary_limited_to_364_or_371_days=True,
            statement_local_scale_and_dollar_literal_checks=True,
            explicit_non_GAAP_reconciliation_pro_forma_adjusted_headers_rejected=True,
            signed_decimal_and_physical_cell_evidence_preserved=True,
            missing_or_ambiguous_values_remain_null=True,
            original_source_and_inherited_holds_preserved=True,
            parser_has_no_IO_and_no_admission_authority=True),
        resolved_findings=[dict(
            id='annual_week_boundary_guard', status='resolved',
            previous_issue='The parser accepted any 330..400-day interval, and synthetic fixtures included non-52/53-week years.',
            correction='Final shared parser requires exactly 364 or 371 days; fixtures use consecutive 52/53-week boundaries.',
            regression='Both suites reject 330, 350, 365, 366, 370 and 372-day proposed intervals; valid 364/371 cases pass.')],
        remaining_blockers=[],
        limitations=[
            'This receipt covers code review and synthetic examples only. No real report was passed to either extraction entrypoint during this review.',
            'This is not independent verification of real monetary values, financial-source facts, GAAP comparability or field admission.',
            'Receipt content/status and pinned next-session validity remain caller responsibilities; parser checks digest references and chronological ordering.',
            'Calendar-boundary starts remain explicitly unapproved assumptions, never directly quoted dates.',
            'Dollar literals do not certify ISO currency, first publication, correction history or continuous security identity.',
            'As-adjusted GAAP statements may be conservatively rejected; a null must not be interpreted as proof of source absence or non-GAAP status.',
            'The four early PEP publication dates remain outside the pinned calendar; all original context holds remain.',
            'The 14 noncandidate catalog entries must remain present in subsequent source-only plans/datasets.'],
        real_report_extraction_calls=0, real_monetary_values_extracted=0,
        accounting_field_admission=False, financial_source_actual_verification=False,
        fact_admitted=False, model_admitted=False, strategy_admitted=False,
        factor_calculations=0, return_calculations=0, order_calculations=0,
        runtime=dict(python=sys.version, execution='run_local.sh offline pinned environment'))
    raw = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with OUT.open('xb') as stream:
        stream.write(raw)
    if json.loads(OUT.read_bytes()) != report:
        raise ValueError('Published receipt did not round trip')
    print(json.dumps(dict(passed=True, path=str(OUT.relative_to(ROOT)), sha256=digest(raw),
                          independent_tests=66, scope=report['scope']), sort_keys=True))


if __name__ == '__main__':
    main()
