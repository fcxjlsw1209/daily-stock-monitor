"""Reproduce static-review gaps using fabricated source records only."""
from pathlib import Path
import copy
import hashlib
import json
from decimal import localcontext
import after155_revenue_chain as reviewed

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'work/after155_revenue_chain_implementation_review.json'
CODE = ROOT / 'work/after155_revenue_chain.py'
PIN = '0a3a9ef537f6198848930732170f586a04bc53659d76ae0cdf5e1b2dab573b98'
PASS = 'checked_reported_total_chain_with_declared_source_scope'

def fixture():
    reports, edges = [], []
    for y in range(2018, 2022):
        reports.append(dict(report_id=str(y), known_ns=y, fiscal_year=y,
            source_checked=True, issuer='MSFT', currency='USD',
            measure='consolidated_gaap_total_revenue', period_start=f'{y-1}-07-01',
            period_end=f'{y}-06-30', current='100', scale='1000000',
            comparative='100', comparative_scale='1000000', comparative_currency='USD',
            comparative_period_end=f'{y-1}-06-30'))
        if y > 2018:
            edges.append(dict(edge_id=f'{y-1}_{y}', known_ns=y,
                earlier_fiscal_year=y-1, later_fiscal_year=y,
                status='checked_reported_total_edge'))
    return reports, edges


def main():
    assert hashlib.sha256(CODE.read_bytes()).hexdigest() == PIN, 'Reviewed code changed'
    observed = []
    def record(name, mutate, day='2021-08-01'):
        reports, edges = fixture()
        mutate(reports, edges)
        original = copy.deepcopy((reports, edges))
        with localcontext() as context:
            context.prec = 28
            result = reviewed.evaluate(reports, edges, 2021, day)
        assert result['status'] == PASS, (name, result)
        assert (reports, edges) == original
        observed.append(dict(case=name, observed_status=result['status'],
            synthetic_only=True, input_mutation=False))
    record('quarter_and_segment_comparative_metadata_ignored',
        lambda r, e: r[-1].update(comparative_period_start='2020-04-01',
                                  comparative_measure='segment_revenue'))
    record('different_selected_report_ids_on_edge_ignored',
        lambda r, e: e[-1].update(earlier_report_id='different_2020',
                                  later_report_id='different_2021'))
    record('future_supporting_note_not_checked',
        lambda r, e: e[-1].update(evidence=[dict(note_id='future_note', known_ns=2022)]))
    record('different_scale_equal_monetary_products_accepted',
        lambda r, e: r[-1].update(comparative='100000', comparative_scale='1000'))
    def precision_case(reports, edges):
        for report in reports:
            report.update(current='10000000000000000000000000001', scale='1',
                          comparative='10000000000000000000000000002', comparative_scale='1')
    record('unequal_29_digit_values_round_to_same_product', precision_case)
    record('decision_date_precedes_latest_period_end', lambda r, e: None, day='2020-01-01')
    reports, edges = fixture()
    del reports[-1]['current']
    try:
        reviewed.evaluate(reports, edges, 2021, '2021-08-01')
    except KeyError as exc:
        observed.append(dict(case='missing_current_field', observed_exception=type(exc).__name__,
                             synthetic_only=True, admitted=False))
    else:
        raise AssertionError('Expected missing-key exception was not reproduced')
    data = json.loads(REVIEW.read_text())
    data['input_hashes']['work/after155_revenue_chain_implementation_review.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['validation']['new_tests_executed'] = len(observed)
    data['validation']['method'] = 'Static inspection plus seven fabricated-input reproduction checks through run_local.sh; producer suite not rerun'
    data['validation']['synthetic_reproductions'] = observed
    data['validation']['reproduction_cli'] = './run_local.sh work/after155_revenue_chain_implementation_review.py'
    data['validation']['reproduction_note'] = 'These assert observed current behavior, not policy acceptance. Wrong-vintage/note cases show unenforced caller-contract fields; they do not allege an actual source error.'
    REVIEW.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    reparsed = json.loads(REVIEW.read_text())
    assert len(reparsed['validation']['synthetic_reproductions']) == 7
    print(json.dumps(dict(reproductions_completed=7, actual_source_chains_evaluated=0,
        review_sha256=hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
        reproduction_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), sort_keys=True))

if __name__ == '__main__':
    main()
