"""Offline source-only extraction of explicit Shell period-end share totals."""
from pathlib import Path
import argparse
from calendar import monthrange
from decimal import Decimal
import hashlib
import io
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
PLAN_SHA = '2ff82efd4ac8abdca1ca9a0a97addad081ceb4dc8152195b38f873499193d606'
LABEL = 'Shares outstanding at the end of the period'
NUMBER = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def verify():
    planfile = OUT / 'TOTAL_OUTSTANDING_PLAN.json'
    assert sha(planfile) == PLAN_SHA
    plan = json.loads(planfile.read_text())
    for key in ('original_plan', 'pdf_runtime_plan'):
        item = plan[key]
        assert sha(ROOT / item['path']) == item['sha256'], key
    original = json.loads((ROOT / plan['original_plan']['path']).read_text())
    assert len(plan['records']) == len(original['records']) == 20
    for record, old in zip(plan['records'], original['records']):
        assert record['binary'] == old['binary']
        assert record['sha256'] == old['inputs'][old['binary']]
        assert sha(ROOT / record['binary']) == record['sha256']
    for path, digest in plan['protected_existing_outputs'].items():
        assert sha(ROOT / path) == digest, path
    pins = json.loads((ROOT / plan['pdf_runtime_plan']['path']).read_text())
    for path, digest in pins['pdf_runtime_hashes'].items():
        assert sha(Path(path)) == digest, path
    sys.path.append(pins['pdf_runtime']['bundle_site_packages'])
    return plan


def conditional_interval(displayed):
    """Decimal arithmetic only; this is a stated rounding assumption, not a fact."""
    value = Decimal(displayed.replace(',', ''))
    step = Decimal(1).scaleb(value.as_tuple().exponent)
    return dict(
        assumption='Nearest rounding to the displayed precision; source rounding rule is not stated.',
        conditional_only=True,
        displayed_step_millions=str(step),
        lower_ordinary_shares=str((value - step / 2) * Decimal(1000000)),
        upper_ordinary_shares=str((value + step / 2) * Decimal(1000000)),
        endpoints_inclusive=True,
        source_certified_interval=False,
    )


def target_row(text, year, quarter):
    lines = text.splitlines()
    hits = [(i, line) for i, line in enumerate(lines) if LABEL.lower() in norm(line).lower()]
    if not hits:
        return None
    assert len(hits) == 1, 'Ambiguous multiple target rows'
    i, line = hits[0]
    assert 'Millions' in text and re.search(r'ordinary shares of\s*€\s*0\.07 each', text, re.I)
    assert 'Quarters' in text and re.search(rf'Q{quarter}\s+{year}', text)
    left, right = re.split(LABEL, line, flags=re.I)
    cells = re.findall(NUMBER, left)
    following = re.findall(NUMBER, right)
    # This table is current quarter, prior quarter, year-ago quarter; Q4 also
    # repeats the current and prior year totals in the Full year columns.
    assert len(cells) == 3 and len(following) == 2 and quarter == 4
    assert cells[0] == following[0], 'Quarter-end and current Full year disagree'
    header = '\n'.join(lines[max(0, i - 12):i + 5]).strip()
    assert re.search(rf'Q{quarter}\s+{year}\s+Q{quarter - 1}\s+{year}\s+Q{quarter}\s+{year - 1}', header)
    return dict(
        label=LABEL,
        displayed_value=cells[0],
        source_units='Millions',
        security_unit='ordinary shares',
        nominal_value_note='€0.07 each; nominal value is not a share count',
        security_unit_evidence=next(norm(x) for x in lines if re.search(r'ordinary shares of', x, re.I)),
        ads=False,
        exact_integer_count=None,
        source_row=norm(line),
        table_context=header,
        comparator_cells={'prior_quarter':cells[1], 'year_ago_quarter':cells[2], 'full_year_current':following[0], 'full_year_previous':following[1]},
        conditional_rounding_interval=conditional_interval(cells[0]),
    )


def contexts(text, patterns, radius=170):
    flat = norm(text)
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, flat, re.I):
            snippet = flat[max(0, match.start() - radius):min(len(flat), match.end() + radius)]
            if snippet not in found:
                found.append(snippet)
    return found


def inspect_record(record):
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    payload = (ROOT / record['binary']).read_bytes()
    pdf = PdfReader(io.BytesIO(payload))
    result = dict(
        period=f"{record['year']}Q{record['quarter']}",
        period_end=f"{record['year']}-{record['quarter'] * 3:02d}-{monthrange(record['year'], record['quarter'] * 3)[1]:02d}",
        source={'binary':record['binary'], 'sha256':record['sha256'], 'url':record['source_url']},
        pages_scanned=len(pdf.pages),
        explicit_period_end_total=None,
        candidate_mentions=[],
        eps_denominator_pages=[],
        trust_treasury_context=[],
        trust_exclusion_explicit=None,
        treasury_exclusion_explicit=None,
        netting_basis='Not established: do not infer a class-specific net count or zero treasury/trust shares.',
        no_interpolation_or_substitution=True,
    )
    pypdf_targets = []
    with pdfium.PdfDocument(payload) as other:
        assert len(other) == len(pdf.pages)
        pdfium_targets = []
        for num, page in enumerate(pdf.pages, 1):
            text = page.extract_text(extraction_mode='layout')
            other_page = other[num - 1]
            textpage = other_page.get_textpage()
            try:
                alternative = textpage.get_text_range()
            finally:
                textpage.close()
                other_page.close()
            if LABEL.lower() in norm(alternative).lower():
                pdfium_targets.append(num)
            row = target_row(text, record['year'], record['quarter'])
            if row is not None:
                pypdf_targets.append(num)
                assert result['explicit_period_end_total'] is None
                assert LABEL.lower() in norm(alternative).lower()
                # Separate engine must find each literal row cell and the
                # source's ordinary-share label on the same original page.
                assert row['displayed_value'] in alternative
                assert 'ordinary shares' in norm(alternative).lower()
                row.update(source_page_1based=num, second_engine_target_confirmed=True)
                result['explicit_period_end_total'] = row
            if re.search(r'weighted\s+average\s+(?:number\s+of\s+)?shares', text, re.I):
                result['eps_denominator_pages'].append(num)
            mentions = contexts(text, [r'\boutstanding\b', r'\bshares.{0,45}end of (?:the )?period\b'])
            if mentions:
                result['candidate_mentions'].append({'source_page_1based':num, 'context':mentions})
            trust = contexts(text, [r'shares held in trust', r'treasury shares', r'(?:excluding|excluded).{0,80}(?:trust|treasury)'])
            if trust:
                result['trust_treasury_context'].append({'source_page_1based':num, 'context':trust})
        assert pypdf_targets == pdfium_targets, (result['period'], pypdf_targets, pdfium_targets)
    result['text_engine_agreement'] = {'pypdf_target_pages':pypdf_targets, 'pdfium_target_pages':pdfium_targets}
    if result['explicit_period_end_total'] is None:
        result['status'] = 'missing_explicit_period_end_total'
        result['missing_reason'] = 'No explicit period-end shares-outstanding total identified in either original-PDF text engine; weighted-average EPS and issued-share tables not substituted.'
    else:
        result['status'] = 'reported_rounded_total_netting_basis_unresolved'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--render-target', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert target_row('Weighted average number of shares 6000.0', 2015, 4) is None
        synthetic = 'SHARES\nQuarters Millions Full year\nQ4 2015 Q3 2015 Q4 2014 2015 2014\n6,397.5 6,348.4 6,295.0 Shares outstanding at the end of the period 6,397.5 6,295.0\n1 Royal Dutch Shell plc ordinary shares of € 0.07 each'
        row = target_row(synthetic, 2015, 4)
        assert row['displayed_value'] == '6,397.5' and row['security_unit'] == 'ordinary shares'
        assert row['conditional_rounding_interval']['lower_ordinary_shares'] == '6397450000.00'
        assert row['conditional_rounding_interval']['upper_ordinary_shares'] == '6397550000.00'
        try:
            target_row(synthetic.replace('6,397.5 6,295.0\n', '6,397.6 6,295.0\n'), 2015, 4)
        except AssertionError:
            pass
        else:
            raise AssertionError('Mismatched repeated total was accepted')
        print('4 synthetic source-parser checks passed')
        return
    plan = verify()
    records = [inspect_record(row) for row in plan['records']]
    if args.render_target:
        import pypdfium2 as pdfium
        for rec in records:
            target = rec['explicit_period_end_total']
            if target:
                with pdfium.PdfDocument((ROOT / rec['source']['binary']).read_bytes()) as pdf:
                    dest = Path('/tmp') / ('shell168_outstanding_' + rec['period'] + '.png')
                    pdf[target['source_page_1based'] - 1].render(scale=1.6).to_pil().save(dest)
                    print(dest)
        if not args.extract:
            return
    output = dict(
        schema='shell_pool_source168_total_outstanding_v1',
        plan_sha256=PLAN_SHA,
        extractor_sha256=sha(Path(__file__)),
        records=records,
        summary={'documents':len(records), 'reported_period_end_totals':sum(r['explicit_period_end_total'] is not None for r in records), 'missing_period_end_totals':sum(r['explicit_period_end_total'] is None for r in records)},
        limits=['No A/B net outstanding split has been established.', 'Ordinary-share totals are not ADS totals.', 'Any rounding interval is conditional on nearest rounding, not source-certified.', 'Trust/treasury exclusion is unresolved, and no zero balance is inferred.', 'No later disclosure fills a missing earlier date.', 'No cap, rank, factor, transaction or return computation.'],
        protected_inputs_verified=True,
        pool_modified=False,
    )
    if args.extract:
        raw = (json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        target = OUT / 'TOTAL_OUTSTANDING.json'
        with target.open('xb') as stream:
            stream.write(raw)
        print(json.dumps(output['summary']))
        print('sha256', sha(target))
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
