"""Freeze a bounded, offline Boeing source review; never import strategy code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_ba_source_plan.py'
OUTPUT = 'work/after158_ba_source_plan.json'
BASE = 'outputs/monthly_earnings_repair/batch_health_energy_foreign/raw/BA/'
# Fiscal year, original basename, raw, receipt, text, text-receipt SHA256.
FILES = [
    (2014, 'ce52820c486a4be1c9df', 'db020dddd1207c430c83bec32f17ca79b5411f71100116047869c912eae7ae0a', '98d0533a39837b1c30b7b36f08eab09d3ae0ecc325ddeb5babf56389c28a4bdb', '502035d2e7eb866f9b0dbfdea1753a1acd89d8b9addaaf6f86b07b52b32dad32', '458440c529fab4b722b733028df720d8c71cffd5f34a764c66f92cf35404d516'),
    (2015, 'e072480d78cdfaa66f15', '8d12754388ceb04004c80f1eefbdfe9a1c463da0c88f9f01e199b63e415f0d78', 'c83a7d75abb88f612adbb660565133834991b506f8bd6ff63a589952d053a2e7', '53bd7f39e7b91dbecd360f8c684f199cad5afc5d7471dd3d6c5d1137bf4656c1', 'bb050d2adc598694a6b33b995db505fb2efd3a11ad1f7be0b30587102c7db766'),
    (2016, '6d80f6f4b64f038bad15', 'a64082bdb248a4259f818ca4725e0d5bbc4ee212f2e5950b15abb3c568095789', '330785606d3e8bfaca74ce59579ec4823beceb57e249b060c929d86b8344d7ef', 'a5428750e55f35b77deb6f56284bb82d667a92a4608ee7ac53d04ac9c8155526', 'efcb4bd9e8813105c0e3090c86045066af78573614fa18f23cc42db11c1ea385'),
    (2017, '4a9d117b0d5dbb2d7763', '97becd4a04aa95899fd9f69df1074717f582e266adb815cf0730aa7075b08f81', '5e62be80738965bdd05567d5c16235b9b86a43cf34587bf4d8a9bb13168475ca', 'c7c154eeecfdfa14e08411043020945b2e36beedb00d9876d8604224620de1a8', 'ed7af31502837e6089dcb04354f0db7c1eab85db8207fa01e4b5284cc32890df'),
    (2018, '339b67802f5437c7cc48', 'bc7b55dd02e49e2954060a4958183289613d367b6b7a56ed4e90f77ba5cc8bf8', 'a99bbbb3801aa9d9e3e1d60a29c348fc13cfbf87a3cbdff4b639a1284701a57a', 'f7cb00009854378bc1b2d04ecf409acac3624129652715a6bf0becb4ee4d1cc7', 'bac05ca1410d058be62dc30b1bab3ee969d868d7c97bac8c0e8cba1500438c11'),
]
ARCHIVE = BASE + '0bfa47080dc9bf32a7da.bin'
ARCHIVE_RECEIPT = BASE + '0bfa47080dc9bf32a7da.meta.json'
CONTEXT_PINS = {
    'outputs/monthly_local_linear70/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'outputs/monthly_revenue_pilot_source156/ROWS.json': '1732c3611fdbd14b00ef39d370a95493aec91f1a2c3ee60a2ce2ec3e27080196',
    'outputs/monthly_revenue_pilot_source156/COVERAGE_GATE.json': 'c846d13349682655e8a18c8a31f822c3323a4c3d383ca38d080c01bdf3c58057',
    'outputs/monthly_revenue_coverage156/ANNUAL_REPORTS.json': '92462d7c09b60b1713c90b85f789b8a6c988998841a6a78eb9d5315e38b653ef',
    'outputs/monthly_earnings_repair/batch_health_energy_foreign/BA_candidates.json': '4e5c4a958c8d1421469ba485b26261815dbce887e1112e2afe0ec6f57f55ce40',
    'work/after155_revenue_pilot_source_plan.json': '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10',
    'work/after155_revenue_chain_policy_review.json': '8564d0d052e4a7cfb89bd73c553b46c6ea831afacd384a5d69fe63c52732ce36',
    'work/after155_msft_revenue_basis_review.json': 'ade332bf2cab7c8f8a06ce832fc3389dd0a08902ac4c05515ba22c487e055b40',
    'work/after155_msft_chain_output_independent.json': '8dd79daa2891cd093fb16dd01015be92e59d38c9545d4ff04a3e45826ae4dadc',
    'work/after155_aapl_revenue_independent.json': 'f66e126f875b3af5698d931c9d06cfe9fc813d4989ec9e49952891ac9afed86a',
    'work/after155_multi_issuer_source_map.json': 'f4bb4432920b6e57c46626a148be1b58555a64a05704a89483630251726ba041',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Nonlocal path')
    require(not any('onedrive' in x.lower() or 'cloudstorage' in x.lower() for x in p.parts), 'Cloud path')
    result = root / p
    require(result.resolve().is_relative_to(root.resolve()), 'Path escape')
    require(not any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts) + 1)), 'Symlink')
    return result


def read(root, path, expected=None):
    p = local(root, path)
    a = p.stat()
    raw = p.read_bytes()
    b = p.stat()
    require((a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_ino, b.st_size, b.st_mtime_ns), 'Input changed')
    require(expected is None or digest(raw) == expected, 'Hash mismatch: ' + path)
    return raw


def source_pins():
    pins = {ARCHIVE: '3cad29afe7edcb86748f92d80faab1dbcf7ac0cf6bc643f1219d8f7dcbc40e4a',
            ARCHIVE_RECEIPT: 'bb4f6ad50d085761b160541b67acc05c412ea02969534114ab2bf04e24d923e9'}
    for _, stem, *hashes in FILES:
        pins.update({BASE + stem + suffix: sha for suffix, sha in zip(('.bin', '.meta.json', '.txt', '.text.json'), hashes)})
    require(len(pins) == 22, 'Source object count')
    return pins


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)


def make_plan(root, script_sha):
    pins = source_pins()
    context = dict(CONTEXT_PINS, **{SELF: script_sha})
    blobs = {p: read(root, p, sha) for p, sha in {**pins, **context}.items()}
    archive = json.loads(blobs[ARCHIVE])
    archive_receipt = json.loads(blobs[ARCHIVE_RECEIPT])
    require(archive_receipt['status'] == 200 and archive_receipt['sha256'] == pins[ARCHIVE], 'Archive receipt')
    require(archive_receipt['bytes'] == len(blobs[ARCHIVE]), 'Archive bytes')
    require(archive_receipt['url'] == archive_receipt['final_url'] ==
            'https://investors.boeing.com/Services/FinancialReportService.svc/GetFinancialReportList', 'Archive publisher')
    archive_links = set(strings(archive))
    reports = json.loads(blobs['outputs/monthly_revenue_coverage156/ANNUAL_REPORTS.json'])
    targets = []
    for year, stem, *_ in FILES:
        paths = {k: BASE + stem + suffix for k, suffix in (
            ('raw', '.bin'), ('receipt', '.meta.json'), ('text', '.txt'), ('text_binding', '.text.json'))}
        receipt = json.loads(blobs[paths['receipt']])
        text_binding = json.loads(blobs[paths['text_binding']])
        require(blobs[paths['raw']].startswith(b'%PDF-'), 'Expected PDF')
        require(receipt['status'] == 200 and receipt['sha256'] == pins[paths['raw']], 'Raw receipt')
        require(receipt['bytes'] == len(blobs[paths['raw']]), 'Raw byte count')
        require(receipt['url'] == receipt['final_url'] and receipt['url'] in archive_links, 'Publisher exact link')
        require(text_binding['source_sha256'] == pins[paths['raw']] and
                text_binding['text_sha256'] == pins[paths['text']], 'Text binding')
        rows = [r for r in reports if r['symbol'] == 'BA' and r['fiscal_year'] == year and r['path'] == paths['raw']]
        require(len(rows) == 1 and rows[0]['sha256'] == pins[paths['raw']], 'Catalog mapping')
        targets.append(dict(fiscal_year=year, role='boundary_only' if year == 2018 else 'original_chain_member_candidate',
                            paths=paths, url=receipt['url'], retrieved_at=receipt['retrieved_at'],
                            catalog_publication_date=rows[0]['publication_date'],
                            catalog_period_end=rows[0]['official_actual_period_end'],
                            catalog_dates_require_original_recheck=True))
    meta = json.loads(blobs['outputs/monthly_local_linear70/META.json'])
    rows = json.loads(blobs['outputs/monthly_revenue_pilot_source156/ROWS.json'])
    require(len(meta['dates']) == 122 and len(rows) == 5978, 'Full original axis')
    for k, row in enumerate(rows):
        i, j = divmod(k, 49)
        require((row['signal_index'], row['candidate_index'], row['symbol'], row['decision_date']) ==
                (i, j, meta['symbols'][i][j], meta['dates'][i]), 'Original axis mismatch')
    ba = [dict(signal_index=r['signal_index'], candidate_index=r['candidate_index'],
               decision_date=r['decision_date'], decision_cutoff=r['decision_cutoff'])
          for r in rows if r['signal_index'] < 53 and r['symbol'] == 'BA']
    require(len(ba) == 21 and ba[0]['decision_date'] == '2018-02-28', 'Unexpected BA metadata coverage')
    plan = dict(
        schema='after158_ba_original_statement_source_plan_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        stage='frozen_source_only_before_statement_review', symbol='BA', source_objects=22,
        source_hashes=pins, context_hashes=context, targets=targets,
        selection_basis='Among original early53 candidate issuers in alphabetical order, inspect the first with four physical annual originals needed at its first early candidate occurrence. AAPL required early statements unestablished, ADBE index-only, AMZN FY2013/FY2015 index-only; BA FY2014-2017 has four physical originals. No prices, returns, holdings or learned scores used.',
        source_scope='FY2014-2017 original consolidated annual revenue statements and exactly three adjacent edges. FY2018 is negative/transition/latest-no-fallback context only, never a replacement chain member in this batch.',
        original_axis=dict(months=122, candidates_per_month=49, cells=5978, early53_ba_slots=ba),
        review_contract={
            'original_table': 'Identify the formal consolidated statements of operations; annual current/prior columns, exact units, period dates, and original versus supplemental restated/non-GAAP tables. Retain printed values with physical page/row provenance; no ratios.',
            'periods': 'Verify twelve months ended December31 in original headers; any January1 start is a declared calendar-period derivation, not a quoted start. No normalization.',
            'three_edges': [[2014, 2015], [2015, 2016], [2016, 2017]],
            'comparative_check': 'Exact original-current versus next original annual-comparative equality is a source consistency check only. It is necessary but not sufficient for an accounting-continuity conclusion. Never overwrite or rebase an earlier original.',
            'accounting': 'Read all five original PDF texts for ASC606/715, adoption, retrospective restatement, reclassification, consolidation/discontinuation and program accounting. Inspect complete relevant rendered pages. Distinguish segment allocations, non-GAAP reconciliation and consolidated total changes; retain unresolved boundaries.',
            'boundary_2018': 'FY2017 contains announced adoption and supplemental restated2017/2016. FY2018 has adopted standards. Identify affected total-revenue scope without promoting a later comparative into an old original. A newer invalid latest remains a blocker; no fallback to2017.',
            'timing': 'Recheck publication dateline and retain actual 2026 retrieval. Sources are original-dated currently retrieved bytes, not certified contemporaneous capture. Later notes cannot be used before their own availability. No source-to-signal availability map is admitted in this review.',
            'currency': 'Preserve literal Dollars in millions. Do not certify ISO USD from a dollar sign or silently import another issuer currency policy.',
            'failure_allowed': 'May conclude no usable chain; lack of a keyword does not prove unchanged accounting. No source qualification or strategy admission changes.',
        },
        context_use='Related prior reviews are bounded methodology and gap context, not inherited BA admission or a claim that their entire transitive closure was re-audited.',
        activity=dict(new_strategy_trials=0, new_factors=0, model_fits=0, ranks=0, orders=0, returns=0, downloads=0),
        source_admission_changed=False, strategy_admitted=False,
    )
    for path, sha in {**pins, **context}.items():
        read(root, path, sha)
    return plan


class Tests(unittest.TestCase):
    def test_fixed_source_count(self):
        self.assertEqual(len(source_pins()), 22)
        self.assertEqual([x[0] for x in FILES], list(range(2014, 2019)))

    def test_path_escapes(self):
        with tempfile.TemporaryDirectory() as d:
            for p in ('../x', '/tmp/x', 'CloudStorage/x'):
                with self.assertRaises(ValueError):
                    local(Path(d), p)
            (Path(d) / 'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                local(Path(d), 'link/x')

    def test_digest_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'x').write_bytes(b'original')
            self.assertEqual(read(Path(d), 'x', digest(b'original')), b'original')
            with self.assertRaises(ValueError):
                read(Path(d), 'x', '0' * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--script-sha256')
    args = parser.parse_args()
    if args.test_only:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    require(args.write and args.script_sha256, 'Use --write and --script-sha256')
    require(not local(ROOT, OUTPUT).exists(), 'Plan already exists; never overwrite')
    plan = make_plan(ROOT, args.script_sha256)
    raw = (json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with local(ROOT, OUTPUT).open('xb') as handle:
        handle.write(raw)
    print(json.dumps(dict(path=OUTPUT, sha256=digest(raw), source_objects=22,
                          source_admission_changed=False, new_strategy_trials=0)))


if __name__ == '__main__':
    main()
