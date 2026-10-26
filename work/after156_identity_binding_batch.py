"""Offline, bounded identity-evidence binding; no factors or security-epoch inference."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after156_identity_binding_batch.json'
PRIOR = 'work/profitability141_identity_admission_review.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
DESIGN = 'work/after156_filing_drift_design.json'
PINS = {
    PRIOR: '590446b750bb2d96d79056be4f0d86fe5cc61ec360cbfe48e75295f584c5f171',
    INVENTORY: 'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
    DESIGN: '4a902f27fd542a0c00f75661d231346c63f99c61e1c8e95bbf45cb94fd73ee75',
}
BATCH_HASHES = ('b9096e587cc6460b4007fe85cddd6a0ebfbbd211736a31fb899405b5f02fb4f3',
                '04a2d2410a35db0a02034fab75b781cd43e5f52f569fb5272f34f3219cee2b33',
                '7a74e1d529ea0f2c36d73dead551582cb6dc157373dec6369ecd6576b5e6c719',
                '8605fc0e0520f39591ed9ff3aa82a75abfbb14f6f57d3cf865e83e5a853db324',
                '8358d49a9d18a2fd072746cfafdecb63eaa7e6013b3a625ca59a210b06073711')
NAMES = {'MSFT': r'Microsoft', 'NVDA': r'NVIDIA', 'research_equity_FB': r'Meta Platforms,? Inc\.?',
         'AMZN': r'Amazon\.com(?:,? Inc\.?)?', 'CSCO': r'Cisco(?: Systems,? Inc\.?)?',
         'INTC': r'Intel(?: Corporation)?', 'KLAC': r'KLA(?:-Tencor)?(?: Corporation)?',
         'KO': r'(?:The )?Coca-Cola Company', 'LRCX': r'Lam Research(?: Corporation)?',
         'MU': r'Micron Technology(?:,? Inc\.?)?', 'PEP': r'PepsiCo(?:,? Inc\.?)?',
         'PLTR': r'Palantir Technologies(?: Inc\.?)?', 'PYPL': r'PayPal(?: Holdings,? Inc\.?)?',
         'TMO': r'Thermo Fisher Scientific(?: Inc\.?)?'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def path(self, relative):
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT), 'Outside research root')
        return path

    def read(self, relative, expected=None):
        relative = str(relative)
        raw = self.path(relative).read_bytes()
        digest = sha(raw)
        require(expected is None or digest == expected, 'Hash mismatch: ' + relative)
        require(relative not in self.hashes or self.hashes[relative] == digest, 'Input changed')
        self.hashes[relative] = digest
        return raw

    def obj(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.read(path, digest)


def normalize(text):
    return re.sub(r'\s+', ' ', html.unescape(text).replace('\\u0000', '').replace('\x00', '')).strip()


class Visible(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.published = [], 0, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'):
            self.skip += 1
        if tag == 'meta' and attrs.get('property', attrs.get('name', '')).lower() in ('article:published_time', 'datepublished', 'pubdate'):
            self.published.append(attrs.get('content', ''))

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def document(raw):
    require(not raw.startswith(b'%PDF'), 'PDF requires existing reviewed extraction')
    parser = Visible()
    parser.feed(raw.decode('utf-8', errors='replace'))
    return normalize(' '.join(parser.parts)), parser.published


def timestamp(raw):
    require(isinstance(raw, str), 'Missing receipt retrieval time')
    parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'Naive retrieval time')
    return parsed.astimezone(timezone.utc).isoformat()


def check_receipt(receipt, raw, expected_url=None):
    require(receipt.get('sha256') == sha(raw), 'Receipt does not bind original bytes')
    status = receipt.get('http_status', receipt.get('status'))
    require(status == 200, 'Receipt is not HTTP200')
    url = receipt.get('url', receipt.get('source_url', receipt.get('source')))
    require(isinstance(url, str) and url.startswith('https://'), 'Missing original URL')
    require(expected_url is None or expected_url in (url, receipt.get('final_url')), 'Receipt URL differs')
    for field in ('bytes', 'size'):
        if field in receipt:
            require(receipt[field] == len(raw), 'Receipt byte count differs')
    return dict(url=url, final_url=receipt.get('final_url', url), http_status=status,
                retrieved_at=timestamp(receipt.get('retrieved_at')), original_sha256=sha(raw),
                bytes=len(raw), original_bytes_receipt_verified=True)


def manifest_entry(manifest, receipt_parent, original_path, expected_url=None):
    matches = [(url, entry) for url, entry in manifest.items()
               if isinstance(entry, dict) and isinstance(entry.get('evidence_path'), str)
               and (receipt_parent / entry['evidence_path']).resolve() == original_path.resolve()
               and (expected_url is None or expected_url == url)]
    require(len(matches) == 1, 'Manifest must bind exactly one original path and URL')
    url, entry = matches[0]
    require(entry.get('url') == url, 'Manifest URL key differs from entry')
    return url, entry


def bind_source(inputs, source, receipt_reference=None):
    path = source['path']
    raw = inputs.read(path, source['sha256'])
    expected_url = source.get('url')
    specified = source.get('receipt_path', source.get('receipt'))
    specified_sha = source.get('receipt_sha256')
    if receipt_reference:
        specified, specified_sha = receipt_reference['path'], receipt_reference['sha256']
    candidates = [specified] if specified else []
    candidates += [str(Path(path).with_suffix('.meta.json')), path + '.meta.json']
    receipt_path = next((p for p in candidates if p and inputs.path(p).is_file()), None)
    require(receipt_path is not None, 'Missing local original receipt for ' + path)
    receipt = inputs.obj(receipt_path, specified_sha if receipt_path == specified else None)
    pointer = None
    if receipt_path.endswith('/manifest.json'):
        pointer, receipt = manifest_entry(receipt, inputs.path(receipt_path).parent,
                                          inputs.path(path), expected_url)
    try:
        proof = check_receipt(receipt, raw, expected_url)
    except (ValueError, TypeError) as error:
        raise ValueError(path + ' / ' + receipt_path + ': ' + str(error)) from error
    proof.update(path=path, receipt_path=receipt_path, receipt_sha256=inputs.hashes[receipt_path],
                 receipt_pointer=pointer, receipt_located_from_exact_source_sibling=not specified)
    return raw, proof


def literal(text, pattern):
    match = re.search(pattern, text, flags=re.I)
    return dict(text=match.group(), start=match.start(), end=match.end(),
                coordinate='normalized_visible_document_text') if match else None


def publication(text, metadata_dates, declared, fallback=None):
    candidate = declared or fallback
    if not candidate:
        return dict(date=None, supported=False, reason='No declared original date')
    day = date.fromisoformat(candidate)
    require(day.isoformat() == candidate, 'Noncanonical date')
    month = day.strftime('%B')
    short = day.strftime('%b')
    patterns = [re.escape(candidate), rf'\b{month}\s+0?{day.day},?\s+{day.year}\b',
                rf'\b{short}\.?\s+0?{day.day},?\s+{day.year}\b',
                rf'\b0?{day.day}\s+{month}\s+{day.year}\b',
                rf'\b0?{day.month}/0?{day.day}/{day.year}\b']
    hit = next((x for x in (literal(text, p) for p in patterns) if x), None)
    meta = [x for x in metadata_dates if x[:10] == candidate]
    return dict(date=candidate if hit or meta else None, candidate_date=candidate,
                prior_declared_original_date=declared, candidate_from_event_id_only=declared is None,
                supported=bool(hit or meta), literal=hit, explicit_publication_metadata=meta,
                first_public_time_verified=False,
                semantics='Prior reviewed publication date corroborated by original date literal/metadata; literal occurrence alone is not independent first-public-time proof.')


def identity_evidence(text, symbol, source_identity=None):
    name = literal(text, NAMES[symbol])
    ticker = 'FB' if symbol == 'research_equity_FB' else symbol
    listing = literal(text, rf'\b(?:NASDAQ|NYSE)(?:\s+Global\s+(?:Select\s+)?Market)?\s*[:\(\s“”\"\u2018\u2019]*{re.escape(ticker)}\b')
    listing_basis = 'Explicit exchange/ticker phrase'
    if not listing:
        listing = literal(text, r'Title of each class Trading symbol\(s\) Name of each exchange on which registered '
            r'Class [ABC] Common Stock, par value \$[\d.]+ per share ' + re.escape(ticker) + r' New York Stock Exchange\b')
        listing_basis = 'Single registration-table header followed by class, ticker, exchange row'
    if not listing and symbol == 'research_equity_FB':
        listing = literal(text, r"ticker symbol [‘’'\"]?META[‘’'\"]?")
    cls = literal(text, r'Class\s+[ABC]\s+common\s+stock|American Depositary (?:Shares|Receipts)|ordinary shares|common stock')
    return dict(issuer_name=name, exchange_ticker=listing, exchange_ticker_basis=listing_basis if listing else None,
                security_class_literal=cls,
                source_local_issuer_supported=name is not None,
                source_local_issuer_ticker_supported=name is not None and listing is not None,
                prior_identity=source_identity or {}, historical_security_epoch_admitted=False,
                scope='Dated document observation, not continuous exchange/class/security identity.')


def gate_reasons(identity):
    reasons = ['No independently admitted SEC acceptance clock attached to this binding.',
               'This batch does not enumerate complete original-plus-amendment SEC chronology.',
               'Security identity over the consumed event/formation interval is not established by a dated source-local observation.']
    if not identity['source_local_issuer_ticker_supported']:
        reasons.append('Explicit exchange/ticker pair not independently located in this source text.')
    return reasons


def tests():
    names = []
    def ok(name, condition):
        require(condition, name); names.append(name)
    def reject(name, fn):
        try: fn()
        except (ValueError, TypeError): names.append(name)
        else: raise ValueError(name)
    data = b'<p>Example</p>'
    rec = dict(sha256=sha(data), http_status=200, url='https://example.test/x', retrieved_at='2020-01-02T03:00:00Z', bytes=len(data))
    ok('exact receipt binds bytes and UTC retrieval', check_receipt(rec, data)['retrieved_at'] == '2020-01-02T03:00:00+00:00')
    reject('wrong bytes fail', lambda: check_receipt(rec, b'changed'))
    reject('HTTP403 fails', lambda: check_receipt(dict(rec, http_status=403), data))
    reject('wrong URL fails', lambda: check_receipt(rec, data, 'https://other.test/x'))
    reject('wrong size fails', lambda: check_receipt(dict(rec, bytes=1), data))
    reject('naive receipt time fails', lambda: check_receipt(dict(rec, retrieved_at='2020-01-02T03:00:00'), data))
    manifest = {rec['url']: dict(rec, evidence_path='raw/x.html')}
    ok('manifest entry binds exact original path', manifest_entry(manifest, Path('/synthetic'), Path('/synthetic/raw/x.html'))[0] == rec['url'])
    reject('manifest wrong original path fails', lambda: manifest_entry(manifest, Path('/synthetic'), Path('/synthetic/raw/y.html')))
    reject('manifest wrong URL fails', lambda: manifest_entry(manifest, Path('/synthetic'), Path('/synthetic/raw/x.html'), 'https://other.test/x'))
    text, dates = document(b'<script>Microsoft (NASDAQ: MSFT)</script><p>Other company</p>')
    ok('script identity is not visible evidence', not identity_evidence(text, 'MSFT')['source_local_issuer_supported'])
    text, dates = document(b'<meta property="article:published_time" content="2020-02-29T12:00:00Z"><p>Microsoft (NASDAQ: MSFT)</p>')
    ok('visible issuer ticker found', identity_evidence(text, 'MSFT')['source_local_issuer_ticker_supported'])
    cover = 'Palantir Technologies Inc. Title of each class Trading symbol(s) Name of each exchange on which registered Class A Common Stock, par value $0.001 per share PLTR New York Stock Exchange'
    ok('registration table preserves class ticker exchange binding', identity_evidence(cover, 'PLTR')['source_local_issuer_ticker_supported'])
    ok('registration table wrong ticker rejected', not identity_evidence(cover.replace('PLTR', 'OTHER'), 'PLTR')['source_local_issuer_ticker_supported'])
    ok('explicit publication metadata retained', publication(text, dates, '2020-02-29')['supported'])
    ok('date mismatch not silently repaired', not publication(text, dates, '2020-03-01')['supported'])
    ok('abbreviated original date found', publication('Feb. 29, 2020', [], '2020-02-29')['supported'])
    reject('invalid calendar date fails', lambda: publication('', [], '2021-02-29'))
    ok('event id alone not publication proof', not publication('nothing', [], None, '2020-02-29')['supported'])
    ident = identity_evidence('Microsoft (NASDAQ: MSFT)', 'MSFT', {'identity_valid_from': '2015-01-01'})
    ok('source use window does not become epoch', ident['historical_security_epoch_admitted'] is False)
    ok('document identity cannot satisfy SEC gate', len(gate_reasons(ident)) == 3)
    return names


def current_links(inputs, inventory, symbols):
    result = {}
    for stock in inventory['stocks']:
        symbol = stock['symbol']
        if symbol not in symbols:
            continue
        ciks = stock['exact_current_ciks']
        require(len(ciks) == 1, 'No unique current CIK for batch identity')
        cik = ciks[0]
        paths = stock['local_snapshot_paths'][cik]
        path = min(paths, key=lambda p: (not p.startswith('outputs/dynamic_top100/registry_sec/'), p))
        original = inputs.read(path, inventory['input_hashes'][path])
        metadata = json.loads(original)
        ticker = 'META' if symbol == 'research_equity_FB' else symbol
        require(str(metadata['cik']).zfill(10) == cik and ticker in metadata['tickers'], 'Current issuer mismatch')
        receipt_path = str(Path(path).with_suffix('.provenance.json'))
        receipt = inputs.obj(receipt_path, inventory['input_hashes'].get(receipt_path))
        require(receipt['sha256'] == sha(original), 'Current CIK receipt hash')
        require(receipt.get('source', receipt.get('url', receipt.get('source_url'))) ==
                'https://data.sec.gov/submissions/CIK' + cik + '.json', 'Current CIK URL: ' + receipt_path)
        result[symbol] = dict(cik=cik, current_name=metadata['name'], path=path, sha256=sha(original),
            source_url='https://data.sec.gov/submissions/CIK' + cik + '.json',
            receipt_path=receipt_path, receipt_sha256=inputs.hashes[receipt_path], retrieved_at=timestamp(receipt['retrieved_at']),
            current_link_verified=True, historical_security_mapping_certified=False)
    require(set(result) == set(symbols), 'Missing current link')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    test_names = tests()
    if args.test_only:
        print(json.dumps(dict(passed=True, synthetic_checks=len(test_names), names=test_names)))
        return
    inputs = Inputs()
    prior = inputs.obj(PRIOR, PINS[PRIOR])
    inventory = inputs.obj(INVENTORY, PINS[INVENTORY])
    inputs.obj(DESIGN, PINS[DESIGN])
    for path in ('AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb', 'work/after156_identity_binding_batch.py'):
        inputs.read(path)
    bindings = []
    annual = prior['focused_original_identity_evidence']['annual_issuer_ticker_observations']
    require(len(annual) == 24 and Counter(x['symbol'] for x in annual) == {'MSFT': 12, 'NVDA': 12}, 'Fixed annual scope')
    for row in annual:
        source = dict(row['raw'], url=row['source_url'])
        raw, proof = bind_source(inputs, source, row['receipt'])
        text, dates = document(raw)
        require(normalize(row['issuer_ticker_literal']).casefold() in text.casefold(), 'Prior issuer literal not reproduced')
        identity = identity_evidence(text, row['symbol'])
        bindings.append(dict(binding_id=row['symbol'] + '_annual_' + str(row['fiscal_year_label']),
            symbol=row['symbol'], kind='dated_annual_issuer_observation', fiscal_year_label=row['fiscal_year_label'],
            source=proof, publication=publication(text, dates, row['release_date']), identity=identity,
            effective_from=None, effective_through=None, source_use_window=None,
            filing_gate_satisfied=False, filing_gate_reasons=gate_reasons(identity)))
    for number, expected in enumerate(BATCH_HASHES, 1):
        base = 'outputs/monthly_financial_coverage_admission_batch' + str(number) + '_v1/'
        dataset = inputs.obj(base + 'DATASET.json', expected)
        manifest = inputs.obj(base + 'MANIFEST.json')
        require(manifest['outputs']['DATASET.json'] == expected, 'Dataset/manifest mismatch')
        for row in dataset['events']:
            raw, proof = bind_source(inputs, row['source'])
            text, dates = document(raw)
            ident = row['identity']
            require(ident['scope'] == 'source_local_report_binding_and_use_window_not_continuous_security_master', 'Unexpected prior identity scope')
            identity = identity_evidence(text, row['symbol'], ident)
            bindings.append(dict(binding_id=row['event_id'], symbol=row['symbol'], kind='source_local_report_binding',
                parent_dataset=base + 'DATASET.json', source=proof,
                publication=publication(text, dates, row.get('publication_date'), row['event_id'][-10:]),
                identity=identity, effective_from=None, effective_through=None,
                source_use_window=dict(from_date=ident['identity_valid_from'], through_date=ident['identity_valid_through'],
                    inherited_available_from=row['available_from'], historical_security_epoch=False),
                filing_gate_satisfied=False, filing_gate_reasons=gate_reasons(identity)))
    meta = prior['focused_original_identity_evidence']['meta']
    note = inputs.obj(meta['identity_note']['path'], meta['identity_note']['sha256'])
    raw, proof = bind_source(inputs, dict(meta['original'], url=meta['source_url']))
    require(raw.startswith(b'%PDF'), 'Meta original is not PDF')
    text_path = str(Path(meta['original']['path']).with_suffix('.txt'))
    text = normalize(inputs.read(text_path).decode('utf-8'))
    require(text == normalize(note['source_text']), 'Existing Meta extraction differs from reviewed identity note')
    identity = identity_evidence(text, 'research_equity_FB')
    require(literal(text, r'CUSIP number will remain unchanged') and literal(text, r'Class A common stock'), 'Missing same-security bridge literals')
    effective = literal(text, r"Class A common stock will begin trading on NASDAQ under the ticker symbol 'META' prior to market open on June 9, 2022")
    require(effective and note['ticker_effective_date'] == '2022-06-09', 'Missing exact effective-date bridge')
    bindings.append(dict(binding_id='research_equity_FB_ticker_change_2022-06-09', symbol='research_equity_FB',
        kind='same_security_ticker_transition', source=proof, publication=publication(text, [], '2022-05-31'),
        identity=identity, effective_from=note['ticker_effective_date'], effective_through=None,
        transition=dict(from_ticker='FB', to_ticker='META', effective_date=note['ticker_effective_date'],
                        same_class_A_and_unchanged_CUSIP_supported=True, continuous_epoch_certified=False,
                        effective_date_literal=effective,
                        unchanged_CUSIP_literal=literal(text, r"Class A common stock will continue to be listed on NASDAQ and its CUSIP number will remain unchanged")),
        extraction=dict(path=text_path, sha256=inputs.hashes[text_path], basis='Existing reviewed PDF extraction, exact text equality to pinned identity note; no fresh PDF parser claim.'),
        source_use_window=None, filing_gate_satisfied=False, filing_gate_reasons=gate_reasons(identity)))
    require(len(bindings) == 110 and len({x['binding_id'] for x in bindings}) == 110, 'Fixed110-row scope/uniqueness')
    symbols = sorted({x['symbol'] for x in bindings})
    links = current_links(inputs, inventory, symbols)
    for binding in bindings:
        link = links[binding['symbol']]
        binding['current_registration_reference'] = link
        inherited = binding['identity']['prior_identity'].get('issuer_id', '')
        if re.fullmatch(r'\d{10}', inherited):
            require(inherited == link['cik'], 'Prior numeric CIK differs from current source')
        binding['CIK_basis'] = 'Current SEC registration is separately verified. Historical CIK/class identity is not inferred from issuer-name similarity or report-use window.'
    inputs.recheck()
    summary = dict(bindings=len(bindings), symbols=len(symbols), kinds=dict(Counter(x['kind'] for x in bindings)),
        original_bytes_and_receipts_verified=sum(x['source']['original_bytes_receipt_verified'] for x in bindings),
        original_date_literals_or_metadata_supported=sum(x['publication']['supported'] for x in bindings),
        issuer_names_supported=sum(x['identity']['source_local_issuer_supported'] for x in bindings),
        exchange_ticker_pairs_supported=sum(x['identity']['source_local_issuer_ticker_supported'] for x in bindings),
        exact_source_sibling_receipts_recovered=sum(x['source']['receipt_located_from_exact_source_sibling'] for x in bindings),
        current_registration_links_verified=len(links), same_security_transitions_supported=1,
        historical_security_epochs_admitted=0, new_filing_gates_satisfied=0, source_files_hashed=len(inputs.hashes))
    gaps = [dict(binding_id=x['binding_id'], missing_original_date=not x['publication']['supported'],
        missing_issuer_name=not x['identity']['source_local_issuer_supported'],
        missing_exchange_ticker_pair=not x['identity']['source_local_issuer_ticker_supported'])
        for x in bindings if not x['publication']['supported'] or not x['identity']['source_local_issuer_ticker_supported']]
    result = dict(schema_version=1, stage='LOCAL_DATED_IDENTITY_EVIDENCE_BINDING_BATCH_AFTER156',
        status='SOURCE_BINDINGS_BUILT_NOT_HISTORICAL_SECURITY_OR_FILING_CLOCK_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(), summary=summary, bindings=bindings, local_identity_or_date_gaps=gaps,
        synthetic_validation=dict(passed=True, checks=len(test_names), names=test_names),
        input_hashes=dict(sorted(inputs.hashes.items())),
        resolved_subconditions=['Original-byte and successful-retrieval provenance for all 110 fixed-scope records, including 16 KLAC receipts recovered by exact local source siblings.',
            'Dated source-local issuer/ticker evidence for the fixed report observations; this is usable identity evidence for a later interval-specific mapping review.',
            'Explicit FB-to-META Class A ticker transition effective before market open 2022-06-09, with unchanged CUSIP; identity continuity across that specific rename is supported.',
            'Current SEC CIK/ticker references for 14 issuers are separately hash-bound and are not backdated.'],
        remaining_source_gates=['Map the consumed historical event/formation intervals to the correct security class using these dated observations plus any required transition/continuity evidence; source-use windows cannot fill gaps.',
            'Bind a complete declared-scope original and amendment filing chronology with independently admitted availability clock; these earnings documents and current registration snapshots do not supply it.'],
        limits=['All 110 requested records preserved, including dates outside the prospective filing interval. No outcome-based filtering.',
            'Original date occurrence and current issuer registration do not certify first public versions, SEC acceptance instants, or uninterrupted security-class epochs.',
            'Inherited source-use windows and available_from fields are retained as prior-source metadata only, never promoted to effective security epochs.',
            'Meta supports a specific same-security transition, not all earlier/later daily history. Other rows support dated source observations only.'],
        activity_counts=dict(network_requests=0, price_values_read=0, financial_values_extracted=0, factors=0, ranks=0, orders=0, returns=0, registrations=0, frozen_files_modified=0))
    if args.write:
        with inputs.path(OUT).open('x') as file:
            json.dump(result, file, sort_keys=True, indent=2)
            file.write('\n')
    print(json.dumps(dict(summary=summary, gaps=gaps, output_sha256=sha(inputs.path(OUT).read_bytes()) if args.write else None), indent=2))


if __name__ == '__main__':
    main()
