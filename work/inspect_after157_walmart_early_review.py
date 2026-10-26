"""Independent offline chain checks and bounded raw-text inspection; no market data."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'work/after157_walmart_bodies_early_scope.json'
SCOPE_SHA = '4bba503d03789791e8880b1ff50f5d1c4d8e24b97ea7874c8a6df3bb45a15393'
OUT = 'outputs/after157_walmart_bodies_early_v1'
MANIFEST_SHA = 'ba0614ad2dde21b4d1cd61abd3befe033d338f0afc195d4e0a30ce812863126e'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plain(value):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]*>', '', value))).strip()


def verified_inputs():
    hashes = {}

    def read(name, expected=None):
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        raw = path.read_bytes()
        digest = sha(raw)
        assert expected is None or expected == digest, name
        assert name not in hashes or hashes[name] == digest, name
        hashes[name] = digest
        return raw

    scope = json.loads(read(SCOPE, SCOPE_SHA))
    assert scope['output_dir'] == OUT
    for name, expected in scope['input_hashes'].items():
        read(name, expected)
    manifest_name = OUT + '/derived_text/MANIFEST.json'
    manifest = json.loads(read(manifest_name, MANIFEST_SHA))
    assert manifest['scope_sha256'] == SCOPE_SHA
    for name, expected in manifest['input_hashes'].items():
        read(name, expected)
    assert manifest['source_admission_changed'] is False
    collection = json.loads(read(OUT + '/COLLECTION.json', manifest['collection_sha256']))
    assert collection['all_successful'] is True and collection['stopped_providers'] == []
    binding = dict(schema=scope['schema'], scope_sha256=SCOPE_SHA,
                   helper_sha256=scope['helper_sha256'])
    assert json.loads(read(OUT + '/.collection-scope.json')) == binding
    assert all(collection[k] == v for k, v in binding.items())
    assert len(scope['targets']) == len(collection['targets']) == len(manifest['records']) == 6
    metadata = json.loads(read('work/after157_annual_pair_source_feasibility.json'))[
        'selected_original_and_amendment_metadata']
    records = []
    for year, target, collected, derived in zip(range(2015, 2021), scope['targets'],
                                               collection['targets'], manifest['records']):
        key = target['id']
        assert key == 'wmt_fy%d_original_10k' % year
        assert collected['id'] == derived['id'] == key
        expected = target['expected_original']
        assert derived['expected_original'] == expected
        assert all(metadata[expected['event_id']][k] == v for k, v in expected.items())
        assert expected['form'] == '10-K' and expected['cik'] == '0000104169'
        assert expected['reportDate'] == '%d-01-31' % year
        assert target['url'].endswith('/' + expected['accessionNumber'] + '/' + expected['primaryDocument'])
        location = target['original_metadata_location']
        page = json.loads(read(location['source'], location['source_sha256']))
        assert all(page[k][location['row']] == expected[k] for k in
                   ('accessionNumber', 'form', 'reportDate', 'filingDate', 'primaryDocument'))
        evidence = target['publisher_evidence']
        index = read(evidence['path']).decode('utf-8')
        assert target['url'] in index.splitlines()[evidence['raw_link_line']-1]
        receipt_name = OUT + '/' + target['receipt_file']
        receipt = json.loads(read(receipt_name, collected['receipt_sha256']))
        assert hashes[receipt_name] == derived['receipt_sha256']
        tb = dict(binding, id=key, provider='walmart', source_url=target['url'], body_file=target['body_file'])
        assert all(receipt[k] == v for k, v in tb.items())
        assert receipt['attempted'] is True and receipt['outcome'] == collected['outcome'] == derived['outcome'] == 'success'
        assert receipt['http_status'] == derived['http_status'] == 200
        assert receipt['stop_provider'] is False and receipt['body_truncated_at_limit'] is False
        assert receipt['final_url'] == target['url'] and receipt['redirect_chain'] == []
        attempt_name = OUT + '/.attempt_' + key + '.json'
        attempt = json.loads(read(attempt_name, receipt['attempt_sha256']))
        assert all(attempt[k] == v for k, v in tb.items())
        assert hashes[attempt_name] == derived['attempt_sha256']
        raw_name = OUT + '/' + target['body_file']
        raw = read(raw_name, receipt['sha256'])
        assert hashes[raw_name] == derived['raw_sha256'] and len(raw) == receipt['bytes'] == derived['raw_bytes']
        assert derived['raw_file'] == target['body_file'] and derived['source_url'] == target['url']
        text_name = OUT + '/derived_text/' + derived['text_file']
        text = read(text_name, derived['text_sha256']).decode('utf-8')
        assert len(text.splitlines()) == derived['lines']
        content = raw.decode('utf-8')
        assert '</html>' in content.lower()
        records.append(dict(year=year, target=target, receipt=receipt, raw=content,
                            text=text, derived=derived))
    read('work/inspect_after157_walmart_early_review.py')
    return hashes, records


def inspect(records):
    for record in records:
        raw = record['raw']
        lines = raw.splitlines()
        normalized = [plain(s) for s in lines]
        found = []
        terms = ('Our fiscal year ends', 'legal name of our corporation',
                 'named "Wal-Mart Stores, Inc." prior', 'Annual report pursuant',
                 'Transition report pursuant', 'Exact name of registrant',
                 'SIGNATURES', 'Date:', 'DATE:')
        for term in terms:
            indices = [i for i, line in enumerate(normalized) if term in line]
            if term in ('Date:', 'DATE:'):
                indices = indices[:1]
            for i in indices[:2]:
                start = max(0, i-1) if term not in ('Our fiscal year ends',) else i
                stop = min(len(lines), i+4)
                found.append(dict(term=term, raw_start_line=start+1, raw_end_line=stop,
                                  text=' '.join(normalized[start:stop])[:2300]))
        canonical = [i+1 for i, line in enumerate(lines) if 'canonical' in line and record['target']['url'] in line]
        print(json.dumps(dict(year=record['year'], canonical_raw_lines=canonical, evidence=found), ensure_ascii=False))
        if record['year'] == 2020:
            for pattern in (r'<xbrli:context\b[^>]*id="FD2020Q4YTD"[^>]*>.*?</xbrli:context>',
                            r'<ix:nonnumeric\b[^>]*name="dei:(?:DocumentType|DocumentFiscalYearFocus|DocumentFiscalPeriodFocus|AmendmentFlag|EntityCentralIndexKey|DocumentPeriodEndDate)"[^>]*>.*?</ix:nonnumeric>'):
                for match in re.finditer(pattern, raw, flags=re.I|re.S):
                    print(json.dumps(dict(year=2020, raw_line=raw[:match.start()].count('\n')+1,
                                          markup=match.group(0))))


def review(hashes, records):
    """Serialize the six accession-specific, manually reviewed document findings."""
    originals = []
    for record in records:
        year, target, receipt = record['year'], record['target'], record['receipt']
        expected, derived = target['expected_original'], record['derived']
        raw, text = record['raw'], record['text']
        lines, texts = raw.splitlines(), text.splitlines()
        clean = [plain(line) for line in lines]

        def hit(term, start=0):
            return next(i for i in range(start, len(clean)) if term in clean[i])

        def text_hit(term):
            return next(i+1 for i, line in enumerate(texts) if term in line)

        transition = hit('Transition report pursuant')
        annual = max(i for i in range(transition) if 'Annual report pursuant' in clean[i])
        end = hit('fiscal year ended January 31, %d' % year, annual)
        registrant = hit('Exact name of registrant', end)-1
        name = 'WAL-MART STORES, INC.' if year <= 2017 else 'WALMART INC.'
        assert clean[registrant] == name
        checkbox, emptybox = ('ý', '¨') if year < 2020 else ('☒', '☐')
        assert checkbox in lines[annual-1] and emptybox in lines[transition-1]
        assert ('Wingdings' if year < 2020 else 'boolballotbox') in lines[annual-1]
        assert ('Wingdings' if year < 2020 else 'boolballotbox') in lines[transition-1]
        convention = hit('Our fiscal year ends on January 31')
        paragraph = clean[convention]
        assert all('January 31, %d' % y in paragraph for y in (year, year-1, year-2))
        assert 'one-month lag' in paragraph
        signature = hit('SIGNATURES')
        signatory = hit('/s/ C. Douglas McMillon', signature)
        sig_end = hit('President and Chief Executive Officer', signatory)
        sig_text = ' '.join(clean[signature:sig_end+1])
        date = datetime.strptime(expected['filingDate'], '%Y-%m-%d').strftime('%B %d, %Y').replace(' 0', ' ')
        assert date in sig_text and name.lower() in sig_text.lower()
        canonical = [i+1 for i, line in enumerate(lines) if 'rel="canonical"' in line and target['url'] in line]
        assert canonical
        ein = hit('71-0415188', registrant)
        file_number = hit('Commission file number', end)
        period = dict(fiscal_year=year, start='%d-02-01' % (year-1), end=expected['reportDate'],
                      start_directly_printed_as_full_year_boundary=year == 2020,
                      end_directly_printed=True,
                      basis='direct_unsegmented_document_annual_context' if year == 2020 else
                      'derived_from_ordinary_annual_cover_and_explicit_recurring_January31_convention',
                      detail='February1 of the preceding year is derived for FY2015-2019 from the ordinary full-year cover and each document\'s explicit recurring January31 convention, not merely the spacing of report dates. The same document labels the current and two preceding fiscal years. No claim that foreign subsidiary accounting cutoffs match the parent reporting year.',
                      explicit_convention_prose=dict(raw_line=convention+1,
                          derived_line=text_hit('Our fiscal year ends on January 31'),
                          quote=paragraph.split(' During fiscal ')[0]))
        if year == 2020:
            context = re.search(r'<xbrli:context\b[^>]*id="FD2020Q4YTD"[^>]*>.*?</xbrli:context>', raw, re.I|re.S)
            assert context and 'segment' not in context.group(0)
            assert all(value in context.group(0) for value in ('0000104169', '2019-02-01', '2020-01-31'))
            facts = {}
            for field, value in [('AmendmentFlag', 'false'), ('DocumentFiscalPeriodFocus', 'FY'),
                                 ('DocumentFiscalYearFocus', '2020'), ('EntityCentralIndexKey', '0000104169'),
                                 ('DocumentType', '10-K'), ('DocumentPeriodEndDate', 'January 31, 2020')]:
                m = re.search(r'<ix:nonnumeric\b[^>]*name="dei:' + field + r'"[^>]*>.*?</ix:nonnumeric>', raw, re.I|re.S)
                assert m and 'contextref="FD2020Q4YTD"' in m.group(0) and plain(m.group(0)) == value
                facts[field] = dict(value=value, raw_line=raw[:m.start()].count('\n')+1)
            period['direct_context'] = dict(id='FD2020Q4YTD', unsegmented=True,
                raw_start_line=raw[:context.start()].count('\n')+1,
                raw_end_line=raw[:context.end()].count('\n')+1,
                cik='0000104169', start='2019-02-01', end='2020-01-31', linked_dei_facts=facts)
        evidence = dict(
            accession_bridge=dict(kind='direct_official_mirror_mapping', source_url=target['url'],
                final_url=receipt['final_url'], canonical_raw_lines=canonical,
                publisher_index_evidence=target['publisher_evidence'],
                frozen_metadata_location=target['original_metadata_location'],
                detail='Pinned issuer index directly links exact accession and primaryDocument; current successful no-redirect response retains exact canonical URL and contains the ordinary report. FY2018 embeds the report within an outer issuer-site page; both outer and inner canonicals match. No year-only or ticker-only inference.'),
            ordinary_annual=dict(kind='direct_font_aware_cover', form='10-K',
                raw_lines=[annual, annual+1, end+1, transition, transition+1],
                derived_annual_line=text_hit('Annual report pursuant to section'),
                derived_end_line=text_hit('fiscal year ended January 31, %d' % year),
                annual_checkbox_selected=True, transition_checkbox_selected=False,
                annual_glyph=checkbox, transition_glyph=emptybox,
                transition_from_to_fields_present=False,
                detail='Annual checkbox is selected; transition checkbox is empty. These covers do not print a transition from/to interval; absence is not described as a blank printed interval.'),
            registrant=dict(kind='direct_legal_name_and_cover_identifiers', legal_name=name,
                raw_lines=[registrant+1, registrant+2, file_number+1, ein+1],
                file_number='1-6991' if year == 2015 else '001-6991', ein='71-0415188',
                cik='0000104169', cik_source='linked document annual XBRL context and frozen original metadata' if year == 2020 else 'frozen accession-specific original metadata; not inferred from current wrapper ticker',
                historical_price_identity_certified=False),
            annual_period=period,
            signature=dict(kind='direct_cross_check_only', raw_start_line=signature+1,
                raw_end_line=sig_end+1, derived_start_line=text_hit('SIGNATURES'),
                derived_end_line=text_hit('SIGNATURES')+7,
                signatory='C. Douglas McMillon', role='President and Chief Executive Officer',
                date=expected['filingDate'], date_matches_frozen_filing_date=True,
                detail='Signature date is cross-check only; it is not acceptance or historical-public-availability certification.'))
        if year >= 2018:
            rename = hit('named "Wal-Mart Stores, Inc." prior to February 1, 2018')
            evidence['explicit_legal_name_bridge'] = dict(raw_line=rename+1,
                derived_line=text_hit('named "Wal-Mart Stores, Inc." prior to February 1, 2018'),
                effective_date='2018-02-01', from_name='Wal-Mart Stores, Inc.', to_name='Walmart Inc.',
                quote=clean[rename], historical_price_identity_certified=False)
        originals.append(dict(target_id=target['id'], symbol='WMT', event_id=expected['event_id'],
            cik=expected['cik'], accession=expected['accessionNumber'], form='10-K',
            primary_document=expected['primaryDocument'], fiscal_year=year,
            report_date=expected['reportDate'], filing_date_from_frozen_metadata=expected['filingDate'],
            exact_original_version_bound=True, normal_annual_context_verified=True,
            raw_path=OUT+'/'+target['body_file'], raw_sha256=derived['raw_sha256'], raw_bytes=derived['raw_bytes'],
            receipt_path=OUT+'/'+target['receipt_file'], receipt_sha256=derived['receipt_sha256'],
            attempt_path=OUT+'/.attempt_'+target['id']+'.json', attempt_sha256=derived['attempt_sha256'],
            derived_path=OUT+'/derived_text/'+derived['text_file'], derived_sha256=derived['text_sha256'],
            retrieved_utc=receipt['retrieved_utc'], evidence=evidence,
            missing_or_contradictory_document_evidence=[],
            limitations=['Current official mirror mapping is not SEC-host byte comparison or proof that these bytes were historically public at the cutoff.',
                         'FY2015-2017 incorporate portions of the annual report as Exhibit13. That linked exhibit is not collected here; this finding concerns the primary10-K cover, explicit reporting convention and signature, not completeness of financial exhibits.'] if year < 2018 else
                        ['Current official mirror mapping is not SEC-host byte comparison or proof that these bytes were historically public at the cutoff.']))
    pairs = []
    for i in range(5):
        two = originals[i:i+2]
        pairs.append(dict(fiscal_years=[o['fiscal_year'] for o in two],
            event_ids=[o['event_id'] for o in two], target_ids=[o['target_id'] for o in two],
            exact_original_versions_bound=True, normal_consecutive_pair_context_verified=True,
            source_admission_changed=False,
            periods=[dict(event_id=o['event_id'], start=o['evidence']['annual_period']['start'],
                          end=o['report_date'], start_is_derived=o['fiscal_year']<2020) for o in two],
            evidence_refs=['#/originals/%d/evidence/annual_period' % j for j in (i,i+1)],
            continuity_basis='Each exact ordinary annual report states the recurring January31 reporting convention and labels current and prior fiscal years; adjacent named parent fiscal years are consecutive full years. February1 starts are labelled derivations except the direct FY2020 annual XBRL context.',
            legal_registrant_bridge_ref='#/originals/3/evidence/explicit_legal_name_bridge' if i == 2 else None,
            continuity_scope='Only fiscal-period and legal-registrant document context. No assertion of unchanged accounting policy, consolidated business perimeter, subsidiary reporting cutoffs, historical mirror availability or price identity.',
            remaining_source_and_slot_gates='All original timing, page, selection, competing-original, amendment, unresolved141 and identity gates remain; specifically WMT2018-05-31 signal_index22 candidate_index40 LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42 remains blocked.'))
    rule = json.loads((ROOT/'work/after157_annual_pair_source_feasibility_plan.json').read_bytes())['rules']['positive_tier3']
    return dict(schema_version=1, reviewed_utc=datetime.now(timezone.utc).isoformat(),
        status='SIX_EXACT_ORIGINALS_AND_FIVE_NORMAL_CONSECUTIVE_PAIRS_DOCUMENT_CONTEXT_VERIFIED_ONLY',
        purpose='Independent semantic review of scoped WMT FY2015-2020 original primary10-K bodies.',
        source_admission_changed=False, exact_acceptance_certified=False,
        historical_price_identity_certified=False,
        exact_original_versions_bound=True, normal_consecutive_pair_context_verified=True,
        boolean_scope='Top-level true values apply only to these six originals and five exact listed pairs, only at document context level. They do not satisfy t-42 availability or change frozen tier3/normal_period_admission.',
        target_ids=[o['target_id'] for o in originals], reviewed_target_ids=[o['target_id'] for o in originals],
        expected_event_ids=[o['event_id'] for o in originals], originals=originals, pairs=pairs,
        input_hashes=hashes, counts=dict(originals=6, pairs=5, raw_bytes=sum(o['raw_bytes'] for o in originals),
            derived_lines=sum(r['derived']['lines'] for r in records), input_hashes_verified=len(hashes)),
        integrity=dict(all_input_hashes_verified=True, scope_sha256=SCOPE_SHA,
            collection_sha256=hashes[OUT+'/COLLECTION.json'], derived_manifest_sha256=MANIFEST_SHA,
            original_metadata_seven_fields_and_raw_submission_rows_verified=True,
            receipts_attempts_raw_bytes_and_derived_hashes_verified=True,
            method='Independent stdlib read-only verifier; no collector/producer imports or extraction rerun. Original covers, font/checkbox markup, period statements and signatures manually inspected against raw HTML. Derived text only aids location.',
            completeness_limit='Receipts report successful EOF/untruncated 200 bodies; raw closes HTML and includes cover/signature. Linked exhibits/images not claimed collected.'),
        frozen_rule=dict(path='work/after157_annual_pair_source_feasibility_plan.json', field='rules.positive_tier3', unchanged=True, text=rule),
        inherited_gates_not_relaxed=['t-42 availability including WMT2018-05-31', 'exact acceptance interpretation',
            'historical price-security identity', 'latest-original and amendment chronology', 'snapshot/page closure',
            'unresolved141 original flags', 'full-grid source floors and prospective numerical registration'],
        remaining_blockers=['No frozen row, tier3 flag or source-admission decision is changed.',
            'Document period verification does not remove LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42 for WMT2018-05-31.',
            'This does not prove original financial statement content for incorporated but uncollected exhibits.'],
        activity=dict(network_requests=0, market_arrays_opened=0, filing_lag_calculations=0,
            actual_factors_calculated=0, actual_ranks_calculated=0, actual_orders_calculated=0,
            actual_returns_calculated=0, frozen_files_changed=0, new_strategy_trials=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--write-review', action='store_true')
    args = parser.parse_args()
    hashes, records = verified_inputs()
    if args.inspect:
        inspect(records)
    if args.write_review:
        result = review(hashes, records)
        # Recheck the complete consumed closure before a write-once new review.
        for name, expected in hashes.items():
            assert sha((ROOT/name).read_bytes()) == expected, name
        destination = ROOT / 'work/after157_walmart_early_review.json'
        encoded = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
        with destination.open('xb') as handle:
            handle.write(encoded)
        assert json.loads(destination.read_bytes()) == result
        print(json.dumps(dict(review_path=str(destination), review_sha256=sha(encoded))))
    print(json.dumps(dict(passed=True, input_hashes=len(hashes), originals=len(records),
                          raw_bytes=sum(x['derived']['raw_bytes'] for x in records),
                          derived_lines=sum(x['derived']['lines'] for x in records),
                          source_admission_changed=False)))
