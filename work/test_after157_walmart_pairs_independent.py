"""Independent synthetic checks for complete, fail-closed WMT pair adjudication.

Only artificial metadata is supplied to collect_pairs. No source bodies, prices,
research outputs, collector or admission main are read or executed here.
"""
import copy
import random
import unittest

from after157_walmart_context_admission import collect_pairs


YEARS = tuple(range(2015, 2027))
EDGES = ((2015, 2016), (2016, 2017), (2017, 2018), (2018, 2019),
         (2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023),
         (2023, 2024), (2024, 2025), (2025, 2026))


def event(year):
    return 'synthetic-original-' + str(year)


def expected_edges(excluded=()):
    return {frozenset((event(a), event(b))) for a, b in EDGES
            if (a, b) not in excluded}


def fixture():
    metadata, targets = {}, []
    reviews = [dict(source_admission_changed=False, originals=[], pairs=[])
               for _ in range(2)]
    for year in YEARS:
        item = dict(event_id=event(year), cik='0000104169', form='10-K',
                    accessionNumber='synthetic-accession-' + str(year),
                    primaryDocument='synthetic-' + str(year) + '.htm',
                    reportDate=str(year) + '-01-31', filingDate=str(year) + '-03-15')
        metadata[event(year)] = copy.deepcopy(item)
        targets.append(dict(id='target-' + str(year), expected_original=item))
        reviews[int(year >= 2021)]['originals'].append(dict(
            target_id='target-' + str(year), symbol='WMT', cik=item['cik'],
            form=item['form'], fiscal_year=year, event_id=event(year),
            accession=item['accessionNumber'], primary_document=item['primaryDocument'],
            report_date=item['reportDate'], filing_date_from_frozen_metadata=item['filingDate'],
            exact_original_version_bound=True, normal_annual_context_verified=True))
    for a, b in EDGES:
        reviews[int(b >= 2021)]['pairs'].append(dict(
            fiscal_years=[a, b], event_ids=[event(a), event(b)],
            exact_original_versions_bound=True,
            normal_consecutive_pair_context_verified=True,
            source_admission_changed=False))
    return reviews, targets, metadata


def original(reviews, year):
    return next(o for r in reviews for o in r['originals'] if o['event_id'] == event(year))


def pair(reviews, a, b):
    return next(p for r in reviews for p in r['pairs'] if p['fiscal_years'] == [a, b])


class IndependentPairs(unittest.TestCase):
    def test_complete_twelve_originals_eleven_pairs_and_no_input_mutation(self):
        data = fixture()
        before = copy.deepcopy(data)
        self.assertEqual(collect_pairs(*data), expected_edges())
        self.assertEqual(data, before)

    def test_review_target_and_endpoint_order_do_not_change_result(self):
        for seed in range(16):
            data = fixture()
            rng = random.Random(seed)
            for review in data[0]:
                rng.shuffle(review['originals'])
                rng.shuffle(review['pairs'])
                for value in review['pairs']:
                    rng.shuffle(value['event_ids'])
            rng.shuffle(data[0])
            rng.shuffle(data[1])
            with self.subTest(seed=seed):
                self.assertEqual(collect_pairs(*data), expected_edges())

    def test_negative_original_and_incident_negative_pairs_preserve_other_pairs(self):
        for year in YEARS:
            for flag in ('exact_original_version_bound', 'normal_annual_context_verified'):
                data = fixture()
                original(data[0], year)[flag] = False
                excluded = []
                for a, b in EDGES:
                    if year in (a, b):
                        pair(data[0], a, b)['normal_consecutive_pair_context_verified'] = False
                        excluded.append((a, b))
                with self.subTest(year=year, flag=flag):
                    self.assertEqual(collect_pairs(*data), expected_edges(excluded))

    def test_negative_original_cannot_be_rescued_by_positive_pair(self):
        for year in YEARS:
            data = fixture()
            original(data[0], year)['exact_original_version_bound'] = False
            with self.subTest(year=year), self.assertRaisesRegex(ValueError, 'unresolved original'):
                collect_pairs(*data)

    def test_negative_pair_is_excluded_even_when_both_originals_positive(self):
        for edge in EDGES:
            for flag in ('exact_original_versions_bound', 'normal_consecutive_pair_context_verified'):
                data = fixture()
                pair(data[0], *edge)[flag] = False
                with self.subTest(edge=edge, flag=flag):
                    self.assertEqual(collect_pairs(*data), expected_edges((edge,)))

    def test_all_explicit_negative_reviews_return_empty_without_missing_adjudication(self):
        data = fixture()
        for review in data[0]:
            for o in review['originals']:
                o['normal_annual_context_verified'] = False
            for p in review['pairs']:
                p['normal_consecutive_pair_context_verified'] = False
        self.assertEqual(collect_pairs(*data), set())

    def test_missing_cross_batch_2020_2021_pair_is_not_inferred(self):
        data = fixture()
        data[0][1]['pairs'].remove(pair(data[0], 2020, 2021))
        with self.assertRaisesRegex(ValueError, 'Incomplete pair review'):
            collect_pairs(*data)

    def test_missing_original_is_not_rescued_from_context_only_prior_reference(self):
        data = fixture()
        missing = original(data[0], 2020)
        data[0][0]['originals'].remove(missing)
        data[0][1]['prior_original_reference'] = missing
        with self.assertRaisesRegex(ValueError, 'Incomplete original review'):
            collect_pairs(*data)

    def test_missing_boundary_originals_rejected(self):
        for year in (2015, 2026):
            data = fixture()
            data[0][int(year >= 2021)]['originals'].remove(original(data[0], year))
            with self.subTest(year=year), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_duplicate_original_rejected_even_if_duplicate_is_identical(self):
        data = fixture()
        data[0][1]['originals'].append(copy.deepcopy(original(data[0], 2020)))
        with self.assertRaisesRegex(ValueError, 'duplicate original'):
            collect_pairs(*data)

    def test_unknown_original_rejected(self):
        data = fixture()
        data[0][0]['originals'][0]['event_id'] = 'unscoped-original'
        with self.assertRaises(ValueError):
            collect_pairs(*data)

    def test_duplicate_pair_rejected_in_either_order(self):
        for reverse in (False, True):
            data = fixture()
            duplicate = copy.deepcopy(pair(data[0], 2020, 2021))
            if reverse:
                duplicate['event_ids'].reverse()
            data[0][0]['pairs'].append(duplicate)
            with self.subTest(reverse=reverse), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_unknown_repeated_nonconsecutive_or_three_endpoint_pair_rejected(self):
        for ids in ([event(2020), 'unknown'], [event(2020), event(2020)],
                    [event(2020), event(2022)], [event(2020), event(2021), event(2022)]):
            data = fixture()
            pair(data[0], 2020, 2021)['event_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_duplicate_target_year_or_event_rejected(self):
        for key, value in (('reportDate', '2015-01-31'), ('event_id', event(2015))):
            data = fixture()
            data[1][-1]['expected_original'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_wrong_metadata_identity_rejected_even_if_scope_matches_it(self):
        for key, value in (('cik', '0000000001'), ('form', '10-K/A'), ('form', '10-KT')):
            data = fixture()
            data[2][event(2024)][key] = value
            data[1][9]['expected_original'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_scope_metadata_drift_rejected(self):
        data = fixture()
        data[2][event(2024)]['filingDate'] = '2024-04-15'
        with self.assertRaisesRegex(ValueError, 'Frozen target metadata changed'):
            collect_pairs(*data)

    def test_original_accession_document_or_report_date_drift_rejected(self):
        for key, value in (('accession', 'wrong-original'), ('primary_document', 'other.htm'),
                           ('report_date', '2024-12-31')):
            data = fixture()
            original(data[0], 2024)[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Original binding mismatch'):
                collect_pairs(*data)

    def test_original_declared_identity_contradiction_rejected(self):
        # A valid event id must not silently authorize a self-contradictory review.
        for key, value in (('cik', '0000000001'), ('form', '10-K/A'),
                           ('symbol', 'OTHER'), ('fiscal_year', 2023)):
            data = fixture()
            original(data[0], 2024)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect_pairs(*data)

    def test_original_and_pair_judgements_require_actual_booleans(self):
        for value in (0, 1, None, 'false', [], {}):
            for kind, field in (('original', 'exact_original_version_bound'),
                                ('original', 'normal_annual_context_verified'),
                                ('pair', 'exact_original_versions_bound'),
                                ('pair', 'normal_consecutive_pair_context_verified')):
                data = fixture()
                target = original(data[0], 2024) if kind == 'original' else pair(data[0], 2023, 2024)
                target[field] = value
                with self.subTest(kind=kind, field=field, value=value), self.assertRaises(ValueError):
                    collect_pairs(*data)

    def test_review_cannot_claim_prior_source_admission_changed(self):
        for value in (True, None, 0, 'false'):
            data = fixture()
            data[0][1]['source_admission_changed'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect_pairs(*data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
