"""Source-only USD corroboration for the five frozen Boeing annual releases."""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_ba_currency_bridge.py'
OUT = 'work/after158_ba_currency_bridge.json'
PLAN = 'work/after158_ba_source_plan.json'
REVIEW = 'work/after158_ba_independent_review.json'
FACTS = 'outputs/dynamic_top100/registry_sec/0000012927_facts.json'
PROVENANCE = 'outputs/dynamic_top100/registry_sec/0000012927_facts.provenance.json'
EXPECTED = {
    PLAN: '4b7067b5a0e60aece954ba6617d4aa050375d323c6d13fd3899eabdadba7f422',
    REVIEW: 'a26da326feafc899773e8c17897b7d6fee91c17f68614d720ab0cff988f84b74',
    FACTS: '0c63f1ae7cc10b3c27e0c94f2add0c461e6293036dd4e52e52ae6dbb6f206e15',
    PROVENANCE: '8a99e72c3d13c1412e962a10662d39fc2254a595e3a9945ddbf8cf5a18b3368e',
}
TAGS = ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT/name).read_text())


def exact_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Invalid date')
    return date.fromisoformat(value)


def compare_row(row, unit, year, millions):
    """Exact unit/context comparison, never a nearest-year or nearest-value match."""
    target = Decimal(millions.replace(',', '')) * Decimal(1000000)
    value = row.get('val')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Invalid monetary value')
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError('Non-finite monetary value')
    for key in ('start', 'end', 'filed'):
        exact_date(row[key])
    if not isinstance(row.get('accn'), str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', row['accn']):
        raise ValueError('Invalid accession')
    period_matches = row['start'] == f'{year}-01-01' and row['end'] == f'{year}-12-31'
    return dict(exact_annual_period_matches=period_matches,
                numeric_value_matches=value == target,
                unit_is_explicit_usd=unit == 'USD',
                currency_corroboration_matches=period_matches and value == target and unit == 'USD')


def collect(facts, year, millions):
    groups = {}
    for tag in TAGS:
        concept = facts['facts']['us-gaap'].get(tag)
        records, excluded = [], []
        if concept is not None:
            for unit, rows in sorted(concept['units'].items()):
                for index, row in enumerate(rows):
                    if row.get('end') != f'{year}-12-31':
                        continue
                    if row.get('start') != f'{year}-01-01':
                        excluded.append(dict(unit=unit, row_index=index, start=row.get('start'),
                                             end=row.get('end'), accn=row.get('accn'),
                                             reason='Not exact January 1 to December 31 annual context'))
                        continue
                    records.append(dict(json_pointer=f'/facts/us-gaap/{tag}/units/{unit}/{index}',
                                        unit=unit, row_index=index, source_row=row,
                                        comparison=compare_row(row, unit, year, millions)))
        matches = [r for r in records if r['comparison']['currency_corroboration_matches']]
        mismatches = [r for r in records if not r['comparison']['currency_corroboration_matches']]
        groups[tag] = dict(label=concept.get('label') if concept else None,
                           role='requested primary concept' if tag == 'Revenues' else 'explicit supplemental concept; not a silent substitute',
                           exact_annual_contexts=records, matched_context_count=len(matches),
                           mismatched_context_count=len(mismatches),
                           excluded_nonannual_same_end_contexts=excluded,
                           earliest_matching_filed=min((r['source_row']['filed'] for r in matches), default=None),
                           distinct_annual_values_by_unit={u:sorted({str(r['source_row']['val']) for r in records if r['unit']==u})
                                                          for u in sorted({r['unit'] for r in records})})
    return groups


def build():
    pins = dict(EXPECTED)
    for name, digest in pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    plan, review, facts, provenance = (read(p) for p in (PLAN, REVIEW, FACTS, PROVENANCE))
    if facts['cik'] != 12927 or facts['entityName'] != 'THE BOEING COMPANY':
        raise ValueError('Wrong issuer')
    if provenance['sha256'] != pins[FACTS] or provenance['source'] != 'https://data.sec.gov/api/xbrl/companyfacts/CIK0000012927.json':
        raise ValueError('Unbound companyfacts provenance')
    pins.update(plan['source_hashes'])
    if review['source_plan_sha256'] != pins[PLAN] or review['source_admission_changed'] is not False:
        raise ValueError('Wrong frozen review')
    targets={t['fiscal_year']:t for t in plan['targets']}
    if set(targets) != set(range(2014, 2019)) or {r['fiscal_year'] for r in review['reports']} != set(targets):
        raise ValueError('Unexpected release scope')
    reports=[]
    for original in review['reports']:
        year=original['fiscal_year']; target=targets[year]
        if original['raw'] != target['paths']['raw'] or original['raw_sha256'] != pins[original['raw']]:
            raise ValueError('Unbound release')
        millions=original['total_revenues_current_comparative'][0]
        groups=collect(facts, year, millions)
        primary_matches=groups['Revenues']['matched_context_count']
        all_matches=sum(x['matched_context_count'] for x in groups.values())
        reports.append(dict(fiscal_year=year, role=original['role'], release_raw=original['raw'],
                            release_raw_sha256=original['raw_sha256'],
                            release_receipt=target['paths']['receipt'],
                            release_cell=dict(label='Total revenues', column='current full year',
                                              printed_value=millions, unit_literal=original['unit_literal'],
                                              physical_pdf_page=original['statement_physical_page'],
                                              annual_header=original['annual_header'],
                                              expected_start=f'{year}-01-01', expected_end=f'{year}-12-31',
                                              start_is_calendar_inference_from_annual_header=True,
                                              expected_amount_in_usd_base_units=str(Decimal(millions.replace(',', ''))*Decimal(1000000))),
                            original_publication_dateline=original['original_publication_dateline'],
                            currency_corroborated_by_requested_revenues_concept=primary_matches>0,
                            currency_corroborated_by_any_declared_concept=all_matches>0,
                            concept_results=groups,
                            historical_availability_admitted=False,
                            currency_source_admission_changed=False))
    for path in (SELF, 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        pins[path]=sha(ROOT/path)
    for name,digest in pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Input changed during read: '+name)
    primary_all=all(r['currency_corroborated_by_requested_revenues_concept'] for r in reports)
    all_any=all(r['currency_corroborated_by_any_declared_concept'] for r in reports)
    return dict(schema='after158_ba_currency_bridge_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                passed=True, verification_scope='Bounded source unit-and-value corroboration, not strategy or historical timestamp admission',
                symbol='BA', cik=12927, input_hashes=pins, companyfacts_provenance=provenance,
                source_admission_changed=False, frozen_sources_and_reviews_unchanged=True,
                five_original_cells_match_revenues_usd=primary_all,
                five_original_cells_match_declared_usd_concepts=all_any,
                reports=reports,
                method=dict(source_values='Current annual total-revenue cells in the independently checked five release PDFs; no financial values from a strategy grid',
                            all_contexts_retained='Every exact annual context across every unit for both declared concepts; mismatches and later filings are retained, not overwritten or chosen as a fallback',
                            amount_comparison='Exact decimal release amount multiplied by 1,000,000; no return, growth, rank or feature calculation',
                            fiscal_year_metadata='Annual context selected using exact start/end, not fy/fp. fy/fp belongs to the filing containing the fact and can describe a later comparative filing.',
                            concept_policy=list(TAGS),
                            no_external_requests=True),
                recommendation=dict(currency=('Explicit USD units and exact annual values corroborate treating these original release totals as USD in a separately versioned, declared within-issuer research policy.' if all_any else 'Only cells with explicit exact-period USD matches are corroborated; do not infer the unmatched cells.'),
                                    policy_scope='Five specified current annual total-revenue cells only. This corroborates currency, not organic comparability, an original-release USD declaration, or every dollar amount in the documents.',
                                    mismatches='Preserve all observed later/other-basis values. A mismatch is evidence of a different reported context, not a currency conversion and not authority to replace the original release cell.',
                                    accounting='The frozen original pre-adoption 2014-2017 chain and the 2018 ASC606/715 boundary remain separate.',
                                    availability='Currency corroboration does not date public access. For a future declared-assumption date-only study, original release datelines plus the already-bound official archive can support a conservative next-complete-XNYS-session rule; separately preregister the rule and check cutoffs. Exact-instant study remains unsupported by these sources.',
                                    availability_caveat='This does not certify first publication, contemporaneous bytes, or historical absence of replacement. Do not use companyfacts filed, current retrieval, archive midnight, PDF creation/modification, or rendering times as the release publication instant.'),
                activity=dict(downloads=0, factors=0, ranks=0, strategy_trials=0, orders=0, returns=0, model_fits=0))


class Tests(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(start='2014-01-01',end='2014-12-31',filed='2015-02-12',
                         accn='0000012927-15-000011',val=90762000000,form='10-K',fy=2014,fp='FY'), **changes)
    def test_exact_usd(self):
        self.assertTrue(compare_row(self.row(),'USD',2014,'90,762')['currency_corroboration_matches'])
    def test_same_number_other_currency_is_not_a_match(self):
        self.assertFalse(compare_row(self.row(),'CAD',2014,'90,762')['currency_corroboration_matches'])
    def test_same_number_quarter_is_not_a_match(self):
        self.assertFalse(compare_row(self.row(start='2014-10-01'),'USD',2014,'90,762')['currency_corroboration_matches'])
    def test_restated_amount_is_retained_as_mismatch(self):
        self.assertFalse(compare_row(self.row(val=94005000000),'USD',2014,'90,762')['currency_corroboration_matches'])
    def test_later_filing_match_retained_without_claiming_publication(self):
        row=self.row(filed='2017-02-12',fy=2016)
        self.assertTrue(compare_row(row,'USD',2014,'90,762')['currency_corroboration_matches'])
    def test_no_boolean_value(self):
        with self.assertRaises(ValueError): compare_row(self.row(val=True),'USD',2014,'90,762')
    def test_invalid_date_rejected(self):
        with self.assertRaises(ValueError): compare_row(self.row(filed='2015-02-30'),'USD',2014,'90,762')
    def test_full_denominator_and_duplicate_retention(self):
        rows=[self.row(),self.row(),self.row(val=90000000000),self.row(start='2014-10-01')]
        facts={'facts':{'us-gaap':{'Revenues':{'units':{'USD':rows,'CAD':[self.row()]}}}}}
        group=collect(facts,2014,'90,762')['Revenues']
        self.assertEqual((group['matched_context_count'],group['mismatched_context_count'],len(group['excluded_nonannual_same_end_contexts'])),(2,2,1))
        self.assertEqual(len(group['exact_annual_contexts']),4)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test',action='store_true');parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        if not result.wasSuccessful(): raise SystemExit(1)
        return
    report=build()
    if args.write:
        data=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        fd=os.open(ROOT/OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
        with os.fdopen(fd,'wb') as handle: handle.write(data)
        print(json.dumps(dict(output=OUT,sha256=sha(ROOT/OUT))))
    print(json.dumps([dict(year=r['fiscal_year'],**{tag:{k:v[k] for k in ('matched_context_count','mismatched_context_count','earliest_matching_filed','distinct_annual_values_by_unit')} for tag,v in r['concept_results'].items()}) for r in report['reports']],indent=2))

if __name__=='__main__': main()
