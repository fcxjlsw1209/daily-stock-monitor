"""Assemble explicit, source-bound exceptions; never suppress unknown gaps."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent


def evidence(path, source_url=None):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_url=source_url)


def main():
    result = []
    folder = P / 'warmup/final_scope_audit'
    file = folder / 'SPCX_IDENTITY_REVIEW.json'
    spcx = json.loads(file.read_text())
    for day in spcx['candidate_dates']:
        if not day < spcx['first_verified_regular_post_ipo_earnings_date']:
            raise ValueError('SPCX structural exception crossed first release')
        refs = [evidence(file)]
        for item in spcx['proofs']:
            ref = evidence(item['evidence_path'], item['source_url'])
            if ref['sha256'] != item['sha256']:
                raise ValueError('SPCX source changed')
            refs.append(ref)
        result.append(dict(issue_id='missing_prior_event|monthly|' + day + '|SPCX',
            resolution_kind='before_first_verified_regular_post_ipo_release',
            reason=spcx['allowed_output'], limitation=spcx['negative_evidence_limit'], evidence=refs))
    file = folder / 'ENDPOINT_RESOLUTIONS.json'
    for row in json.loads(file.read_text()):
        if row['symbol'] != 'AVGO':
            continue
        if not row['last_actual_candidate'] < row['next_actual_release_date']:
            raise ValueError('Next publication already occurred at candidate date')
        response_ref = evidence(row['evidence_path'], row['source_url'])
        if response_ref['sha256'] != row['sha256']:
            raise ValueError('Endpoint publication source changed')
        result.append(dict(issue_id='archive_endpoint_review|monthly|' + row['last_actual_candidate'] + '|AVGO',
            resolution_kind='next_regular_release_after_last_candidate', reason=row['interpretation'],
            next_release=row['next_actual_release_date'], evidence=[evidence(file, row['source_url']), response_ref]))
    file = P / 'batch_health_energy_foreign/BABA_FIRST_LISTED_EARNINGS.json'
    baba = json.loads(file.read_text())
    if not baba['candidate_date'] < baba['first_regular_release_date']:
        raise ValueError('BABA structural exception crossed first release')
    refs = [evidence(file)]
    for key in ('ipo_evidence', 'first_regular_release_evidence', 'official_2014_archive_evidence'):
        item = baba[key]
        ref = evidence(item['evidence_path'], item['source_url'])
        if ref['sha256'] != item['sha256']:
            raise ValueError('BABA source changed')
        refs.append(ref)
    result.append(dict(issue_id='missing_prior_event|training|' + baba['candidate_date'] + '|BABA',
        resolution_kind=baba['classification'], reason=baba['interpretation'],
        limitation=baba['limitation'], evidence=refs))
    file = P / 'warmup/sap_predecessor/BOUNDARY.json'
    sap = json.loads(file.read_text())
    if not sap['signal_date'] < sap['next_release']:
        raise ValueError('SAP boundary crossed actual preliminary publication')
    refs = [evidence(file)]
    for item in sap['evidence']:
        ref = evidence(item['path'], item['source_url'])
        if ref['sha256'] != item['sha256']:
            raise ValueError('SAP boundary source changed')
        refs.append(ref)
    result.append(dict(issue_id=sap['issue_id'], resolution_kind=sap['resolution_kind'],
        reason=sap['reason'], expected_prior_release=sap['previous_release_date'],
        next_release=sap['next_release'], next_release_kind=sap['next_release_kind'],
        limitation=sap['limitation'], evidence=refs))
    file = P / 'batch_financial_consumer/structural_review/DISPOSITIONS.json'
    dispositions = json.loads(file.read_text())['dispositions']
    kinds = {'valid_reporting_interval': 'long_reporting_interval',
             'future_next_quarter': 'missing_intervening_quarter',
             'vendor_hint_wrong_date': 'unmatched_vendor_hint'}
    for row in dispositions:
        if row['symbol'] == 'TM':
            # A real predecessor was collected. Never waive a missing event.
            continue
        refs = [evidence(file)]
        for item in row['evidence']:
            ref = evidence(item['evidence_path'], item['source_url'])
            if ref['sha256'] != item['sha256']:
                raise ValueError('Disposition source changed')
            if ref not in refs:
                refs.append(ref)
        for scope in row['scopes']:
            for day in row['signal_dates']:
                item = dict(issue_id='|'.join((kinds[row['kind']], scope, day, row['symbol'])),
                    resolution_kind=row['kind'], expected_prior_release=row['prior_release'],
                    reason=row['explanation'], evidence=refs)
                if row.get('next_release'):
                    item['next_release'] = row['next_release']
                if row.get('vendor_hint_date'):
                    item['expected_provider_dates'] = [row['vendor_hint_date']]
                result.append(item)
    (P / 'SCOPE_RESOLUTIONS.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Explicit resolutions: ' + str(len(result)))


if __name__ == '__main__':
    main()
