"""Close the registered twenty-document transcription, not a strategy trial."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
PINS = {
    'PLAN.json': '587fa9546ef99a84a947a8f51bb6e66b768454fdb9474faeefac9a2cf41687eb',
    'EXTRACTION.json': 'ce42228a4611565a9b031bb7a37415a8b0a4c54380f17dbb80e8bf61f0a4c9fd',
    'INDEPENDENT_REVIEW.json': 'fc103480d047ae5b8a577a86c7e733164ddd99e68418d1d0df8869d79c3fc2af',
    'EXTERNAL_AUDIT.json': '1d249d8164cf67c43f52f2ff759151850a8f48cea948ab4db019cf129c7cc3e1',
    'FOLLOWUP_REVIEW.json': 'b33d4af2bcc8907d629182482d36b76aa9126499fc895183a5a68bb344b07a7e',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    docs = {}
    for name, digest in PINS.items():
        assert sha(OUT / name) == digest, name
        docs[name] = json.loads((OUT / name).read_text())
    plan = docs['PLAN.json']
    pins = dict(plan['anchors'])
    for row in plan['records']:
        for path, digest in row['inputs'].items():
            assert path not in pins or pins[path] == digest
            pins[path] = digest
    for path, digest in pins.items():
        assert sha(ROOT / path) == digest, path
    extracted = docs['EXTRACTION.json']
    reviewed = docs['INDEPENDENT_REVIEW.json']
    assert len(extracted['records']) == len(reviewed['records']) == 20
    assert extracted['plan_sha256'] == reviewed['plan_sha256'] == PINS['PLAN.json']
    assert reviewed['producer_extraction_sha256'] == PINS['EXTRACTION.json']
    assert sha(ROOT / 'work/shell_pool_source168_extract.py') == extracted['extractor_sha256']
    assert sha(ROOT / 'work/shell_pool_source168_review.py') == reviewed['reviewer_source_sha256']
    assert reviewed['summary']['overall'] == 'PASS_SOURCE_TRANSCRIPTION_ONLY'
    assert reviewed['mismatches'] == []
    for key in ['documents_reviewed', 'count_pairs_matched', 'dates_matched', 'rollforwards_matched']:
        assert reviewed['summary'][key] == 20, key
    assert reviewed['external_audit_review']['audit_sha256'] == PINS['EXTERNAL_AUDIT.json']
    assert all(reviewed['external_audit_review']['checks'].values())
    followup = docs['FOLLOWUP_REVIEW.json']
    assert followup['receipt_and_body_checks_passed']
    assert followup['ratio']['A_ADS_ordinary_shares'] == followup['ratio']['B_ADS_ordinary_shares'] == 2
    result = dict(
        schema='shell_pool_source168_completion_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='SOURCE_TRANSCRIPTION_COMPLETE_CAPITALIZATION_NOT_ADMITTED',
        scope='Registered twenty original QRA PDFs, 2015Q4 through 2020Q3; separately reviewed saved public-source follow-up.',
        artifacts=PINS, input_files_checked=len(pins),
        documents=20, exact_issued_A_B_rows=20, stated_release_dates=20, rollforwards_verified=20,
        independent_all_pages_visually_checked=True,
        supplementary_ratio='Two ordinary A shares per RDS.A ADS and two ordinary B shares per RDS.B ADS explicitly stated in 2016 issuer material; full historical continuity not certified.',
        unresolved=[
            'Class-specific net outstanding basis, including employee trust and treasury treatment.',
            'Compatible unadjusted class-specific quote coverage for all early signals.',
            'Original vendor series class basis; two exact legacy source files remain missing.',
            'Publication timing: current historical documents do not prove contemporaneous bytes; never backdate future disclosures.'
        ],
        next_work='Separately registered total-outstanding extraction and issuer-embedded price source collection; any capitalization or pool comparison requires a new data rule before calculation.',
        strict_point_in_time_pool_certified=False, market_cap_computed=False,
        pool_or_frozen_strategy_modified=False, strategy_trials_added=0,
        strategy_round_counter_unchanged=True, research_goal_complete=False,
        closer_sha256=sha(Path(__file__))
    )
    with (OUT / 'COMPLETION.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
