"""Repeatable offline validation and scoped merge status."""
from collect import *
import subprocess

def main():
    ep=HERE/'events.json';es=json.loads(ep.read_text());assert len(es)==1
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (HERE/'TESTS.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('SAP tests failed; see TESTS.txt')
    e=es[0];n=normalized(e);assert n['status']=='ready'
    meta_file=ROOT/'outputs/monthly_architecture_restart/META.json';meta=json.loads(meta_file.read_text())
    inputs={rel(p):sha(p.read_bytes()) for p in [ep,meta_file,PDF,PUBLICATION]}
    dump(HERE/'CANDIDATE_AUDIT.json',dict(inputs=inputs,scope='SAP training2012-12-31 predecessor only',signal_date='2012-12-31',fiscal_year=2012,fiscal_quarter=3,release_date=e['release_date'],available_session=n['available_session'],known_by_at=n['known_by_at'],age_calendar_days=68,quarter_sequence_boundary='First required actual SAP training candidate. Later SAP events remain in separately frozen health source; no unrelated history required.',no_first_public_instant_claim=True))
    dump(HERE/'STATUS.json',dict(status='verified_ready_to_merge',generated_at=now(),new_events=1,events_sha256=sha(ep.read_bytes()),release_date=e['release_date'],fiscal_period_end_actual=e['fiscal_period_end_actual'],fiscal_year=2012,fiscal_quarter=3,scope='Only training candidate2012-12-31 predecessor',source_class=e['source_class'],official_host_bytes=False,issuer_authored_report_mirror_full_bytes=True,independent_actual_release_evidence_class=e['fiscal_identity_evidence']['source_class'],publication_timestamp=None,precision='date_only',publication_timezone='Europe/Berlin',tests_passed=11,missing_assigned_quarters=[],limitations=e['historical_limitation'],account_or_training_changes=False))
    dump(HERE/'VALIDATION.json',dict(status='passed',validated_at=now(),inputs=inputs,tests=11,network_used=False,checks=['55-page issuer-authored PDF byte and extraction SHA; third-party mirror explicitly labeled','SAP-issued actual PresseBox completed-results dateline October24 independently corroborates report imprint','three-month consolidated current2012 column ends September30; prior2011 comparative excluded','report imprint alone and future announcement rejected by negative tests','date-only Berlin availability October25; same-day October24 excluded','actual training2012-12-31 candidate exists; scoped predecessor remains separate from frozen source','failed requests and current LA-day host stops retained'],event_imports_financial_values=False))
    print('SAP ready',sha(ep.read_bytes()),'11 tests passed')

if __name__=='__main__':main()
