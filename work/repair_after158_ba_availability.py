"""Preserve a source-only failure and generate a narrowly repaired v2 helper."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = 'work/after158_ba_availability.py'
OLD_SHA = '2fc52731112f7b77cf1e133c5ce7ae1a2b234a7d120d77601217f7b15c3ddd2a'
PLAN = 'outputs/after158_ba_availability_v1/PLAN.json'
PLAN_SHA = '6414eb8ea351bff4a08ed9e242b1322418287b5b0b956edffcda249546f3337d'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(ROOT/OLD) == OLD_SHA and digest(ROOT/PLAN) == PLAN_SHA
    assert not (ROOT/'outputs/after158_ba_availability_v1/FEASIBILITY.json').exists()
    failure = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='failed_before_event_or_row_mapping', exception="KeyError: 'market_close'",
        command='./run_local.sh work/after158_ba_availability.py run --plan-sha '+PLAN_SHA,
        stage="schedule['market_close']", source_helper_sha256=OLD_SHA, plan_sha256=PLAN_SHA,
        cause="Pinned exchange_calendars uses schedule['close']; source session axis check passed before access.",
        source_admission_changed=False, financial_factors_computed=False, new_strategy_trials=0)
    fp=ROOT/'outputs/after158_ba_availability_v1/FAILURE.json'
    with fp.open('x') as f:
        json.dump(failure,f,indent=2,sort_keys=True);f.write('\n')
    text=(ROOT/OLD).read_text()
    replacements = {
        "SELF = 'work/after158_ba_availability.py'": "SELF = 'work/after158_ba_availability_v2.py'",
        "OUT = 'outputs/after158_ba_availability_v1'": "OUT = 'outputs/after158_ba_availability_v2'",
        "schedule['market_close']": "schedule['close']",
        "after158_ba_two_clock_source_feasibility_v1": "after158_ba_two_clock_source_feasibility_v2",
        "after158_ba_two_clock_source_plan_v1": "after158_ba_two_clock_source_plan_v2",
    }
    for old,new in replacements.items():
        assert text.count(old)==1, old
        text=text.replace(old,new)
    provenance={OLD:OLD_SHA, PLAN:PLAN_SHA,
        str(fp.relative_to(ROOT)):digest(fp),
        'work/repair_after158_ba_availability.py':digest(Path(__file__))}
    assert text.count('ACCESSIONS = {')==1
    text=text.replace('ACCESSIONS = {','PINS.update('+repr(provenance)+')\nACCESSIONS = {')
    integration='''    def test_real_calendar_close_interface_and_full_input_closure(self):
        local, external = collect_inputs()
        self.assertGreater(len(local), 9)
        spec = importlib.util.spec_from_file_location('ba_test_calendar', ROOT/CAL_CODE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schedule = module.market_calendar('2018-02-01', '2018-02-16').schedule
        self.assertIn('close', schedule)
        close = schedule.loc['2018-02-14', 'close']
        self.assertEqual(close.isoformat(), '2018-02-14T21:00:00+00:00')

'''
    assert text.count('    def test_strict_second_calendar_session(self):')==1
    text=text.replace('    def test_strict_second_calendar_session(self):',
        integration+'    def test_strict_second_calendar_session(self):')
    destination=ROOT/'work/after158_ba_availability_v2.py'
    with destination.open('x') as f:f.write(text)
    print(json.dumps(dict(original_preserved=digest(ROOT/OLD)==OLD_SHA,
        failure=str(fp.relative_to(ROOT)), new_helper=str(destination.relative_to(ROOT)),
        new_helper_sha256=digest(destination), contract_and_source_selection_unchanged=True)))


if __name__=='__main__':main()
