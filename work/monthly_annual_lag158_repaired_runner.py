"""Explicit-amendment runner for the bounded missing-amendment-field repair.

Only prepare temporarily substitutes one pure cell function. Frozen model build,
source validation, score construction, selection and diagnostics stay unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse

import monthly_annual_lag158_runner as original
import annual_lag158_cell_repair as cell

ORIGINAL_CELL = original.model._cell
NAMES, OUT = original.NAMES, original.OUT


@contextmanager
def repaired_cells():
    original.require(original.model._cell is ORIGINAL_CELL,
                     'Cell repair requires the unmodified frozen function; no nested patch')
    original.model._cell = cell.repaired_cell
    try:
        yield
    finally:
        active = original.model._cell
        original.model._cell = ORIGINAL_CELL
        original.require(active is cell.repaired_cell, 'Unexpected cell replacement during prepare')


class Runner(original.Runner):
    def __init__(self, root, plan_path, plan_sha, amendment_sha, backend_factory=original.Backend):
        self.root, self.plan_sha = Path(root).resolve(), plan_sha
        self.amendment_sha = amendment_sha
        self._verify_amendment()
        super().__init__(root, plan_path, plan_sha, backend_factory=backend_factory)

    def _verify_amendment(self):
        from annual_lag158_repair import verify_amendment
        return verify_amendment(self.root, self.plan_sha, self.amendment_sha)

    def refresh(self):
        self._verify_amendment()
        return super().refresh()

    def write(self, name, value):
        # Frozen write calls self.refresh(), so both amendment and original pins
        # are checked once without a second full closure traversal.
        return super().write(name, value)

    def prepare(self):
        self._verify_amendment()
        with repaired_cells():
            return super().prepare()

    def require_prepare_review(self):
        super().require_prepare_review()
        from annual_lag158_repair import verify_repair_report
        return verify_repair_report(self.root,
            original.read(self.out/'INDEPENDENT_PREPARE.json'), self.plan_sha, self.amendment_sha)


def source_files(root):
    root = Path(root)
    paths = (Path(__file__), Path(cell.__file__), root/'work/test_annual_lag158_cell_repair.py',
             root/'work/test_monthly_annual_lag158_repaired_runner.py')
    return dict(original.source_files(root),
                **{str(path.resolve()): original.sha(path) for path in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'select_first88', 'diagnose_later34'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--amendment-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha, args.amendment_sha), args.phase)()


if __name__ == '__main__':
    main()
