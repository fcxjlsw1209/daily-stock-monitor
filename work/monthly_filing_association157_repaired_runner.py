"""Versioned zero-budget adapter repair; frozen round157 math stays unchanged.

The frozen149 correction already adds exactly .01 * H to the original pre-score.
Its final adapter call has zero hand budget, but that adapter unnecessarily
requires H in [-.5, .5]. Only this call uses a neutral dummy. Actual execution
requires both the original PLAN and a separately registered code-only amendment.
Importing this module performs no source, price, feature or return calculation.
"""
from pathlib import Path
import argparse
import numpy as np
import monthly_filing_association157_runner as original


NAMES, HANDS, DOSE, OUT = original.NAMES, original.HANDS, original.DOSE, original.OUT
# Keep the actual157 -> 156 -> 149 call path. The backend adapter is the sole
# numerical integration repair; neither hand construction nor score math changes.
corrected, choose, panels = original.corrected, original.choose, original.panels


class AdapterProxy:
    """Delegate everything except a validated continuous-H zero-budget call."""

    def __init__(self, adapter, selector):
        original.require(callable(selector), 'Original boundary selector required')
        self._adapter, self._selector = adapter, selector

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def score_and_choose(self, base, budget, factor, arrays, meta, n, *,
                         pre_score=None, effects=None, episode_policy=None,
                         boundary_selector=None):
        zero_budget = (isinstance(budget, (int, float, np.integer, np.floating))
                       and not isinstance(budget, (bool, np.bool_)) and budget == 0.)
        explicit_original_call = (base == 'incumbent66' and zero_budget
            and pre_score is not None and effects is not None
            and episode_policy is not None and boundary_selector is self._selector)
        if explicit_original_call:
            hand = np.asarray(factor)
            shape = np.asarray(arrays['baseline']).shape
            valid_prefix = (isinstance(n, (int, np.integer))
                and not isinstance(n, (bool, np.bool_)) and len(shape) == 2
                and 0 <= n <= shape[0])
            # Malformed/noncontinuous calls still reach every frozen guard with
            # their original factor. Inspect only the requested prefix.
            if (valid_prefix and hand.shape == shape and hand.dtype.kind == 'f'
                    and not np.isinf(hand[:n]).any()
                    and np.all(np.abs(hand[:n][np.isfinite(hand[:n])]) <= 1.)):
                factor = np.zeros_like(hand)
        return self._adapter.score_and_choose(base, budget, factor, arrays, meta, n,
            pre_score=pre_score, effects=effects, episode_policy=episode_policy,
            boundary_selector=boundary_selector)


class Backend(original.Backend):
    def __init__(self, root, runner):
        super().__init__(root, runner)
        self.adapter = AdapterProxy(self.adapter, self.selector)


class Runner(original.Runner):
    def __init__(self, root, plan_path, plan_sha, amendment_sha, backend_factory=Backend):
        self.root, self.plan_sha = Path(root).resolve(), plan_sha
        self.amendment_sha = amendment_sha
        self._verify_amendment()
        super().__init__(root, plan_path, plan_sha, backend_factory=backend_factory)

    def _verify_amendment(self):
        # Lazy import avoids a registration-helper/source-inventory import cycle.
        from monthly_filing_association157_repair import verify_amendment
        return verify_amendment(self.root, self.plan_sha, self.amendment_sha)

    def refresh(self):
        self._verify_amendment()
        return super().refresh()

    def write(self, name, value):
        self._verify_amendment()
        return super().write(name, value)

    def require_prepare_review(self):
        super().require_prepare_review()
        from monthly_filing_association157_repair import verify_repair_report
        return verify_repair_report(self.root,
            original.read(self.out/'INDEPENDENT_PREPARE.json'),
            self.plan_sha, self.amendment_sha)


def source_files(root):
    root = Path(root)
    paths = (Path(__file__), root/'work/test_monthly_filing_association157_repaired_runner.py')
    return dict(original.source_files(root),
                **{str(path.resolve()): original.sha(path) for path in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'select_first88', 'diagnose_later34'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--amendment-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha, args.amendment_sha), args.phase)()
