"""Pure149 specification fixtures; no source package or market reads."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import prepare_stock_comp149_proposal as p


def inputs():
    prior = dict(sources={key: dict(path="synthetic/"+key, sha256="1"*64)
                          for key in ("dataset", "admission", "policy")},
                 measurement={"old_measure": "must_be_replaced"})
    refs = {key: dict(path=path, sha256="a"*64) for key, path in p.r.SBC_REFS.items()}
    return prior, refs


class ProposalTests(unittest.TestCase):
    def test_fixed_contract_and_inputs_unchanged(self):
        prior, refs = inputs()
        saved = deepcopy((prior, refs))
        result = p.build_proposal(prior, refs, "2026-01-01T00:00:00+00:00")
        self.assertEqual((prior, refs), saved)
        self.assertEqual(result["round"], 149)
        self.assertEqual(result["measurement"], p.r.measurement_contract())
        self.assertEqual(result["selection_contract"], p.r.selection_contract())
        self.assertEqual(result["actual_new_factors"], 0)
        self.assertNotIn("old_measure", result["measurement"])

    def test_missing_wrong_or_placeholder_source_reference_fails(self):
        for operation in ("missing", "wrong_path", "short_hash", "extra"):
            prior, refs = inputs()
            if operation == "missing": refs.pop("sbc_review")
            elif operation == "wrong_path": refs["sbc_review"]["path"] = "other.json"
            elif operation == "short_hash": refs["sbc_review"]["sha256"] = "pending"
            else: refs["sbc_review"]["extra"] = True
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                p.build_proposal(prior, refs, "synthetic")

    def test_existing_local_regular_file_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/"file").write_bytes(b"synthetic")
            self.assertEqual(len(p.checked_ref(root, "file")["sha256"]), 64)
            (root/"link").symlink_to(root/"file")
            for name in ("missing", "link", "../escape"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    p.checked_ref(root, name)

    def test_wrong_original_source_package_fails(self):
        prior, refs = inputs()
        prior["sources"].pop("dataset")
        with self.assertRaises(ValueError):
            p.build_proposal(prior, refs, "synthetic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
