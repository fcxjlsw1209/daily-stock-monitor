"""Synthetic parser/batch interface plus metadata-only fixed-scope check."""
import copy
import importlib
import unittest

import after151_source_batch as batch


class ParserInterfaceTests(unittest.TestCase):
    def fixtures(self):
        for symbol, name in batch.MODULES.items():
            module = importlib.import_module(name)
            raw, ctx = importlib.import_module("test_"+name).fixture()
            yield symbol, module, raw, ctx

    def test_transport_of_both_parsers(self):
        for symbol, module, raw, ctx in self.fixtures():
            with self.subTest(symbol=symbol):
                out = module.extract_document(raw, ctx)
                batch.validate_record(out, ctx)
                self.assertTrue(all(v is not None for v in out["fields"].values()))
                self.assertEqual(out["source"], ctx["source"])
                self.assertEqual(out["context_holds"], ctx["context_holds"])

    def test_serialized_output_stays_valid(self):
        for symbol, module, raw, ctx in self.fixtures():
            with self.subTest(symbol=symbol):
                out = batch.shared.json.loads(batch.shared.json.dumps(module.extract_document(raw, ctx)))
                batch.validate_record(out, ctx)

    def test_immutable_source_and_holds_not_aliased(self):
        for symbol, module, raw, ctx in self.fixtures():
            with self.subTest(symbol=symbol):
                before = copy.deepcopy(ctx)
                out = module.extract_document(raw, ctx)
                out["source"]["sha256"] = "0"*64
                out["context_holds"].append("bad")
                self.assertEqual(ctx, before)
                with self.assertRaises(ValueError):
                    batch.validate_record(out, ctx)

    def test_raw_changed_source_rejected(self):
        for symbol, module, raw, ctx in self.fixtures():
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                module.extract_document(raw+b"tampered", ctx)

    def test_exact_scope_metadata_and_policy(self):
        ctx = batch.shared.load(batch.CONTEXT)
        policy = batch.shared.load(batch.POLICY)
        self.assertEqual(batch.shared.sha(batch.CONTEXT), batch.CONTEXT_SHA)
        batch.validate_policy(policy, ctx, batch.CONTEXT_SHA)
        self.assertEqual(len(batch.records(ctx)), 10)
        self.assertEqual(len(policy["fixed_catalog_metadata"]), 32)
        policy["fixed_catalog_metadata"][0]["publication_date"] = "2099-01-01"
        with self.assertRaises(ValueError):
            batch.validate_policy(policy, ctx, batch.CONTEXT_SHA)


if __name__ == "__main__":
    unittest.main()
