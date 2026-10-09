import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from config import FRAMEWORK_VERSION
from evaluation.summarize import load_reports
from utils.io import append_jsonl


def ledger_row(sample_key: str = "easy/1"):
    return {
        "framework_version": FRAMEWORK_VERSION,
        "experiment": "base_direct",
        "sample_key": sample_key,
        "split": sample_key.split("/", 1)[0],
        "metrics": {"SVR": 1, "SCR": 1, "PSR": 1, "VPR": 1},
        "abstraction_metrics": {
            "RMA": None,
            "GER": None,
            "rma_correct": 0,
            "rma_eligible": 0,
            "ger_wrong": 0,
            "ger_eligible": 0,
            "acs_decisions": [],
        },
        "pipeline_diagnostics": {},
        "generation_usage": {},
    }


class SummaryLedgerTests(unittest.TestCase):
    def test_master_ledger_recovers_missing_sample_tree(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_jsonl(root / "master_sample_reports_ledger.jsonl", ledger_row())
            (root / "base_direct").mkdir(parents=True)
            (root / "base_direct" / "run_completion.json").write_text(
                json.dumps({"selected_samples": 1}), encoding="utf-8"
            )
            reports = load_reports(root)
            self.assertEqual(len(reports["base_direct"]), 1)
            self.assertEqual(reports["base_direct"][0]["sample_key"], "easy/1")

    def test_completion_count_mismatch_is_fatal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_jsonl(root / "master_sample_reports_ledger.jsonl", ledger_row())
            (root / "base_direct").mkdir(parents=True)
            (root / "base_direct" / "run_completion.json").write_text(
                json.dumps({"selected_samples": 2}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Incomplete summary input"):
                load_reports(root)

    def test_missing_completion_marker_is_fatal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_jsonl(root / "master_sample_reports_ledger.jsonl", ledger_row())
            with self.assertRaisesRegex(RuntimeError, "run_completion.json.*is missing"):
                load_reports(root)


if __name__ == "__main__":
    unittest.main()
