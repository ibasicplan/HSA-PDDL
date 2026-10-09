from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from abstraction.air_builder import build_air
from abstraction.relation_catalog import RelationCatalog
from generation.eir_coverage import (
    evaluate_eir_coverage,
    extract_explicit_classifications,
    extract_fact_statements,
    extract_object_inventory,
    inspect_nl_coverage_contract,
)
from generation.eir_generator import build_eir_prompt, parse_eir
from main import run_structured_sample
from planner.fd_runner import FastDownwardRunner
from planner.val_runner import VALRunner
from utils.pddl import parse_domain, parse_problem


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "logistics" / "data" / "easy" / "1"
MOCK_EIR = ROOT / "fixtures" / "logistics" / "mock" / "easy" / "1" / "eir.txt"
CATALOG_PATH = PROJECT / "catalog" / "logistics_relation_catalog.json"


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, stage, system_prompt, user_prompt, max_new_tokens, sample_key, attempt):
        self.prompts.append({
            "stage": stage,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "attempt": attempt,
        })
        if stage != "eir":
            raise AssertionError(f"Unexpected disambiguation call: {stage}")
        raw = self.responses[attempt - 1]
        return {
            "raw_output": raw,
            "clean_output": raw,
            "stats": {
                "backend": "mock",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "request_attempt": 1,
                "cache_hit": False,
            },
        }


class EIRCoverageTests(unittest.TestCase):
    def setUp(self):
        self.domain_pddl = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        self.domain_nl = (FIXTURE / "domain.nl").read_text(encoding="utf-8")
        self.problem_nl = (FIXTURE / "problem.nl").read_text(encoding="utf-8")
        self.reference = (FIXTURE / "problem.pddl").read_text(encoding="utf-8")
        self.domain = parse_domain(self.domain_pddl)
        self.catalog = RelationCatalog(CATALOG_PATH, self.domain)
        self.full_raw = MOCK_EIR.read_text(encoding="utf-8")
        self.full_eir, _ = parse_eir(self.full_raw)

    def _partial_without_classes(self):
        partial = {
            **self.full_eir,
            "facts": [fact for fact in self.full_eir["facts"] if len(fact["arguments"]) == 2],
        }
        lines = [
            f"OBJECT|{obj['name']}|{obj['type']}|{obj['evidence']}"
            for obj in partial["objects"]
        ]
        for fact in partial["facts"]:
            polarity = "POS" if fact["polarity"] else "NEG"
            lines.append(
                f"FACT|{fact['fact_id']}|{fact['stage']}|{polarity}|{fact['relation']}|"
                f"{','.join(fact['arguments'])}|{fact['evidence']}"
            )
        lines.append("END_EIR")
        return partial, "\n".join(lines) + "\n"

    def test_input_only_coverage_extracts_logistics_contract(self):
        inventory = extract_object_inventory(self.problem_nl)
        self.assertEqual(inventory["declared_count"], 13)
        self.assertEqual(len(inventory["objects"]), 13)
        statements = extract_fact_statements(self.problem_nl, inventory["objects"])
        self.assertEqual(statements["init_count"], 26)
        self.assertEqual(statements["goal_count"], 1)
        classes = extract_explicit_classifications(
            statements["init"], inventory["objects"], self.domain, self.catalog
        )
        self.assertEqual(len(classes), 16)
        l0_classes = {
            row["predicate"] for row in classes if row["arguments"] == ["l0-0"]
        }
        self.assertEqual(l0_classes, {"location", "airport"})

        contract = inspect_nl_coverage_contract(
            self.problem_nl, self.domain, self.catalog
        )
        self.assertTrue(contract["ok"])
        self.assertEqual(contract["expected_objects"], 13)
        self.assertEqual(contract["expected_init_fact_statements"], 26)
        self.assertEqual(contract["expected_goal_fact_statements"], 1)
        self.assertEqual(contract["expected_explicit_classification_facts"], 16)

    def test_unknown_nl_template_fails_closed_before_generation(self):
        contract = inspect_nl_coverage_contract(
            "Cargo p0 starts somewhere and should move.", self.domain, self.catalog
        )
        self.assertFalse(contract["ok"])
        error_types = {row["error_type"] for row in contract["errors"]}
        self.assertIn("nl_object_inventory_unavailable", error_types)
        self.assertIn("nl_init_assertions_unavailable", error_types)
        self.assertIn("nl_goal_assertions_unavailable", error_types)

    def test_complete_eir_passes_and_omission_fails_before_planning(self):
        full_air = build_air(self.full_eir, self.domain, self.catalog)
        complete = evaluate_eir_coverage(
            self.problem_nl, self.full_eir, full_air, self.domain, self.catalog
        )
        self.assertTrue(complete["ok"])
        self.assertEqual(complete["classification_facts"], {
            "expected": 16, "missing": 0, "coverage": 1.0
        })

        partial, _ = self._partial_without_classes()
        partial_air = build_air(partial, self.domain, self.catalog)
        broken = evaluate_eir_coverage(
            self.problem_nl, partial, partial_air, self.domain, self.catalog
        )
        self.assertFalse(broken["ok"])
        self.assertEqual(broken["fact_statement_counts"]["expected_init"], 26)
        self.assertEqual(broken["fact_statement_counts"]["observed_init"], 10)
        self.assertEqual(broken["classification_facts"]["missing"], 16)
        self.assertEqual(
            sum(error["error_type"] == "missing_explicit_classification_fact" for error in broken["errors"]),
            16,
        )

    def test_untyped_prompt_explicitly_requires_classification_facts(self):
        prompt = build_eir_prompt(
            self.domain_pddl, self.domain_nl, self.problem_nl, None
        )
        self.assertIn("CRITICAL UNTYPED-STRIPS RULES", prompt)
        self.assertIn("13 objects", prompt)
        self.assertIn("26 INIT assertions", prompt)
        self.assertIn("1 GOAL assertions", prompt)
        self.assertIn("metadata only", prompt)
        self.assertIn("multiple category FACTs", prompt)

    def test_first_omission_is_repaired_on_second_eir_attempt(self):
        _, partial_raw = self._partial_without_classes()
        llm = SequenceLLM([partial_raw, self.full_raw])
        args = SimpleNamespace(
            eir_max_new_tokens=4096,
            catalog_score_threshold=3.0,
            catalog_score_margin=0.5,
            relation_max_new_tokens=512,
            grounding_max_new_tokens=512,
            max_attempts=3,
        )
        item = {
            "domain_pddl": self.domain_pddl,
            "domain_nl": self.domain_nl,
            "problem_nl": self.problem_nl,
            "reference_pddl": self.reference,
            "sample_key": "easy/1",
            "split": "easy",
            "sample_id": "1",
        }
        with TemporaryDirectory() as tmp:
            sample_dir = Path(tmp)
            domain_path = sample_dir / "domain.pddl"
            domain_path.write_text(self.domain_pddl, encoding="utf-8")
            result = run_structured_sample(
                item,
                sample_dir,
                llm,
                self.domain,
                self.catalog,
                domain_path,
                FastDownwardRunner("/nonexistent/mock-fd", "astar(lmcut())", 10, backend="mock"),
                VALRunner("/nonexistent/mock-val", 10, backend="mock"),
                args,
            )
            self.assertEqual(result["attempt"], 2)
            self.assertEqual(result["status"], "val_valid")
            self.assertEqual(result["pipeline"]["eir_coverage_ok"], 1)
            self.assertFalse((sample_dir / "attempt_1" / "fd.log").exists())
            self.assertTrue((sample_dir / "attempt_1" / "eir_coverage.json").is_file())
            self.assertIn("eir_init_fact_count_mismatch", llm.prompts[1]["user_prompt"])
            self.assertIn("missing_explicit_classification_fact", llm.prompts[1]["user_prompt"])
            parsed = parse_problem(result["pddl"])
            predicates = [atom["predicate"] for atom in parsed["atoms"]]
            self.assertIn("airplane", predicates)
            self.assertIn("airport", predicates)
            self.assertIn("truck", predicates)
            self.assertEqual(len(parsed["atoms"]), 27)


if __name__ == "__main__":
    unittest.main()
