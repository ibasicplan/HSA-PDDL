from pathlib import Path
import unittest

from abstraction.air_builder import build_air, validate_air
from abstraction.grounding import build_gir, deterministic_grounding, resolve_object_types, validate_gir
from abstraction.relation_catalog import RelationCatalog
from evaluation.abstraction_metrics import evaluate_abstraction
from evaluation.pddl_metrics import evaluate_pddl
from generation.eir_coverage import (
    evaluate_eir_coverage,
    extract_fact_statements,
    extract_object_inventory,
    inspect_nl_coverage_contract,
)
from generation.eir_generator import build_eir_prompt, parse_eir, validate_eir
from utils.pddl import domain_uses_typing, parse_domain, parse_problem, render_problem, static_report, syntax_report


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "quantum" / "data" / "easy" / "01"
MOCK = ROOT / "fixtures" / "quantum" / "mock" / "easy" / "01"
CATALOG_PATH = PROJECT / "catalog" / "quantum_relation_catalog.json"


class QuantumTests(unittest.TestCase):
    def setUp(self):
        self.domain_pddl = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        self.domain_nl = (FIXTURE / "domain.nl").read_text(encoding="utf-8")
        self.problem_nl = (FIXTURE / "problem.nl").read_text(encoding="utf-8")
        self.reference = (FIXTURE / "problem.pddl").read_text(encoding="utf-8")
        self.domain = parse_domain(self.domain_pddl)
        self.catalog = RelationCatalog(CATALOG_PATH, self.domain)
        self.eir, status = parse_eir((MOCK / "eir.txt").read_text(encoding="utf-8"))
        self.assertTrue(status.startswith("ok"))
        self.assertIsNotNone(self.eir)

    def _pipeline(self):
        self.assertEqual(validate_eir(self.eir, self.domain), [])
        air = build_air(self.eir, self.domain, self.catalog)
        self.assertEqual(validate_air(air), [])
        mapping, unresolved = deterministic_grounding(air)
        self.assertEqual(unresolved, [])
        gir = build_gir(air, mapping, self.catalog)
        self.assertEqual(validate_gir(gir, self.domain), [])
        return air, gir

    def test_quantum_typed_inventory_and_constant_contract(self):
        inventory = extract_object_inventory(self.problem_nl)
        self.assertEqual(inventory["source_kind"], "typed_counted_inventory_groups")
        self.assertEqual(inventory["declared_count"], 24)
        self.assertEqual(len(inventory["objects"]), 24)
        self.assertEqual(inventory["object_types"]["d0"], "depth")
        self.assertEqual(inventory["object_types"]["l0"], "lqubit")
        self.assertEqual(inventory["object_types"]["p13"], "pqubit")

        statements = extract_fact_statements(self.problem_nl, inventory["objects"])
        self.assertEqual(statements["init_count"], 49)
        self.assertEqual(statements["goal_count"], 9)
        self.assertTrue(statements["goal"][3].startswith("it is not the case that"))

        contract = inspect_nl_coverage_contract(self.problem_nl, self.domain, self.catalog)
        self.assertTrue(contract["ok"], contract)
        self.assertEqual(contract["expected_entities_including_constants"], 24)
        self.assertEqual(contract["expected_objects"], 23)
        self.assertEqual(contract["domain_constants_in_nl"], ["d0"])
        self.assertEqual(contract["expected_init_fact_statements"], 49)
        self.assertEqual(contract["expected_goal_fact_statements"], 9)
        self.assertFalse(contract["reference_pddl_used"])

    def test_quantum_catalog_covers_exact_schema_and_guidance(self):
        self.assertEqual(set(self.catalog.predicate_to_relation), {
            "occupied_pqubit", "initialized", "occupied_lqubit", "mapped",
            "connected", "rcnot", "current_depth", "next_depth",
        })
        self.assertIn("lexicalized-negation", self.catalog.eir_guidance())
        self.assertIn("d0", self.catalog.eir_guidance())
        direct = self.catalog.direct_pddl_guidance()
        self.assertIn("rcnot(L1,L2,D)", direct)
        self.assertIn("MUST NOT be redeclared", direct)

    def test_quantum_prompt_explains_typed_objects_constant_and_rcnot(self):
        prompt = build_eir_prompt(
            self.domain_pddl,
            self.domain_nl,
            self.problem_nl,
            None,
            self.catalog.eir_guidance(),
        )
        self.assertIn("CRITICAL TYPED-OBJECT RULES", prompt)
        self.assertIn("d0: fixed domain constant", prompt)
        self.assertIn("Do NOT emit a fixed domain constant as an OBJECT line", prompt)
        self.assertIn("lexicalized-negation", prompt)
        self.assertIn("49", str(extract_fact_statements(self.problem_nl, extract_object_inventory(self.problem_nl)["objects"])["init_count"]))

    def test_quantum_eir_air_gir_render_and_metrics(self):
        air, gir = self._pipeline()
        coverage = evaluate_eir_coverage(
            self.problem_nl, self.eir, air, self.domain, self.catalog
        )
        self.assertTrue(coverage["ok"], coverage)
        self.assertEqual(coverage["object_inventory"]["expected_count"], 23)
        self.assertEqual(coverage["object_inventory"]["domain_constants_in_nl"], ["d0"])
        self.assertEqual(coverage["fact_statement_counts"], {
            "expected_init": 49,
            "observed_init": 49,
            "expected_goal": 9,
            "observed_goal": 9,
        })

        resolved = resolve_object_types(gir, self.domain)
        constants = set(self.domain["constants"])
        renderable = [obj for obj in resolved if obj["name"] not in constants]
        generated = render_problem(
            "quantum-01-generated",
            self.domain["domain_name"],
            renderable,
            gir["facts"],
            typed_objects=domain_uses_typing(self.domain),
        )
        object_block = generated.split("(:objects", 1)[1].split("(:init", 1)[0]
        self.assertNotIn("d0", object_block)
        self.assertIn("(current_depth d0)", generated)
        self.assertIn("(not (rcnot l0 l1 d4))", generated)
        self.assertTrue(syntax_report(generated)["ok"])
        self.assertTrue(static_report(self.domain, generated)["ok"])

        metrics = evaluate_pddl(
            generated,
            self.reference,
            self.domain_pddl,
            {"status": "solution_found"},
            {"status": "valid"},
        )
        self.assertEqual({key: metrics[key] for key in ("SVR", "SCR", "PSR", "VPR")}, {
            "SVR": 1, "SCR": 1, "PSR": 1, "VPR": 1
        })
        abstraction = evaluate_abstraction(air, gir, self.reference, self.catalog)
        self.assertEqual(abstraction["RMA"], 1.0)
        self.assertEqual(abstraction["GER"], 0.0)
        self.assertEqual(abstraction["alignable_facts"], 58)
        self.assertEqual(abstraction["rma_eligible"], 58)
        self.assertEqual(abstraction["ger_eligible"], 58)

    def test_quantum_wrong_object_type_and_constant_redeclaration_fail_coverage(self):
        wrong = {
            **self.eir,
            "objects": [dict(obj) for obj in self.eir["objects"]],
        }
        next(obj for obj in wrong["objects"] if obj["name"] == "l0")["type"] = "pqubit"
        air = build_air(wrong, self.domain, self.catalog)
        report = evaluate_eir_coverage(self.problem_nl, wrong, air, self.domain, self.catalog)
        self.assertFalse(report["ok"])
        self.assertIn("wrong_eir_object_type", {e["error_type"] for e in report["errors"]})

        with_constant = {
            **self.eir,
            "objects": [*self.eir["objects"], {"name": "d0", "type": "depth", "evidence": "NL inventory"}],
        }
        air = build_air(with_constant, self.domain, self.catalog)
        report = evaluate_eir_coverage(self.problem_nl, with_constant, air, self.domain, self.catalog)
        self.assertFalse(report["ok"])
        self.assertIn("extra_eir_object", {e["error_type"] for e in report["errors"]})

    def test_quantum_reference_scope(self):
        problem = parse_problem(self.reference)
        self.assertEqual(problem["domain_name"], "quantum")
        self.assertEqual(len(problem["objects"]), 23)
        self.assertNotIn("d0", problem["objects"])
        self.assertEqual(self.domain["constants"], {"d0": "depth"})
        self.assertEqual(sum(a["stage"] == "INIT" for a in problem["atoms"]), 49)
        self.assertEqual(sum(a["stage"] == "GOAL" for a in problem["atoms"]), 9)


if __name__ == "__main__":
    unittest.main()
