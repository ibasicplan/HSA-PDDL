from pathlib import Path
import unittest

from abstraction.air_builder import build_air, validate_air
from abstraction.grounding import build_gir, deterministic_grounding, resolve_object_types, validate_gir
from abstraction.relation_catalog import RelationCatalog
from evaluation.abstraction_metrics import evaluate_abstraction
from evaluation.pddl_metrics import evaluate_pddl
from generation.eir_coverage import evaluate_eir_coverage, extract_fact_statements, extract_object_inventory, inspect_nl_coverage_contract
from generation.eir_generator import build_eir_prompt, parse_eir, validate_eir
from utils.pddl import domain_uses_typing, parse_domain, parse_problem, render_problem, static_report, syntax_report


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "mystery" / "data" / "easy" / "10"
MOCK = ROOT / "fixtures" / "mystery" / "mock" / "easy" / "10"
CATALOG_PATH = PROJECT / "catalog" / "mystery_4ops_relation_catalog.json"


class MysteryTests(unittest.TestCase):
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
        self.assertEqual(validate_eir(self.eir), [])
        air = build_air(self.eir, self.domain, self.catalog)
        self.assertEqual(validate_air(air), [])
        mapping, unresolved = deterministic_grounding(air)
        self.assertEqual(unresolved, [])
        gir = build_gir(air, mapping, self.catalog)
        self.assertEqual(validate_gir(gir, self.domain), [])
        return air, gir

    def test_mystery_nl_contract_uses_only_inline_object_mentions(self):
        inventory = extract_object_inventory(self.problem_nl)
        self.assertEqual(inventory["source_kind"], "inline_object_mentions")
        self.assertEqual(inventory["objects"], ["a", "d", "b", "c"])
        self.assertEqual(inventory["declared_count"], 4)

        statements = extract_fact_statements(self.problem_nl, inventory["objects"])
        self.assertEqual(statements["init_template"], "as_initial_conditions")
        self.assertEqual(statements["init_count"], 7)
        self.assertEqual(statements["goal_count"], 1)

        contract = inspect_nl_coverage_contract(self.problem_nl, self.domain, self.catalog)
        self.assertTrue(contract["ok"], contract)
        self.assertEqual(contract["object_inventory_source"], "inline_object_mentions")
        self.assertEqual(contract["expected_objects"], 4)
        self.assertEqual(contract["expected_init_fact_statements"], 7)
        self.assertEqual(contract["expected_goal_fact_statements"], 1)
        self.assertEqual(contract["expected_explicit_classification_facts"], 0)
        self.assertFalse(contract["reference_pddl_used"])

    def test_two_assertions_joined_only_by_and_are_split(self):
        problem_nl = (
            "As initial conditions I have that, province object a and planet object b. "
            "My goal is to have that province object c and object a craves object b."
        )
        inventory = extract_object_inventory(problem_nl)
        statements = extract_fact_statements(problem_nl, inventory["objects"])
        self.assertEqual(inventory["objects"], ["a", "b", "c"])
        self.assertEqual(statements["init"], ["province object a", "planet object b"])
        self.assertEqual(
            statements["goal"],
            ["province object c", "object a craves object b"],
        )
        self.assertEqual(statements["init_count"], 2)
        self.assertEqual(statements["goal_count"], 2)

    def test_two_assertions_can_start_with_harmony_or_pain(self):
        problem_nl = (
            "As initial conditions I have that, harmony and pain object a. "
            "My goal is to have that harmony and planet object b."
        )
        inventory = extract_object_inventory(problem_nl)
        statements = extract_fact_statements(problem_nl, inventory["objects"])
        self.assertEqual(statements["init"], ["harmony", "pain object a"])
        self.assertEqual(statements["goal"], ["harmony", "planet object b"])
        self.assertEqual(statements["init_count"], 2)
        self.assertEqual(statements["goal_count"], 2)

    def test_zero_arity_harmony_survives_eir_air_gir_and_render(self):
        air, gir = self._pipeline()
        harmony = next(fact for fact in gir["facts"] if fact["predicate"] == "harmony")
        self.assertEqual(harmony["arguments"], [])
        self.assertEqual(harmony["source_arguments"], [])
        self.assertEqual(harmony["abstract_relation"], "global_harmony")

        coverage = evaluate_eir_coverage(
            self.problem_nl, self.eir, air, self.domain, self.catalog
        )
        self.assertTrue(coverage["ok"], coverage)
        self.assertEqual(coverage["fact_statement_counts"], {
            "expected_init": 7,
            "observed_init": 7,
            "expected_goal": 1,
            "observed_goal": 1,
        })

        generated = render_problem(
            "mystery-10-generated",
            self.domain["domain_name"],
            resolve_object_types(gir, self.domain),
            gir["facts"],
            typed_objects=domain_uses_typing(self.domain),
        )
        self.assertIn("(harmony)", generated)
        self.assertNotIn("(harmony ", generated)
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
        self.assertEqual(abstraction["alignable_facts"], 8)
        self.assertEqual(abstraction["rma_eligible"], 8)
        self.assertEqual(abstraction["ger_eligible"], 8)

    def test_omitting_harmony_fails_coverage_before_planning(self):
        partial = {
            **self.eir,
            "facts": [fact for fact in self.eir["facts"] if fact["relation"] != "global harmony holds"],
        }
        air = build_air(partial, self.domain, self.catalog)
        report = evaluate_eir_coverage(
            self.problem_nl, partial, air, self.domain, self.catalog
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["fact_statement_counts"]["expected_init"], 7)
        self.assertEqual(report["fact_statement_counts"]["observed_init"], 6)
        self.assertIn(
            "eir_init_fact_count_mismatch",
            {error["error_type"] for error in report["errors"]},
        )

    def test_prompt_and_catalog_are_mystery_scoped(self):
        prompt = build_eir_prompt(
            self.domain_pddl, self.domain_nl, self.problem_nl, None
        )
        self.assertIn("4 objects", prompt)
        self.assertIn("7 INIT assertions", prompt)
        self.assertIn("1 GOAL assertions", prompt)
        self.assertIn("ZERO-ARITY FACT RULE", prompt)
        self.assertIn("two adjacent separators", prompt)
        self.assertIn("harmony", prompt)

        guidance = self.catalog.direct_pddl_guidance()
        self.assertIn("harmony/0", guidance)
        self.assertIn("craves(X,Y)", guidance)
        self.assertIn("first argument", guidance)

    def test_reference_shape_and_known_four_step_plan(self):
        problem = parse_problem(self.reference)
        self.assertEqual(problem["domain_name"], "mystery-4ops")
        self.assertEqual(set(problem["objects"]), {"a", "b", "c", "d"})
        self.assertEqual(len(problem["atoms"]), 8)
        self.assertTrue(all(value == "object" for value in problem["objects"].values()))

        state = {
            (atom["predicate"], tuple(atom["arguments"]))
            for atom in problem["atoms"] if atom["stage"] == "INIT" and atom["polarity"]
        }
        self.assertTrue({("craves", ("d", "c")), ("province", ("d",)), ("harmony", ())} <= state)
        state -= {("craves", ("d", "c")), ("province", ("d",)), ("harmony", ())}
        state |= {("pain", ("d",)), ("province", ("c",))}
        self.assertIn(("pain", ("d",)), state)
        state -= {("pain", ("d",))}
        state |= {("province", ("d",)), ("planet", ("d",)), ("harmony", ())}
        self.assertTrue({("province", ("c",)), ("planet", ("c",)), ("harmony", ())} <= state)
        state -= {("province", ("c",)), ("planet", ("c",)), ("harmony", ())}
        state |= {("pain", ("c",))}
        self.assertTrue({("province", ("a",)), ("pain", ("c",))} <= state)
        state -= {("province", ("a",)), ("pain", ("c",))}
        state |= {("harmony", ()), ("province", ("c",)), ("craves", ("c", "a"))}
        self.assertIn(("craves", ("c", "a")), state)


if __name__ == "__main__":
    unittest.main()
