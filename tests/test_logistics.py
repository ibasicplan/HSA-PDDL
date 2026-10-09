from pathlib import Path
import unittest

from abstraction.air_builder import build_air, validate_air
from abstraction.grounding import build_gir, deterministic_grounding, resolve_object_types, validate_gir
from abstraction.relation_catalog import RelationCatalog
from evaluation.abstraction_metrics import evaluate_abstraction
from evaluation.pddl_metrics import evaluate_pddl
from generation.eir_generator import parse_eir, validate_eir
from utils.pddl import domain_uses_typing, parse_domain, parse_problem, render_problem, static_report, syntax_report


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "logistics" / "data" / "easy" / "1"
MOCK = ROOT / "fixtures" / "logistics" / "mock" / "easy" / "1"
CATALOG = PROJECT / "catalog" / "logistics_relation_catalog.json"


class LogisticsTests(unittest.TestCase):
    def _pipeline(self):
        domain_pddl = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        domain = parse_domain(domain_pddl)
        catalog = RelationCatalog(CATALOG, domain)
        eir, status = parse_eir((MOCK / "eir.txt").read_text(encoding="utf-8"))
        self.assertTrue(status.startswith("ok"))
        self.assertIsNotNone(eir)
        self.assertEqual(validate_eir(eir), [])
        air = build_air(eir, domain, catalog)
        self.assertEqual(validate_air(air), [])
        mapping, unresolved = deterministic_grounding(air)
        self.assertEqual(unresolved, [])
        gir = build_gir(air, mapping, catalog)
        self.assertEqual(validate_gir(gir, domain), [])
        return domain_pddl, domain, catalog, eir, air, gir

    def test_untyped_logistics_catalog_air_and_grounding(self):
        _, domain, catalog, eir, air, gir = self._pipeline()
        self.assertEqual(domain["domain_name"], "logistics-strips")
        self.assertFalse(domain_uses_typing(domain))
        self.assertEqual(len(domain["predicates"]), 9)
        self.assertEqual(len(catalog.predicate_to_relation), 9)
        self.assertEqual(len(eir["objects"]), 13)
        self.assertEqual(len(eir["facts"]), 27)
        self.assertEqual(len(gir["facts"]), 27)

        by_id = {fact["fact_id"]: fact for fact in gir["facts"]}
        self.assertEqual((by_id["F2"]["predicate"], by_id["F2"]["arguments"]), ("at", ["a0", "l2-0"]))
        self.assertEqual((by_id["F8"]["predicate"], by_id["F8"]["arguments"]), ("in-city", ["l0-0", "c0"]))
        self.assertEqual((by_id["F15"]["predicate"], by_id["F15"]["arguments"]), ("obj", ["p0"]))
        self.assertEqual((by_id["F27"]["predicate"], by_id["F27"]["arguments"]), ("at", ["p0", "l2-0"]))

    def test_untyped_render_is_semantically_complete(self):
        domain_pddl, domain, catalog, _, air, gir = self._pipeline()
        objects = resolve_object_types(gir, domain)
        self.assertTrue(all(obj["type"] == "object" for obj in objects))
        generated = render_problem(
            problem_name="logistics-c3-s1-p1-a1-generated",
            domain_name=domain["domain_name"],
            objects=objects,
            facts=gir["facts"],
            typed_objects=domain_uses_typing(domain),
        )
        object_block = generated.split("(:objects", 1)[1].split("(:init", 1)[0]
        self.assertNotIn(" - ", object_block)
        self.assertTrue(syntax_report(generated)["ok"])
        self.assertTrue(static_report(domain, generated)["ok"])
        reference = (FIXTURE / "problem.pddl").read_text(encoding="utf-8")
        metrics = evaluate_pddl(
            generated,
            reference,
            domain_pddl,
            {"status": "solution_found"},
            {"status": "valid"},
        )
        self.assertEqual({key: metrics[key] for key in ("SVR", "SCR", "PSR", "VPR")}, {
            "SVR": 1,
            "SCR": 1,
            "PSR": 1,
            "VPR": 1,
        })
        abstraction = evaluate_abstraction(air, gir, reference, catalog)
        self.assertEqual(abstraction["RMA"], 1.0)
        self.assertEqual(abstraction["GER"], 0.0)
        self.assertEqual(abstraction["alignable_facts"], 27)
        self.assertEqual(abstraction["rma_eligible"], 27)
        self.assertEqual(abstraction["ger_eligible"], 27)

    def test_logistics_guidance_is_domain_scoped(self):
        domain = parse_domain((FIXTURE / "domain.pddl").read_text(encoding="utf-8"))
        catalog = RelationCatalog(CATALOG, domain)
        guidance = catalog.direct_pddl_guidance()
        self.assertIn("untyped STRIPS domain", guidance)
        self.assertIn("(in-city LOCATION CITY)", guidance)
        self.assertIn("(in CARGO VEHICLE)", guidance)

    def test_reference_problem_has_expected_untyped_shape(self):
        problem = parse_problem((FIXTURE / "problem.pddl").read_text(encoding="utf-8"))
        self.assertEqual(problem["domain_name"], "logistics-strips")
        self.assertEqual(len(problem["objects"]), 13)
        self.assertTrue(all(value == "object" for value in problem["objects"].values()))
        self.assertEqual(len(problem["atoms"]), 27)


if __name__ == "__main__":
    unittest.main()
