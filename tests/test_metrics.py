from pathlib import Path
import unittest

from abstraction.air_builder import build_air
from abstraction.grounding import build_gir, deterministic_grounding
from abstraction.relation_catalog import RelationCatalog
from evaluation.abstraction_metrics import consistency_score, evaluate_abstraction
from evaluation.pddl_metrics import evaluate_pddl, semantic_completeness
from generation.eir_generator import parse_eir
from utils.pddl import parse_domain, render_problem


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "data" / "easy" / "1"
MOCK = ROOT / "fixtures" / "mock" / "easy" / "1"


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.domain_text = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        self.reference = (FIXTURE / "problem.pddl").read_text(encoding="utf-8")
        self.generated = (MOCK / "direct_pddl.txt").read_text(encoding="utf-8")

    def test_primary_pddl_metrics(self):
        fd = {"status": "solution_found"}
        val = {"status": "valid"}
        report = evaluate_pddl(self.generated, self.reference, self.domain_text, fd, val)
        self.assertEqual({k: report[k] for k in ("SVR", "SCR", "PSR", "VPR")}, {
            "SVR": 1, "SCR": 1, "PSR": 1, "VPR": 1
        })

    def test_semantic_omission_fails_scr(self):
        broken = self.generated.replace("    (next fl0 fl1)\n", "")
        report = semantic_completeness(broken, self.reference, self.domain_text)
        self.assertEqual(report["SCR"], 0)
        self.assertIn("missing_predicate", report["error_counts"])
        self.assertFalse(report["checks"]["missing_action"]["applicable"])

    def test_abstraction_metrics(self):
        domain = parse_domain(self.domain_text)
        catalog = RelationCatalog(PROJECT / "catalog" / "zenotravel_relation_catalog.json", domain)
        eir, _ = parse_eir((MOCK / "eir.txt").read_text(encoding="utf-8"))
        air = build_air(eir, domain, catalog)
        mapping, _ = deterministic_grounding(air)
        gir = build_gir(air, mapping, catalog)
        metrics = evaluate_abstraction(air, gir, self.reference, catalog)
        self.assertEqual(metrics["RMA"], 1.0)
        self.assertEqual(metrics["GER"], 0.0)
        self.assertEqual(metrics["alignable_facts"], len(air["facts"]))
        ordering = next(x for x in metrics["fact_reports"] if x["fact_id"] == "F4")
        self.assertEqual(ordering["grounded_arguments"], ["fl0", "fl1"])
        self.assertTrue(ordering["argument_order_correct"])
        acs = consistency_score(metrics["acs_decisions"])
        self.assertEqual(acs["ACS"], 1.0)

        generated = render_problem("role-order-test", domain["domain_name"], gir["objects"], gir["facts"])
        report = semantic_completeness(generated, self.reference, self.domain_text)
        self.assertEqual(report["SCR"], 1)

    def test_role_order_error_is_counted_not_dropped(self):
        domain = parse_domain(self.domain_text)
        catalog = RelationCatalog(PROJECT / "catalog" / "zenotravel_relation_catalog.json", domain)
        eir, _ = parse_eir((MOCK / "eir.txt").read_text(encoding="utf-8"))
        air = build_air(eir, domain, catalog)
        mapping, _ = deterministic_grounding(air)
        wrongly_ordered_gir = build_gir(air, mapping)
        metrics = evaluate_abstraction(air, wrongly_ordered_gir, self.reference, catalog)
        self.assertEqual(metrics["RMA"], 1.0)
        self.assertEqual(metrics["GER"], 1 / len(air["facts"]))
        self.assertEqual(metrics["alignable_facts"], len(air["facts"]))
        ordering = next(x for x in metrics["fact_reports"] if x["fact_id"] == "F4")
        self.assertEqual(ordering["alignment_mode"], "catalog_expected_arguments")
        self.assertFalse(ordering["argument_order_correct"])


if __name__ == "__main__":
    unittest.main()
