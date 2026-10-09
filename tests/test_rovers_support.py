from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from abstraction.relation_catalog import RelationCatalog
from main import dataset_partitions, load_dataset
from scripts.build_rovers_catalog import build_catalog, build_relations
from utils.io import atomic_write_json
from utils.pddl import parse_domain, static_report


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "fixtures" / "rovers" / "data"
FULL_SCHEMA = ROOT / "fixtures" / "rovers" / "full_schema"


class RoversSupportTests(unittest.TestCase):
    def test_flat_dataset_is_explicitly_loaded_as_all(self):
        partitions = dataset_partitions(DATA, "all")
        self.assertEqual(partitions, [("all", DATA)])
        rows = load_dataset(DATA, "all")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_key"], "all/1")
        with self.assertRaises(FileNotFoundError):
            load_dataset(DATA, "easy")

    def test_catalog_is_built_from_actual_schema_without_gold_problem(self):
        payload, diagnostics = build_catalog(DATA, "auto")
        self.assertEqual(payload["domain"], "rover")
        self.assertFalse(payload["provenance"]["reference_problem_pddl_used"])
        self.assertFalse(payload["provenance"]["true_plan_used"])
        self.assertTrue(diagnostics["coverage_enabled"])
        self.assertEqual(diagnostics["predicate_count"], 9)
        self.assertEqual(
            {item["name"] for item in payload["relations"]["physical_location"]["predicates"]},
            {"at", "at_lander"},
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "rovers_catalog.json"
            atomic_write_json(path, payload)
            domain_text = (DATA / "1" / "domain.pddl").read_text(encoding="utf-8")
            domain = parse_domain(domain_text)
            catalog = RelationCatalog(path, domain)
            self.assertEqual(set(catalog.predicate_to_relation), set(domain["predicates"]))
            self.assertEqual(catalog.map_predicate("store_of"), "store_assignment")

    def test_reference_problem_is_schema_valid(self):
        domain_text = (DATA / "1" / "domain.pddl").read_text(encoding="utf-8")
        problem_text = (DATA / "1" / "problem.pddl").read_text(encoding="utf-8")
        self.assertTrue(static_report(parse_domain(domain_text), problem_text)["ok"])

    def test_actual_25_predicate_rovers_schema_has_exact_semantics(self):
        domain_text = (FULL_SCHEMA / "domain.pddl").read_text(encoding="utf-8")
        problem_text = (FULL_SCHEMA / "problem.pddl").read_text(encoding="utf-8")
        domain = parse_domain(domain_text)
        self.assertEqual(len(domain["predicates"]), 25)
        self.assertTrue(static_report(domain, problem_text)["ok"])

        relations = build_relations(domain)
        predicate_specs = {
            item["name"]: item
            for relation in relations.values()
            for item in relation["predicates"]
        }
        self.assertEqual(set(predicate_specs), set(domain["predicates"]))
        self.assertEqual(predicate_specs["supports"]["argument_roles"], ["camera", "mode"])
        self.assertEqual(
            predicate_specs["calibration_target"]["argument_roles"],
            ["camera", "objective"],
        )
        self.assertEqual(predicate_specs["on_board"]["argument_roles"], ["camera", "rover"])
        self.assertIn("imaging mode", relations["camera_support"]["description"])
        self.assertNotIn("objective", relations["camera_support"]["description"])
        self.assertEqual(
            {item["name"] for item in relations["calibration_target"]["predicates"]},
            {"calibration_target"},
        )
        self.assertEqual(
            {item["name"] for item in relations["camera_mounting"]["predicates"]},
            {"on_board"},
        )


if __name__ == "__main__":
    unittest.main()
