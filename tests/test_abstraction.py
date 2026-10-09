from pathlib import Path
import unittest

from abstraction.air_builder import build_air, validate_air
from abstraction.grounding import build_gir, deterministic_grounding, validate_gir
from abstraction.relation_catalog import RelationCatalog, canonical_domain_name
from generation.eir_generator import parse_eir
from utils.pddl import parse_domain


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIXTURE = ROOT / "fixtures" / "data" / "easy" / "1"
MOCK = ROOT / "fixtures" / "mock" / "easy" / "1"


class AbstractionTests(unittest.TestCase):
    def test_domain_name_separator_normalization(self):
        self.assertEqual(canonical_domain_name("zenotravel"), canonical_domain_name("Zeno-Travel"))
        self.assertEqual(canonical_domain_name("zeno_travel"), canonical_domain_name("zeno-travel"))
        domain_text = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        domain = parse_domain(domain_text)
        RelationCatalog(PROJECT / "catalog" / "zenotravel_relation_catalog.json", domain)

    def test_catalog_air_grounding(self):
        domain = parse_domain((FIXTURE / "domain.pddl").read_text(encoding="utf-8"))
        catalog = RelationCatalog(PROJECT / "catalog" / "zenotravel_relation_catalog.json", domain)
        eir, status = parse_eir((MOCK / "eir.txt").read_text(encoding="utf-8"))
        self.assertTrue(status.startswith("ok"))
        air = build_air(eir, domain, catalog)
        self.assertEqual(validate_air(air), [])
        self.assertEqual([x["selected_relation"] for x in air["facts"]], [
            "location", "location", "resource", "ordering", "location"
        ])
        mapping, unresolved = deterministic_grounding(air)
        self.assertFalse(unresolved)
        gir = build_gir(air, mapping, catalog)
        self.assertEqual(validate_gir(gir, domain), [])
        ordering = next(x for x in gir["facts"] if x["fact_id"] == "F4")
        self.assertEqual(ordering["source_arguments"], ["fl1", "fl0"])
        self.assertEqual(ordering["arguments"], ["fl0", "fl1"])

    def test_catalog_argument_permutation_and_direct_guidance(self):
        domain = parse_domain((FIXTURE / "domain.pddl").read_text(encoding="utf-8"))
        catalog = RelationCatalog(PROJECT / "catalog" / "zenotravel_relation_catalog.json", domain)
        self.assertEqual(catalog.ground_arguments("next", ["fl1", "fl0"]), ["fl0", "fl1"])
        self.assertEqual(catalog.ground_arguments("at", ["plane1", "city0"]), ["plane1", "city0"])
        guidance = catalog.direct_pddl_guidance()
        self.assertIn("emit (next B A)", guidance)
        self.assertIn("lower_or_predecessor_level", guidance)


if __name__ == "__main__":
    unittest.main()
