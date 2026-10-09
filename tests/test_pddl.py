from pathlib import Path
import unittest

from utils.pddl import parse_domain, parse_problem, render_problem, static_report, syntax_report


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "data" / "easy" / "1"


class PDDLTests(unittest.TestCase):
    def test_parse_and_render(self):
        domain_text = (FIXTURE / "domain.pddl").read_text(encoding="utf-8")
        problem_text = (FIXTURE / "problem.pddl").read_text(encoding="utf-8")
        domain = parse_domain(domain_text)
        problem = parse_problem(problem_text)
        self.assertEqual(domain["domain_name"], "zeno-travel")
        self.assertEqual(len(domain["predicates"]), 4)
        self.assertEqual(len(problem["atoms"]), 5)
        objects = [{"name": name, "type": typ} for name, typ in problem["objects"].items()]
        rendered = render_problem("roundtrip", problem["domain_name"], objects, problem["atoms"])
        self.assertTrue(syntax_report(rendered)["ok"])
        self.assertTrue(static_report(domain, rendered)["ok"])

    def test_unbalanced_fails(self):
        self.assertFalse(syntax_report("(define (problem broken)")["ok"])


if __name__ == "__main__":
    unittest.main()
