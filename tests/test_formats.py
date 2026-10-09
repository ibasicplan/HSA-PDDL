import unittest

from generation.eir_generator import detect_output_format, parse_eir
from generation.local_llm import sanitize_output


class FormatTests(unittest.TestCase):
    def test_gpt_oss_return_token_is_removed(self):
        raw = "(define (problem demo) (:domain d) (:objects) (:init) (:goal (and)))<|eot_id|><|return|>"
        clean = sanitize_output(raw)
        self.assertNotIn("<|", clean)
        self.assertEqual(detect_output_format(clean), "pddl_problem")

    def test_spaced_eir_delimiters_parse(self):
        raw = """OBJECT | p1 | person | p1 is a person
OBJECT | c1 | city | c1 is a city
FACT | F1 | INIT | POS | located at a city | p1,c1 | p1 is at c1
FACT | F2 | GOAL | POS | located at a city | p1,c1 | p1 must be at c1
END_EIR
"""
        eir, status = parse_eir(raw)
        self.assertTrue(status.startswith("ok"))
        self.assertEqual(len(eir["objects"]), 2)
        self.assertEqual(len(eir["facts"]), 2)

if __name__ == "__main__":
    unittest.main()
