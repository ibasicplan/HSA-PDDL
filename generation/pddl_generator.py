from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from generation.prompts import PDDL_SYSTEM_PROMPT
from utils.pddl import extract_complete_define


def build_pddl_prompt(
    domain_pddl: str,
    domain_nl: str,
    problem_nl: str,
    feedback: Optional[Dict[str, Any]],
    schema_guidance: str = "",
) -> str:
    repair = f"\nPrevious validation feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n" if feedback else ""
    guidance = f"\nFrozen dataset-specific predicate-slot conventions:\n{schema_guidance}\n" if schema_guidance else ""
    return f"""Domain PDDL:
{domain_pddl}

Domain NL:
{domain_nl}

Problem NL:
{problem_nl}
{guidance}{repair}
Return only one complete PDDL problem. Preserve every explicitly stated object, initial fact, and goal fact.
Use only predicates, arities, types, and constants declared by the fixed Domain PDDL.
"""


class PDDLGenerator:
    def __init__(self, llm: Any, max_new_tokens: int = 4096, schema_guidance: str = ""):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.schema_guidance = schema_guidance

    def generate(
        self,
        domain_pddl: str,
        domain_nl: str,
        problem_nl: str,
        feedback: Optional[Dict[str, Any]],
        sample_key: str,
        attempt: int,
    ) -> Dict[str, Any]:
        prompt = build_pddl_prompt(
            domain_pddl, domain_nl, problem_nl, feedback, self.schema_guidance
        )
        generation = self.llm.generate(
            "direct_pddl", PDDL_SYSTEM_PROMPT, prompt, self.max_new_tokens, sample_key, attempt
        )
        return {
            "prompt": prompt,
            "generation": generation,
            "pddl": extract_complete_define(generation["clean_output"]),
        }


def parse_relation_mapping(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*REL\s*\|\s*([^|]+)\s*\|\s*([a-zA-Z0-9_-]+)\s*$", line)
        if match:
            mapping[match.group(1).strip()] = match.group(2).strip().lower()
    return mapping


def parse_grounding_mapping(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*MAP\s*\|\s*([^|]+)\s*\|\s*([a-zA-Z0-9_-]+)\s*$", line)
        if match:
            mapping[match.group(1).strip()] = match.group(2).strip().lower()
    return mapping
