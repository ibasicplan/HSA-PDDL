from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from generation.local_llm import sanitize_output
from generation.prompts import EIR_SYSTEM_PROMPT
from generation.eir_coverage import extract_object_inventory, prompt_coverage_summary
from utils.pddl import domain_uses_typing, parse_domain


def _normalise(eir: Dict[str, Any]) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    seen_objects = set()
    for raw in eir.get("objects", []):
        name = str(raw.get("name", "")).strip().lower()
        if not name or name in seen_objects:
            continue
        seen_objects.add(name)
        objects.append({
            "name": name,
            "type": raw.get("type", "unknown") or "unknown",
            "evidence": str(raw.get("evidence", "")),
        })
    facts: List[Dict[str, Any]] = []
    seen_facts = set()
    for index, raw in enumerate(eir.get("facts", []), 1):
        stage = str(raw.get("stage", "")).upper()
        relation = str(raw.get("relation", raw.get("relation_phrase", ""))).strip()
        arguments = [str(x).strip().lower() for x in raw.get("arguments", []) if str(x).strip()]
        polarity = bool(raw.get("polarity", True))
        key = (stage, relation.lower(), tuple(arguments), polarity)
        if stage not in {"INIT", "GOAL"} or not relation or key in seen_facts:
            continue
        seen_facts.add(key)
        facts.append({
            "fact_id": str(raw.get("fact_id") or f"F{index}"),
            "stage": stage,
            "polarity": polarity,
            "relation": relation,
            "arguments": arguments,
            "evidence": str(raw.get("evidence", "")),
        })
    return {"objects": objects, "facts": facts}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def detect_output_format(text: str) -> str:
    clean = sanitize_output(text)
    if re.search(r"^\s*OBJECT\s*\|", clean, re.I | re.M) or re.search(r"^\s*FACT\s*\|", clean, re.I | re.M):
        return "eir_lines"
    if re.search(r"\(\s*define\s+\(\s*problem\b", clean, re.I):
        return "pddl_problem"
    if _extract_json_object(clean) is not None:
        return "json"
    return "unknown"


def parse_eir(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    clean = sanitize_output(text)
    obj = _extract_json_object(clean)
    if obj is not None:
        eir = _normalise(obj)
        if eir["objects"] and eir["facts"]:
            return eir, "json_ok"
    objects: List[Dict[str, Any]] = []
    facts: List[Dict[str, Any]] = []
    saw_end = False
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "END_EIR":
            saw_end = True
            break
        if re.match(r"^OBJECT\s*\|", line, re.I):
            parts = re.split(r"\s*\|\s*", line, maxsplit=3)
            if len(parts) >= 3:
                objects.append({
                    "name": parts[1].strip(),
                    "type": parts[2].strip() or "unknown",
                    "evidence": parts[3].strip() if len(parts) > 3 else "",
                })
        elif re.match(r"^FACT\s*\|", line, re.I):
            parts = re.split(r"\s*\|\s*", line, maxsplit=6)
            if len(parts) >= 7:
                polarity = parts[3].strip().upper() != "NEG"
                relation_index, args_index, evidence_index = 4, 5, 6
            elif len(parts) >= 6:
                polarity = True
                relation_index, args_index, evidence_index = 3, 4, 5
            else:
                continue
            facts.append({
                "fact_id": parts[1].strip(),
                "stage": parts[2].strip().upper(),
                "polarity": polarity,
                "relation": parts[relation_index].strip(),
                "arguments": [x.strip() for x in parts[args_index].split(",") if x.strip()],
                "evidence": parts[evidence_index].strip(),
            })
    eir = _normalise({"objects": objects, "facts": facts})
    if not eir["objects"] or not eir["facts"]:
        return None, "missing_objects_or_facts"
    if not any(x["stage"] == "GOAL" for x in eir["facts"]):
        return None, "missing_goal"
    return eir, "ok" if saw_end else "ok_without_end_marker"


def validate_eir(eir: Dict[str, Any], domain: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    objects = {str(x.get("name", "")).lower() for x in eir.get("objects", [])}
    allowed_arguments = set(objects)
    if domain:
        allowed_arguments.update(str(name).lower() for name in domain.get("constants", {}))
    if not objects:
        errors.append({"error_type": "missing_objects"})
    if not any(x.get("stage") == "GOAL" for x in eir.get("facts", [])):
        errors.append({"error_type": "missing_goal"})
    ids = set()
    for fact in eir.get("facts", []):
        fact_id = str(fact.get("fact_id", ""))
        if not fact_id or fact_id in ids:
            errors.append({"error_type": "duplicate_or_missing_fact_id", "fact_id": fact_id})
        ids.add(fact_id)
        for argument in fact.get("arguments", []):
            if argument not in allowed_arguments:
                errors.append({"error_type": "undeclared_eir_object_or_domain_constant", "fact_id": fact_id, "argument": argument})
    return errors


def build_eir_prompt(
    domain_pddl: str,
    domain_nl: str,
    problem_nl: str,
    feedback: Optional[Dict[str, Any]],
    schema_guidance: str = "",
) -> str:
    repair = f"\nPrevious validation feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n" if feedback else ""
    parsed_domain = parse_domain(domain_pddl)
    coverage = prompt_coverage_summary(problem_nl, parsed_domain)
    untyped_block = ""
    zero_arity_predicates = sorted(
        name for name, spec in parsed_domain.get("predicates", {}).items()
        if int(spec.get("arity", -1)) == 0
    )
    zero_arity_block = ""
    if zero_arity_predicates:
        zero_arity_block = f"""
ZERO-ARITY FACT RULE:
The schema contains zero-arity predicates: {', '.join(zero_arity_predicates)}. A zero-arity fact has
no entity arguments. Preserve it as a FACT with an empty argument field (two adjacent separators),
for example: FACT|F1|INIT|POS|global condition holds||the global condition currently holds
Never invent a dummy object argument for a zero-arity fact.
"""
    typed_block = ""
    inventory = extract_object_inventory(problem_nl)
    if domain_uses_typing(parsed_domain) and inventory.get("object_types"):
        constants = {str(name).lower(): typ for name, typ in parsed_domain.get("constants", {}).items()}
        problem_object_types = {
            str(name).lower(): str(typ).lower()
            for name, typ in inventory.get("object_types", {}).items()
            if str(name).lower() not in constants
        }
        grouped = {}
        for name, typ in problem_object_types.items():
            grouped.setdefault(typ, []).append(name)
        type_lines = "\n".join(
            f"- {typ}: {', '.join(sorted(names))}" for typ, names in sorted(grouped.items())
        )
        constant_lines = "\n".join(
            f"- {name}: fixed domain constant of type {typ}"
            for name, typ in sorted(constants.items())
            if name in set(inventory.get("objects", []))
        )
        typed_block = f"""
CRITICAL TYPED-OBJECT RULES:
Problem OBJECT lines must preserve the exact PDDL type names declared by the input inventory and Domain PDDL.
The expected problem objects are:
{type_lines or '- none'}
Fixed domain constants mentioned in Problem NL are schema symbols, not problem OBJECT declarations:
{constant_lines or '- none mentioned'}
Do NOT emit a fixed domain constant as an OBJECT line. A domain constant may still appear as a FACT argument.
"""
    guidance_block = (
        f"\nFROZEN DATASET-SPECIFIC EIR SEMANTIC RULES:\n{schema_guidance}\n"
        if schema_guidance else ""
    )

    format_example = (
        "OBJECT|demo_person|person|demo_person is explicitly declared as a person\n"
        "OBJECT|demo_city|city|demo_city is explicitly declared as a city\n"
        "FACT|F1|INIT|POS|located at a city|demo_person,demo_city|demo_person is at demo_city\n"
        "FACT|F2|GOAL|POS|located at a city|demo_person,demo_city|demo_person must be at demo_city\n"
    )
    if not domain_uses_typing(parsed_domain):
        unary_predicates = sorted(
            name for name, spec in parsed_domain.get("predicates", {}).items()
            if int(spec.get("arity", -1)) == 1
        )
        count_line = ""
        if coverage["object_inventory_available"] and coverage["fact_statement_counts_available"]:
            count_line = (
                f"The input explicitly declares {coverage['expected_objects']} objects, "
                f"{coverage['expected_init_fact_statements']} INIT assertions, and "
                f"{coverage['expected_goal_fact_statements']} GOAL assertions. Emit all of them.\n"
            )
        untyped_block = f"""
CRITICAL UNTYPED-STRIPS RULES:
This domain does not use PDDL :typing. The type_or_unknown field on an OBJECT line is descriptive
metadata only; it NEVER creates a state fact. Every explicit category assertion in Problem NL must
also be emitted as its own INIT FACT. Unary category predicates in this schema are:
{', '.join(unary_predicates) if unary_predicates else 'none'}.
An entity may require multiple category FACTs. For example, if a place is explicitly both a location
and an airport, emit two FACT lines for it. Do not replace either FACT with the OBJECT type label.
{count_line}Before END_EIR, verify that every comma-separated current-state assertion and every goal assertion
has exactly one FACT line. Never omit category/classification assertions.
"""
        if {"obj", "location", "airport"}.issubset(set(unary_predicates)):
            format_example = (
                "OBJECT|demo_cargo|object|demo_cargo is explicitly declared as an object\n"
                "OBJECT|demo_hub|location|demo_hub is explicitly declared as a location and airport\n"
                "FACT|F1|INIT|POS|object classification|demo_cargo|demo_cargo is an object\n"
                "FACT|F2|INIT|POS|location classification|demo_hub|demo_hub is a location\n"
                "FACT|F3|INIT|POS|airport classification|demo_hub|demo_hub is an airport\n"
                "FACT|F4|INIT|POS|located at a location|demo_cargo,demo_hub|demo_cargo is at demo_hub\n"
                "FACT|F5|GOAL|POS|located at a location|demo_cargo,demo_hub|demo_cargo must be at demo_hub\n"
            )
    return f"""Domain PDDL (fixed schema):
{domain_pddl}

Domain NL:
{domain_nl}

Problem NL:
{problem_nl}
{repair}{untyped_block}{typed_block}{zero_arity_block}{guidance_block}
FORMAT EXAMPLE ONLY (do not copy these demo entities):
{format_example}END_EIR

Now output the actual problem in exactly this format:
OBJECT|name|type_or_unknown|short evidence
FACT|F1|INIT|POS|natural-language relation|arg1,arg2|short evidence
FACT|F2|GOAL|POS|natural-language relation|arg1,arg2|short evidence
END_EIR

Use NEG instead of POS only according to the benchmark semantics and the dataset-specific EIR rules above.
Every FACT argument must either be declared as an OBJECT or be a fixed constant declared by Domain PDDL.
Do not copy a concrete PDDL predicate name merely as the relation description; preserve the NL meaning.
Keep FACT arguments in the same semantic/mention order as the natural-language relation unless a frozen
dataset-specific EIR rule above explicitly defines a canonical semantic role order. Do not otherwise reorder
arguments merely to imitate a concrete PDDL predicate signature; the later catalog grounding stage does that.
"""


class EIRGenerator:
    def __init__(self, llm: Any, max_new_tokens: int = 2048, schema_guidance: str = ""):
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
        prompt = build_eir_prompt(
            domain_pddl, domain_nl, problem_nl, feedback, self.schema_guidance
        )
        generation = self.llm.generate(
            "eir", EIR_SYSTEM_PROMPT, prompt, self.max_new_tokens, sample_key, attempt
        )
        eir, parse_status = parse_eir(generation["clean_output"])
        detected_format = detect_output_format(generation["clean_output"])
        return {
            "prompt": prompt,
            "generation": generation,
            "eir": eir,
            "parse_status": parse_status,
            "detected_output_format": detected_format,
            "source": "native_eir",
            "adapter_triggered": False,
        }
