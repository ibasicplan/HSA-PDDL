from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from abstraction.relation_catalog import RelationCatalog
from utils.pddl import domain_uses_typing


def _identifier_pattern(value: str) -> str:
    return r"(?<![a-z0-9_-])" + re.escape(value) + r"(?![a-z0-9_-])"


def extract_object_inventory(problem_nl: str) -> Dict[str, Any]:
    """Extract an inference-visible entity inventory from Problem NL."""
    typed_pattern = re.compile(
        r"\bthere\s+(?:are|is)\s+(\d+)\s+objects?\s+that\s+(?:are|is)\s+"
        r"(?:(?:a|an|the)\s+)?([a-z][a-z0-9_-]*)\s*:\s*([^\r\n]+)",
        re.I,
    )
    typed_matches = list(typed_pattern.finditer(problem_nl))
    if typed_matches:
        unique: List[str] = []
        seen = set()
        object_types: Dict[str, str] = {}
        type_conflicts: List[Dict[str, str]] = []
        declared_total = 0
        parsed_total = 0
        groups: List[Dict[str, Any]] = []
        for match in typed_matches:
            declared_count = int(match.group(1))
            type_name = match.group(2).lower()
            names = [
                token.lower()
                for token in re.findall(r"[a-z][a-z0-9_-]*", match.group(3), re.I)
                if token.lower() != "and"
            ]
            group_unique: List[str] = []
            group_seen = set()
            for name in names:
                if name not in group_seen:
                    group_seen.add(name)
                    group_unique.append(name)
                previous = object_types.get(name)
                if previous and previous != type_name:
                    type_conflicts.append({"object": name, "left": previous, "right": type_name})
                object_types[name] = type_name
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            declared_total += declared_count
            parsed_total += len(group_unique)
            groups.append({
                "type": type_name,
                "declared_count": declared_count,
                "objects": group_unique,
                "count_matches_list": declared_count == len(group_unique),
            })
        return {
            "available": True,
            "declared_count": declared_total,
            "objects": unique,
            "object_types": object_types,
            "count_matches_list": (
                declared_total == parsed_total
                and all(group["count_matches_list"] for group in groups)
                and not type_conflicts
            ),
            "type_conflicts": type_conflicts,
            "groups": groups,
            "source_text": "typed counted object inventory groups in problem.nl",
            "source_kind": "typed_counted_inventory_groups",
        }

    match = re.search(
        r"\bthere\s+(?:are|is)\s+(\d+)\s+objects?\s*:\s*([^\r\n]+)",
        problem_nl,
        re.I,
    )
    if not match:
        inline_names = [
            token.lower()
            for token in re.findall(r"\bobject\s+([a-z][a-z0-9_-]*)", problem_nl, re.I)
        ]
        unique_inline: List[str] = []
        seen_inline = set()
        for name in inline_names:
            if name not in seen_inline:
                seen_inline.add(name)
                unique_inline.append(name)
        if not unique_inline:
            return {
                "available": False,
                "declared_count": None,
                "objects": [],
                "object_types": {},
                "source_kind": None,
            }
        return {
            "available": True,
            "declared_count": len(unique_inline),
            "objects": unique_inline,
            "object_types": {},
            "count_matches_list": True,
            "source_text": "inline 'object NAME' mentions in problem.nl",
            "source_kind": "inline_object_mentions",
        }
    declared_count = int(match.group(1))
    names = [
        token.lower()
        for token in re.findall(r"[a-z][a-z0-9_-]*", match.group(2), re.I)
        if token.lower() != "and"
    ]
    unique: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return {
        "available": True,
        "declared_count": declared_count,
        "objects": unique,
        "object_types": {},
        "count_matches_list": declared_count == len(unique),
        "source_text": match.group(0).strip(),
        "source_kind": "explicit_counted_inventory",
    }


def _problem_object_names(inventory: Dict[str, Any], domain: Dict[str, Any]) -> List[str]:
    constants = {str(name).lower() for name in domain.get("constants", {})}
    return [
        str(name).lower()
        for name in inventory.get("objects", [])
        if str(name).lower() not in constants
    ]


def _split_assertions(text: str, object_names: Sequence[str]) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"[,;\r\n]+", text) if chunk.strip()]
    def clean(chunk: str) -> str:
        chunk = re.sub(r"^\s*(?:that\s*,?\s*)?and\s+", "", chunk, flags=re.I)
        chunk = re.sub(r"^\s*that\s*,?\s*", "", chunk, flags=re.I)
        return chunk.strip().rstrip(". ")

    if not object_names:
        return [clean(chunk) for chunk in chunks if clean(chunk)]
    object_union = "|".join(re.escape(name) for name in sorted(object_names, key=len, reverse=True))
    output: List[str] = []
    # Restrict "and" splitting to known assertion heads to avoid breaking conjunctions within a statement.
    assertion_head = (
        rf"(?:in\s+the\s+end\s+)?(?:"
        rf"(?:{object_union})(?![a-z0-9_-])|"
        rf"object\s+(?:{object_union})(?![a-z0-9_-])|"
        rf"(?:province|planet|pain)\s+(?:object\s+)?(?:{object_union})(?![a-z0-9_-])|"
        rf"(?:that\s+)?it\s+is\s+not\s+the\s+case\s+that\s+(?:{object_union})(?![a-z0-9_-])|"
        rf"harmony(?![a-z0-9_-])"
        rf")"
    )
    splitter = re.compile(
        rf"\s+and\s+(?={assertion_head})",
        re.I,
    )
    for chunk in chunks:
        output.extend(clean(part) for part in splitter.split(chunk) if clean(part))
    return output


def extract_fact_statements(problem_nl: str, object_names: Sequence[str]) -> Dict[str, Any]:
    """Count explicit INIT/GOAL clauses in supported benchmark NL templates."""
    init_match = re.search(
        r"\bcurrently\s*,\s*(.*?)(?=(?:\r?\n\s*){1,2}my\s+goal\b|\bmy\s+goal\b|$)",
        problem_nl,
        re.I | re.S,
    )
    init_template = "currently"
    if init_match is None:
        init_match = re.search(
            r"\bas\s+initial\s+conditions?\s*,?\s*i\s+have\s+that\s*,?\s*"
            r"(.*?)(?=\bmy\s+goal\b|$)",
            problem_nl,
            re.I | re.S,
        )
        init_template = "as_initial_conditions"
    goal_match = re.search(
        r"\bmy\s+goal\s+is\s+(?:to\s+have\s+that|that)\s*(.*)$",
        problem_nl,
        re.I | re.S,
    )
    init = _split_assertions(init_match.group(1), object_names) if init_match else []
    goal = _split_assertions(goal_match.group(1), object_names) if goal_match else []
    goal = [re.sub(r"^\s*in\s+the\s+end\s+", "", clause, flags=re.I) for clause in goal]
    return {
        "init_available": init_match is not None,
        "goal_available": goal_match is not None,
        "init": init,
        "goal": goal,
        "init_count": len(init),
        "goal_count": len(goal),
        "init_template": init_template if init_match else None,
        "goal_template": "my_goal" if goal_match else None,
    }


def extract_explicit_classifications(
    init_statements: Sequence[str],
    object_names: Sequence[str],
    domain: Dict[str, Any],
    catalog: RelationCatalog,
) -> List[Dict[str, Any]]:
    """Extract explicit unary class assertions such as ``t0 is a truck``."""
    output: List[Dict[str, Any]] = []
    seen = set()
    for statement in init_statements:
        for name in object_names:
            name_pattern = _identifier_pattern(name)
            if not re.search(name_pattern, statement, re.I):
                continue
            for spec in catalog.classification_specs(domain):
                for phrase in spec["phrases"]:
                    phrase_pattern = re.escape(phrase).replace(r"\ ", r"\s+")
                    assertion = re.compile(
                        name_pattern
                        + r"\s+(?:is|is\s+classified\s+as|is\s+declared\s+as)\s+"
                        + r"(?:(?:a|an|the)\s+)?"
                        + phrase_pattern
                        + r"(?![a-z0-9_-])",
                        re.I,
                    )
                    if not assertion.search(statement):
                        continue
                    key = (spec["relation"], name)
                    if key not in seen:
                        seen.add(key)
                        output.append({
                            "stage": "INIT",
                            "polarity": True,
                            "relation": spec["relation"],
                            "predicate": spec["predicate"],
                            "arguments": [name],
                            "evidence": statement,
                        })
                    break
    return output


def prompt_coverage_summary(problem_nl: str, domain: Dict[str, Any] | None = None) -> Dict[str, Any]:
    inventory = extract_object_inventory(problem_nl)
    all_entities = inventory.get("objects", [])
    problem_objects = _problem_object_names(inventory, domain or {}) if domain else list(all_entities)
    statements = extract_fact_statements(problem_nl, all_entities)
    constants_in_nl = sorted(
        set(all_entities).intersection({str(name).lower() for name in (domain or {}).get("constants", {})})
    )
    return {
        "object_inventory_available": inventory["available"],
        "expected_objects": len(problem_objects) if inventory.get("available") else None,
        "expected_entities_including_constants": len(all_entities) if inventory.get("available") else None,
        "object_types": dict(inventory.get("object_types", {})),
        "domain_constants_in_nl": constants_in_nl,
        "fact_statement_counts_available": bool(
            statements["init_available"] and statements["goal_available"]
        ),
        "expected_init_fact_statements": statements["init_count"],
        "expected_goal_fact_statements": statements["goal_count"],
    }


def inspect_nl_coverage_contract(
    problem_nl: str,
    domain: Dict[str, Any],
    catalog: RelationCatalog,
) -> Dict[str, Any]:
    """Verify that the inference-visible NL supports the configured coverage gate."""
    policy = dict(catalog.eir_coverage or {})
    if not bool(policy.get("enabled", False)):
        return {
            "applicable": False,
            "ok": True,
            "errors": [],
            "reference_pddl_used": False,
        }

    inventory = extract_object_inventory(problem_nl)
    entity_names = inventory.get("objects", [])
    object_names = _problem_object_names(inventory, domain)
    statements = extract_fact_statements(problem_nl, entity_names)
    errors: List[Dict[str, Any]] = []

    if bool(policy.get("strict_object_inventory", True)):
        if not inventory.get("available"):
            errors.append({
                "error_type": "nl_object_inventory_unavailable",
                "instruction": (
                    "Expected either an explicit 'There are N objects: ...' inventory or "
                    "unambiguous inline 'object NAME' mentions in problem.nl."
                ),
            })
        elif not inventory.get("count_matches_list", True):
            errors.append({
                "error_type": "nl_object_inventory_count_mismatch",
                "declared_count": inventory.get("declared_count"),
                "parsed_entity_count": len(entity_names),
                "parsed_problem_object_count": len(object_names),
            })

    declared_types = {str(k).lower(): str(v).lower() for k, v in inventory.get("object_types", {}).items()}
    valid_types = {"object", *[str(name).lower() for name in domain.get("type_hierarchy", {})]}
    valid_types.update(
        str(value).lower() if not isinstance(value, list) else str(value[0]).lower()
        for value in domain.get("constants", {}).values()
        if value
    )
    for name, declared_type in sorted(declared_types.items()):
        if declared_type not in valid_types:
            errors.append({
                "error_type": "nl_unknown_object_type",
                "object": name,
                "declared_type": declared_type,
                "valid_types": sorted(valid_types),
            })
    for name, constant_type in domain.get("constants", {}).items():
        lname = str(name).lower()
        if lname not in declared_types:
            continue
        observed_type = declared_types[lname]
        expected_type = str(constant_type[0] if isinstance(constant_type, list) else constant_type).lower()
        if observed_type != expected_type:
            errors.append({
                "error_type": "nl_domain_constant_type_mismatch",
                "constant": lname,
                "declared_type": observed_type,
                "expected_type": expected_type,
            })

    if bool(policy.get("strict_fact_statement_counts", True)):
        if not statements["init_available"]:
            errors.append({
                "error_type": "nl_init_assertions_unavailable",
                "instruction": (
                    "Expected a supported 'Currently, ...' or "
                    "'As initial conditions I have that, ...' section in problem.nl."
                ),
            })
        if not statements["goal_available"]:
            errors.append({
                "error_type": "nl_goal_assertions_unavailable",
                "instruction": (
                    "Expected a supported 'My goal is that ...' or "
                    "'My goal is to have that ...' section in problem.nl."
                ),
            })

    expected_classifications: List[Dict[str, Any]] = []
    classification_specs = catalog.classification_specs(domain)
    if bool(policy.get("require_untyped_classification_facts", False)) and not domain_uses_typing(domain):
        if not classification_specs:
            errors.append({
                "error_type": "catalog_classification_phrases_unavailable",
                "instruction": "Add classification_phrases to unary catalog relations.",
            })
        elif statements["init_available"] and inventory.get("available"):
            expected_classifications = extract_explicit_classifications(
                statements["init"], object_names, domain, catalog
            )
            if not expected_classifications:
                errors.append({
                    "error_type": "nl_explicit_classifications_unavailable",
                    "instruction": (
                        "The untyped domain requires unary classification FACTs, but none could be "
                        "extracted from problem.nl with the configured catalog phrases."
                    ),
                })

    return {
        "applicable": True,
        "ok": not errors,
        "errors": errors,
        "policy": policy,
        "reference_pddl_used": False,
        "object_inventory_available": bool(inventory.get("available")),
        "object_inventory_source": inventory.get("source_kind"),
        "expected_objects": len(object_names) if inventory.get("available") else None,
        "expected_entities_including_constants": len(entity_names) if inventory.get("available") else None,
        "domain_constants_in_nl": sorted(
            set(entity_names).intersection({str(name).lower() for name in domain.get("constants", {})})
        ),
        "object_types": dict(inventory.get("object_types", {})),
        "expected_init_fact_statements": (
            statements["init_count"] if statements["init_available"] else None
        ),
        "expected_goal_fact_statements": (
            statements["goal_count"] if statements["goal_available"] else None
        ),
        "expected_explicit_classification_facts": len(expected_classifications),
    }


def evaluate_eir_coverage(
    problem_nl: str,
    eir: Dict[str, Any],
    air: Dict[str, Any],
    domain: Dict[str, Any],
    catalog: RelationCatalog,
) -> Dict[str, Any]:
    """Validate input-visible EIR coverage without reading reference PDDL."""
    policy = dict(catalog.eir_coverage or {})
    if not bool(policy.get("enabled", False)):
        return {
            "applicable": False,
            "ok": True,
            "errors": [],
            "reference_pddl_used": False,
        }

    input_contract = inspect_nl_coverage_contract(problem_nl, domain, catalog)
    errors: List[Dict[str, Any]] = list(input_contract.get("errors", []))
    inventory = extract_object_inventory(problem_nl)
    entity_names = [str(name).lower() for name in inventory.get("objects", [])]
    expected_objects = set(_problem_object_names(inventory, domain))
    observed_objects = {str(obj.get("name", "")).lower() for obj in eir.get("objects", [])}
    if inventory.get("available") and not inventory.get("count_matches_list", True):
        errors.append({
            "error_type": "nl_object_inventory_count_mismatch",
            "declared_count": inventory.get("declared_count"),
            "parsed_names": sorted(entity_names),
        })
    if inventory.get("available") and bool(policy.get("strict_object_inventory", True)):
        for name in sorted(expected_objects - observed_objects):
            errors.append({"error_type": "missing_eir_object", "object": name})
        for name in sorted(observed_objects - expected_objects):
            errors.append({"error_type": "extra_eir_object", "object": name})

        expected_types = {
            str(name).lower(): str(typ).lower()
            for name, typ in inventory.get("object_types", {}).items()
            if str(name).lower() in expected_objects
        }
        observed_types = {
            str(obj.get("name", "")).lower(): str(obj.get("type", "unknown") or "unknown").lower()
            for obj in eir.get("objects", [])
        }
        for name, expected_type in sorted(expected_types.items()):
            observed_type = observed_types.get(name)
            if observed_type is not None and observed_type != expected_type:
                errors.append({
                    "error_type": "wrong_eir_object_type",
                    "object": name,
                    "expected_type": expected_type,
                    "observed_type": observed_type,
                })

    statements = extract_fact_statements(problem_nl, entity_names or sorted(observed_objects))
    observed_init = sum(1 for fact in eir.get("facts", []) if fact.get("stage") == "INIT")
    observed_goal = sum(1 for fact in eir.get("facts", []) if fact.get("stage") == "GOAL")
    if bool(policy.get("strict_fact_statement_counts", True)):
        if statements["init_available"] and observed_init != statements["init_count"]:
            errors.append({
                "error_type": "eir_init_fact_count_mismatch",
                "expected_from_problem_nl": statements["init_count"],
                "observed_in_eir": observed_init,
                "instruction": "Emit one INIT FACT for every comma-separated current-state assertion.",
            })
        if statements["goal_available"] and observed_goal != statements["goal_count"]:
            errors.append({
                "error_type": "eir_goal_fact_count_mismatch",
                "expected_from_problem_nl": statements["goal_count"],
                "observed_in_eir": observed_goal,
                "instruction": "Emit one GOAL FACT for every explicit goal assertion.",
            })

    expected_classifications: List[Dict[str, Any]] = []
    if bool(policy.get("require_untyped_classification_facts", False)) and not domain_uses_typing(domain):
        expected_classifications = extract_explicit_classifications(
            statements["init"], sorted(expected_objects or observed_objects), domain, catalog
        )
        expected_index = {
            (row["stage"], row["relation"], tuple(row["arguments"]), row["polarity"]): row
            for row in expected_classifications
        }
        classification_relations = {
            spec["relation"] for spec in catalog.classification_specs(domain)
        }
        actual_index: Dict[Tuple[str, str, Tuple[str, ...], bool], Dict[str, Any]] = {}
        for fact in air.get("facts", []):
            relation = str(fact.get("selected_relation") or "")
            if relation not in classification_relations:
                continue
            key = (
                str(fact.get("stage", "")).upper(),
                relation,
                tuple(str(arg).lower() for arg in fact.get("arguments", [])),
                bool(fact.get("polarity", True)),
            )
            actual_index[key] = fact
        for key in sorted(set(expected_index) - set(actual_index)):
            expected = expected_index[key]
            errors.append({
                "error_type": "missing_explicit_classification_fact",
                "entity": expected["arguments"][0],
                "classification": expected["predicate"],
                "expected_relation": expected["relation"],
                "evidence": expected["evidence"],
                "required_eir_shape": (
                    f"FACT|new_id|INIT|POS|{expected['predicate']} classification|"
                    f"{expected['arguments'][0]}|{expected['evidence']}"
                ),
            })
        if bool(policy.get("reject_extra_classification_facts", True)):
            for key in sorted(set(actual_index) - set(expected_index)):
                fact = actual_index[key]
                errors.append({
                    "error_type": "extra_classification_fact_not_in_problem_nl",
                    "fact_id": fact.get("fact_id"),
                    "relation": fact.get("selected_relation"),
                    "arguments": fact.get("arguments", []),
                })

    missing_classes = sum(
        1 for error in errors if error.get("error_type") == "missing_explicit_classification_fact"
    )
    return {
        "applicable": True,
        "ok": not errors,
        "errors": errors,
        "policy": policy,
        "input_contract": input_contract,
        "reference_pddl_used": False,
        "object_inventory": {
            "available": inventory.get("available"),
            "source_kind": inventory.get("source_kind"),
            "expected_count": len(expected_objects) if inventory.get("available") else None,
            "expected_entities_including_constants": len(entity_names) if inventory.get("available") else None,
            "domain_constants_in_nl": sorted(
                set(entity_names).intersection({str(name).lower() for name in domain.get("constants", {})})
            ),
            "observed_count": len(observed_objects),
            "missing": sorted(expected_objects - observed_objects),
            "extra": sorted(observed_objects - expected_objects) if inventory.get("available") else [],
        },
        "fact_statement_counts": {
            "expected_init": statements["init_count"] if statements["init_available"] else None,
            "observed_init": observed_init,
            "expected_goal": statements["goal_count"] if statements["goal_available"] else None,
            "observed_goal": observed_goal,
        },
        "classification_facts": {
            "expected": len(expected_classifications),
            "missing": missing_classes,
            "coverage": (
                (len(expected_classifications) - missing_classes) / len(expected_classifications)
                if expected_classifications else None
            ),
        },
    }
