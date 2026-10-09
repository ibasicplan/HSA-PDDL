from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from abstraction.relation_catalog import RelationCatalog
from utils.pddl import domain_uses_typing


def deterministic_grounding(air: Dict[str, Any]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    mapping: Dict[str, str] = {}
    unresolved: List[Dict[str, Any]] = []
    for fact in air.get("facts", []):
        candidates = sorted(set(str(x).lower() for x in fact.get("candidate_predicates", []) if str(x)))
        if len(candidates) == 1:
            mapping[str(fact["fact_id"])] = candidates[0]
        else:
            unresolved.append(fact)
    return mapping, unresolved


def build_gir(
    air: Dict[str, Any],
    mapping: Dict[str, str],
    catalog: Optional[RelationCatalog] = None,
) -> Dict[str, Any]:
    facts: List[Dict[str, Any]] = []
    for source in air.get("facts", []):
        candidates = sorted(set(str(x).lower() for x in source.get("candidate_predicates", []) if str(x)))
        predicate = str(mapping.get(str(source.get("fact_id")), "")).lower()
        source_arguments = [str(x).lower() for x in source.get("arguments", [])]
        grounded_arguments = (
            catalog.ground_arguments(predicate, source_arguments)
            if catalog and predicate
            else source_arguments
        )
        facts.append({
            "fact_id": str(source.get("fact_id")),
            "stage": str(source.get("stage", "")).upper(),
            "polarity": bool(source.get("polarity", True)),
            "abstract_relation": source.get("selected_relation"),
            "predicate": predicate,
            "source_arguments": source_arguments,
            "arguments": grounded_arguments,
            "evidence": str(source.get("evidence", "")),
            "candidate_predicates": candidates,
        })
    return {"objects": air.get("objects", []), "facts": facts}


def validate_gir(gir: Dict[str, Any], domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    object_names = {str(x.get("name", "")).lower() for x in gir.get("objects", [])}
    object_names.update(str(name).lower() for name in domain.get("constants", {}))
    for fact in gir.get("facts", []):
        fact_id = fact.get("fact_id")
        predicate = str(fact.get("predicate", "")).lower()
        candidates = fact.get("candidate_predicates", [])
        if not predicate:
            errors.append({"error_type": "missing_grounding", "fact_id": fact_id})
            continue
        if predicate not in candidates:
            errors.append({"error_type": "predicate_not_in_whitelist", "fact_id": fact_id, "predicate": predicate, "allowed": candidates})
            continue
        signature = domain.get("predicates", {}).get(predicate)
        if not signature:
            errors.append({"error_type": "unknown_predicate", "fact_id": fact_id, "predicate": predicate})
            continue
        if len(fact.get("arguments", [])) != int(signature.get("arity", -1)):
            errors.append({"error_type": "wrong_parameter_count", "fact_id": fact_id, "predicate": predicate})
        for argument in fact.get("arguments", []):
            if argument not in object_names:
                errors.append({"error_type": "undeclared_object", "fact_id": fact_id, "argument": argument})
    return errors


def resolve_object_types(gir: Dict[str, Any], domain: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Infer unknown EIR object types from grounded predicate slots."""
    if not domain_uses_typing(domain):
        return [{**source, "type": "object"} for source in gir.get("objects", [])]

    constraints: Dict[str, Set[str]] = defaultdict(set)
    for fact in gir.get("facts", []):
        signature = domain.get("predicates", {}).get(str(fact.get("predicate", "")).lower())
        if not signature:
            continue
        for index, argument in enumerate(fact.get("arguments", [])):
            if index >= len(signature.get("argument_types", [])):
                continue
            expected = signature["argument_types"][index]
            if isinstance(expected, list):
                continue
            constraints[str(argument).lower()].add(str(expected).lower())
    output: List[Dict[str, Any]] = []
    for source in gir.get("objects", []):
        obj = dict(source)
        name = str(obj.get("name", "")).lower()
        current = obj.get("type", "unknown")
        if str(current).lower() == "unknown":
            inferred = constraints.get(name, set())
            obj["type"] = next(iter(inferred)) if len(inferred) == 1 else "object"
        output.append(obj)
    return output
