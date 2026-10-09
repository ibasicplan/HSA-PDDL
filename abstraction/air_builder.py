from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from abstraction.relation_catalog import RelationCatalog


def build_air(
    eir: Dict[str, Any],
    domain: Dict[str, Any],
    catalog: RelationCatalog,
    score_threshold: float = 3.0,
    score_margin: float = 0.5,
) -> Dict[str, Any]:
    object_types = {
        str(name).lower(): typ
        for name, typ in domain.get("constants", {}).items()
    }
    object_types.update({
        str(x["name"]).lower(): x.get("type", "unknown")
        for x in eir.get("objects", [])
    })
    facts: List[Dict[str, Any]] = []
    for source in eir.get("facts", []):
        arguments = [str(x).lower() for x in source.get("arguments", [])]
        feasible = catalog.feasible_relations(arguments, object_types, domain)
        semantic_text = " ".join([
            str(source.get("relation", source.get("relation_phrase", ""))),
            str(source.get("evidence", "")),
        ]).strip()
        ranking = catalog.rank(semantic_text, feasible)
        selected: Optional[str] = None
        selection_source = "unresolved"
        if len(feasible) == 1:
            selected = feasible[0]
            selection_source = "type_arity_singleton"
        elif ranking:
            best_score = ranking[0][1]
            runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
            if best_score >= score_threshold and best_score - runner_up >= score_margin:
                selected = ranking[0][0]
                selection_source = "offline_catalog_match"
        facts.append({
            "fact_id": str(source.get("fact_id")),
            "stage": str(source.get("stage", "")).upper(),
            "polarity": bool(source.get("polarity", True)),
            "relation_text": str(source.get("relation", source.get("relation_phrase", ""))),
            "arguments": arguments,
            "evidence": str(source.get("evidence", "")),
            "candidate_relations": feasible,
            "relation_scores": [{"relation": rid, "score": score} for rid, score in ranking],
            "selected_relation": selected,
            "relation_selection_source": selection_source,
            "candidate_predicates": (
                catalog.compatible_predicates(selected, arguments, object_types, domain) if selected else []
            ),
        })
    return {
        "objects": eir.get("objects", []),
        "facts": facts,
        "relation_abstraction_enabled": True,
        "catalog_domain": catalog.domain_name,
    }


def unresolved_facts(air: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [x for x in air.get("facts", []) if not x.get("selected_relation")]


def apply_relation_mapping(
    air: Dict[str, Any],
    mapping: Dict[str, str],
    domain: Dict[str, Any],
    catalog: RelationCatalog,
    source: str = "local_llm_whitelist",
) -> Dict[str, Any]:
    object_types = {
        str(name).lower(): typ
        for name, typ in domain.get("constants", {}).items()
    }
    object_types.update({
        str(x["name"]).lower(): x.get("type", "unknown")
        for x in air.get("objects", [])
    })
    out = {**air, "facts": []}
    for fact in air.get("facts", []):
        updated = dict(fact)
        requested = str(mapping.get(str(fact.get("fact_id")), fact.get("selected_relation") or "")).lower()
        if requested and requested in fact.get("candidate_relations", []):
            updated["selected_relation"] = requested
            if not fact.get("selected_relation"):
                updated["relation_selection_source"] = source
            updated["candidate_predicates"] = catalog.compatible_predicates(
                requested, fact.get("arguments", []), object_types, domain
            )
        out["facts"].append(updated)
    return out


def validate_air(air: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for fact in air.get("facts", []):
        fact_id = fact.get("fact_id")
        selected = fact.get("selected_relation")
        allowed = fact.get("candidate_relations", [])
        if not allowed:
            errors.append({"error_type": "no_feasible_relation", "fact_id": fact_id})
        elif not selected:
            errors.append({"error_type": "missing_relation_mapping", "fact_id": fact_id, "allowed": allowed})
        elif selected not in allowed:
            errors.append({"error_type": "relation_not_in_whitelist", "fact_id": fact_id, "selected": selected, "allowed": allowed})
        elif not fact.get("candidate_predicates"):
            errors.append({"error_type": "no_grounding_candidate", "fact_id": fact_id, "relation": selected})
    return errors

