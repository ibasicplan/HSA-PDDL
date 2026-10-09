from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from abstraction.relation_catalog import RelationCatalog
from utils.pddl import atom_key, parse_problem


def _annotation_index(annotation_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not annotation_path or not annotation_path.is_file():
        return {}
    obj = json.loads(annotation_path.read_text(encoding="utf-8"))
    facts = obj.get("facts", []) if isinstance(obj, dict) else []
    return {str(x.get("fact_id")): x for x in facts if isinstance(x, dict) and x.get("fact_id")}


def evaluate_abstraction(
    air: Optional[Dict[str, Any]],
    gir: Optional[Dict[str, Any]],
    reference_pddl: str,
    catalog: RelationCatalog,
    annotation_path: Optional[Path] = None,
    require_annotations: bool = False,
) -> Dict[str, Any]:
    if not air or not air.get("relation_abstraction_enabled", False):
        return {
            "RMA": None,
            "GER": None,
            "rma_correct": 0,
            "rma_eligible": 0,
            "ger_wrong": 0,
            "ger_eligible": 0,
            "acs_decisions": [],
            "abstraction_applicable": False,
        }
    annotations = _annotation_index(annotation_path)
    if require_annotations and not annotations:
        raise FileNotFoundError(f"Required human annotation missing: {annotation_path}")
    reference = parse_problem(reference_pddl)
    gold_by_stage_args: Dict[Tuple[str, Tuple[str, ...], bool], Set[str]] = defaultdict(set)
    for atom in reference.get("atoms", []):
        stage, predicate, arguments, polarity = atom_key(atom)
        gold_by_stage_args[(stage, arguments, polarity)].add(predicate)
    gir_by_id = {
        str(x.get("fact_id")): x for x in (gir or {}).get("facts", []) if isinstance(x, dict)
    }

    rma_correct = 0
    rma_eligible = 0
    ger_wrong = 0
    ger_eligible = 0
    alignable = 0
    fact_reports: List[Dict[str, Any]] = []
    decisions: List[Dict[str, str]] = []
    for fact in air.get("facts", []):
        fact_id = str(fact.get("fact_id"))
        annotation = annotations.get(fact_id)
        stage = str(fact.get("stage", "")).upper()
        air_arguments = tuple(str(x).lower() for x in fact.get("arguments", []))
        gir_fact = gir_by_id.get(fact_id) or {}
        grounded_arguments = tuple(
            str(x).lower() for x in gir_fact.get("arguments", fact.get("arguments", []))
        )
        polarity = bool(fact.get("polarity", True))
        alignment_arguments = grounded_arguments
        alignment_mode = "gir_grounded_arguments" if gir_fact else "air_arguments_fallback"
        gold_disambiguation = "not_needed"
        if annotation:
            gold_predicates = {str(x).lower() for x in annotation.get("gold_predicates", [])}
            gold_relations = {str(x).lower() for x in annotation.get("gold_relations", [])}
            semantic_key = str(annotation.get("semantic_key") or "|".join(sorted(gold_predicates)))
            gold_source = "human_annotation"
            gold_disambiguation = "human_annotation"
        else:
            gold_predicates = gold_by_stage_args.get((stage, alignment_arguments, polarity), set())
            if not gold_predicates:
                # Re-align via the catalog EIR-to-PDDL permutation so role-order errors stay in RMA/GER denominators.
                expected_matches: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
                for candidate in fact.get("candidate_predicates", []):
                    predicate = str(candidate).lower()
                    expected = tuple(catalog.ground_arguments(predicate, air_arguments))
                    predicates = gold_by_stage_args.get((stage, expected, polarity), set())
                    if predicate in predicates:
                        expected_matches[expected].add(predicate)
                if len(expected_matches) == 1:
                    alignment_arguments, gold_predicates = next(iter(expected_matches.items()))
                    alignment_mode = "catalog_expected_arguments"
            # Disambiguate via NL evidence and the catalog; exclude ambiguous facts from automatic RMA/ACS/GER.
            if len(gold_predicates) > 1:
                possible_relations = sorted({
                    relation
                    for predicate in gold_predicates
                    for relation in [catalog.map_predicate(predicate)]
                    if relation
                })
                semantic_text = " ".join([
                    str(fact.get("relation_text", "")),
                    str(fact.get("evidence", "")),
                ]).strip()
                ranking = catalog.rank(semantic_text, possible_relations)
                best_score = ranking[0][1] if ranking else 0.0
                runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
                if ranking and best_score >= 3.0 and best_score - runner_up >= 0.5:
                    chosen_relation = ranking[0][0]
                    gold_predicates = {
                        predicate for predicate in gold_predicates
                        if catalog.map_predicate(predicate) == chosen_relation
                    }
                    gold_disambiguation = "source_text_catalog_unique"
                else:
                    gold_predicates = set()
                    gold_disambiguation = "ambiguous_reference_atoms_require_annotation"
            gold_relations = {catalog.map_predicate(x) for x in gold_predicates if catalog.map_predicate(x)}
            semantic_key = "predicate:" + "|".join(sorted(gold_predicates)) if gold_predicates else ""
            gold_source = "reference_pddl_derived"
        if not gold_predicates or not gold_relations:
            fact_reports.append({
                "fact_id": fact_id,
                "alignable": False,
                "reason": "no_gold_alignment",
                "air_arguments": list(air_arguments),
                "grounded_arguments": list(grounded_arguments),
                "alignment_mode": alignment_mode,
                "gold_disambiguation": gold_disambiguation,
            })
            continue
        alignable += 1
        selected_relation = str(fact.get("selected_relation") or "").lower()
        rma_eligible += 1
        relation_correct = selected_relation in gold_relations
        rma_correct += int(relation_correct)
        if selected_relation:
            decisions.append({"semantic_key": semantic_key, "predicted_relation": selected_relation})
        grounded = str(gir_fact.get("predicate", "")).lower()
        grounding_correct: Optional[bool] = None
        if relation_correct:
            ger_eligible += 1
            grounding_correct = (
                grounded in gold_predicates and grounded_arguments == alignment_arguments
            )
            ger_wrong += int(not grounding_correct)
        fact_reports.append({
            "fact_id": fact_id,
            "alignable": True,
            "gold_source": gold_source,
            "gold_predicates": sorted(gold_predicates),
            "gold_relations": sorted(gold_relations),
            "selected_relation": selected_relation,
            "relation_correct": relation_correct,
            "grounded_predicate": grounded,
            "air_arguments": list(air_arguments),
            "grounded_arguments": list(grounded_arguments),
            "gold_arguments": list(alignment_arguments),
            "argument_order_correct": grounded_arguments == alignment_arguments,
            "alignment_mode": alignment_mode,
            "gold_disambiguation": gold_disambiguation,
            "grounding_correct_conditional": grounding_correct,
            "semantic_key": semantic_key,
        })
    return {
        "RMA": rma_correct / rma_eligible if rma_eligible else None,
        "GER": ger_wrong / ger_eligible if ger_eligible else None,
        "rma_correct": rma_correct,
        "rma_eligible": rma_eligible,
        "ger_wrong": ger_wrong,
        "ger_eligible": ger_eligible,
        "alignable_facts": alignable,
        "air_facts": len(air.get("facts", [])),
        "alignment_rate": alignable / len(air.get("facts", [])) if air.get("facts") else None,
        "acs_decisions": decisions,
        "fact_reports": fact_reports,
        "abstraction_applicable": True,
    }


def consistency_score(decisions: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for decision in decisions:
        key = str(decision.get("semantic_key", ""))
        relation = str(decision.get("predicted_relation", ""))
        if key and relation:
            grouped[key][relation] += 1
    total = sum(sum(counts.values()) for counts in grouped.values())
    conflicts = sum(sum(counts.values()) - max(counts.values()) for counts in grouped.values())
    return {
        "ACS": 1.0 - conflicts / total if total else None,
        "acs_conflicts": conflicts,
        "acs_decisions": total,
        "acs_semantic_clusters": len(grouped),
        "cluster_distributions": {key: dict(sorted(counts.items())) for key, counts in sorted(grouped.items())},
    }
