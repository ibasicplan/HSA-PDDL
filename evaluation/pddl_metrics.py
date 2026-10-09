from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.pddl import atom_key, parse_domain, parse_problem, static_report, syntax_report, type_spec_from_json


def syntax_validity(problem_pddl: str) -> Dict[str, Any]:
    report = syntax_report(problem_pddl)
    return {"SVR": int(report["ok"]), "syntax_ok": report["ok"], "syntax_errors": report["errors"]}


def _same_type(left: Any, right: Any) -> bool:
    return type_spec_from_json(left) == type_spec_from_json(right)


def semantic_completeness(
    generated_pddl: str,
    reference_pddl: str,
    domain_pddl: str,
    generated_domain_pddl: Optional[str] = None,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    generated_syntax = syntax_report(generated_pddl)
    reference_syntax = syntax_report(reference_pddl)
    if not generated_syntax["ok"]:
        return {
            "SCR": 0,
            "semantic_complete": False,
            "semantic_errors": generated_syntax["errors"],
            "checks": {"missing_action": {"applicable": generated_domain_pddl is not None}},
        }
    if not reference_syntax["ok"]:
        raise ValueError(f"Reference PDDL is invalid: {reference_syntax['errors']}")
    domain = parse_domain(domain_pddl)
    generated = generated_syntax["parsed"]
    reference = reference_syntax["parsed"]

    generated_objects = generated.get("objects", {})
    reference_objects = reference.get("objects", {})
    for name in sorted(set(reference_objects) - set(generated_objects)):
        errors.append({"error_type": "missing_object", "object": name})
    for name in sorted(set(generated_objects) - set(reference_objects)):
        errors.append({"error_type": "extra_object", "object": name})
    for name in sorted(set(generated_objects) & set(reference_objects)):
        if not _same_type(generated_objects[name], reference_objects[name]):
            errors.append({
                "error_type": "wrong_type",
                "object": name,
                "expected": reference_objects[name],
                "observed": generated_objects[name],
            })

    static = static_report(domain, generated_pddl)
    errors.extend(x for x in static["errors"] if x.get("error_type") == "wrong_type")

    generated_atoms = {atom_key(x) for x in generated.get("atoms", [])}
    reference_atoms = {atom_key(x) for x in reference.get("atoms", [])}
    missing = reference_atoms - generated_atoms
    extra = generated_atoms - reference_atoms
    generated_by_stage_args: Dict[Tuple[str, Tuple[str, ...], bool], Set[str]] = defaultdict(set)
    generated_by_stage_pred: Dict[Tuple[str, str, bool], Set[Tuple[str, ...]]] = defaultdict(set)
    for stage, predicate, arguments, polarity in generated_atoms:
        generated_by_stage_args[(stage, arguments, polarity)].add(predicate)
        generated_by_stage_pred[(stage, predicate, polarity)].add(arguments)

    for stage, predicate, arguments, polarity in sorted(missing):
        alternative_predicates = generated_by_stage_args.get((stage, arguments, polarity), set())
        alternative_arguments = generated_by_stage_pred.get((stage, predicate, polarity), set())
        if alternative_predicates:
            error_type = "wrong_predicate"
        elif alternative_arguments:
            error_type = "wrong_parameter"
        else:
            error_type = "missing_predicate"
        errors.append({
            "error_type": error_type,
            "stage": stage,
            "expected_predicate": predicate,
            "expected_arguments": list(arguments),
            "polarity": polarity,
            "observed_predicates_same_arguments": sorted(alternative_predicates),
            "observed_arguments_same_predicate": [list(x) for x in sorted(alternative_arguments)],
        })
    for stage, predicate, arguments, polarity in sorted(extra):
        errors.append({
            "error_type": "extra_semantic_fact",
            "stage": stage,
            "predicate": predicate,
            "arguments": list(arguments),
            "polarity": polarity,
        })

    missing_action_check: Dict[str, Any]
    if generated_domain_pddl:
        generated_domain = parse_domain(generated_domain_pddl)
        missing_actions = sorted(set(domain.get("actions", {})) - set(generated_domain.get("actions", {})))
        missing_action_check = {"applicable": True, "missing_actions": missing_actions}
        for action in missing_actions:
            errors.append({"error_type": "missing_action", "action": action})
    else:
        missing_action_check = {
            "applicable": False,
            "reason": "The evaluated task generates only a PDDL problem; the domain and its actions are fixed inputs.",
        }

    counts = defaultdict(int)
    for error in errors:
        counts[str(error.get("error_type", "unknown"))] += 1
    return {
        "SCR": int(not errors),
        "semantic_complete": not errors,
        "semantic_errors": errors,
        "error_counts": dict(sorted(counts.items())),
        "checks": {
            "objects": {"generated": len(generated_objects), "reference": len(reference_objects)},
            "atoms": {"generated": len(generated_atoms), "reference": len(reference_atoms)},
            "missing_action": missing_action_check,
        },
    }


def planning_metrics(fd: Optional[Dict[str, Any]], val: Optional[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "PSR": int(bool(fd and fd.get("status") == "solution_found")),
        "VPR": int(bool(val and val.get("status") == "valid")),
    }


def evaluate_pddl(
    generated_pddl: str,
    reference_pddl: str,
    domain_pddl: str,
    fd: Optional[Dict[str, Any]],
    val: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    syntax = syntax_validity(generated_pddl)
    semantic = semantic_completeness(generated_pddl, reference_pddl, domain_pddl)
    planning = planning_metrics(fd, val)
    return {**syntax, **semantic, **planning}

