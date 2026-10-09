#!/usr/bin/env python3
"""Build a frozen Rovers relation catalog from the domain schema (reference PDDL is never read)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from generation.eir_coverage import extract_fact_statements, extract_object_inventory  # noqa: E402
from utils.io import atomic_write_json, stable_hash  # noqa: E402
from utils.pddl import parse_domain, parse_sexpressions  # noqa: E402


GROUPS: Dict[str, Tuple[str, str]] = {
    "at": ("physical_location", "An entity is located at a waypoint."),
    "at_lander": ("physical_location", "An entity is located at a waypoint."),
    "available": ("rover_availability", "A rover is available for an operation."),
    "visible": ("visibility", "One waypoint or target is visible from another location."),
    "visible_from": ("visibility", "One waypoint or target is visible from another location."),
    "can_traverse": ("traversability", "A rover can traverse between two waypoints."),
    "equipped_for_soil_analysis": ("analysis_equipment", "A rover has equipment for material analysis."),
    "equipped_for_rock_analysis": ("analysis_equipment", "A rover has equipment for material analysis."),
    "equipped_for_imaging": ("imaging_equipment", "A rover has equipment for imaging."),
    "store_of": ("store_assignment", "A sample store belongs to a rover."),
    "empty": ("store_state", "A sample store has an empty or full capacity state."),
    "full": ("store_state", "A sample store has an empty or full capacity state."),
    "at_soil_sample": ("sample_presence", "A soil or rock sample is present at a waypoint."),
    "at_rock_sample": ("sample_presence", "A soil or rock sample is present at a waypoint."),
    "have_soil_analysis": ("analysis_acquired", "A rover has acquired analysis data at a waypoint."),
    "have_rock_analysis": ("analysis_acquired", "A rover has acquired analysis data at a waypoint."),
    "have_image": ("image_acquired", "A rover has acquired an image in an imaging mode."),
    "calibrated": ("camera_calibration", "A camera is calibrated for use by a rover."),
    "supports": ("camera_support", "A camera supports an imaging mode."),
    "calibration_target": (
        "calibration_target",
        "A camera uses an objective as its calibration target.",
    ),
    "on_board": (
        "camera_mounting",
        "A camera is mounted on and carried by a rover.",
    ),
    "channel_free": ("communication_channel", "A lander's communication channel is free."),
    "communicated_soil_data": ("analysis_communicated", "Analysis data for a waypoint has been communicated."),
    "communicated_rock_data": ("analysis_communicated", "Analysis data for a waypoint has been communicated."),
    "communicated_image_data": ("image_communicated", "Image data for an objective and mode has been communicated."),
}


ALIASES: Dict[str, List[str]] = {
    "at": ["at", "located at", "positioned at"],
    "at_lander": ["lander at", "lander located at"],
    "available": ["available", "rover available"],
    "visible": ["visible", "waypoint visible from waypoint"],
    "visible_from": ["visible from", "objective visible from waypoint"],
    "can_traverse": ["can traverse", "traversable", "can navigate from to"],
    "equipped_for_soil_analysis": ["equipped for soil analysis", "soil analysis equipment"],
    "equipped_for_rock_analysis": ["equipped for rock analysis", "rock analysis equipment"],
    "equipped_for_imaging": ["equipped for imaging", "imaging equipment"],
    "store_of": ["store of", "sample store belongs to rover"],
    "empty": ["empty", "store is empty"],
    "full": ["full", "store is full"],
    "at_soil_sample": ["soil sample at", "soil sample present at waypoint"],
    "at_rock_sample": ["rock sample at", "rock sample present at waypoint"],
    "have_soil_analysis": ["have soil analysis", "soil analysis acquired"],
    "have_rock_analysis": ["have rock analysis", "rock analysis acquired"],
    "have_image": ["have image", "image acquired"],
    "calibrated": ["calibrated", "camera calibrated"],
    "supports": [
        "supports",
        "supports imaging mode",
        "camera supports mode",
        "compatible imaging mode",
    ],
    "calibration_target": [
        "calibration target",
        "camera calibration target",
        "objective used to calibrate camera",
    ],
    "on_board": [
        "on board",
        "camera on board rover",
        "camera is on rover",
        "camera mounted on rover",
        "camera installed on rover",
        "carried by rover",
    ],
    "channel_free": ["channel free", "communication channel is free"],
    "communicated_soil_data": ["communicated soil data", "soil data communicated"],
    "communicated_rock_data": ["communicated rock data", "rock data communicated"],
    "communicated_image_data": ["communicated image data", "image data communicated"],
}


def numeric_key(path: Path) -> Tuple[int, Any, str]:
    return (0, int(path.name), path.name) if path.name.isdigit() else (1, path.name, path.name)


def discover_samples(source: Path) -> List[Path]:
    return sorted(
        (path for path in source.iterdir() if path.is_dir() and path.name.isdigit()),
        key=numeric_key,
    ) if source.is_dir() else []


def normalized_predicate(name: str) -> str:
    return str(name).lower().replace("-", "_")


def full_domain_fingerprint(text: str) -> str:
    """Hash the complete comment-insensitive PDDL S-expression tree."""
    return stable_hash(parse_sexpressions(text))


def argument_roles(signature: Dict[str, Any]) -> List[str]:
    raw_roles: List[str] = []
    for parameter in signature.get("parameters", []):
        typ = parameter.get("type", "object")
        if isinstance(typ, list):
            label = "_or_".join(str(item) for item in typ) or "object"
        else:
            label = str(typ or "object")
        if label == "object":
            label = str(parameter.get("name", "argument")).lstrip("?") or "argument"
        raw_roles.append(label.replace("-", "_"))
    totals = Counter(raw_roles)
    seen: Counter[str] = Counter()
    roles: List[str] = []
    for role in raw_roles:
        seen[role] += 1
        roles.append(f"{role}_{seen[role]}" if totals[role] > 1 else role)
    return roles


def inference_contract_ready(problem_nl: str, domain: Dict[str, Any]) -> Tuple[bool, List[str]]:
    inventory = extract_object_inventory(problem_nl)
    statements = extract_fact_statements(problem_nl, inventory.get("objects", []))
    reasons: List[str] = []
    if not inventory.get("available"):
        reasons.append("object_inventory_unavailable")
    elif not inventory.get("count_matches_list", True):
        reasons.append("object_inventory_count_mismatch")
    if not statements.get("init_available"):
        reasons.append("init_assertions_unavailable")
    if not statements.get("goal_available"):
        reasons.append("goal_assertions_unavailable")
    valid_types = {"object", *[str(name).lower() for name in domain.get("type_hierarchy", {})]}
    for name, value in domain.get("constants", {}).items():
        del name
        valid_types.add(str(value[0] if isinstance(value, list) and value else value).lower())
    for name, typ in inventory.get("object_types", {}).items():
        if str(typ).lower() not in valid_types:
            reasons.append(f"unknown_type:{name}:{typ}")
    return not reasons, reasons


def build_relations(domain: Dict[str, Any]) -> Dict[str, Any]:
    """Build complete semantic relation entries for one parsed domain schema."""
    relation_members: Dict[str, List[str]] = defaultdict(list)
    relation_descriptions: Dict[str, str] = {}
    for predicate in sorted(domain.get("predicates", {})):
        normalized = normalized_predicate(predicate)
        relation_id, description = GROUPS.get(
            normalized,
            (f"schema_{normalized}", f"The semantic relation represented by schema predicate {predicate}."),
        )
        relation_members[relation_id].append(predicate)
        relation_descriptions[relation_id] = description

    relations: Dict[str, Any] = {}
    for relation_id in sorted(relation_members):
        predicates = relation_members[relation_id]
        aliases: List[str] = []
        predicate_specs: List[Dict[str, Any]] = []
        for predicate in predicates:
            normalized = normalized_predicate(predicate)
            candidates = [predicate, predicate.replace("-", " ").replace("_", " ")]
            candidates.extend(ALIASES.get(normalized, []))
            for alias in candidates:
                clean = " ".join(str(alias).lower().split())
                if clean and clean not in aliases:
                    aliases.append(clean)
            signature = domain["predicates"][predicate]
            roles = argument_roles(signature)
            slot_text = " ".join(roles)
            predicate_specs.append({
                "name": predicate,
                "argument_permutation": list(range(int(signature.get("arity", 0)))),
                "argument_roles": roles,
                "direct_pddl_instruction": (
                    f"Use ({predicate}{(' ' + slot_text) if slot_text else ''}) with arguments in this exact "
                    "domain-declared slot order."
                ),
            })
        relations[relation_id] = {
            "description": relation_descriptions[relation_id],
            "aliases": aliases,
            "patterns": [],
            "predicates": predicate_specs,
        }
    return relations


def build_catalog(source: Path, coverage_policy: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples = discover_samples(source)
    if not samples:
        raise ValueError(f"No numeric Rovers sample directories found under {source}")

    required = ("domain.pddl", "domain.nl", "problem.nl", "problem.pddl")
    missing_rows = []
    domains: List[Dict[str, Any]] = []
    domain_fingerprints: Dict[str, List[str]] = defaultdict(list)
    contract_failures: List[Dict[str, Any]] = []
    for sample in samples:
        missing = [name for name in required if not (sample / name).is_file()]
        if missing:
            missing_rows.append({"sample_id": sample.name, "missing": missing})
            continue
        domain_text = (sample / "domain.pddl").read_text(encoding="utf-8", errors="ignore")
        domain = parse_domain(domain_text)
        domains.append(domain)
        domain_fingerprints[full_domain_fingerprint(domain_text)].append(sample.name)
        problem_nl = (sample / "problem.nl").read_text(encoding="utf-8", errors="ignore")
        ready, reasons = inference_contract_ready(problem_nl, domain)
        if not ready:
            contract_failures.append({"sample_id": sample.name, "reasons": reasons})
    if missing_rows:
        raise ValueError(f"Incomplete Rovers samples: {json.dumps(missing_rows[:20], ensure_ascii=False)}")
    if len(domain_fingerprints) != 1:
        variants = {key: ids[:10] for key, ids in domain_fingerprints.items()}
        raise ValueError(
            "Rovers samples contain multiple complete domain.pddl definitions; "
            f"one 2x2 run must use one fixed domain. Variants={variants}"
        )

    domain = domains[0]
    relations = build_relations(domain)

    all_contracts_ready = not contract_failures
    if coverage_policy == "on" and not all_contracts_ready:
        raise ValueError(
            "Strict EIR coverage was requested, but Problem-NL parsing is unsupported for some samples: "
            + json.dumps(contract_failures[:20], ensure_ascii=False)
        )
    coverage_enabled = coverage_policy == "on" or (
        coverage_policy == "auto" and all_contracts_ready
    )
    domain_fingerprint = next(iter(domain_fingerprints))
    catalog = {
        "schema_version": "1.1",
        "domain": domain["domain_name"],
        "offline": True,
        "provenance": {
            "builder": "scripts/build_rovers_catalog.py",
            "builder_version": "1.1",
            "source_domain_fingerprint": domain_fingerprint,
            "source_sample_count": len(samples),
            "reference_problem_pddl_used": False,
            "true_plan_used": False,
            "coverage_policy_requested": coverage_policy,
            "coverage_policy_enabled": coverage_enabled,
            "unsupported_nl_contract_count": len(contract_failures),
        },
        "eir_coverage": {
            "enabled": coverage_enabled,
            "strict_object_inventory": True,
            "strict_fact_statement_counts": True,
            "require_untyped_classification_facts": False,
        },
        "eir_instructions": [
            "Preserve the exact Rovers entity types stated in the problem inventory.",
            "Keep soil analysis, rock analysis, imaging, calibration, visibility, and communication facts semantically distinct.",
            "For camera-mode compatibility, keep semantic arguments in [camera, imaging mode] order.",
            "For a camera calibration-target relation, keep semantic arguments in [camera, objective] order.",
            "For a camera mounted on a rover, keep semantic arguments in [camera, rover] order.",
            "The three canonical camera role orders above override surface mention order; for all other relations, keep natural-language mention order.",
        ],
        "direct_pddl_instructions": [
            "Use only the predicates and object types declared by the supplied Rovers domain PDDL.",
            "Do not add static connectivity, visibility, equipment, store, or camera facts unless explicitly stated in Problem NL.",
        ],
        "relations": relations,
    }
    diagnostics = {
        "sample_count": len(samples),
        "domain": domain["domain_name"],
        "domain_fingerprint": domain_fingerprint,
        "predicate_count": len(domain.get("predicates", {})),
        "relation_count": len(relations),
        "coverage_enabled": coverage_enabled,
        "nl_contract_failure_count": len(contract_failures),
        "nl_contract_failures": contract_failures[:20],
    }
    return catalog, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen catalog for the actual Rovers PDDL schema")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-policy", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args()

    catalog, diagnostics = build_catalog(args.source.resolve(), args.coverage_policy)
    atomic_write_json(args.output.resolve(), catalog)
    print(json.dumps({"status": "ok", "catalog": str(args.output.resolve()), **diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
