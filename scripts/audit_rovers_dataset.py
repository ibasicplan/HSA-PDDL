#!/usr/bin/env python3
"""Offline integrity audit for a flat Rovers corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from abstraction.relation_catalog import RelationCatalog, canonical_domain_name  # noqa: E402
from generation.eir_coverage import inspect_nl_coverage_contract  # noqa: E402
from scripts.build_rovers_catalog import discover_samples, full_domain_fingerprint  # noqa: E402
from utils.io import atomic_write_json  # noqa: E402
from utils.pddl import parse_domain, parse_problem, static_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Rovers data before the 2x2 model runs")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    samples = discover_samples(source)
    if not samples:
        raise SystemExit(f"No numeric Rovers sample directories found under {source}")
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    domains = set()
    fingerprints: Dict[str, List[str]] = defaultdict(list)
    object_counts: Counter[int] = Counter()
    init_counts: Counter[int] = Counter()
    goal_counts: Counter[int] = Counter()
    true_plan_count = 0

    catalog = RelationCatalog(args.catalog.resolve())
    required = ("domain.pddl", "domain.nl", "problem.nl", "problem.pddl")
    for sample in samples:
        missing = [name for name in required if not (sample / name).is_file()]
        if missing:
            failures.append({"sample_id": sample.name, "error": "missing_files", "files": missing})
            continue
        if (sample / "true.plan").is_file():
            true_plan_count += 1
        else:
            warnings.append({"sample_id": sample.name, "warning": "true_plan_missing"})
        try:
            domain_text = (sample / "domain.pddl").read_text(encoding="utf-8", errors="ignore")
            domain_nl = (sample / "domain.nl").read_text(encoding="utf-8", errors="ignore").strip()
            problem_nl = (sample / "problem.nl").read_text(encoding="utf-8", errors="ignore").strip()
            reference_text = (sample / "problem.pddl").read_text(encoding="utf-8", errors="ignore")
            domain = parse_domain(domain_text)
            reference = parse_problem(reference_text)
            catalog.validate_domain(domain)
            coverage = inspect_nl_coverage_contract(problem_nl, domain, catalog)
            static = static_report(domain, reference_text)
        except Exception as exc:
            failures.append({
                "sample_id": sample.name,
                "error": "parse_or_catalog_failure",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue
        domains.add(canonical_domain_name(domain["domain_name"]))
        fingerprints[full_domain_fingerprint(domain_text)].append(sample.name)
        if not domain_nl:
            failures.append({"sample_id": sample.name, "error": "empty_domain_nl"})
        if not problem_nl:
            failures.append({"sample_id": sample.name, "error": "empty_problem_nl"})
        if canonical_domain_name(reference["domain_name"]) != canonical_domain_name(domain["domain_name"]):
            failures.append({
                "sample_id": sample.name,
                "error": "problem_domain_mismatch",
                "domain": domain["domain_name"],
                "problem_domain": reference["domain_name"],
            })
        if not static["ok"]:
            failures.append({"sample_id": sample.name, "error": "invalid_reference_problem", "details": static["errors"][:20]})
        if not coverage["ok"]:
            failures.append({"sample_id": sample.name, "error": "nl_coverage_contract_failure", "details": coverage["errors"][:20]})
        object_counts[len(reference.get("objects", {}))] += 1
        init_counts[sum(atom["stage"] == "INIT" for atom in reference.get("atoms", []))] += 1
        goal_counts[sum(atom["stage"] == "GOAL" for atom in reference.get("atoms", []))] += 1

    if len(domains) != 1:
        failures.append({"error": "expected_one_domain_name", "domains": sorted(domains)})
    if len(fingerprints) != 1:
        failures.append({
            "error": "multiple_complete_domain_definitions",
            "variants": {fingerprint: ids[:10] for fingerprint, ids in fingerprints.items()},
        })
    report = {
        "status": "ok" if not failures else "failed",
        "source": str(source),
        "sample_count": len(samples),
        "sample_ids": [sample.name for sample in samples],
        "domain_names": sorted(domains),
        "complete_domain_fingerprints": sorted(fingerprints),
        "catalog": str(args.catalog.resolve()),
        "catalog_predicate_count": len(catalog.predicate_to_relation),
        "catalog_relation_count": len(catalog.relations),
        "catalog_eir_coverage_enabled": bool(catalog.eir_coverage.get("enabled", False)),
        "true_plan_present": true_plan_count,
        "reference_problem_pddl_used_for_offline_audit": True,
        "reference_problem_pddl_used_for_catalog": False,
        "reference_problem_pddl_used_at_inference": False,
        "reference_object_count_distribution": dict(sorted(object_counts.items())),
        "reference_init_count_distribution": dict(sorted(init_counts.items())),
        "reference_goal_count_distribution": dict(sorted(goal_counts.items())),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures[:100],
        "warnings": warnings[:100],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        atomic_write_json(args.json_output.resolve(), report)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
