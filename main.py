#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from abstraction.air_builder import apply_relation_mapping, build_air, unresolved_facts, validate_air
from abstraction.grounding import (
    build_gir,
    deterministic_grounding,
    resolve_object_types,
    validate_gir,
)
from abstraction.relation_catalog import RelationCatalog, canonical_domain_name
from config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_FD_ENTRY,
    DEFAULT_FD_SEARCH,
    DEFAULT_FD_TIMEOUT_S,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RESULTS,
    DEFAULT_SEED,
    DEFAULT_SFT_MODEL,
    DEFAULT_VAL_BIN,
    DEFAULT_VAL_TIMEOUT_S,
    EXPERIMENTS,
    FRAMEWORK_VERSION,
    experiment_spec,
)
from evaluation.abstraction_metrics import evaluate_abstraction
from evaluation.pddl_metrics import evaluate_pddl
from generation.eir_coverage import evaluate_eir_coverage, inspect_nl_coverage_contract
from generation.eir_generator import EIRGenerator, validate_eir
from generation.local_llm import LocalLLM
from generation.pddl_generator import PDDLGenerator, parse_grounding_mapping, parse_relation_mapping
from generation.prompts import GROUNDING_SYSTEM_PROMPT, RELATION_SYSTEM_PROMPT
from planner.fd_runner import FastDownwardRunner
from planner.val_runner import VALRunner
from utils.io import append_jsonl, atomic_write_json, atomic_write_text, ensure_dir, sha256_file, stable_hash
from utils.pddl import domain_uses_typing, parse_domain, render_problem, static_report, syntax_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-OSS-20B full 2x2 SFT x HSA ablation for reliable PDDL generation")
    parser.add_argument("--framework_version", action="store_true")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--data_root", default=DEFAULT_DATA)
    parser.add_argument("--output", default=None)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft_model", default=DEFAULT_SFT_MODEL)
    parser.add_argument(
        "--model", default=None,
        help="Explicit single-run override. Do not use this in the official 2x2 runner.",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--annotations_root", default=None)
    parser.add_argument("--require_gold_annotations", action="store_true")
    parser.add_argument("--split", choices=["easy", "hard", "all"], default="all")
    parser.add_argument("--begin_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--generator_backend", choices=["transformers", "mock"], default="transformers")
    parser.add_argument("--planner_backend", choices=["real", "mock"], default="real")
    parser.add_argument("--val_backend", choices=["real", "mock"], default="real")
    parser.add_argument("--mock_response_dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max_context_tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--eir_max_new_tokens", type=int, default=int(os.environ.get("HSA_EIR_MAX_NEW_TOKENS", "4096")))
    parser.add_argument("--relation_max_new_tokens", type=int, default=int(os.environ.get("HSA_RELATION_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--grounding_max_new_tokens", type=int, default=int(os.environ.get("HSA_GROUNDING_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--catalog_score_threshold", type=float, default=3.0)
    parser.add_argument("--catalog_score_margin", type=float, default=0.5)

    parser.add_argument("--fd_entry", default=DEFAULT_FD_ENTRY)
    parser.add_argument("--fd_search", default=DEFAULT_FD_SEARCH)
    parser.add_argument("--fd_timeout_s", type=int, default=DEFAULT_FD_TIMEOUT_S)
    parser.add_argument("--val_bin", default=DEFAULT_VAL_BIN)
    parser.add_argument("--val_timeout_s", type=int, default=DEFAULT_VAL_TIMEOUT_S)
    return parser.parse_args()


def dataset_partitions(data_root: Path, split: str) -> List[Tuple[str, Path]]:
    """Resolve ``{easy,hard}/<id>`` or flat ``<id>`` layouts (flat requires ``split=all``)."""
    split_dirs = {name: data_root / name for name in ("easy", "hard")}
    has_split_layout = any(path.is_dir() for path in split_dirs.values())
    if has_split_layout:
        requested = ("easy", "hard") if split == "all" else (split,)
        missing = [name for name in requested if not split_dirs[name].is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing dataset split(s) {missing} under split-layout root: {data_root}"
            )
        return [(name, split_dirs[name]) for name in requested]
    if split != "all":
        raise FileNotFoundError(
            f"Flat dataset root {data_root} has no '{split}' split; use --split all"
        )
    return [("all", data_root)]


def load_dataset(data_root: Path, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, split_dir in dataset_partitions(data_root, split):
        sample_dirs = [x for x in split_dir.iterdir() if x.is_dir()]
        sample_dirs.sort(key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name))
        for sample_dir in sample_dirs:
            required = {
                "domain_pddl": sample_dir / "domain.pddl",
                "problem_nl": sample_dir / "problem.nl",
                "reference_pddl": sample_dir / "problem.pddl",
            }
            if not all(path.is_file() for path in required.values()):
                continue
            domain_nl_path = sample_dir / "domain.nl"
            rows.append({
                "split": split_name,
                "sample_id": sample_dir.name,
                "sample_key": f"{split_name}/{sample_dir.name}",
                "source_dir": str(sample_dir),
                "domain_pddl": required["domain_pddl"].read_text(encoding="utf-8", errors="ignore"),
                "domain_nl": domain_nl_path.read_text(encoding="utf-8", errors="ignore") if domain_nl_path.is_file() else "",
                "problem_nl": required["problem_nl"].read_text(encoding="utf-8", errors="ignore"),
                "reference_pddl": required["reference_pddl"].read_text(encoding="utf-8", errors="ignore"),
            })
    return rows


def select_items(items: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for position, item in enumerate(items, 1):
        if args.begin_idx is not None and position < args.begin_idx:
            continue
        if args.end_idx is not None and position > args.end_idx:
            break
        output.append(item)
        if args.max_samples is not None and len(output) >= args.max_samples:
            break
    return output


def relation_prompt(air: Dict[str, Any], catalog: RelationCatalog, feedback: Optional[Dict[str, Any]]) -> str:
    unresolved = unresolved_facts(air)
    relation_ids = sorted({rid for fact in unresolved for rid in fact.get("candidate_relations", [])})
    facts = [
        {
            "fact_id": fact["fact_id"],
            "stage": fact["stage"],
            "relation_text": fact["relation_text"],
            "arguments": fact["arguments"],
            "evidence": fact["evidence"],
            "allowed_relations": fact["candidate_relations"],
        }
        for fact in unresolved
    ]
    return f"""Offline semantic relation catalog:
{catalog.descriptions(relation_ids, expose_predicates=False)}

Unresolved facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Previous feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else 'none'}

Output one line per fact, then END_REL:
REL|F1|one_allowed_relation
END_REL
"""


def grounding_prompt(facts: Sequence[Dict[str, Any]], feedback: Optional[Dict[str, Any]]) -> str:
    payload = [
        {
            "fact_id": fact["fact_id"],
            "stage": fact["stage"],
            "relation_text": fact["relation_text"],
            "selected_relation": fact["selected_relation"],
            "arguments": fact["arguments"],
            "allowed_predicates": fact["candidate_predicates"],
            "evidence": fact["evidence"],
        }
        for fact in facts
    ]
    return f"""Ground each fact using only its allowed predicate whitelist:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Previous feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else 'none'}

Output one line per fact, then END_MAP:
MAP|F1|one_allowed_predicate
END_MAP
"""


def feedback_from_validation(
    syntax: Dict[str, Any],
    static: Dict[str, Any],
    fd: Optional[Dict[str, Any]],
    val: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not syntax.get("ok"):
        return {"source": "syntax", "repair_layer": "generation", "errors": syntax.get("errors", [])}
    if not static.get("ok"):
        return {"source": "static", "repair_layer": "generation", "errors": static.get("errors", [])[:20]}
    if not fd or fd.get("status") != "solution_found":
        return {
            "source": "fast_downward",
            "repair_layer": "generation",
            "status": (fd or {}).get("status", "not_run"),
            "raw_excerpt": (fd or {}).get("raw_excerpt", "")[-1500:],
        }
    if not val or val.get("status") != "valid":
        return {
            "source": "VAL",
            "repair_layer": "generation",
            "status": (val or {}).get("status", "not_run"),
            "raw_excerpt": (val or {}).get("raw_excerpt", "")[-1500:],
        }
    return {"source": "validation", "repair_layer": "none", "errors": []}


def summarize_generation_usage(stats_rows: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = [x for x in stats_rows if isinstance(x, dict)]
    input_tokens = sum(int(x.get("input_tokens") or 0) for x in rows)
    output_tokens = sum(int(x.get("output_tokens") or x.get("generated_tokens") or 0) for x in rows)
    return {
        "logical_generation_calls": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "generation_time_s": round(sum(float(x.get("generation_time_s") or 0.0) for x in rows), 4),
    }


def validate_candidate(
    pddl: str,
    domain: Dict[str, Any],
    domain_path: Path,
    candidate_path: Path,
    attempt_dir: Path,
    fd_runner: FastDownwardRunner,
    val_runner: VALRunner,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    syntax = syntax_report(pddl) if pddl else {
        "ok": False,
        "errors": [{"error_type": "missing_complete_pddl"}],
        "parsed": None,
    }
    static = static_report(domain, pddl) if syntax["ok"] else {"ok": False, "errors": []}
    fd: Optional[Dict[str, Any]] = None
    val: Optional[Dict[str, Any]] = None
    if syntax["ok"] and static["ok"]:
        atomic_write_text(candidate_path, pddl + ("\n" if not pddl.endswith("\n") else ""))
        fd = fd_runner.run(domain_path, candidate_path, attempt_dir)
        if fd.get("status") == "solution_found":
            plan_path = Path(fd["plan_path"]) if fd.get("plan_path") else None
            val = val_runner.run(domain_path, candidate_path, plan_path, attempt_dir)
    return syntax, static, fd, val


def run_direct_sample(
    item: Dict[str, Any],
    sample_dir: Path,
    llm: Any,
    domain: Dict[str, Any],
    catalog: Optional[RelationCatalog],
    domain_path: Path,
    fd_runner: FastDownwardRunner,
    val_runner: VALRunner,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    generator = PDDLGenerator(
        llm,
        args.max_new_tokens,
        schema_guidance=catalog.direct_pddl_guidance() if catalog else "",
    )
    feedback: Optional[Dict[str, Any]] = None
    generation_stats: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {"pddl": "", "fd": None, "val": None, "attempt": None, "status": "not_started"}
    for attempt in range(1, args.max_attempts + 1):
        attempt_dir = ensure_dir(sample_dir / f"attempt_{attempt}")
        generated = generator.generate(
            item["domain_pddl"], item["domain_nl"], item["problem_nl"],
            feedback, item["sample_key"], attempt,
        )
        generation_stats.append(generated["generation"]["stats"])
        atomic_write_text(attempt_dir / "prompt.txt", generated["prompt"])
        atomic_write_text(attempt_dir / "raw_output.txt", generated["generation"]["raw_output"])
        pddl = generated["pddl"]
        syntax, static, fd, val = validate_candidate(
            pddl, domain, domain_path, attempt_dir / "problem.pddl", attempt_dir, fd_runner, val_runner
        )
        feedback = feedback_from_validation(syntax, static, fd, val)
        attempt_report = {
            "attempt": attempt,
            "mode": "direct_pddl",
            "generation_stats": generated["generation"]["stats"],
            "syntax": {"ok": syntax["ok"], "errors": syntax["errors"]},
            "static": static,
            "fd": fd,
            "val": val,
            "feedback": feedback,
        }
        atomic_write_json(attempt_dir / "attempt_report.json", attempt_report)
        final = {
            "pddl": pddl,
            "fd": fd,
            "val": val,
            "attempt": attempt,
            "status": "val_valid" if val and val.get("status") == "valid" else feedback.get("source", "failed"),
            "eir": None,
            "air": None,
            "gir": None,
            "provenance": {"mode": "direct_pddl"},
            "generation_usage": summarize_generation_usage(generation_stats),
        }
        if val and val.get("status") == "valid":
            break
        if fd and fd.get("status") == "timeout":
            final["status"] = "planner_timeout_terminal"
            break
    final["generation_usage"] = summarize_generation_usage(generation_stats)
    return final


def run_structured_sample(
    item: Dict[str, Any],
    sample_dir: Path,
    llm: Any,
    domain: Dict[str, Any],
    catalog: RelationCatalog,
    domain_path: Path,
    fd_runner: FastDownwardRunner,
    val_runner: VALRunner,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    eir_generator = EIRGenerator(llm, args.eir_max_new_tokens, catalog.eir_guidance())
    feedback: Optional[Dict[str, Any]] = None
    generation_stats: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {"pddl": "", "fd": None, "val": None, "attempt": None, "status": "not_started"}
    for attempt in range(1, args.max_attempts + 1):
        attempt_dir = ensure_dir(sample_dir / f"attempt_{attempt}")
        generated = eir_generator.generate(
            item["domain_pddl"], item["domain_nl"], item["problem_nl"],
            feedback, item["sample_key"], attempt,
        )
        generation_stats.append(generated["generation"]["stats"])
        atomic_write_text(attempt_dir / "eir_prompt.txt", generated["prompt"])
        atomic_write_text(attempt_dir / "eir_raw.txt", generated["generation"]["raw_output"])
        provenance = {
            "mode": "semantic_abstraction",
            "eir_source": generated["source"],
            "eir_parse_status": generated["parse_status"],
            "detected_output_format": generated.get("detected_output_format"),
            "pddl_eir_adapter_enabled": False,
            "pddl_eir_adapter_triggered": generated["adapter_triggered"],
        }
        eir = generated["eir"]
        eir_errors = validate_eir(eir, domain) if eir else [{"error_type": "eir_parse_failed", "status": generated["parse_status"]}]
        if eir_errors:
            feedback = {"source": "eir", "repair_layer": "eir", "errors": eir_errors}
            atomic_write_json(attempt_dir / "attempt_report.json", {"attempt": attempt, "provenance": provenance, "feedback": feedback})
            final = {
                "pddl": "", "fd": None, "val": None, "attempt": attempt,
                "status": "eir_failed", "eir": eir, "air": None, "gir": None,
                "coverage": None, "provenance": provenance,
                "pipeline": {
                    "eir_parse_ok": 0, "eir_coverage_ok": None,
                    "eir_ok": 0, "air_ok": 0, "gir_ok": 0,
                },
            }
            continue
        atomic_write_json(attempt_dir / "eir.json", eir)

        air = build_air(
            eir, domain, catalog,
            score_threshold=args.catalog_score_threshold,
            score_margin=args.catalog_score_margin,
        )
        unresolved = unresolved_facts(air)
        relation_generation = None
        if unresolved:
            prompt = relation_prompt(air, catalog, feedback)
            relation_generation = llm.generate(
                "relation", RELATION_SYSTEM_PROMPT, prompt, args.relation_max_new_tokens,
                item["sample_key"], attempt,
            )
            generation_stats.append(relation_generation["stats"])
            atomic_write_text(attempt_dir / "relation_prompt.txt", prompt)
            atomic_write_text(attempt_dir / "relation_raw.txt", relation_generation["raw_output"])
            mapping = parse_relation_mapping(relation_generation["clean_output"])
            air = apply_relation_mapping(air, mapping, domain, catalog)
        air_errors = validate_air(air)
        atomic_write_json(attempt_dir / "air.json", air)
        if air_errors:
            feedback = {"source": "air", "repair_layer": "relation", "errors": air_errors}
            atomic_write_json(attempt_dir / "attempt_report.json", {"attempt": attempt, "provenance": provenance, "feedback": feedback})
            final = {
                "pddl": "", "fd": None, "val": None, "attempt": attempt,
                "status": "air_failed", "eir": eir, "air": air, "gir": None,
                "coverage": None, "provenance": provenance,
                "pipeline": {
                    "eir_parse_ok": 1, "eir_coverage_ok": None,
                    "eir_ok": 1, "air_ok": 0, "gir_ok": 0,
                },
            }
            continue

        coverage = evaluate_eir_coverage(item["problem_nl"], eir, air, domain, catalog)
        atomic_write_json(attempt_dir / "eir_coverage.json", coverage)
        if not coverage["ok"]:
            feedback = {
                "source": "eir_coverage",
                "repair_layer": "eir",
                "instruction": (
                    "Regenerate the complete EIR from Problem NL. In an untyped STRIPS domain, "
                    "OBJECT type labels do not replace unary classification FACT lines."
                ),
                "coverage_summary": {
                    "object_inventory": coverage.get("object_inventory"),
                    "fact_statement_counts": coverage.get("fact_statement_counts"),
                    "classification_facts": coverage.get("classification_facts"),
                },
                "errors": coverage["errors"],
            }
            atomic_write_json(attempt_dir / "attempt_report.json", {
                "attempt": attempt,
                "mode": "semantic_abstraction",
                "provenance": provenance,
                "pipeline": {
                    "eir_parse_ok": 1, "eir_coverage_ok": 0,
                    "eir_ok": 0, "air_ok": 1, "gir_ok": 0,
                },
                "eir_generation_stats": generated["generation"]["stats"],
                "relation_generation_stats": (relation_generation or {}).get("stats"),
                "eir_coverage": coverage,
                "feedback": feedback,
                "fd": None,
                "val": None,
            })
            final = {
                "pddl": "", "fd": None, "val": None, "attempt": attempt,
                "status": "eir_coverage_failed", "eir": eir, "air": air,
                "gir": None, "coverage": coverage, "provenance": provenance,
                "pipeline": {
                    "eir_parse_ok": 1, "eir_coverage_ok": 0,
                    "eir_ok": 0, "air_ok": 1, "gir_ok": 0,
                },
            }
            continue

        coverage_value = 1 if coverage.get("applicable") else None

        grounding_map, ambiguous = deterministic_grounding(air)
        grounding_generation = None
        if ambiguous:
            prompt = grounding_prompt(ambiguous, feedback)
            grounding_generation = llm.generate(
                "grounding", GROUNDING_SYSTEM_PROMPT, prompt, args.grounding_max_new_tokens,
                item["sample_key"], attempt,
            )
            generation_stats.append(grounding_generation["stats"])
            atomic_write_text(attempt_dir / "grounding_prompt.txt", prompt)
            atomic_write_text(attempt_dir / "grounding_raw.txt", grounding_generation["raw_output"])
            grounding_map.update(parse_grounding_mapping(grounding_generation["clean_output"]))
        gir = build_gir(air, grounding_map, catalog)
        gir_errors = validate_gir(gir, domain)
        atomic_write_json(attempt_dir / "gir.json", gir)
        if gir_errors:
            feedback = {"source": "gir", "repair_layer": "grounding", "errors": gir_errors}
            atomic_write_json(attempt_dir / "attempt_report.json", {"attempt": attempt, "provenance": provenance, "feedback": feedback})
            final = {
                "pddl": "", "fd": None, "val": None, "attempt": attempt,
                "status": "grounding_failed", "eir": eir, "air": air, "gir": gir,
                "coverage": coverage, "provenance": provenance,
                "pipeline": {
                    "eir_parse_ok": 1, "eir_coverage_ok": coverage_value,
                    "eir_ok": 1, "air_ok": 1, "gir_ok": 0,
                },
            }
            continue

        resolved_objects = resolve_object_types(gir, domain)
        domain_constants = {str(name).lower() for name in domain.get("constants", {})}
        resolved_objects = [
            obj for obj in resolved_objects
            if str(obj.get("name", "")).lower() not in domain_constants
        ]
        pddl = render_problem(
            f"omsa-{item['split']}-{item['sample_id']}",
            domain["domain_name"],
            resolved_objects,
            gir["facts"],
            typed_objects=domain_uses_typing(domain),
        )
        syntax, static, fd, val = validate_candidate(
            pddl, domain, domain_path, attempt_dir / "problem.pddl", attempt_dir, fd_runner, val_runner
        )
        feedback = feedback_from_validation(syntax, static, fd, val)
        attempt_report = {
            "attempt": attempt,
            "mode": "semantic_abstraction",
            "provenance": provenance,
            "pipeline": {
                "eir_parse_ok": 1, "eir_coverage_ok": coverage_value,
                "eir_ok": 1, "air_ok": 1, "gir_ok": 1,
            },
            "eir_coverage": coverage,
            "eir_generation_stats": generated["generation"]["stats"],
            "relation_generation_stats": (relation_generation or {}).get("stats"),
            "grounding_generation_stats": (grounding_generation or {}).get("stats"),
            "syntax": {"ok": syntax["ok"], "errors": syntax["errors"]},
            "static": static,
            "fd": fd,
            "val": val,
            "feedback": feedback,
        }
        atomic_write_json(attempt_dir / "attempt_report.json", attempt_report)
        final = {
            "pddl": pddl,
            "fd": fd,
            "val": val,
            "attempt": attempt,
            "status": "val_valid" if val and val.get("status") == "valid" else feedback.get("source", "failed"),
            "eir": eir,
            "air": air,
            "gir": gir,
            "coverage": coverage,
            "provenance": provenance,
            "pipeline": {
                "eir_parse_ok": 1, "eir_coverage_ok": coverage_value,
                "eir_ok": 1, "air_ok": 1, "gir_ok": 1,
            },
            "generation_usage": summarize_generation_usage(generation_stats),
        }
        if val and val.get("status") == "valid":
            break
        if fd and fd.get("status") == "timeout":
            final["status"] = "planner_timeout_terminal"
            break
    final["generation_usage"] = summarize_generation_usage(generation_stats)
    return final


def build_manifest(
    args: argparse.Namespace,
    spec: Any,
    data_root: Path,
    catalog_path: Path,
) -> Dict[str, Any]:
    dataset_files = []
    for _, split_dir in dataset_partitions(data_root, args.split):
        sample_dirs = sorted(
            (path for path in split_dir.iterdir() if path.is_dir()),
            key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name),
        )
        for sample_dir in sample_dirs:
            for path in sorted(sample_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".pddl", ".nl"}:
                    dataset_files.append({
                        "relative_path": str(path.relative_to(data_root)),
                        "sha256": sha256_file(path),
                    })
    annotation_files = []
    if args.annotations_root:
        annotations_root = Path(args.annotations_root).resolve()
        annotation_files = [
            {"relative_path": str(path.relative_to(annotations_root)), "sha256": sha256_file(path)}
            for path in sorted(annotations_root.glob("*/*.json")) if path.is_file()
        ]
    config = {
        "framework_version": FRAMEWORK_VERSION,
        "experiment": spec.name,
        "experiment_id": spec.experiment_id,
        "use_memory": spec.use_memory,
        "use_abstraction": spec.use_abstraction,
        "llm_source": spec.llm_source,
        "data_root": str(data_root),
        "dataset_fingerprint": stable_hash(dataset_files),
        "dataset_file_count": len(dataset_files),
        "split": args.split,
        "begin_idx": args.begin_idx,
        "end_idx": args.end_idx,
        "max_samples": args.max_samples,
        "factor_sft": int(spec.use_sft),
        "factor_hsa": int(spec.use_abstraction),
        "model_variant": "sft" if spec.use_sft else "base",
        "model": str(Path(args.model).resolve()),
        "local_model": str(Path(args.model).resolve()),
        "base_model": str(Path(args.base_model).resolve()) if args.base_model else None,
        "sft_model": str(Path(args.sft_model).resolve()) if args.sft_model else None,
        "catalog": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_usage": "hsa_grounding" if spec.use_abstraction else "flattened_direct_schema_guidance",
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "eir_max_new_tokens": args.eir_max_new_tokens,
        "relation_max_new_tokens": args.relation_max_new_tokens,
        "grounding_max_new_tokens": args.grounding_max_new_tokens,
        "generator_backend": args.generator_backend,
        "mock_response_dir": str(Path(args.mock_response_dir).resolve()) if args.mock_response_dir else None,
        "planner_backend": args.planner_backend,
        "val_backend": args.val_backend,
        "fd_entry": str(Path(args.fd_entry).resolve()),
        "fd_search": args.fd_search,
        "fd_timeout_s": args.fd_timeout_s,
        "val_bin": str(Path(args.val_bin).resolve()),
        "val_timeout_s": args.val_timeout_s,
        "native_eir_only": True,
        "catalog_score_threshold": args.catalog_score_threshold,
        "catalog_score_margin": args.catalog_score_margin,
        "annotations_root": str(Path(args.annotations_root).resolve()) if args.annotations_root else None,
        "annotations_fingerprint": stable_hash(annotation_files) if annotation_files else None,
        "require_gold_annotations": args.require_gold_annotations,
        "reference_used_online": False,
        "online_api_used": False,
        "eir_coverage_policy": (
            RelationCatalog(catalog_path).eir_coverage if spec.use_abstraction else None
        ),
    }
    return {**config, "config_hash": stable_hash(config), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def compact_ledger_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a small append-only summary record outside purge-prone sample trees."""
    abstraction = dict(report.get("abstraction_metrics") or {})
    abstraction.pop("fact_reports", None)
    keys = (
        "framework_version", "run_config_hash", "experiment", "experiment_id",
        "split", "sample_id", "sample_key", "domain", "use_memory",
        "use_abstraction", "use_sft", "model_variant", "llm_source", "final_status",
        "final_attempt", "pipeline_diagnostics", "coverage_diagnostics",
        "generation_usage", "metrics",
    )
    output = {key: report.get(key) for key in keys}
    output["abstraction_metrics"] = abstraction
    return output


def main() -> None:
    args = parse_args()
    if args.framework_version:
        print(FRAMEWORK_VERSION)
        return
    if not args.experiment:
        raise ValueError("--experiment is required")
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")

    spec = experiment_spec(args.experiment)
    selected_model = args.model or (args.sft_model if spec.use_sft else args.base_model)
    if not selected_model:
        factor = "SFT" if spec.use_sft else "base"
        raise ValueError(
            f"No {factor} model path configured. Set "
            f"{'HSA_SFT_MODEL' if spec.use_sft else 'HSA_BASE_MODEL'} or pass the corresponding CLI option."
        )
    # All downstream provenance uses the model actually selected by the factorial cell.
    args.model = selected_model
    random.seed(args.seed)
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output or (Path(DEFAULT_RESULTS) / spec.name)).resolve()
    catalog_path = Path(args.catalog).resolve()
    annotations_root = Path(args.annotations_root).resolve() if args.annotations_root else None
    ensure_dir(output_root)

    manifest = build_manifest(args, spec, data_root, catalog_path)
    manifest_path = output_root / "run_manifest.json"
    lenient_resume = os.environ.get("HSA_LENIENT_RESUME") == "1"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != manifest["config_hash"] and not lenient_resume:
            raise RuntimeError(
                f"Output contains a different run configuration: {manifest_path}. Use a new output path."
            )
        if not args.resume:
            raise RuntimeError(f"Output already exists; pass --resume to continue: {output_root}")
        if lenient_resume and previous.get("config_hash") != manifest["config_hash"]:
            print(
                f"[LENIENT-RESUME] config_hash changed (old={previous.get('config_hash')[:8]}... "
                f"new={manifest['config_hash'][:8]}...); keeping existing manifest, "
                f"re-running only samples missing or from a different config."
            )
        manifest = previous
    else:
        atomic_write_json(manifest_path, manifest)

    items = select_items(load_dataset(data_root, args.split), args)
    if not items:
        raise ValueError(f"No dataset samples found under {data_root} for split={args.split}")
    parsed_domains = {parse_domain(item["domain_pddl"])["domain_name"] for item in items}
    canonical_domains = {canonical_domain_name(name) for name in parsed_domains}
    if len(canonical_domains) != 1:
        raise ValueError(f"One experiment must use exactly one PDDL domain, observed={sorted(parsed_domains)}")
    dataset_domain_name = sorted(parsed_domains)[0]
    catalog: Optional[RelationCatalog] = RelationCatalog(catalog_path)
    for item in items:
        catalog.validate_domain(parse_domain(item["domain_pddl"]))
    if spec.use_abstraction and catalog:
        invalid_contracts = []
        for item in items:
            contract = inspect_nl_coverage_contract(
                item["problem_nl"], parse_domain(item["domain_pddl"]), catalog
            )
            if not contract["ok"]:
                invalid_contracts.append({
                    "sample_key": item["sample_key"],
                    "errors": contract["errors"],
                })
        if invalid_contracts:
            raise ValueError(
                "NL coverage contract failed before model initialization; "
                "no generation request was sent. First failures="
                + json.dumps(invalid_contracts[:5], ensure_ascii=False)
            )
    llm: Any = LocalLLM(
        args.model,
        backend=args.generator_backend,
        device=args.device,
        seed=args.seed,
        max_context_tokens=args.max_context_tokens,
        mock_response_dir=args.mock_response_dir,
    )
    fd_runner = FastDownwardRunner(args.fd_entry, args.fd_search, args.fd_timeout_s, args.planner_backend)
    val_runner = VALRunner(args.val_bin, args.val_timeout_s, args.val_backend)

    processed = 0
    skipped = 0
    started = time.time()
    for item in items:
        sample_dir = ensure_dir(output_root / "samples" / item["split"] / item["sample_id"])
        sample_report_path = sample_dir / "sample_report.json"
        if args.resume and sample_report_path.is_file():
            previous = json.loads(sample_report_path.read_text(encoding="utf-8"))
            if previous.get("run_config_hash") == manifest["config_hash"]:
                skipped += 1
                print(f"[SKIP] {item['sample_key']} already complete")
                continue

        domain = parse_domain(item["domain_pddl"])
        if catalog:
            catalog.validate_domain(domain)
        domain_path = sample_dir / "domain.pddl"
        atomic_write_text(domain_path, item["domain_pddl"])
        atomic_write_text(sample_dir / "domain.nl", item["domain_nl"])
        atomic_write_text(sample_dir / "problem.nl", item["problem_nl"])
        atomic_write_text(sample_dir / "reference_OFFLINE_ONLY.pddl", item["reference_pddl"])

        if spec.use_abstraction:
            final = run_structured_sample(
                item, sample_dir, llm, domain, catalog, domain_path,
                fd_runner, val_runner, args,
            )
        else:
            final = run_direct_sample(
                item, sample_dir, llm, domain, catalog, domain_path,
                fd_runner, val_runner, args,
            )
        final.setdefault("provenance", {})["llm_source"] = spec.llm_source
        final["provenance"]["llm_model"] = str(Path(args.model).resolve())
        final["provenance"]["model_variant"] = "sft" if spec.use_sft else "base"
        final["provenance"]["factor_sft"] = int(spec.use_sft)
        final["provenance"]["factor_hsa"] = int(spec.use_abstraction)
        final_pddl = final.get("pddl", "")
        if final_pddl:
            atomic_write_text(sample_dir / "final_output.pddl", final_pddl + ("\n" if not final_pddl.endswith("\n") else ""))
        if final.get("eir") is not None:
            atomic_write_json(sample_dir / "final_eir.json", final["eir"])
        if final.get("air") is not None:
            atomic_write_json(sample_dir / "final_air.json", final["air"])
        if final.get("gir") is not None:
            atomic_write_json(sample_dir / "final_gir.json", final["gir"])
        if final.get("coverage") is not None:
            atomic_write_json(sample_dir / "final_eir_coverage.json", final["coverage"])

        metrics = evaluate_pddl(
            final_pddl,
            item["reference_pddl"],
            item["domain_pddl"],
            final.get("fd"),
            final.get("val"),
        )
        annotation_path = annotations_root / item["split"] / f"{item['sample_id']}.json" if annotations_root else None
        coverage_failed = (final.get("pipeline") or {}).get("eir_coverage_ok") == 0
        abstraction_metrics = evaluate_abstraction(
            final.get("air"),
            final.get("gir"),
            item["reference_pddl"],
            catalog,
            annotation_path=annotation_path,
            require_annotations=args.require_gold_annotations,
        ) if spec.use_abstraction and catalog and not coverage_failed else {
            "RMA": None, "GER": None, "rma_correct": 0, "rma_eligible": 0,
            "ger_wrong": 0, "ger_eligible": 0, "acs_decisions": [],
            "abstraction_applicable": False,
            "reason": "upstream_eir_coverage_failed" if coverage_failed else "no_explicit_abstraction_layer",
        }
        report = {
            "framework_version": FRAMEWORK_VERSION,
            "run_config_hash": manifest["config_hash"],
            "experiment": spec.name,
            "experiment_id": spec.experiment_id,
            "split": item["split"],
            "sample_id": item["sample_id"],
            "sample_key": item["sample_key"],
            "domain": domain["domain_name"],
            "source_dir": item["source_dir"],
            "input_hashes": {
                "domain_pddl": stable_hash(item["domain_pddl"]),
                "domain_nl": stable_hash(item["domain_nl"]),
                "problem_nl": stable_hash(item["problem_nl"]),
                "reference_pddl_offline_only": stable_hash(item["reference_pddl"]),
                "final_output_pddl": stable_hash(final_pddl) if final_pddl else None,
            },
            "use_memory": spec.use_memory,
            "use_abstraction": spec.use_abstraction,
            "use_sft": spec.use_sft,
            "model_variant": "sft" if spec.use_sft else "base",
            "llm_source": spec.llm_source,
            "catalog_sha256": sha256_file(catalog_path) if catalog else None,
            "final_status": final.get("status"),
            "final_attempt": final.get("attempt"),
            "provenance": final.get("provenance"),
            "pipeline_diagnostics": final.get("pipeline") or {
                "eir_parse_ok": None, "eir_coverage_ok": None,
                "eir_ok": None, "air_ok": None, "gir_ok": None,
            },
            "coverage_diagnostics": final.get("coverage"),
            "generation_usage": final.get("generation_usage") or summarize_generation_usage([]),
            "fd": final.get("fd"),
            "val": final.get("val"),
            "metrics": {key: metrics[key] for key in ("SVR", "SCR", "PSR", "VPR")},
            "pddl_metric_details": metrics,
            "abstraction_metrics": abstraction_metrics,
            "reference_used_online": False,
            "online_api_used": False,
        }
        atomic_write_json(sample_report_path, report)
        ledger_report = compact_ledger_report(report)
        append_jsonl(output_root / "sample_reports_ledger.jsonl", ledger_report)
        append_jsonl(output_root.parent / "master_sample_reports_ledger.jsonl", ledger_report)
        processed += 1
        print(
            f"[{time.strftime('%X')}] {spec.name} {item['sample_key']} "
            f"SVR={metrics['SVR']} SCR={metrics['SCR']} PSR={metrics['PSR']} VPR={metrics['VPR']} "
            f"RMA={abstraction_metrics.get('RMA')} GER={abstraction_metrics.get('GER')} "
            f"status={final.get('status')} source={spec.llm_source} "
            f"format={(final.get('provenance') or {}).get('detected_output_format')} "
            f"EIR-COV={(final.get('pipeline') or {}).get('eir_coverage_ok')}"
        )
    elapsed = time.time() - started
    atomic_write_json(output_root / "run_completion.json", {
        "framework_version": FRAMEWORK_VERSION,
        "experiment": spec.name,
        "processed_this_invocation": processed,
        "skipped_complete": skipped,
        "selected_samples": len(items),
        "elapsed_s": round(elapsed, 4),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"[DONE] experiment={spec.name} processed={processed} skipped={skipped} output={output_root}")


if __name__ == "__main__":
    main()
