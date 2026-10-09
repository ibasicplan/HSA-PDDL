#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from config import EXPERIMENTS, FRAMEWORK_VERSION
from evaluation.summarize import (
    _fmt,
    _write_csv,
    aggregate,
    factorial_interactions,
    load_reports,
    paired_effects,
)
from utils.io import atomic_write_json, atomic_write_text


DEFAULT_DOMAINS = ("zenotravel", "logistics", "mystery", "quantum", "rovers")


def prefix_reports(domain: str, reports: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for experiment, rows in reports.items():
        output[experiment] = []
        for row in rows:
            cloned = deepcopy(row)
            cloned["benchmark_domain"] = domain
            cloned["sample_key"] = f"{domain}/{row.get('sample_key')}"
            output[experiment].append(cloned)
    return output


def merge_reports(parts: List[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for part in parts:
        for experiment, rows in part.items():
            merged.setdefault(experiment, []).extend(rows)
    return merged


def markdown(summary: List[Dict[str, Any]], effects: List[Dict[str, Any]], interactions: List[Dict[str, Any]]) -> str:
    lines = [
        f"# Cross-domain 2x2 SFT x HSA results ({FRAMEWORK_VERSION})",
        "",
        "| Domain | Method | N | SVR | SCR | PSR | VPR | Calls/task | Tokens/task |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['domain']} | {row['experiment']} | {row['N']} | {_fmt(row['SVR'])} | "
            f"{_fmt(row['SCR'])} | {_fmt(row['PSR'])} | {_fmt(row['VPR'])} | "
            f"{_fmt(row['Mean_generation_calls'], percent=False)} | {_fmt(row['Mean_total_tokens'], percent=False)} |"
        )
    lines.extend([
        "", "## Overall paired causal contrasts", "",
        "| Contrast (A-B) | Metric | N | Delta | 95% CI | McNemar p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in effects:
        if row.get("domain") != "overall" or row.get("split") != "overall":
            continue
        if row.get("metric") not in {"SVR", "SCR", "PSR", "VPR"}:
            continue
        ci = "N/A" if row["bootstrap_95ci_low"] is None else f"[{row['bootstrap_95ci_low']:.4f}, {row['bootstrap_95ci_high']:.4f}]"
        p = row.get("mcnemar_exact_p")
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['N_paired']} | "
            f"{_fmt(row['delta_A_minus_B'], percent=False)} | {ci} | {'N/A' if p is None else f'{p:.6f}'} |"
        )
    lines.extend([
        "", "## Factor interaction", "",
        "The interaction is `(SFT-HSA - SFT-Direct) - (Base-HSA - Base-Direct)`.", "",
        "| Domain | Metric | N | Interaction | 95% CI |", "|---|---|---:|---:|---:|",
    ])
    key = "interaction_(sft_hsa-sft_direct)-(base_hsa-base_direct)"
    for row in interactions:
        if row.get("split") != "overall":
            continue
        ci = "N/A" if row.get("bootstrap_95ci_low") is None else f"[{row['bootstrap_95ci_low']:.4f}, {row['bootstrap_95ci_high']:.4f}]"
        lines.append(f"| {row['domain']} | {row['metric']} | {row['N_common']} | {_fmt(row.get(key), percent=False)} | {ci} |")
    lines.extend([
        "", "Notes:",
        "- All success metrics use a fixed denominator; failures and timeouts count as zero.",
        "- Mystery and Quantum use a technical `easy` view; flat Rovers data are recorded as `all`; these labels are not claimed as benchmark difficulty annotations.",
        "- McNemar tests use paired per-instance outcomes; confidence intervals use paired bootstrap resampling.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a configurable multi-domain GPT-OSS 2x2 experiment")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()

    parts: List[Dict[str, List[Dict[str, Any]]]] = []
    summary_rows: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    interaction_rows: List[Dict[str, Any]] = []
    found_domains: List[str] = []
    for domain in args.domains:
        domain_root = root / domain
        if not domain_root.is_dir():
            continue
        raw = load_reports(domain_root)
        if not raw:
            continue
        found_domains.append(domain)
        prefixed = prefix_reports(domain, raw)
        parts.append(prefixed)
        for experiment in EXPERIMENTS:
            if experiment in prefixed:
                summary_rows.append({"domain": domain, "experiment": experiment, **aggregate(prefixed[experiment], "overall")})
        effect_rows.extend({"domain": domain, **row} for row in paired_effects(prefixed, args.bootstrap_iterations, args.seed))
        interaction_rows.extend({"domain": domain, **row} for row in factorial_interactions(prefixed, args.bootstrap_iterations, args.seed))

    merged = merge_reports(parts)
    for experiment in EXPERIMENTS:
        if experiment in merged:
            summary_rows.append({"domain": "overall", "experiment": experiment, **aggregate(merged[experiment], "overall")})
    effect_rows.extend({"domain": "overall", **row} for row in paired_effects(merged, args.bootstrap_iterations, args.seed))
    interaction_rows.extend({"domain": "overall", **row} for row in factorial_interactions(merged, args.bootstrap_iterations, args.seed))

    _write_csv(root / "cross_domain_summary.csv", summary_rows)
    _write_csv(root / "cross_domain_paired_effects.csv", effect_rows)
    _write_csv(root / "cross_domain_interactions.csv", interaction_rows)
    atomic_write_json(root / "cross_domain_summary.json", {
        "framework_version": FRAMEWORK_VERSION,
        "domains": found_domains,
        "summary": summary_rows,
        "paired_effects": effect_rows,
        "factorial_interactions": interaction_rows,
    })
    report = markdown(summary_rows, effect_rows, interaction_rows)
    atomic_write_text(root / "cross_domain_summary.md", report)
    print(report)


if __name__ == "__main__":
    main()
