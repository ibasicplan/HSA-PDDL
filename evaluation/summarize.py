from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import EXPERIMENTS, FRAMEWORK_VERSION
from evaluation.abstraction_metrics import consistency_score
from utils.io import atomic_write_json, atomic_write_text, iter_jsonl


PRIMARY_METRICS = ["SVR", "SCR", "PSR", "VPR", "RMA", "ACS", "GER"]
PAIRWISE_COMPARISONS = [
    ("sft_effect_without_hsa", "sft_direct", "base_direct"),
    ("hsa_effect_on_base", "base_hsa", "base_direct"),
    ("hsa_effect_on_sft", "sft_hsa", "sft_direct"),
    ("sft_effect_with_hsa", "sft_hsa", "base_hsa"),
    ("full_vs_base_direct", "sft_hsa", "base_direct"),
]


def observed_splits(rows: Iterable[Dict[str, Any]]) -> List[str]:
    labels = {str(row.get("split")) for row in rows if row.get("split")}
    preferred = [name for name in ("easy", "hard", "all") if name in labels]
    preferred.extend(sorted(labels - set(preferred)))
    return preferred + ["overall"]


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _fmt(value: Any, percent: bool = True) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.2f}%" if percent else f"{float(value):.4f}"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_reports(master_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    master_ledger = list(iter_jsonl(master_root / "master_sample_reports_ledger.jsonl"))
    for experiment in EXPERIMENTS:
        indexed: Dict[str, Dict[str, Any]] = {}
        ledger_rows = [
            row for row in master_ledger
            if row.get("experiment") == experiment
        ]
        ledger_rows.extend(iter_jsonl(master_root / experiment / "sample_reports_ledger.jsonl"))
        for report in ledger_rows:
            if report.get("framework_version") != FRAMEWORK_VERSION:
                continue
            sample_key = str(report.get("sample_key") or "")
            if sample_key:
                indexed[sample_key] = report
        for path in sorted((master_root / experiment / "samples").glob("*/*/sample_report.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("framework_version") != FRAMEWORK_VERSION:
                raise ValueError(f"Framework mismatch in {path}: {report.get('framework_version')}")
            indexed[str(report.get("sample_key"))] = report
        reports = [indexed[key] for key in sorted(indexed)]
        completion_path = master_root / experiment / "run_completion.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            expected = int(completion.get("selected_samples") or 0)
            if expected and len(reports) != expected:
                raise RuntimeError(
                    f"Incomplete summary input for {experiment}: found={len(reports)}, "
                    f"expected={expected}. Restore sample reports or the append-only ledger."
                )
        elif reports:
            raise RuntimeError(
                f"Incomplete summary input for {experiment}: reports exist but "
                f"{completion_path} is missing. Resume the experiment before summarizing."
            )
        if reports:
            result[experiment] = reports
    return result


def aggregate(reports: Sequence[Dict[str, Any]], split: str) -> Dict[str, Any]:
    rows = [x for x in reports if split == "overall" or x.get("split") == split]
    n = len(rows)
    binary = {
        metric: (sum(int((x.get("metrics") or {}).get(metric, 0)) for x in rows) / n if n else None)
        for metric in ("SVR", "SCR", "PSR", "VPR")
    }
    rma_correct = sum(int(((x.get("abstraction_metrics") or {}).get("rma_correct") or 0)) for x in rows)
    rma_eligible = sum(int(((x.get("abstraction_metrics") or {}).get("rma_eligible") or 0)) for x in rows)
    ger_wrong = sum(int(((x.get("abstraction_metrics") or {}).get("ger_wrong") or 0)) for x in rows)
    ger_eligible = sum(int(((x.get("abstraction_metrics") or {}).get("ger_eligible") or 0)) for x in rows)
    decisions = [
        decision
        for row in rows
        for decision in ((row.get("abstraction_metrics") or {}).get("acs_decisions") or [])
    ]
    consistency = consistency_score(decisions)
    usages = [(x.get("generation_usage") or {}) for x in rows]
    pipeline_rates = {}
    for metric in ("eir_parse_ok", "eir_coverage_ok", "eir_ok", "air_ok", "gir_ok"):
        eligible = [
            (x.get("pipeline_diagnostics") or {}).get(metric)
            for x in rows
            if (x.get("pipeline_diagnostics") or {}).get(metric) is not None
        ]
        pipeline_rates[metric.upper() + "_rate"] = _mean([float(x) for x in eligible])
    return {
        "split": split,
        "N": n,
        **binary,
        "RMA": rma_correct / rma_eligible if rma_eligible else None,
        "ACS": consistency["ACS"],
        "GER": ger_wrong / ger_eligible if ger_eligible else None,
        "RMA_correct": rma_correct,
        "RMA_eligible": rma_eligible,
        "ACS_conflicts": consistency["acs_conflicts"],
        "ACS_decisions": consistency["acs_decisions"],
        "GER_wrong": ger_wrong,
        "GER_eligible": ger_eligible,
        "Mean_attempts": _mean([float(x.get("final_attempt") or 0) for x in rows]) if rows else None,
        "Mean_generation_calls": _mean([float(x.get("logical_generation_calls") or 0) for x in usages]) if rows else None,
        "Mean_input_tokens": _mean([float(x.get("input_tokens") or 0) for x in usages]) if rows else None,
        "Mean_output_tokens": _mean([float(x.get("output_tokens") or 0) for x in usages]) if rows else None,
        "Mean_total_tokens": _mean([float(x.get("total_tokens") or 0) for x in usages]) if rows else None,
        "Mean_generation_time_s": _mean([float(x.get("generation_time_s") or 0) for x in usages]) if rows else None,
        **pipeline_rates,
    }


def _binomial_two_sided(k: int, n: int) -> Optional[float]:
    if n == 0:
        return 1.0
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower)


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    a_only = sum(1 for left, right in zip(a, b) if left == 1 and right == 0)
    b_only = sum(1 for left, right in zip(a, b) if left == 0 and right == 1)
    discordant = a_only + b_only
    return {
        "A_only": a_only,
        "B_only": b_only,
        "discordant": discordant,
        "mcnemar_exact_p": _binomial_two_sided(min(a_only, b_only), discordant),
    }


def bootstrap_difference(
    paired: Sequence[Tuple[float, float]],
    iterations: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not paired:
        return None, None, None
    observed = _mean([a - b for a, b in paired])
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [paired[rng.randrange(len(paired))] for _ in range(len(paired))]
        values.append(sum(a - b for a, b in sample) / len(sample))
    values.sort()
    lo = values[max(0, int(0.025 * iterations) - 1)]
    hi = values[min(iterations - 1, int(0.975 * iterations))]
    return observed, lo, hi


def paired_effects(
    reports: Dict[str, List[Dict[str, Any]]],
    iterations: int,
    seed: int,
) -> List[Dict[str, Any]]:
    indexed = {
        experiment: {str(x.get("sample_key")): x for x in rows}
        for experiment, rows in reports.items()
    }
    output: List[Dict[str, Any]] = []
    for comparison, a_name, b_name in PAIRWISE_COMPARISONS:
        if a_name not in indexed or b_name not in indexed:
            continue
        keys = sorted(set(indexed[a_name]) & set(indexed[b_name]))
        paired_rows = [indexed[a_name][key] for key in keys]
        for split in observed_splits(paired_rows):
            split_keys = [key for key in keys if split == "overall" or indexed[a_name][key].get("split") == split]
            for metric in ("SVR", "SCR", "PSR", "VPR", "RMA", "GER"):
                paired: List[Tuple[float, float]] = []
                for key in split_keys:
                    if metric in {"RMA", "GER"}:
                        left = (indexed[a_name][key].get("abstraction_metrics") or {}).get(metric)
                        right = (indexed[b_name][key].get("abstraction_metrics") or {}).get(metric)
                    else:
                        left = (indexed[a_name][key].get("metrics") or {}).get(metric)
                        right = (indexed[b_name][key].get("metrics") or {}).get(metric)
                    if left is not None and right is not None:
                        paired.append((float(left), float(right)))
                difference, ci_low, ci_high = bootstrap_difference(paired, iterations, seed)
                row: Dict[str, Any] = {
                    "comparison": comparison,
                    "A": a_name,
                    "B": b_name,
                    "split": split,
                    "metric": metric,
                    "N_paired": len(paired),
                    "A_mean": _mean([x[0] for x in paired]),
                    "B_mean": _mean([x[1] for x in paired]),
                    "delta_A_minus_B": difference,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                }
                if metric in {"SVR", "SCR", "PSR", "VPR"}:
                    row.update(mcnemar_exact([int(x[0]) for x in paired], [int(x[1]) for x in paired]))
                output.append(row)
    return output


def factorial_interactions(
    reports: Dict[str, List[Dict[str, Any]]], iterations: int = 5000, seed: int = 123,
) -> List[Dict[str, Any]]:
    required = {"base_direct", "sft_direct", "base_hsa", "sft_hsa"}
    if not required.issubset(reports):
        return []
    indexed = {name: {x["sample_key"]: x for x in reports[name]} for name in required}
    common = sorted(set.intersection(*(set(indexed[name]) for name in required)))
    rows = []
    for split in observed_splits(indexed["base_direct"].values()):
        keys = [key for key in common if split == "overall" or indexed["base_direct"][key].get("split") == split]
        for metric in ("SVR", "SCR", "PSR", "VPR"):
            values = []
            for key in keys:
                bd = float(indexed["base_direct"][key]["metrics"][metric])
                sd = float(indexed["sft_direct"][key]["metrics"][metric])
                bh = float(indexed["base_hsa"][key]["metrics"][metric])
                sh = float(indexed["sft_hsa"][key]["metrics"][metric])
                values.append((sh - sd) - (bh - bd))
            rng = random.Random(seed + sum(ord(ch) for ch in split + metric))
            boot = []
            if values:
                for _ in range(iterations):
                    boot.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
                boot.sort()
            lo = boot[max(0, int(0.025 * iterations) - 1)] if boot else None
            hi = boot[min(iterations - 1, int(0.975 * iterations))] if boot else None
            rows.append({
                "split": split,
                "metric": metric,
                "N_common": len(values),
                "interaction_(sft_hsa-sft_direct)-(base_hsa-base_direct)": _mean(values),
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
            })
    return rows


def build_markdown(summary_rows: Sequence[Dict[str, Any]], effects: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"# GPT-OSS-20B 2x2 SFT x HSA Summary ({FRAMEWORK_VERSION})", "",
        "All four PDDL metrics use the fixed denominator N. The four cells differ only in SFT (off/on) and HSA (off/on); memory and remote APIs are disabled.", "",
        "| Experiment | Split | N | SVR | SCR | PSR | VPR | RMA | ACS | GER | EIR-Parse | EIR-Cov | EIR-SR | AIR-SR | GIR-SR | Calls/task | Tokens/task | Sec/task | RMA-n | ACS-n | GER-n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['N']} | {_fmt(row['SVR'])} | {_fmt(row['SCR'])} | "
            f"{_fmt(row['PSR'])} | {_fmt(row['VPR'])} | {_fmt(row['RMA'])} | {_fmt(row['ACS'])} | {_fmt(row['GER'])} | "
            f"{_fmt(row['EIR_PARSE_OK_rate'])} | {_fmt(row['EIR_COVERAGE_OK_rate'])} | "
            f"{_fmt(row['EIR_OK_rate'])} | {_fmt(row['AIR_OK_rate'])} | {_fmt(row['GIR_OK_rate'])} | "
            f"{_fmt(row['Mean_generation_calls'], percent=False)} | {_fmt(row['Mean_total_tokens'], percent=False)} | "
            f"{_fmt(row['Mean_generation_time_s'], percent=False)} | "
            f"{row['RMA_eligible']} | {row['ACS_decisions']} | {row['GER_eligible']} |"
        )
    lines.extend(["", "## Primary paired effects", "",
                  "| Comparison | Split | Metric | N | Delta | 95% CI | McNemar p |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in effects:
        if row["split"] != "overall" or row["metric"] not in PRIMARY_METRICS:
            continue
        ci = "N/A" if row["bootstrap_95ci_low"] is None else f"[{row['bootstrap_95ci_low']:.4f}, {row['bootstrap_95ci_high']:.4f}]"
        p = row.get("mcnemar_exact_p")
        lines.append(
            f"| {row['comparison']} | {row['split']} | {row['metric']} | {row['N_paired']} | "
            f"{_fmt(row['delta_A_minus_B'], percent=False)} | {ci} | {'N/A' if p is None else f'{p:.6f}'} |"
        )
    lines.extend([
        "", "Notes:",
        "- PSR = number of Fast Downward successes / N; it is not conditioned on syntax validity.",
        "- VPR = number of VAL-valid plans / N.",
        "- GER is conditional on a correct abstract relation mapping and checks both the grounded predicate and argument order.",
        "- ACS is a corpus-level majority-consistency score within gold semantic clusters.",
        "- EIR-Cov checks only inference-visible Problem NL object/fact counts and explicit untyped classification assertions; it never reads reference PDDL.",
        "- RMA/ACS/GER are N/A when the upstream EIR coverage gate fails, preventing correct mappings on an incomplete subset from appearing as 100% abstraction quality.",
        "- Human annotations take precedence; otherwise gold mappings are derived offline from reference PDDL plus the frozen catalog.",
        "- Direct cells receive a flattened version of the same frozen catalog used by HSA, reducing information-access confounding.",
        "- HSA cells require native EIR; the PDDL-to-EIR adapter is disabled and malformed EIR fails closed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the GPT-OSS-20B SFT x HSA experiments")
    parser.add_argument("--master_root", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    master_root = Path(args.master_root).resolve()
    reports = load_reports(master_root)
    summary_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    for experiment, rows in reports.items():
        for split in observed_splits(rows):
            summary_rows.append({"experiment": experiment, **aggregate(rows, split)})
        for report in rows:
            metrics = report.get("metrics") or {}
            abstraction = report.get("abstraction_metrics") or {}
            per_sample_rows.append({
                "experiment": experiment,
                "sample_key": report.get("sample_key"),
                "split": report.get("split"),
                "sample_id": report.get("sample_id"),
                "final_status": report.get("final_status"),
                "final_attempt": report.get("final_attempt"),
                **{metric: metrics.get(metric) for metric in ("SVR", "SCR", "PSR", "VPR")},
                **{metric: abstraction.get(metric) for metric in ("RMA", "GER")},
                "RMA_eligible": abstraction.get("rma_eligible"),
                "GER_eligible": abstraction.get("ger_eligible"),
                "model_variant": report.get("model_variant"),
                "factor_sft": int(bool(report.get("use_sft"))),
                "factor_hsa": int(bool(report.get("use_abstraction"))),
                **{key: (report.get("generation_usage") or {}).get(key) for key in (
                    "logical_generation_calls", "input_tokens", "output_tokens", "total_tokens", "generation_time_s",
                )},
                **{
                    metric: (report.get("pipeline_diagnostics") or {}).get(metric)
                    for metric in ("eir_parse_ok", "eir_coverage_ok", "eir_ok", "air_ok", "gir_ok")
                },
            })
    effects = paired_effects(reports, args.bootstrap_iterations, args.seed)
    interactions = factorial_interactions(reports, args.bootstrap_iterations, args.seed)
    _write_csv(master_root / "all_experiments_summary.csv", summary_rows)
    _write_csv(master_root / "all_samples_metrics.csv", per_sample_rows)
    _write_csv(master_root / "paired_effects.csv", effects)
    _write_csv(master_root / "factorial_interactions.csv", interactions)
    atomic_write_json(master_root / "all_experiments_summary.json", {
        "framework_version": FRAMEWORK_VERSION,
        "experiments_found": sorted(reports),
        "summary": summary_rows,
        "paired_effects": effects,
        "factorial_interactions": interactions,
    })
    markdown = build_markdown(summary_rows, effects)
    atomic_write_text(master_root / "all_experiments_summary.md", markdown)
    print(markdown)
    print(f"[DONE] Summary written to {master_root}")


if __name__ == "__main__":
    main()
