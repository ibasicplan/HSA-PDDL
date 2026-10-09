from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.io import read_text


def find_plan(work_dir: Path) -> Optional[Path]:
    candidates = sorted(
        x for x in work_dir.glob("plan*") if x.is_file() and not x.name.endswith((".log", ".json"))
    )
    return candidates[0] if candidates else None


def _int_stat(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, re.I)
    return int(match.group(1).replace(",", "")) if match else None


def _float_stat(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.I)
    return float(match.group(1)) if match else None


class FastDownwardRunner:
    def __init__(self, entry: str, search: str, timeout_s: int, backend: str = "real"):
        self.entry = Path(entry).resolve()
        self.search = search
        self.timeout_s = timeout_s
        self.backend = backend
        if backend == "real" and not self.entry.is_file():
            raise FileNotFoundError(f"Fast Downward entry not found: {self.entry}")

    def run(self, domain_path: Path, problem_path: Path, work_dir: Path) -> Dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        plan_prefix = work_dir / "plan"
        log_path = work_dir / "fd.log"
        if self.backend == "mock":
            plan_path = work_dir / "plan"
            plan_path.write_text("; mock plan\n", encoding="utf-8")
            log_path.write_text("Solution found.\nPlan length: 0 step(s).\nPlan cost: 0\n", encoding="utf-8")
            return {
                "status": "solution_found",
                "exit_code": 0,
                "duration_s": 0.0,
                "plan_path": str(plan_path),
                "log_path": str(log_path),
                "plan_length": 0,
                "plan_cost": 0.0,
                "expanded_states": 0,
            }
        command = [
            str(self.entry), "--plan-file", str(plan_prefix), str(domain_path), str(problem_path),
            "--search", self.search,
        ]
        started = time.time()
        with log_path.open("w", encoding="utf-8") as output:
            try:
                process = subprocess.run(
                    command, stdout=output, stderr=subprocess.STDOUT, timeout=self.timeout_s, check=False
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                output.write(f"\n[TIMEOUT] Reached {self.timeout_s} seconds.\n")
                return_code = 124
        elapsed = time.time() - started
        text = read_text(log_path)
        plan_path = find_plan(work_dir)
        if return_code == 124:
            status = "timeout"
        elif re.search(r"Could not parse task file|Syntax error|ParserError|translate exit code: [^0]", text, re.I):
            status = "could_not_parse"
        elif re.search(r"Solution found\.", text, re.I) and plan_path:
            status = "solution_found"
        elif re.search(r"Search stopped without finding a solution|Completely explored state space", text, re.I):
            status = "unsolved"
        elif return_code != 0:
            status = "error"
        else:
            status = "unsolved"
        return {
            "status": status,
            "exit_code": return_code,
            "duration_s": round(elapsed, 4),
            "plan_path": str(plan_path) if plan_path else None,
            "log_path": str(log_path),
            "plan_length": _int_stat(r"Plan length:\s*([0-9,]+)\s*step", text),
            "plan_cost": _float_stat(r"Plan cost:\s*([-+]?\d+(?:\.\d+)?)", text),
            "expanded_states": _int_stat(r"Expanded\s+([0-9,]+)\s+state", text),
            "evaluated_states": _int_stat(r"Evaluated\s+([0-9,]+)\s+state", text),
            "generated_states": _int_stat(r"Generated\s+([0-9,]+)\s+state", text),
            "search_time_s": _float_stat(r"Search time:\s*([0-9.]+)s", text),
            "total_time_s": _float_stat(r"Total time:\s*([0-9.]+)s", text),
            "raw_excerpt": text[-3000:],
        }

