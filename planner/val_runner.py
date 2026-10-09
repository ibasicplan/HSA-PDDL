from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.io import read_text


class VALRunner:
    def __init__(self, binary: str, timeout_s: int, backend: str = "real"):
        self.binary = Path(binary).resolve()
        self.timeout_s = timeout_s
        self.backend = backend
        if backend == "real" and not self.binary.is_file():
            raise FileNotFoundError(f"VAL binary not found: {self.binary}")

    def run(
        self,
        domain_path: Path,
        problem_path: Path,
        plan_path: Optional[Path],
        work_dir: Path,
    ) -> Dict[str, Any]:
        log_path = work_dir / "val.log"
        if not plan_path or not plan_path.is_file():
            log_path.write_text("VAL skipped: no plan file\n", encoding="utf-8")
            return {"status": "skipped_no_plan", "exit_code": None, "duration_s": 0.0, "log_path": str(log_path)}
        if self.backend == "mock":
            log_path.write_text("Plan valid\n", encoding="utf-8")
            return {"status": "valid", "exit_code": 0, "duration_s": 0.0, "log_path": str(log_path)}
        command = [str(self.binary), str(domain_path), str(problem_path), str(plan_path)]
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
            except OSError as exc:
                output.write(f"\n[ERROR] {exc}\n")
                return_code = 1
        elapsed = time.time() - started
        text = read_text(log_path)
        lower = text.lower()
        if return_code == 124:
            status = "timeout"
        elif any(marker in lower for marker in ("plan valid", "successful plan", "solution valid")):
            status = "valid"
        elif return_code == 0 and not any(marker in lower for marker in ("invalid", "failed", "not satisfied")):
            status = "valid"
        else:
            status = "invalid"
        return {
            "status": status,
            "exit_code": return_code,
            "duration_s": round(elapsed, 4),
            "log_path": str(log_path),
            "raw_excerpt": text[-3000:],
        }
