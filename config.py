from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


FRAMEWORK_VERSION = "v12.2-gptoss-2x2-hsa-pddl"
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: int
    name: str
    use_sft: bool
    use_abstraction: bool
    description: str

    @property
    def use_memory(self) -> bool:
        """Retained in reports for backward compatibility; memory is not a factor."""
        return False

    @property
    def llm_source(self) -> str:
        return "local"


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "base_direct": ExperimentSpec(0, "base_direct", False, False, "Base GPT-OSS-20B: direct NL-to-PDDL"),
    "sft_direct": ExperimentSpec(1, "sft_direct", True, False, "SFT GPT-OSS-20B: direct NL-to-PDDL"),
    "base_hsa": ExperimentSpec(2, "base_hsa", False, True, "Base GPT-OSS-20B: native EIR-to-AIR-to-GIR-to-PDDL"),
    "sft_hsa": ExperimentSpec(3, "sft_hsa", True, True, "SFT GPT-OSS-20B: native EIR-to-AIR-to-GIR-to-PDDL"),
}


DEFAULT_BASE_MODEL = os.environ.get("HSA_BASE_MODEL", "")
DEFAULT_SFT_MODEL = os.environ.get("HSA_SFT_MODEL", "")
DEFAULT_DATA = os.environ.get("HSA_DATA_ROOT", str(PROJECT_ROOT / "data" / "zenotravel"))
DEFAULT_CATALOG = os.environ.get(
    "HSA_CATALOG", str(PROJECT_ROOT / "catalog" / "zenotravel_relation_catalog.json")
)
DEFAULT_RESULTS = os.environ.get("HSA_RESULTS_ROOT", str(PROJECT_ROOT / "results"))
DEFAULT_FD_ENTRY = os.environ.get("HSA_FD_ENTRY", "")
DEFAULT_VAL_BIN = os.environ.get("HSA_VAL_BIN", "")

DEFAULT_SEED = int(os.environ.get("HSA_SEED", "123"))
DEFAULT_MAX_ATTEMPTS = int(os.environ.get("HSA_MAX_ATTEMPTS", "3"))
DEFAULT_FD_TIMEOUT_S = int(os.environ.get("HSA_FD_TIMEOUT_S", "1800"))
DEFAULT_VAL_TIMEOUT_S = int(os.environ.get("HSA_VAL_TIMEOUT_S", "360"))
DEFAULT_FD_SEARCH = os.environ.get("HSA_FD_SEARCH", "astar(lmcut())")
DEFAULT_MAX_CONTEXT_TOKENS = int(os.environ.get("HSA_MAX_CONTEXT_TOKENS", "16384"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("HSA_MAX_NEW_TOKENS", "4096"))

def experiment_spec(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}") from exc
