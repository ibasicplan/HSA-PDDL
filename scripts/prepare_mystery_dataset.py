#!/usr/bin/env python3
"""Create a non-destructive HSA-runner split view over a flat Mystery dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


REQUIRED = ("domain.pddl", "problem.nl", "problem.pddl")


def numeric_key(path: Path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def discover(source: Path) -> List[Path]:
    return sorted(
        (path for path in source.iterdir() if path.is_dir() and path.name.isdigit()),
        key=numeric_key,
    )


def validate_samples(samples: Iterable[Path]) -> None:
    failures = []
    for sample in samples:
        missing = [name for name in REQUIRED if not (sample / name).is_file()]
        if missing:
            failures.append(f"{sample.name}: missing {', '.join(missing)}")
    if failures:
        raise SystemExit("Invalid source samples:\n" + "\n".join(failures[:30]))


def install_sample(source: Path, target: Path, mode: str) -> str:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return "reused"
        raise SystemExit(f"Refusing to replace existing target: {target}")
    if mode == "symlink":
        target.symlink_to(source.resolve(), target_is_directory=True)
    else:
        shutil.copytree(source, target)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt flat <root>/<id>/ Mystery data to <root>/<split>/<id>/ without changing source data."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("easy", "hard"), default="easy")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="include every numeric sample directory")
    group.add_argument("--sample-id", action="append", help="include one ID; repeat for multiple IDs")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source dataset does not exist: {source}")
    if output == source or source in output.parents:
        raise SystemExit("Output must not equal or be nested inside the source dataset")

    candidates = discover(source)
    by_id = {path.name: path for path in candidates}
    if args.all:
        selected = candidates
    else:
        requested = list(dict.fromkeys(args.sample_id or []))
        unknown = [sample_id for sample_id in requested if sample_id not in by_id]
        if unknown:
            raise SystemExit(f"Unknown sample IDs: {', '.join(unknown)}")
        selected = [by_id[sample_id] for sample_id in requested]
    if not selected:
        raise SystemExit("No samples selected")
    validate_samples(selected)

    (output / "easy").mkdir(parents=True, exist_ok=True)
    (output / "hard").mkdir(parents=True, exist_ok=True)
    counts = {"created": 0, "reused": 0}
    for sample in selected:
        status = install_sample(sample, output / args.split / sample.name, args.mode)
        counts[status] += 1
    print(
        f"[OK] Mystery dataset view: source={source} output={output} split={args.split} "
        f"selected={len(selected)} created={counts['created']} reused={counts['reused']} mode={args.mode}"
    )


if __name__ == "__main__":
    main()
