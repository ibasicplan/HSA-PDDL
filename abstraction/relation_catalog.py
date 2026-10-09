from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.pddl import is_type_compatible, parse_domain, type_spec_from_json


def _tokens(text: str) -> Set[str]:
    stop = {"a", "an", "the", "is", "are", "was", "were", "of", "to", "and", "or"}
    return {x for x in re.findall(r"[a-z][a-z0-9_-]*", text.lower()) if x not in stop}


def canonical_domain_name(name: str) -> str:
    """Normalize PDDL domain-name spelling (e.g. ``zenotravel`` == ``zeno-travel``)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


class RelationCatalog:
    """Validated, local-only semantic relation catalog."""

    def __init__(self, path: Path, domain: Optional[Dict[str, Any]] = None):
        self.path = path.resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.domain_name = str(self.data.get("domain", "")).lower()
        raw_coverage = self.data.get("eir_coverage", {})
        if raw_coverage is None:
            raw_coverage = {}
        if not isinstance(raw_coverage, dict):
            raise ValueError("Catalog 'eir_coverage' must be an object when provided")
        self.eir_coverage: Dict[str, Any] = dict(raw_coverage)
        raw_eir_instructions = self.data.get("eir_instructions", [])
        if isinstance(raw_eir_instructions, str):
            raw_eir_instructions = [raw_eir_instructions]
        self.eir_instructions = [
            str(item).strip() for item in raw_eir_instructions if str(item).strip()
        ]
        raw_instructions = self.data.get("direct_pddl_instructions", [])
        if isinstance(raw_instructions, str):
            raw_instructions = [raw_instructions]
        self.direct_pddl_instructions = [
            str(item).strip() for item in raw_instructions if str(item).strip()
        ]
        raw_relations = self.data.get("relations")
        if not isinstance(raw_relations, dict) or not raw_relations:
            raise ValueError("Catalog must contain a non-empty 'relations' object")
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.predicate_to_relation: Dict[str, str] = {}
        self.predicate_specs: Dict[str, Dict[str, Any]] = {}
        for relation_id, raw in raw_relations.items():
            rid = str(relation_id).strip().lower()
            if not rid or not isinstance(raw, dict):
                raise ValueError(f"Invalid relation entry: {relation_id!r}")
            predicates: List[str] = []
            for item in raw.get("predicates", []):
                name = item.get("name") if isinstance(item, dict) else item
                name = str(name or "").strip().lower()
                if name:
                    predicates.append(name)
                    raw_permutation = item.get("argument_permutation") if isinstance(item, dict) else None
                    permutation = (
                        [int(index) for index in raw_permutation]
                        if isinstance(raw_permutation, list)
                        else None
                    )
                    if permutation is not None and sorted(permutation) != list(range(len(permutation))):
                        raise ValueError(
                            f"Predicate {name!r} has invalid argument_permutation={permutation!r}"
                        )
                    self.predicate_specs[name] = {
                        "name": name,
                        "argument_permutation": permutation,
                        "argument_roles": [
                            str(role).strip() for role in (
                                item.get("argument_roles", []) if isinstance(item, dict) else []
                            )
                        ],
                        "direct_pddl_instruction": str(
                            item.get("direct_pddl_instruction", "") if isinstance(item, dict) else ""
                        ).strip(),
                    }
            if not predicates:
                raise ValueError(f"Relation {rid!r} has no predicates")
            duplicate = set(predicates).intersection(self.predicate_to_relation)
            if duplicate:
                raise ValueError(f"Predicates assigned to multiple relations: {sorted(duplicate)}")
            entry = {
                "id": rid,
                "description": str(raw.get("description", "")).strip(),
                "aliases": [str(x).strip().lower() for x in raw.get("aliases", []) if str(x).strip()],
                "patterns": [str(x) for x in raw.get("patterns", []) if str(x).strip()],
                "classification_phrases": [
                    str(x).strip().lower()
                    for x in raw.get("classification_phrases", [])
                    if str(x).strip()
                ],
                "predicates": sorted(set(predicates)),
            }
            for pattern in entry["patterns"]:
                re.compile(pattern, re.I)
            self.relations[rid] = entry
            for predicate in entry["predicates"]:
                self.predicate_to_relation[predicate] = rid
        if domain is not None:
            self.validate_domain(domain)

    def validate_domain(self, domain: Dict[str, Any]) -> None:
        observed_domain = str(domain.get("domain_name", "")).lower()
        if (
            self.domain_name
            and observed_domain
            and canonical_domain_name(self.domain_name) != canonical_domain_name(observed_domain)
        ):
            raise ValueError(
                f"Catalog domain {self.domain_name!r} != PDDL domain {observed_domain!r} "
                f"after identifier normalization"
            )
        domain_predicates = set(domain.get("predicates", {}))
        catalog_predicates = set(self.predicate_to_relation)
        missing = domain_predicates - catalog_predicates
        unknown = catalog_predicates - domain_predicates
        if missing or unknown:
            raise ValueError(
                f"Catalog predicate coverage error: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for predicate, spec in self.predicate_specs.items():
            permutation = spec.get("argument_permutation")
            if permutation is None:
                continue
            arity = int(domain["predicates"][predicate].get("arity", -1))
            if len(permutation) != arity:
                raise ValueError(
                    f"Predicate {predicate!r} permutation length {len(permutation)} != arity {arity}"
                )
            roles = spec.get("argument_roles", [])
            if roles and len(roles) != arity:
                raise ValueError(
                    f"Predicate {predicate!r} role count {len(roles)} != arity {arity}"
                )
        for relation_id, relation in self.relations.items():
            if not relation.get("classification_phrases"):
                continue
            non_unary = [
                predicate for predicate in relation["predicates"]
                if int(domain["predicates"][predicate].get("arity", -1)) != 1
            ]
            if non_unary:
                raise ValueError(
                    f"Relation {relation_id!r} declares classification_phrases "
                    f"for non-unary predicates {non_unary}"
                )

    def map_predicate(self, predicate: str) -> Optional[str]:
        return self.predicate_to_relation.get(predicate.lower())

    def predicates(self, relation_id: str) -> List[str]:
        return list((self.relations.get(relation_id.lower()) or {}).get("predicates", []))

    def classification_specs(self, domain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return catalog-declared NL labels for unary state predicates."""
        specs: List[Dict[str, Any]] = []
        for relation_id, relation in sorted(self.relations.items()):
            phrases = list(relation.get("classification_phrases") or [])
            if not phrases:
                continue
            for predicate in relation["predicates"]:
                signature = domain.get("predicates", {}).get(predicate) or {}
                if int(signature.get("arity", -1)) == 1:
                    specs.append({
                        "relation": relation_id,
                        "predicate": predicate,
                        "phrases": phrases,
                    })
        return specs

    def ground_arguments(self, predicate: str, arguments: Sequence[str]) -> List[str]:
        """Map EIR argument order to PDDL slot order via ``argument_permutation``."""
        normalized = [str(argument).lower() for argument in arguments]
        spec = self.predicate_specs.get(str(predicate).lower(), {})
        permutation = spec.get("argument_permutation")
        if permutation is None:
            return normalized
        if len(permutation) != len(normalized):
            return normalized
        return [normalized[index] for index in permutation]

    def eir_guidance(self) -> str:
        """Return frozen dataset-specific EIR semantic guidance."""
        return "\n".join(f"- {instruction}" for instruction in self.eir_instructions)

    def direct_pddl_guidance(self) -> str:
        """Return frozen predicate-slot guidance for information-matched direct controls."""
        lines: List[str] = [f"- {instruction}" for instruction in self.direct_pddl_instructions]
        for predicate in sorted(self.predicate_specs):
            spec = self.predicate_specs[predicate]
            instruction = str(spec.get("direct_pddl_instruction", "")).strip()
            if not instruction:
                continue
            roles = ", ".join(spec.get("argument_roles", []))
            role_suffix = f" PDDL slot roles: [{roles}]." if roles else ""
            lines.append(f"- {predicate}: {instruction}{role_suffix}")
        return "\n".join(lines)

    def descriptions(self, relation_ids: Optional[Iterable[str]] = None, expose_predicates: bool = False) -> str:
        ids = sorted(set(relation_ids or self.relations))
        lines = []
        for relation_id in ids:
            entry = self.relations[relation_id]
            suffix = f"; predicates={entry['predicates']}" if expose_predicates else ""
            aliases = ", ".join(entry["aliases"][:8])
            lines.append(f"- {relation_id}: {entry['description']}; aliases=[{aliases}]{suffix}")
        return "\n".join(lines)

    def compatible_predicates(
        self,
        relation_id: str,
        arguments: Sequence[str],
        object_types: Dict[str, Any],
        domain: Dict[str, Any],
    ) -> List[str]:
        hierarchy = {str(k): str(v) for k, v in domain.get("type_hierarchy", {}).items()}
        candidates: List[str] = []
        for predicate in self.predicates(relation_id):
            signature = domain.get("predicates", {}).get(predicate)
            if not signature or int(signature.get("arity", -1)) != len(arguments):
                continue
            grounded_arguments = self.ground_arguments(predicate, arguments)
            compatible = True
            for index, argument in enumerate(grounded_arguments):
                actual_raw = object_types.get(argument, "unknown")
                if str(actual_raw).lower() == "unknown":
                    continue
                actual = type_spec_from_json(actual_raw)
                expected = type_spec_from_json(signature["argument_types"][index])
                if not is_type_compatible(actual, expected, hierarchy):
                    compatible = False
                    break
            if compatible:
                candidates.append(predicate)
        return sorted(candidates)

    def feasible_relations(
        self,
        arguments: Sequence[str],
        object_types: Dict[str, Any],
        domain: Dict[str, Any],
    ) -> List[str]:
        return [
            relation_id
            for relation_id in sorted(self.relations)
            if self.compatible_predicates(relation_id, arguments, object_types, domain)
        ]

    def score(self, text: str, relation_id: str) -> float:
        entry = self.relations[relation_id]
        normalized = " ".join(text.lower().split())
        score = 0.0
        for pattern in entry["patterns"]:
            if re.search(pattern, normalized, re.I):
                score += 5.0
        text_tokens = _tokens(normalized)
        description_tokens = _tokens(entry["description"])
        if description_tokens:
            score += len(text_tokens & description_tokens) / len(description_tokens)
        for alias in entry["aliases"]:
            alias_tokens = _tokens(alias)
            if not alias_tokens:
                continue
            if alias == normalized:
                score += 5.0
            elif re.search(r"(?<![a-z0-9_-])" + re.escape(alias) + r"(?![a-z0-9_-])", normalized):
                score += 3.0
            score += 2.0 * len(text_tokens & alias_tokens) / len(alias_tokens)
        return round(score, 6)

    def rank(self, text: str, feasible: Sequence[str]) -> List[Tuple[str, float]]:
        return sorted(((rid, self.score(text, rid)) for rid in feasible), key=lambda x: (-x[1], x[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an offline semantic relation catalog")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()
    domain = parse_domain(Path(args.domain).read_text(encoding="utf-8"))
    catalog = RelationCatalog(Path(args.catalog), domain)
    print(json.dumps({
        "status": "ok",
        "domain": catalog.domain_name,
        "relations": len(catalog.relations),
        "predicates": len(catalog.predicate_to_relation),
        "eir_coverage_enabled": bool(catalog.eir_coverage.get("enabled", False)),
        "classification_specs": len(catalog.classification_specs(domain)),
    }, indent=2))


if __name__ == "__main__":
    main()
