from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


SExpr = Union[str, List["SExpr"]]
TypeSpec = Union[str, Tuple[str, ...]]


class PDDLParseError(ValueError):
    pass


@dataclass(frozen=True)
class Atom:
    stage: str
    predicate: str
    arguments: Tuple[str, ...]
    polarity: bool = True

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["arguments"] = list(self.arguments)
        return obj


LOGICAL_HEADS = {"and", "or", "forall", "exists", "when", "imply"}
NUMERIC_HEADS = {"=", "<", ">", "<=", ">=", "assign", "increase", "decrease", "scale-up", "scale-down"}


def strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines())


def tokenize(text: str) -> List[str]:
    return re.findall(r"\(|\)|[^\s()]+", strip_comments(text))


def parse_sexpressions(text: str) -> List[SExpr]:
    tokens = tokenize(text)
    roots: List[SExpr] = []
    stack: List[List[SExpr]] = []
    for index, token in enumerate(tokens):
        if token == "(":
            node: List[SExpr] = []
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise PDDLParseError(f"Unexpected ')' at token {index}")
            stack.pop()
        else:
            if not stack:
                raise PDDLParseError(f"Token outside an expression at token {index}: {token}")
            stack[-1].append(token.lower())
    if stack:
        raise PDDLParseError(f"Unclosed parenthesis: depth={len(stack)}")
    if not roots:
        raise PDDLParseError("Empty PDDL document")
    return roots


def parse_define(text: str) -> List[SExpr]:
    roots = parse_sexpressions(text)
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise PDDLParseError("Expected exactly one top-level expression")
    root = roots[0]
    if not root or str(root[0]).lower() != "define":
        raise PDDLParseError("Top-level expression must start with define")
    return root


def _head(node: SExpr) -> str:
    if not isinstance(node, list) or not node:
        return ""
    return str(node[0]).lower()


def find_section(root: Sequence[SExpr], keyword: str) -> Optional[List[SExpr]]:
    key = keyword.lower()
    for node in root:
        if isinstance(node, list) and _head(node) == key:
            return node
    return None


def _type_spec(value: SExpr) -> TypeSpec:
    if isinstance(value, list):
        if value and _head(value) == "either":
            members = tuple(str(x).lower() for x in value[1:] if isinstance(x, str))
            return members or ("object",)
        flat = tuple(str(x).lower() for x in value if isinstance(x, str))
        return flat or ("object",)
    return str(value).lower()


def parse_typed_items(items: Sequence[SExpr], default_type: TypeSpec = "object") -> Dict[str, TypeSpec]:
    result: Dict[str, TypeSpec] = {}
    pending: List[str] = []
    index = 0
    while index < len(items):
        item = items[index]
        if isinstance(item, str) and item == "-":
            if index + 1 >= len(items):
                raise PDDLParseError("Typed list ends after '-'")
            spec = _type_spec(items[index + 1])
            for name in pending:
                result[name.lower()] = spec
            pending = []
            index += 2
            continue
        if not isinstance(item, str):
            raise PDDLParseError(f"Unexpected nested expression in typed item list: {item}")
        pending.append(item.lower())
        index += 1
    for name in pending:
        result[name] = default_type
    return result


def type_spec_to_json(spec: TypeSpec) -> Union[str, List[str]]:
    return list(spec) if isinstance(spec, tuple) else spec


def type_spec_from_json(spec: Any) -> TypeSpec:
    if isinstance(spec, list):
        return tuple(str(x).lower() for x in spec)
    return str(spec or "object").lower()


def _as_strings(items: Iterable[SExpr]) -> List[str]:
    return [str(x).lower() for x in items if isinstance(x, str)]


def parse_domain(text: str) -> Dict[str, Any]:
    root = parse_define(text)
    domain_decl = next((x for x in root if isinstance(x, list) and _head(x) == "domain"), None)
    if not domain_decl or len(domain_decl) < 2:
        raise PDDLParseError("Missing (domain NAME) declaration")
    domain_name = str(domain_decl[1]).lower()

    requirements = _as_strings((find_section(root, ":requirements") or [])[1:])
    type_items = (find_section(root, ":types") or [])[1:]
    raw_hierarchy = parse_typed_items(type_items, "object") if type_items else {}
    type_hierarchy = {
        name: (spec[0] if isinstance(spec, tuple) and spec else spec)
        for name, spec in raw_hierarchy.items()
    }
    constants = parse_typed_items((find_section(root, ":constants") or [])[1:], "object")

    predicates: Dict[str, Dict[str, Any]] = {}
    pred_sec = find_section(root, ":predicates") or []
    for declaration in pred_sec[1:]:
        if not isinstance(declaration, list) or not declaration or not isinstance(declaration[0], str):
            continue
        name = declaration[0].lower()
        params = parse_typed_items(declaration[1:], "object")
        predicates[name] = {
            "name": name,
            "arity": len(params),
            "parameters": [
                {"name": pname, "type": type_spec_to_json(ptype)} for pname, ptype in params.items()
            ],
            "argument_types": [type_spec_to_json(x) for x in params.values()],
        }

    actions: Dict[str, Dict[str, Any]] = {}
    for node in root:
        if not isinstance(node, list) or _head(node) not in {":action", ":durative-action"} or len(node) < 2:
            continue
        name = str(node[1]).lower()
        fields: Dict[str, SExpr] = {}
        index = 2
        while index < len(node):
            key = node[index]
            if isinstance(key, str) and key.startswith(":") and index + 1 < len(node):
                fields[key.lower()] = node[index + 1]
                index += 2
            else:
                index += 1
        param_node = fields.get(":parameters", [])
        params = parse_typed_items(param_node if isinstance(param_node, list) else [], "object")
        actions[name] = {
            "name": name,
            "parameters": [
                {"name": pname, "type": type_spec_to_json(ptype)} for pname, ptype in params.items()
            ],
        }

    return {
        "domain_name": domain_name,
        "requirements": requirements,
        "uses_typing": (
            ":typing" in requirements
            or bool(type_hierarchy)
            or any(
                str(value).lower() != "object"
                for signature in predicates.values()
                for value in signature.get("argument_types", [])
            )
        ),
        "type_hierarchy": type_hierarchy,
        "constants": {k: type_spec_to_json(v) for k, v in constants.items()},
        "predicates": predicates,
        "actions": actions,
    }


def domain_uses_typing(domain: Dict[str, Any]) -> bool:
    """Return whether problem objects should be rendered with PDDL types."""
    return bool(domain.get("uses_typing", False))


def _collect_atoms(node: SExpr, stage: str, polarity: bool = True) -> List[Atom]:
    if not isinstance(node, list) or not node:
        return []
    head = _head(node)
    if head == "not":
        return _collect_atoms(node[1], stage, not polarity) if len(node) > 1 else []
    if head in LOGICAL_HEADS:
        start = 1
        if head in {"forall", "exists"}:
            start = 2
        atoms: List[Atom] = []
        for child in node[start:]:
            atoms.extend(_collect_atoms(child, stage, polarity))
        return atoms
    if head in NUMERIC_HEADS or head.startswith(":"):
        return []
    if any(isinstance(x, list) for x in node[1:]):
        return []
    return [Atom(stage=stage, predicate=head, arguments=tuple(_as_strings(node[1:])), polarity=polarity)]


def parse_problem(text: str) -> Dict[str, Any]:
    root = parse_define(text)
    problem_decl = next((x for x in root if isinstance(x, list) and _head(x) == "problem"), None)
    if not problem_decl or len(problem_decl) < 2:
        raise PDDLParseError("Missing (problem NAME) declaration")
    domain_sec = find_section(root, ":domain")
    if not domain_sec or len(domain_sec) < 2:
        raise PDDLParseError("Missing (:domain NAME) section")
    objects = parse_typed_items((find_section(root, ":objects") or [])[1:], "object")
    init_sec = find_section(root, ":init")
    goal_sec = find_section(root, ":goal")
    if init_sec is None or goal_sec is None:
        raise PDDLParseError("Problem must contain both :init and :goal")
    atoms: List[Atom] = []
    for child in init_sec[1:]:
        atoms.extend(_collect_atoms(child, "INIT"))
    for child in goal_sec[1:]:
        atoms.extend(_collect_atoms(child, "GOAL"))
    return {
        "problem_name": str(problem_decl[1]).lower(),
        "domain_name": str(domain_sec[1]).lower(),
        "objects": {k: type_spec_to_json(v) for k, v in objects.items()},
        "atoms": [x.to_dict() for x in atoms],
    }


def syntax_report(problem_pddl: str) -> Dict[str, Any]:
    try:
        parsed = parse_problem(problem_pddl)
        return {"ok": True, "errors": [], "parsed": parsed}
    except PDDLParseError as exc:
        return {"ok": False, "errors": [{"error_type": "pddl_syntax_error", "message": str(exc)}], "parsed": None}


def _ancestors(actual: str, hierarchy: Dict[str, str]) -> List[str]:
    out = [actual]
    seen = {actual}
    current = actual
    while current in hierarchy:
        current = str(hierarchy[current]).lower()
        if current in seen:
            break
        out.append(current)
        seen.add(current)
    if "object" not in seen:
        out.append("object")
    return out


def is_type_compatible(actual: TypeSpec, expected: TypeSpec, hierarchy: Dict[str, str]) -> bool:
    actuals = tuple(actual) if isinstance(actual, tuple) else (actual,)
    expecteds = tuple(expected) if isinstance(expected, tuple) else (expected,)
    for actual_name in actuals:
        ancestors = set(_ancestors(str(actual_name).lower(), hierarchy))
        if any(str(expected_name).lower() in ancestors for expected_name in expecteds):
            return True
    return False


def static_report(domain: Dict[str, Any], problem_pddl: str) -> Dict[str, Any]:
    syn = syntax_report(problem_pddl)
    if not syn["ok"]:
        return {"ok": False, "errors": syn["errors"]}
    problem = syn["parsed"]
    errors: List[Dict[str, Any]] = []
    if problem["domain_name"] != domain["domain_name"]:
        errors.append({
            "error_type": "domain_name_mismatch",
            "expected": domain["domain_name"],
            "observed": problem["domain_name"],
        })
    objects: Dict[str, Any] = dict(domain.get("constants", {}))
    objects.update(problem.get("objects", {}))
    hierarchy = {str(k).lower(): str(v).lower() for k, v in domain.get("type_hierarchy", {}).items()}
    for atom in problem.get("atoms", []):
        predicate = atom["predicate"]
        signature = domain.get("predicates", {}).get(predicate)
        if signature is None:
            errors.append({"error_type": "unknown_predicate", "atom": atom})
            continue
        arguments = atom.get("arguments", [])
        if len(arguments) != int(signature["arity"]):
            errors.append({
                "error_type": "wrong_parameter_count",
                "predicate": predicate,
                "expected": signature["arity"],
                "observed": len(arguments),
                "atom": atom,
            })
            continue
        for index, argument in enumerate(arguments):
            if argument not in objects:
                errors.append({"error_type": "undeclared_object", "argument": argument, "atom": atom})
                continue
            actual = type_spec_from_json(objects[argument])
            expected = type_spec_from_json(signature["argument_types"][index])
            if not is_type_compatible(actual, expected, hierarchy):
                errors.append({
                    "error_type": "wrong_type",
                    "predicate": predicate,
                    "argument": argument,
                    "actual_type": type_spec_to_json(actual),
                    "expected_type": type_spec_to_json(expected),
                    "atom": atom,
                })
    return {"ok": not errors, "errors": errors}


def atom_key(atom: Dict[str, Any]) -> Tuple[str, str, Tuple[str, ...], bool]:
    return (
        str(atom.get("stage", "")).upper(),
        str(atom.get("predicate", "")).lower(),
        tuple(str(x).lower() for x in atom.get("arguments", [])),
        bool(atom.get("polarity", True)),
    )


def render_problem(
    problem_name: str,
    domain_name: str,
    objects: Sequence[Dict[str, Any]],
    facts: Sequence[Dict[str, Any]],
    typed_objects: bool = True,
) -> str:
    grouped: Dict[str, List[str]] = {}
    for obj in objects:
        name = str(obj["name"]).lower()
        typ = obj.get("type", "object")
        if isinstance(typ, list):
            typ = typ[0] if typ else "object"
        typ = str(typ or "object").lower()
        grouped.setdefault(typ, []).append(name)
    if typed_objects:
        object_lines = [f"    {' '.join(sorted(names))} - {typ}" for typ, names in sorted(grouped.items())]
    else:
        object_lines = [f"    {' '.join(sorted(name for names in grouped.values() for name in names))}"]

    def render_atom(fact: Dict[str, Any], indent: str) -> str:
        args = " ".join(str(x).lower() for x in fact.get("arguments", []))
        core = f"({str(fact['predicate']).lower()}{(' ' + args) if args else ''})"
        return f"{indent}{core}" if fact.get("polarity", True) else f"{indent}(not {core})"

    init = [x for x in facts if str(x.get("stage", "")).upper() == "INIT"]
    goal = [x for x in facts if str(x.get("stage", "")).upper() == "GOAL"]
    return "\n".join([
        f"(define (problem {problem_name.lower()})",
        f"  (:domain {domain_name.lower()})",
        "  (:objects",
        *object_lines,
        "  )",
        "  (:init",
        *[render_atom(x, "    ") for x in init],
        "  )",
        "  (:goal",
        "    (and",
        *[render_atom(x, "      ") for x in goal],
        "    )",
        "  )",
        ")",
        "",
    ])


def extract_complete_define(text: str) -> str:
    match = re.search(r"\(\s*define\b", text, re.I)
    if not match:
        return ""
    segment = text[match.start():]
    depth = 0
    started = False
    in_string = False
    escaped = False
    for index, char in enumerate(segment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
            started = True
        elif char == ")":
            depth -= 1
            if started and depth == 0:
                return segment[:index + 1].strip()
    return ""
