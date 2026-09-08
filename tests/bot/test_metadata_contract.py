import ast
import collections
import pathlib


# Flat allowlist keyed by stable relative paths:
# - bot production: relative to src/bot (e.g. abilities/processing.py)
# - provider production: relative to src/ (e.g. providers/livekit/src/bot/providers/livekit/phone.py)
# Only production .py under those roots is scanned (provider package tests and
# hatch/vendor trees outside each provider's src/ are excluded).
_LEGACY_COORDINATION_METADATA_REFERENCES: dict[str, dict[str, int]] = {
    # for-key iteration over telemetry metadata maps (not coordination keys).
    "abilities/processing.py": {},
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _production_source_entries() -> list[tuple[str, pathlib.Path]]:
    """Production Python sources under core bot and provider packages.

    Returns (allowlist_key, absolute_path) pairs. Keys form one flat map:
    bot paths stay relative to ``src/bot``; provider paths are relative to
    ``src/`` so they cannot collide with bot module names.
    """
    src = _repo_root() / "src"
    entries: list[tuple[str, pathlib.Path]] = []
    bot_root = src / "bot"
    for path in sorted(bot_root.rglob("*.py")):
        entries.append((path.relative_to(bot_root).as_posix(), path))

    providers_root = src / "providers"
    if providers_root.is_dir():
        for provider in sorted(providers_root.iterdir()):
            provider_src = provider / "src"
            if not provider_src.is_dir():
                continue
            for path in sorted(provider_src.rglob("*.py")):
                if "tests" in path.parts:
                    continue
                entries.append((path.relative_to(src).as_posix(), path))
    return entries


def _coordination_metadata_references() -> dict[str, dict[str, int]]:
    references: dict[str, dict[str, int]] = {}
    for key, path in _production_source_entries():
        counts = _coordination_metadata_counts(path.read_text())
        if counts:
            references[key] = dict(counts)
    return references


def _coordination_metadata_counts(source: str) -> collections.Counter[str]:
    tree = ast.parse(source)
    mapping_aliases: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Dict) or _is_dict_constructor(node.value):
                mapping_aliases[node.targets[0].id] = node.value
    aliases = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg) and "metadata" in node.arg.lower()}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value: ast.AST | None = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = [node.target]
            if value is not None and _is_metadata_expression(value, aliases):
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in aliases:
                        aliases.add(target.id)
                        changed = True

    counts: collections.Counter[str] = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.endswith("METADATA_KEY"):
            counts[node.id] += 1
        elif isinstance(node, ast.Attribute) and node.attr.endswith("METADATA_KEY"):
            counts[node.attr] += 1
        key: ast.AST | None = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"get", "pop", "setdefault"} and _is_metadata_expression(node.func.value, aliases):
                key = node.args[0] if node.args else None
            elif node.func.attr == "update" and _is_metadata_expression(node.func.value, aliases):
                if node.args:
                    _count_mapping_keys(counts, node.args[0], mapping_aliases)
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        counts[f"literal:{keyword.arg}"] += 1
                    else:
                        _count_mapping_keys(counts, keyword.value, mapping_aliases)
        elif isinstance(node, ast.Subscript) and _is_metadata_expression(node.value, aliases):
            key = node.slice
        elif isinstance(node, ast.Compare):
            for index, comparator in enumerate(node.comparators):
                operator = node.ops[index]
                if isinstance(operator, ast.In | ast.NotIn) and _is_metadata_expression(comparator, aliases):
                    candidate = node.left if index == 0 else node.comparators[index - 1]
                    if not _is_named_metadata_key(candidate):
                        _count_unmodeled_metadata_key(counts, candidate)
        elif isinstance(node, ast.For) and _is_metadata_expression(node.iter, aliases):
            counts["operation:iteration"] += 1
        elif isinstance(node, ast.keyword) and node.arg in {"metadata", "stage_metadata"}:
            _count_mapping_keys(counts, node.value, mapping_aliases)
        elif isinstance(node, ast.AugAssign) and _is_metadata_expression(node.target, aliases):
            _count_mapping_keys(counts, node.value, mapping_aliases)
        if key is not None and not _is_named_metadata_key(key):
            _count_unmodeled_metadata_key(counts, key)
    return counts


def _is_metadata_expression(node: ast.AST, aliases: set[str] | None = None) -> bool:
    if isinstance(node, ast.Name):
        return node.id in (aliases or set())
    if isinstance(node, ast.Attribute) and node.attr == "metadata":
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"copy", "items", "keys", "values"}
    ):
        return _is_metadata_expression(node.func.value, aliases)
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return _is_metadata_expression(node.args[0], aliases)
    return False


def _is_dict_constructor(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"


def _count_mapping_keys(
    counts: collections.Counter[str],
    node: ast.AST,
    aliases: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> None:
    if isinstance(node, ast.Name) and node.id in aliases:
        if node.id in seen:
            return
        _count_mapping_keys(counts, aliases[node.id], aliases, seen | {node.id})
    elif isinstance(node, ast.Dict):
        for item, value in zip(node.keys, node.values, strict=True):
            if item is None:
                _count_mapping_keys(counts, value, aliases, seen)
            elif not _is_named_metadata_key(item):
                _count_unmodeled_metadata_key(counts, item)
    elif _is_dict_constructor(node):
        assert isinstance(node, ast.Call)
        for keyword in node.keywords:
            if keyword.arg is not None:
                counts[f"literal:{keyword.arg}"] += 1
            else:
                _count_mapping_keys(counts, keyword.value, aliases, seen)


def _is_named_metadata_key(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id.endswith("METADATA_KEY")
        or isinstance(node, ast.Attribute)
        and node.attr.endswith("METADATA_KEY")
    )


def _count_unmodeled_metadata_key(counts: collections.Counter[str], node: ast.AST | None) -> None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        counts[f"literal:{node.value}"] += 1
    elif isinstance(node, ast.Name):
        counts[f"local:{node.id}"] += 1
    elif node is not None:
        counts[f"expression:{type(node).__name__}"] += 1


def test_behavioral_metadata_legacy_allowlist_can_only_shrink() -> None:
    actual = _coordination_metadata_references()
    violations = {
        path: {
            key: count
            for key, count in counts.items()
            if key not in _LEGACY_COORDINATION_METADATA_REFERENCES.get(path, {})
            or count > _LEGACY_COORDINATION_METADATA_REFERENCES[path][key]
        }
        for path, counts in actual.items()
    }
    violations = {path: counts for path, counts in violations.items() if counts}
    assert not violations, f"New behavioral metadata coordination is forbidden: {violations}"
    stale_allowances = {
        path: {key: maximum for key, maximum in counts.items() if actual.get(path, {}).get(key, 0) < maximum}
        for path, counts in _LEGACY_COORDINATION_METADATA_REFERENCES.items()
    }
    stale_allowances = {path: counts for path, counts in stale_allowances.items() if counts}
    assert not stale_allowances, f"Shrink the legacy metadata allowlist with the repaired code: {stale_allowances}"


def test_behavioral_metadata_scanner_rejects_literal_local_and_imported_key_bypasses() -> None:
    counts = _coordination_metadata_counts(
        """
def coordinate(event, local_key, imported):
    event.metadata["literal.key"] = "value"
    event.metadata.get(local_key)
    event.metadata.pop(imported.IMPORTED_METADATA_KEY)
    payload = event.metadata
    payload["aliased.key"] = "value"
    if "compared.key" in event.metadata:
        pass
    for key in event.metadata:
        pass
    dataclasses.replace(event, metadata={"written.key": object()})
    event.metadata.update(operation=value)
    for key in event.metadata.keys():
        pass
    if key in event.metadata.keys():
        pass
    payload_data = {"operation": value}
    dataclasses.replace(event, metadata=payload_data)
    dataclasses.replace(event, metadata={**payload_data})
    event.metadata.update(**payload_data)
    dataclasses.replace(event, stage_metadata=dict(operation=value))
    event.metadata |= {"operation": value}
    message.metadata["operation"] = value
"""
    )

    assert counts == {
        "literal:literal.key": 1,
        "local:local_key": 1,
        "IMPORTED_METADATA_KEY": 1,
        "literal:aliased.key": 1,
        "literal:compared.key": 1,
        "operation:iteration": 2,
        "literal:written.key": 1,
        "literal:operation": 7,
        "local:key": 1,
    }
