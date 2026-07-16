import ast
import collections
import pathlib


_LEGACY_COORDINATION_METADATA_REFERENCES: dict[str, dict[str, int]] = {
    "abilities/cognition/intuition.py": {
        "literal:bot.intuition.confidence": 4,
        "literal:bot.intuition.confidence_threshold": 4,
        "literal:bot.intuition.escalate": 4,
    },
    "abilities/ability.py": {
        "TERMINAL_RESULT_METADATA_KEY": 3,
        "_COMPOSITE_ATTACHMENT_OPERATION_METADATA_KEY": 5,
    },
    "abilities/conversation/conversation.py": {
        "_CONVERSATION_DECODED_METADATA_KEY": 4,
        "_CONVERSATION_MESSAGE_METADATA_KEY": 4,
        "local:conversation_result_metadata_key": 2,
    },
    "abilities/conversation/host_turn.py": {
        "TERMINAL_RESULT_METADATA_KEY": 1,
        "local:conversation_result_metadata_key": 1,
    },
    "abilities/memory/associative.py": {"_ASSOCIATIVE_MEMORY_PHASE_METADATA_KEY": 4},
    "abilities/participating/participating.py": {"_PARTICIPATING_INPUT_METADATA_KEY": 6},
    "abilities/processing.py": {"operation:iteration": 1},
    "abilities/reading/reading.py": {
        "_READING_CLASSIFIED_METADATA_KEY": 6,
        "_READING_INPUT_METADATA_KEY": 4,
    },
    "bot.py": {
        "_LIFECYCLE_OPERATION_METADATA_KEY": 6,
        "_REBOOT_CLEANUP_METADATA_KEY": 5,
        "_STARTED_ABILITIES_METADATA_KEY": 3,
        "_STARTED_ATTACHMENT_GROUP_METADATA_KEY": 3,
        "_STARTED_DEVICES_METADATA_KEY": 3,
    },
    "device/device.py": {"_FIRMWARE_LIFECYCLE_OPERATION_METADATA_KEY": 7},
    "devices/phone/phone.py": {"_PHONE_CALL_ID_METADATA_KEY": 2},
    "habit/verify.py": {"operation:iteration": 1},
    "protocols/attachment/group.py": {
        "_MEMBER_INDEX_METADATA_KEY": 16,
        "_OPERATION_METADATA_KEY": 19,
        "_REQUEST_CONTEXT_METADATA_KEY": 7,
    },
    "protocols/yamux/stream.py": {"_OPERATION_FAILURE_METADATA_KEY": 10},
}


def _coordination_metadata_references(source_root: pathlib.Path) -> dict[str, dict[str, int]]:
    references: dict[str, dict[str, int]] = {}
    for path in sorted(source_root.rglob("*.py")):
        counts = _coordination_metadata_counts(path.read_text())
        if counts:
            references[str(path.relative_to(source_root))] = dict(counts)
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
    source_root = pathlib.Path(__file__).parents[2] / "src" / "bot"
    actual = _coordination_metadata_references(source_root)
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
