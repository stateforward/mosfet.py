import ast
import inspect
import types


def _is_metadata_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "metadata"
        or isinstance(node, ast.Name)
        and node.id.endswith("metadata")
    )


def assert_metadata_is_not_coordination(module: types.ModuleType) -> None:
    """Fail when an HSM module treats telemetry metadata as behavioral state."""

    source = inspect.getsource(module)
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.endswith("METADATA_KEY"):
                    violations.append(f"coordination metadata key {target.id} at line {node.lineno}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault", "update"}
        ):
            owner = node.func.value
            if _is_metadata_expression(owner):
                violations.append(f"metadata {node.func.attr} at line {node.lineno}")
        if isinstance(node, ast.Subscript) and _is_metadata_expression(node.value):
            violations.append(f"metadata subscript at line {node.lineno}")
        if isinstance(node, ast.For) and _is_metadata_expression(node.iter):
            violations.append(f"metadata iteration at line {node.lineno}")
        if isinstance(node, ast.Compare) and (
            _is_metadata_expression(node.left) or any(_is_metadata_expression(item) for item in node.comparators)
        ):
            violations.append(f"metadata comparison at line {node.lineno}")
    assert not violations, "\n".join(violations)


def assert_metadata_key_prefix_is_absent(module: types.ModuleType, prefix: str) -> None:
    """Fail if a retired coordination-key family is reintroduced."""

    tree = ast.parse(inspect.getsource(module))
    names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert not {name for name in names if name.startswith(prefix)}
