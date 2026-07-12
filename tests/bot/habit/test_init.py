from bot import habit

def test_habit_package_exports_public_domain_api() -> None:
    assert habit.Instance is habit.instance.Instance
    assert habit.start is habit.instance.start
    assert habit.Behavior is habit.behavior.Behavior
    assert habit.Source is habit.source.Source
    assert habit.EventContract is habit.source.EventContract
    assert habit.SourceError is habit.source.SourceError
    assert habit.JsonSchema is habit.schema.JsonSchema
    assert habit.build is habit.compiler.build
    assert habit.check is habit.instance.check
    assert habit.Report is habit.diagnostic.Report
    assert habit.Diagnostic is habit.diagnostic.Diagnostic
    assert habit.parse_source is habit.source.parse_source
    assert habit.STARLARK_API is habit.source.STARLARK_API
    assert "hsm.define" in habit.STARLARK_API
    assert callable(habit.define_model)
