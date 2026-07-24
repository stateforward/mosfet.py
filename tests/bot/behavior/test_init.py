from bot import behavior

def test_behavior_package_exports_public_domain_api() -> None:
    assert behavior.Instance is behavior.instance.Instance
    assert behavior.start is behavior.instance.start
    assert behavior.Behavior is behavior.behavior.Behavior
    assert behavior.Source is behavior.source.Source
    assert behavior.EventContract is behavior.source.EventContract
    assert behavior.SourceError is behavior.source.SourceError
    assert behavior.JsonSchema is behavior.schema.JsonSchema
    assert behavior.build is behavior.compiler.build
    assert behavior.check is behavior.instance.check
    assert behavior.Report is behavior.diagnostic.Report
    assert behavior.Diagnostic is behavior.diagnostic.Diagnostic
    assert behavior.parse_source is behavior.source.parse_source
    assert behavior.STARLARK_API is behavior.source.STARLARK_API
    assert "hsm.define" in behavior.STARLARK_API
    assert callable(behavior.define_model)
