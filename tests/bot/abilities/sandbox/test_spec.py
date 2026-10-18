import datetime

import pydantic
import pytest

from mosfet.abilities import sandbox

_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def _spec(**fields: object) -> sandbox.Spec:
    return sandbox.Spec.model_validate({"image": "ghcr.io/stateforward/builder:1.0", **fields})


def _capacity(**fields: object) -> sandbox.Capacity:
    return sandbox.Capacity.model_validate(
        {
            "isolation": "container",
            "networks": ["none", "egress"],
            "max_resources": {"cpus": 4.0, "memory_bytes": 8 * _GIB, "pids": 1024, "workspace_bytes": 2 * _GIB},
            "files": True,
            **fields,
        }
    )


def test_network_defaults_to_none_and_egress_means_outbound_only() -> None:
    assert [network.value for network in sandbox.Network] == ["none", "egress"]
    assert _spec().network is sandbox.Network.NONE


def test_isolation_names_every_host_kind_including_a_user_space_kernel() -> None:
    assert [isolation.value for isolation in sandbox.Isolation] == ["process", "container", "user_kernel", "microvm"]


def test_resources_default_to_the_owner_bench() -> None:
    resources = sandbox.Resources()

    assert resources.cpus == 1.0
    assert resources.memory_bytes == _GIB
    assert resources.pids == 256
    assert resources.workspace_bytes == 512 * _MIB


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("cpus", 0.1, 16.0),
        ("memory_bytes", 64 * _MIB, 32 * _GIB),
        ("pids", 16, 4096),
        ("workspace_bytes", _MIB, 8 * _GIB),
    ],
)
def test_resources_hold_every_bound(field: str, low: float, high: float) -> None:
    assert getattr(sandbox.Resources.model_validate({field: low}), field) == low
    assert getattr(sandbox.Resources.model_validate({field: high}), field) == high
    below = low - 0.01 if isinstance(low, float) else low - 1
    above = high + 0.01 if isinstance(high, float) else high + 1
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.Resources.model_validate({field: below})
    with pytest.raises(pydantic.ValidationError):
        _ = sandbox.Resources.model_validate({field: above})


def test_spec_defaults_are_the_closed_unprivileged_ephemeral_bench() -> None:
    spec = _spec()

    assert spec.resources == sandbox.Resources()
    assert spec.workspace == "/workspace"
    assert spec.env == {"HOME": "/workspace"}
    assert spec.user == "unprivileged"
    assert spec.max_lifetime == datetime.timedelta(hours=8)


@pytest.mark.parametrize(
    "image",
    [
        "python",
        "python:3.13-slim",
        "ghcr.io/stateforward/builder:1.0",
        "localhost:5000/builder",
        "registry.example/team/builder@sha256:" + "a" * 64,
    ],
)
def test_spec_image_takes_an_oci_style_reference(image: str) -> None:
    assert _spec(image=image).image == image


@pytest.mark.parametrize(
    "image",
    ["", "Python", "python:", "builder; rm -rf /", "a b", "x" * 257, "python@sha256:short"],
)
def test_spec_image_refuses_what_is_not_a_reference(image: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(image=image)


def test_spec_root_is_never_offered() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(user="root")


def test_spec_max_lifetime_is_one_minute_to_one_day() -> None:
    assert _spec(max_lifetime=datetime.timedelta(minutes=1)).max_lifetime == datetime.timedelta(minutes=1)
    assert _spec(max_lifetime=datetime.timedelta(hours=24)).max_lifetime == datetime.timedelta(hours=24)
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(max_lifetime=datetime.timedelta(seconds=59))
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(max_lifetime=datetime.timedelta(hours=24, seconds=1))


def test_spec_gives_home_the_workspace_unless_env_sets_it() -> None:
    """DSL-28d: the root is read-only, so ``HOME`` is the workspace, where a tool writing ``~`` can write."""

    assert _spec(env={"LANG": "C.UTF-8"}).env == {"LANG": "C.UTF-8", "HOME": "/workspace"}
    assert _spec(workspace="/work").env == {"HOME": "/work"}
    assert _spec(env={"HOME": "/workspace/home"}).env == {"HOME": "/workspace/home"}


def test_spec_env_is_bounded_plain_text() -> None:
    assert len(_spec(env={f"V{index}": "x" for index in range(31)}).env) == 32
    assert len(_spec(env={"HOME": "/workspace", **{f"V{index}": "x" for index in range(31)}}).env) == 32
    with pytest.raises(pydantic.ValidationError, match="HOME"):
        _ = _spec(env={f"V{index}": "x" for index in range(32)})
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(env={f"V{index}": "x" for index in range(33)})
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(env={"V": "x" * 4097})
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(env={"1V": "x"})
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(env={"V": "a\x00b"})


def test_spec_env_refuses_a_secret_value() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(env={"TOKEN": pydantic.SecretStr("hunter2")})


@pytest.mark.parametrize("workspace", ["/tmp", "/tmp/work", "relative", "/", "/work/../etc", "/work//x", "/work/"])
def test_spec_workspace_is_an_absolute_path_apart_from_tmp(workspace: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(workspace=workspace)


def test_spec_is_frozen_and_closed() -> None:
    spec = _spec()

    with pytest.raises(pydantic.ValidationError):
        spec.network = sandbox.Network.EGRESS
    with pytest.raises(pydantic.ValidationError):
        _ = _spec(privileged=True)


def test_absolute_path_refuses_nul_dot_dot_empty_segments_and_overlong_paths() -> None:
    adapter: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(sandbox.AbsolutePath)

    assert adapter.validate_python("/workspace/src/main.py") == "/workspace/src/main.py"
    assert adapter.validate_python("/" + "a" * 1023) == "/" + "a" * 1023
    for path in ("workspace/main.py", "/workspace/../etc/passwd", "/workspace//main.py", "/workspace/", "/a\x00b"):
        with pytest.raises(pydantic.ValidationError):
            _ = adapter.validate_python(path)
    with pytest.raises(pydantic.ValidationError):
        _ = adapter.validate_python("/" + "a" * 1024)


def test_spec_admits_paths_under_its_workspace_or_tmp_only() -> None:
    spec = _spec(workspace="/home/builder/work")

    assert spec.admits("/home/builder/work")
    assert spec.admits("/home/builder/work/src/main.py")
    assert spec.admits("/tmp")
    assert spec.admits("/tmp/stdin")
    assert not spec.admits("/home/builder/workshop/main.py")
    assert not spec.admits("/home/builder")
    assert not spec.admits("/etc/passwd")
    assert not spec.admits("/tmpfile")


def test_capacity_admits_a_spec_within_it() -> None:
    _capacity().check(_spec(network="egress", resources={"cpus": 4.0, "memory_bytes": 8 * _GIB}))


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"resources": {"cpus": 4.5}}, "cpus"),
        ({"resources": {"memory_bytes": 9 * _GIB}}, "memory_bytes"),
        ({"resources": {"pids": 1025}}, "pids"),
        ({"resources": {"workspace_bytes": 3 * _GIB}}, "workspace_bytes"),
    ],
)
def test_a_spec_beyond_the_capacity_is_refused(fields: dict[str, object], named: str) -> None:
    with pytest.raises(ValueError, match=named):
        _capacity().check(_spec(**fields))


def test_a_network_the_host_does_not_offer_is_refused() -> None:
    with pytest.raises(ValueError, match="egress"):
        _capacity(networks=["none"]).check(_spec(network="egress"))


def test_capacity_declares_at_least_one_network_and_is_frozen() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = _capacity(networks=[])
    capacity = _capacity()

    assert capacity.networks == frozenset({sandbox.Network.NONE, sandbox.Network.EGRESS})
    with pytest.raises(pydantic.ValidationError):
        capacity.files = False


def test_spec_description_is_the_owners_bounded_words_for_what_the_bench_holds() -> None:
    holds = "python 3.13, uv, git, pytest; mosfet installed"

    assert sandbox.Spec(image="localhost/bench:2").description is None
    assert sandbox.Spec(image="localhost/bench:2", description=holds).description == holds
    for refused in ("", "x" * 513):
        with pytest.raises(pydantic.ValidationError):
            _ = sandbox.Spec(image="localhost/bench:2", description=refused)
