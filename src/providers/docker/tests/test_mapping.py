import base64
import datetime
import json
import typing

import pytest

from mosfet.abilities import sandbox
from mosfet.abilities.sandbox import host
from mosfet.providers.docker import mapping

_SPEC = sandbox.Spec(
    image="ghcr.io/stateforward/builder:1.0",
    resources=sandbox.Resources(cpus=1.5, memory_bytes=512 * 1024**2, pids=128, workspace_bytes=256 * 1024**2),
    env={"LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
)
_ROOTLESS: typing.Final[dict[str, object]] = {
    "SecurityOptions": ["name=seccomp,profile=default", "name=rootless", "name=cgroupns"],
    "CgroupVersion": "2",
    "MemoryLimit": True,
    "PidsLimit": True,
    "Runtimes": {"runc": {"path": "runc"}},
}


def test_the_create_body_maps_every_row_of_the_bench() -> None:
    body = mapping.create_body(_SPEC, owner="builder/7f3a", runtime="runc", egress_network=None)

    assert body == {
        "Image": "ghcr.io/stateforward/builder:1.0",
        "Entrypoint": ["sleep", "infinity"],
        "Cmd": [],
        "User": "65532:65532",
        "WorkingDir": "/workspace",
        "Env": ["LANG=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1"],
        "Labels": {"org.stateforward.mosfet.sandbox": "1", "org.stateforward.mosfet.owner": "builder/7f3a"},
        "HostConfig": {
            "NanoCpus": 1_500_000_000,
            "Memory": 536_870_912,
            "MemorySwap": 536_870_912,
            "PidsLimit": 128,
            "Tmpfs": {
                "/workspace": "rw,nosuid,nodev,size=268435456,mode=1777",
                "/tmp": "rw,nosuid,nodev,size=64m",
            },
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "CapAdd": [],
            "SecurityOpt": ["no-new-privileges:true"],
            "Binds": [],
            "Mounts": [],
            "Devices": [],
            "VolumesFrom": [],
            "Init": True,
            "Runtime": "runc",
        },
    }


def test_egress_uses_the_owners_bridge_and_runsc_is_passed_through() -> None:
    spec = _SPEC.model_copy(update={"network": sandbox.Network.EGRESS, "workspace": "/home/work"})

    body = mapping.create_body(spec, owner="o", runtime="runsc", egress_network="mosfet-egress")
    host_config = typing.cast(dict[str, object], body["HostConfig"])

    assert host_config["NetworkMode"] == "mosfet-egress"
    assert host_config["Runtime"] == "runsc"
    assert body["WorkingDir"] == "/home/work"
    assert typing.cast(dict[str, str], host_config["Tmpfs"])["/home/work"].startswith("rw,nosuid,nodev,size=")


def test_egress_without_an_owner_bridge_is_refused() -> None:
    spec = _SPEC.model_copy(update={"network": sandbox.Network.EGRESS})

    with pytest.raises(ValueError, match="egress"):
        _ = mapping.create_body(spec, owner="o", runtime="runc", egress_network=None)


def test_the_host_environment_never_reaches_the_machine() -> None:
    body = mapping.create_body(sandbox.Spec(image="python:3.13-slim"), owner="o", runtime="runc", egress_network=None)

    assert body["Env"] == []
    assert "PATH" not in json.dumps(body)


def test_a_run_is_argv_under_a_kill_timeout_with_quotes_and_dollars_kept_positional() -> None:
    command = sandbox.RunData(argv=("python", "-c", "print('$HOME \"x\"')"), timeout=datetime.timedelta(seconds=1.2))

    assert mapping.timeout_seconds(command) == 2
    assert mapping.run_argv(command, stdin_path=None) == (
        "timeout",
        "-s",
        "KILL",
        "2",
        "python",
        "-c",
        "print('$HOME \"x\"')",
    )


def test_a_run_with_stdin_reads_the_staged_file_through_a_fixed_wrapper() -> None:
    command = sandbox.RunData(argv=("cat", "-n"), stdin="3\n4\n")
    path = mapping.stdin_path("4f2c9e1b")

    argv = mapping.run_argv(command, stdin_path=path)

    assert path == "/tmp/.mosfet-stdin-4f2c9e1b"
    assert argv[:4] == ("timeout", "-s", "KILL", "60")
    assert argv[4:7] == ("sh", "-c", mapping.STDIN_SCRIPT)
    assert argv[7:] == (path, "cat", "-n")


def test_a_write_decodes_base64_argv_chunks_into_the_file_through_a_fixed_wrapper() -> None:
    text = "é" * 100_000 + "\n"

    argv = mapping.write_argv("/workspace/app/main.py", text)

    assert argv[:5] == ("sh", "-c", mapping.WRITE_SCRIPT, "sh", "/workspace/app/main.py")
    chunks = argv[5:]
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= mapping.WRITE_CHUNK_CHARS for chunk in chunks)
    assert base64.b64decode("".join(chunks)).decode() == text
    assert mapping.write_argv("/tmp/empty", "")[5:] == ()


def test_a_read_checks_for_a_regular_file_through_a_fixed_wrapper() -> None:
    assert mapping.read_argv("/workspace/report.txt") == (
        "sh",
        "-c",
        mapping.READ_SCRIPT,
        "sh",
        "/workspace/report.txt",
    )
    assert f"exit {mapping.NOT_FOUND_EXIT}" in mapping.READ_SCRIPT
    assert "-L" in mapping.READ_SCRIPT


def test_an_exec_body_attaches_output_only_as_the_unprivileged_user() -> None:
    assert mapping.exec_body(("id", "-u")) == {
        "Cmd": ["id", "-u"],
        "AttachStdin": False,
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": False,
        "User": "65532:65532",
        "Env": [],
    }
    body = mapping.exec_body(("make",), env={"DEBUG": "1"}, cwd="/workspace/app")
    assert (body["Env"], body["WorkingDir"]) == (["DEBUG=1"], "/workspace/app")


def test_a_machines_limits_are_read_from_its_cgroup_files() -> None:
    assert mapping.LIMITS_ARGV == (
        "cat",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/pids.max",
        "/sys/fs/cgroup/memory.max",
    )


@pytest.mark.parametrize(
    ("limits", "applied"),
    [
        ("150000 100000\n128\n536870912\n", True),
        ("100000 100000\n64\n536866816\n", True),
        ("150000 100000\n128\n", False),
        ("max 100000\n128\n536870912\n", False),
        ("150000 100000\nmax\n536870912\n", False),
        ("150000 100000\n128\nmax\n", False),
        ("200000 100000\n128\n536870912\n", False),
        ("150000 100000\n256\n536870912\n", False),
        ("150000 100000\n128\n1073741824\n", False),
        ("0 100000\n128\n536870912\n", False),
        ("150000 0\n128\n536870912\n", False),
        ("150000 100000\n128\n536870912\nextra\n", False),
        ("", False),
    ],
)
def test_limits_are_applied_only_when_each_is_a_number_no_greater_than_asked(limits: str, applied: bool) -> None:
    assert mapping.limits_applied(limits, _SPEC.resources) is applied


def test_the_oom_kill_count_is_read_from_memory_events() -> None:
    assert mapping.OOM_ARGV == ("cat", "/sys/fs/cgroup/memory.events")
    assert mapping.oom_kills("low 0\nhigh 0\nmax 35\noom 1\noom_kill 2\noom_group_kill 0\n") == 2
    assert mapping.oom_kills("low 0\noom 0\n") == 0
    assert mapping.oom_kills("garbage") == 0


def test_the_sweep_filters_by_both_labels() -> None:
    assert json.loads(mapping.sweep_filters("builder/7f3a")) == {
        "label": ["org.stateforward.mosfet.sandbox=1", "org.stateforward.mosfet.owner=builder/7f3a"]
    }


def test_the_egress_network_is_a_bridge_without_inter_container_traffic() -> None:
    assert mapping.network_body("mosfet-egress") == {
        "Name": "mosfet-egress",
        "Driver": "bridge",
        "Internal": False,
        "Options": {"com.docker.network.bridge.enable_icc": "false"},
        "Labels": {"org.stateforward.mosfet.sandbox": "1"},
    }


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, host.FailureKind.INVALID_INPUT),
        (404, host.FailureKind.UNAVAILABLE),
        (409, host.FailureKind.FAILED),
        (403, host.FailureKind.FAILED),
        (500, host.FailureKind.FAILED),
        (503, host.FailureKind.FAILED),
    ],
)
def test_statuses_map_to_failure_kinds(status: int, kind: host.FailureKind) -> None:
    assert mapping.failure_kind(status) is kind


@pytest.mark.parametrize(("status", "label"), [(200, "2xx"), (204, "2xx"), (304, "3xx"), (404, "4xx"), (503, "5xx")])
def test_statuses_fall_in_closed_classes(status: int, label: str) -> None:
    assert mapping.status_class(status) == label


def test_a_rootless_cgroup_v2_daemon_that_enforces_every_limit_is_accepted() -> None:
    info = mapping.Info.model_validate(_ROOTLESS)

    assert mapping.info_refusal(info, runtime="runc") is None


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"]}, "rootful"),
        ({"SecurityOptions": []}, "rootful"),
        ({"CgroupVersion": "1"}, "cgroup_v1"),
        ({"MemoryLimit": False}, "limit_unsupported"),
        ({"PidsLimit": False}, "limit_unsupported"),
    ],
)
def test_a_daemon_that_is_rootful_or_cannot_enforce_the_limits_is_refused(
    change: dict[str, object], refusal: mapping.Refusal
) -> None:
    info = mapping.Info.model_validate({**_ROOTLESS, **change})

    assert mapping.info_refusal(info, runtime="runc") == refusal


def test_a_daemon_that_reports_no_cfs_quota_is_accepted_since_each_machine_verifies_its_limits() -> None:
    podman = mapping.Info.model_validate({**_ROOTLESS, "CpuCfsQuota": False})

    assert mapping.info_refusal(podman, runtime="runc") is None


def test_runsc_must_be_a_daemon_runtime() -> None:
    without = mapping.Info.model_validate(_ROOTLESS)
    with_runsc = mapping.Info.model_validate({**_ROOTLESS, "Runtimes": {"runc": {}, "runsc": {}}})

    assert mapping.info_refusal(without, runtime="runsc") == "runtime_missing"
    assert mapping.info_refusal(with_runsc, runtime="runsc") is None


def test_missing_info_fields_count_as_unsupported() -> None:
    assert mapping.info_refusal(mapping.Info.model_validate({}), runtime="runc") == "rootful"


@pytest.mark.parametrize(
    ("api", "refusal"), [("1.44", None), ("1.41", None), ("1.40", "api_too_old"), ("x", "api_too_old")]
)
def test_the_daemon_api_must_reach_the_requested_version(api: str, refusal: mapping.Refusal | None) -> None:
    version = mapping.Version.model_validate({"ApiVersion": api, "Version": "29.5.2"})

    assert mapping.version_refusal(version, api_version="v1.41") == refusal


def test_only_engine_ids_are_put_in_a_request_path() -> None:
    assert mapping.is_id("4f2c9e1b7a30")
    assert mapping.is_id("a" * 64)
    for candidate in ("../networks", "4f2c/exec", "ABCDEF123456", "abc", "a" * 65, ""):
        assert not mapping.is_id(candidate)
    assert mapping.Identified.model_validate({"Id": "b" * 64}).id == "b" * 64
    with pytest.raises(ValueError):
        _ = mapping.Identified.model_validate({"Id": "../x"})
