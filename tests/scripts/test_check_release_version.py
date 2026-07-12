from __future__ import annotations

import pytest

import scripts.check_release_version as check_release_version


def test_check_release_version_ignores_branch_ref_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")

    assert check_release_version.main([]) == 0


def test_check_release_version_uses_tag_ref_name(monkeypatch: pytest.MonkeyPatch) -> None:
    package_version = check_release_version.verify_dynamic_metadata(check_release_version.PACKAGE_SPECS[0])
    monkeypatch.setenv("GITHUB_REF_NAME", f"v{package_version}")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")

    assert check_release_version.main([]) == 0
