"""Backend tests for feature 2: project workspace + project-shared memory.

Server side of docs/二开设计/技术方案.md §3.2 (功能二):
  * projects.json attribute extension + ``POST /api/projects/update``
  * ``/api/session/new`` inheritance (workspace / model / personality) and the
    non-blocking ``workspace_fallback`` / ``project_warning`` responses
  * project shared memory storage (``STATE_DIR/project_memory/<id>.md``),
    ``GET /api/memory`` fields and ``POST /api/memory/write`` section
  * per-turn injection via ``api.streaming._webui_project_context_block``
  * ``POST /api/projects/delete`` best-effort memory cleanup

Two layers:
  1. handler/data-layer tests with captured ``j()``/``bad()`` sinks and
     monkeypatched state files (no server needed);
  2. live round trips against the session test server
     (``tests._pytest_port.BASE``), same style as tests/test_sprint15.py.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.config as cfg
import api.models as models
import api.routes as routes
from api.streaming import (
    _PROJECT_MEMORY_INJECT_MAX_CHARS,
    _webui_ephemeral_system_prompt,
    _webui_project_context_block,
)
from tests._pytest_port import BASE, TEST_STATE_DIR


# ── shared helpers ───────────────────────────────────────────────────────────


class _FakeHandler:
    pass


def _capture(monkeypatch):
    """Capture j()/bad() results instead of writing an HTTP response."""
    cap: dict = {}
    monkeypatch.setattr(
        routes, "j", lambda h, o, *a, **k: (cap.__setitem__("ok", o), True)[1]
    )
    monkeypatch.setattr(
        routes,
        "bad",
        lambda h, m, c=400: (cap.__setitem__("bad", (m, c)), True)[1],
    )
    return cap


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point projects.json / project_memory at tmp_path (never prod state)."""
    projects_file = tmp_path / "projects.json"
    monkeypatch.setattr(cfg, "PROJECTS_FILE", projects_file)
    monkeypatch.setattr(models, "PROJECTS_FILE", projects_file)
    monkeypatch.setattr(models, "PROJECT_MEMORY_DIR", tmp_path / "project_memory")
    monkeypatch.setattr(models, "_projects_migrated", True)
    monkeypatch.setattr(routes, "get_active_profile_name", lambda: "default")
    return projects_file


@pytest.fixture
def ws_store(monkeypatch):
    """Replace the per-profile workspace list with an in-memory list."""
    store: dict = {"list": []}
    monkeypatch.setattr(routes, "load_workspaces", lambda profile=None: list(store["list"]))
    monkeypatch.setattr(
        routes,
        "save_workspaces",
        lambda wss, profile=None: store.__setitem__("list", list(wss)) or None,
    )
    return store


def _write_project(projects_file: Path, **fields) -> dict:
    proj = {
        "project_id": "0123456789ab",
        "name": "demo",
        "color": None,
        "profile": "default",
        "created_at": 1.0,
    }
    proj.update(fields)
    projects_file.write_text(json.dumps([proj]), encoding="utf-8")
    return proj


# ── data layer: api/models.py project memory ─────────────────────────────────


def test_project_memory_path_requires_server_generated_id():
    with pytest.raises(ValueError):
        models.project_memory_path("../../etc/passwd")
    with pytest.raises(ValueError):
        models.project_memory_path("short")
    # Server-generated shape (uuid4().hex[:12]) is accepted.
    assert models.project_memory_path("0f3a9c2d51be").name == "0f3a9c2d51be.md"


def test_project_memory_roundtrip_and_delete(isolated_state):
    pid = "0123456789ab"
    assert models.load_project_memory(pid) == ""
    path = models.save_project_memory(pid, "# 共享记忆\n中文内容")
    assert path.exists()
    assert models.load_project_memory(pid) == "# 共享记忆\n中文内容"
    assert models.project_memory_mtime(pid) is not None
    assert models.delete_project_memory(pid) is True
    assert models.load_project_memory(pid) == ""
    # Deleting a missing file is a no-op, never an error.
    assert models.delete_project_memory(pid) is False
    assert models.delete_project_memory("not-an-id") is False


def test_project_memory_read_refuses_symlinked_file(isolated_state):
    target = isolated_state.parent / "outside.md"
    target.write_text("secret", encoding="utf-8")
    models.PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    link = models.project_memory_path("0123456789ab")
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")
    assert models.load_project_memory("0123456789ab") == ""
    assert models.project_memory_mtime("0123456789ab") is None
    assert models.delete_project_memory("0123456789ab") is False
    assert target.read_text(encoding="utf-8") == "secret"


def test_find_project_is_profile_scoped(isolated_state):
    _write_project(isolated_state, profile="other")
    assert models.find_project("0123456789ab", projects=[]) is None
    assert models.find_project("0123456789ab", profile="default") is None
    assert models.find_project("0123456789ab", profile="other") is not None


# ── POST /api/projects/update ────────────────────────────────────────────────


def test_project_update_sets_attributes_and_registers_workspace(
    isolated_state, ws_store, monkeypatch, tmp_path
):
    _write_project(isolated_state)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cap = _capture(monkeypatch)

    routes._handle_project_update(
        _FakeHandler(),
        {
            "project_id": "0123456789ab",
            "name": "renamed",
            "color": "#7cb9ff",
            "workspace": str(workspace),
            "model": "deepseek-v4-flash",
            "model_provider": "deepseek",
            "personality": "concise",
        },
    )

    assert "bad" not in cap, cap
    proj = cap["ok"]["project"]
    assert proj["name"] == "renamed"
    assert proj["color"] == "#7cb9ff"
    assert proj["workspace"] == str(workspace)
    assert proj["model"] == "deepseek-v4-flash"
    assert proj["model_provider"] == "deepseek"
    assert proj["personality"] == "concise"
    assert proj["updated_at"] > 0
    assert cap["ok"]["workspace_registered"] is True
    # Persisted to disk, and the path landed in the workspace list so
    # resolve_trusted_workspace accepts it for the project's new sessions.
    saved = json.loads(isolated_state.read_text(encoding="utf-8"))
    assert saved[0]["workspace"] == str(workspace)
    assert [w["path"] for w in ws_store["list"]] == [str(workspace)]

    # A second update with the same workspace stays idempotent.
    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "workspace": str(workspace)}
    )
    assert cap["ok"]["workspace_registered"] is False
    assert len(ws_store["list"]) == 1


def test_project_update_absent_keys_untouched_and_clears(
    isolated_state, ws_store, monkeypatch, tmp_path
):
    _write_project(isolated_state, model="keep-me", workspace="/old/place")
    cap = _capture(monkeypatch)

    routes._handle_project_update(_FakeHandler(), {"project_id": "0123456789ab"})
    assert cap["ok"]["project"]["model"] == "keep-me"
    assert cap["ok"]["project"]["workspace"] == "/old/place"

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "workspace": None}
    )
    assert cap["ok"]["project"]["workspace"] is None
    assert cap["ok"]["project"]["model"] == "keep-me"

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "model": "x", "clear_workspace": True}
    )
    assert cap["ok"]["project"]["workspace"] is None

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "personality": None}
    )
    assert cap["ok"]["project"]["personality"] is None


def test_project_update_error_matrix(isolated_state, ws_store, monkeypatch, tmp_path):
    _write_project(isolated_state)
    cap = _capture(monkeypatch)

    routes._handle_project_update(_FakeHandler(), {})
    assert cap["bad"] == ("Missing required field(s): project_id", 400)

    routes._handle_project_update(_FakeHandler(), {"project_id": "doesnotexist"})
    assert cap["bad"] == ("Project not found", 404)

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "color": "red"}
    )
    assert cap["bad"] == ("Invalid color format", 400)

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "workspace": str(tmp_path / "missing")}
    )
    assert cap["bad"][1] == 400
    assert "does not exist" in cap["bad"][0] or "exist" in cap["bad"][0]

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "workspace": "/etc"}
    )
    assert cap["bad"][1] == 400
    assert "system directory" in cap["bad"][0]

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "model": "m" * 200}
    )
    assert cap["bad"] == ("invalid model", 400)

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "model_provider": {"not": "a string"}}
    )
    assert cap["bad"] == ("invalid model_provider", 400)

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "personality": "p" * 100}
    )
    assert cap["bad"] == ("invalid personality", 400)

    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "name": "   "}
    )
    assert cap["bad"] == ("name required", 400)

    # Nothing was persisted by the rejected calls.
    assert json.loads(isolated_state.read_text(encoding="utf-8"))[0]["name"] == "demo"


def test_project_update_rejects_foreign_profile(isolated_state, ws_store, monkeypatch):
    _write_project(isolated_state, profile="someone-else")
    cap = _capture(monkeypatch)
    routes._handle_project_update(
        _FakeHandler(), {"project_id": "0123456789ab", "name": "hijack"}
    )
    assert cap["bad"] == ("Project not found", 404)


# ── POST /api/memory/write section=project_memory ────────────────────────────


def test_memory_write_project_memory_section(isolated_state, monkeypatch, tmp_path):
    _write_project(isolated_state)
    cap = _capture(monkeypatch)

    routes._handle_memory_write(
        _FakeHandler(),
        {"section": "project_memory", "project_id": "0123456789ab", "content": "- 回复使用中文"},
    )

    assert "bad" not in cap, cap
    assert cap["ok"]["section"] == "project_memory"
    written = tmp_path / "project_memory" / "0123456789ab.md"
    assert written.read_text(encoding="utf-8") == "- 回复使用中文"
    assert cap["ok"]["path"] == str(written)
    assert cap["ok"]["mtime"] is not None


def test_memory_write_project_memory_error_paths(isolated_state, monkeypatch):
    _write_project(isolated_state)
    cap = _capture(monkeypatch)

    routes._handle_memory_write(
        _FakeHandler(), {"section": "project_memory", "content": "x"}
    )
    assert cap["bad"] == ("project_id is required", 400)

    routes._handle_memory_write(
        _FakeHandler(),
        {"section": "project_memory", "project_id": "../../etc/passwd", "content": "x"},
    )
    assert cap["bad"] == ("invalid project_id", 400)

    routes._handle_memory_write(
        _FakeHandler(), {"section": "project_memory", "project_id": "aaaaaaaaaaaa", "content": "x"}
    )
    assert cap["bad"] == ("Project not found", 404)

    routes._handle_memory_write(
        _FakeHandler(),
        {"section": "project_memory", "project_id": "0123456789ab", "content": {"nope": 1}},
    )
    assert cap["bad"] == ("content must be a string", 400)

    # Foreign profile: same fail-closed 404 as the other project endpoints.
    monkeypatch.setattr(routes, "get_active_profile_name", lambda: "elsewhere")
    routes._handle_memory_write(
        _FakeHandler(),
        {"section": "project_memory", "project_id": "0123456789ab", "content": "x"},
    )
    assert cap["bad"] == ("Project not found", 404)

    # Unknown section keeps the (updated) whitelist message.
    routes._handle_memory_write(
        _FakeHandler(), {"section": "bogus", "content": "x"}
    )
    assert cap["bad"][1] == 400
    assert "project_memory" in cap["bad"][0]


def test_memory_write_project_memory_rejects_symlink(isolated_state, monkeypatch):
    _write_project(isolated_state)
    outside = isolated_state.parent / "outside.md"
    outside.write_text("untouched", encoding="utf-8")
    models.PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(outside), str(models.project_memory_path("0123456789ab")))
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")
    cap = _capture(monkeypatch)
    routes._handle_memory_write(
        _FakeHandler(),
        {"section": "project_memory", "project_id": "0123456789ab", "content": "changed"},
    )
    assert cap["bad"] == ("Cannot write to a symlinked memory file", 400)
    assert outside.read_text(encoding="utf-8") == "untouched"


# ── GET /api/memory project_memory fields ────────────────────────────────────


def _memory_read_payload(monkeypatch, project_id, profile="default"):
    monkeypatch.setattr(
        routes,
        "get_session",
        lambda sid: SimpleNamespace(project_id=project_id, profile=profile, workspace=""),
    )
    parsed = SimpleNamespace(query=urllib.parse.urlencode({"session_id": "sid1"}))
    return routes._memory_project_memory_payload(parsed)


def test_memory_read_includes_project_memory_fields(isolated_state, monkeypatch):
    _write_project(isolated_state)
    models.save_project_memory("0123456789ab", "# 约定\n- 中文回复")

    payload = _memory_read_payload(monkeypatch, "0123456789ab")

    assert payload["project_memory"] == "# 约定\n- 中文回复"
    assert payload["project_memory_project_id"] == "0123456789ab"
    assert payload["project_memory_project_name"] == "demo"
    assert payload["project_memory_path"].endswith("0123456789ab.md")
    assert payload["project_memory_mtime"] is not None


def test_memory_read_empty_without_project_or_for_foreign_profile(
    isolated_state, monkeypatch
):
    _write_project(isolated_state, profile="other")
    models.save_project_memory("0123456789ab", "hidden")

    no_project = _memory_read_payload(monkeypatch, None)
    assert no_project["project_memory"] == ""
    assert no_project["project_memory_project_id"] == ""

    foreign = _memory_read_payload(monkeypatch, "0123456789ab", profile="default")
    assert foreign["project_memory"] == ""
    assert foreign["project_memory_project_id"] == ""

    # A project that no longer exists reads as empty too.
    missing = _memory_read_payload(monkeypatch, "aaaaaaaaaaaa")
    assert missing["project_memory"] == ""


# ── runtime injection: api/streaming.py ──────────────────────────────────────


def _session(project_id, profile="default"):
    return SimpleNamespace(project_id=project_id, profile=profile)


def test_project_context_block_injects_shared_memory(isolated_state):
    _write_project(isolated_state, workspace="/tmp/demo-ws")
    models.save_project_memory("0123456789ab", "---\ntags: [a]\n---\n# 共享\n- 中文回复")

    block = _webui_project_context_block(_session("0123456789ab"))

    assert "Project context (shared by every session in this project):" in block
    assert "demo (project_id: 0123456789ab)" in block
    assert "Project workspace: /tmp/demo-ws" in block
    assert "--- BEGIN PROJECT SHARED MEMORY ---" in block
    assert "--- END PROJECT SHARED MEMORY ---" in block
    # Frontmatter is stripped before injection.
    assert "tags:" not in block
    assert "- 中文回复" in block


def test_project_context_block_degrades_safely(isolated_state):
    # No project on the session → nothing injected.
    assert _webui_project_context_block(_session(None)) == ""
    # Session without the attribute at all (legacy object) → nothing injected.
    assert _webui_project_context_block(SimpleNamespace()) == ""
    # Unknown project id → nothing injected.
    assert _webui_project_context_block(_session("aaaaaaaaaaaa")) == ""
    # Nothing at all → still a valid block with the "not written yet" note.
    _write_project(isolated_state)
    block = _webui_project_context_block(_session("0123456789ab"))
    assert "No shared project memory has been written yet." in block
    assert "Project workspace: (not set)" in block


def test_project_context_block_is_profile_scoped(isolated_state):
    _write_project(isolated_state, profile="other")
    models.save_project_memory("0123456789ab", "other profile secret")
    assert _webui_project_context_block(_session("0123456789ab", profile="default")) == ""
    assert "other profile secret" in _webui_project_context_block(
        _session("0123456789ab", profile="other")
    )


def test_project_context_block_truncates_long_memory(isolated_state):
    _write_project(isolated_state)
    models.save_project_memory("0123456789ab", "x" * (_PROJECT_MEMORY_INJECT_MAX_CHARS + 500))

    block = _webui_project_context_block(_session("0123456789ab"))

    assert "truncated at 8000 characters" in block
    body = block.split("--- BEGIN PROJECT SHARED MEMORY ---", 1)[1]
    assert len(body.split("--- END PROJECT SHARED MEMORY ---")[0]) < _PROJECT_MEMORY_INJECT_MAX_CHARS + 400


def test_project_block_is_appended_last_in_ephemeral_prompt(isolated_state):
    _write_project(isolated_state)
    models.save_project_memory("0123456789ab", "PROJECT-MEMORY-MARKER")

    prompt = _webui_ephemeral_system_prompt(
        "PERSONALITY-MARKER",
        surface_context={"source": "webui", "session_id": "sid1"},
        config_data=None,
        project_block=_webui_project_context_block(_session("0123456789ab")),
    )

    assert "PERSONALITY-MARKER" in prompt
    assert "PROJECT-MEMORY-MARKER" in prompt
    # Stable content first, volatile project block last (prefix-cache friendly).
    assert prompt.rstrip().endswith("--- END PROJECT SHARED MEMORY ---")
    # No project → prompt unchanged from the previous behaviour.
    assert "PROJECT-MEMORY-MARKER" not in _webui_ephemeral_system_prompt(
        None, surface_context={"source": "webui"}, config_data=None
    )


# ── live test-server round trips ─────────────────────────────────────────────


def _get_json(path: str):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.loads(response.read()), response.status


def _post_json(path: str, body: dict | None = None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read()), response.status
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read()), exc.code


@pytest.fixture
def live_projects():
    """Track live-server projects/sessions and clean them up afterwards."""
    pids: list = []
    sids: list = []
    yield pids, sids
    for sid in sids:
        try:
            _post_json("/api/session/delete", {"session_id": sid})
        except Exception:
            pass
    for pid in pids:
        try:
            _post_json("/api/projects/delete", {"project_id": pid})
        except Exception:
            pass


def _new_live_project(pids, **fields):
    body = {"name": fields.pop("name", "ws-feature-project"), **fields}
    payload, status = _post_json("/api/projects/create", body)
    assert status == 200, payload
    pid = payload["project"]["project_id"]
    pids.append(pid)
    return pid


def test_live_project_update_roundtrip_and_workspace_registration(tmp_path, live_projects):
    pids, _ = live_projects
    pid = _new_live_project(pids, name="live-update")
    workspace = tmp_path / "live-ws"
    workspace.mkdir()

    payload, status = _post_json(
        "/api/projects/update",
        {
            "project_id": pid,
            "workspace": str(workspace),
            "model": "openai/gpt-5.4-mini",
            "model_provider": "openai",
        },
    )

    assert status == 200, payload
    assert payload["workspace_registered"] is True
    # GET /api/projects reflects the new fields (additive, no migration).
    listed, _ = _get_json("/api/projects")
    row = next(p for p in listed["projects"] if p["project_id"] == pid)
    assert row["workspace"] == str(workspace)
    assert row["model"] == "openai/gpt-5.4-mini"
    # Auto-registration makes the path trusted for the project's new sessions.
    ws_list, _ = _get_json("/api/workspaces")
    assert str(workspace) in [w["path"] for w in ws_list["workspaces"]]

    # Errors: unknown project → 404, bad color → 400, clearing works.
    _, status = _post_json("/api/projects/update", {"project_id": "aaaaaaaaaaaa"})
    assert status == 404
    _, status = _post_json("/api/projects/update", {"project_id": pid, "color": "nope"})
    assert status == 400
    cleared, status = _post_json(
        "/api/projects/update", {"project_id": pid, "workspace": None}
    )
    assert status == 200
    assert cleared["project"]["workspace"] is None


def test_live_session_new_inherits_project_workspace(tmp_path, live_projects):
    pids, sids = live_projects
    pid = _new_live_project(pids, name="live-inherit")
    workspace = tmp_path / "inherit-ws"
    workspace.mkdir()
    _, status = _post_json(
        "/api/projects/update", {"project_id": pid, "workspace": str(workspace)}
    )
    assert status == 200

    payload, status = _post_json(
        "/api/session/new", {"project_id": pid, "worktree": False}
    )
    assert status == 200, payload
    sid = payload["session"]["session_id"]
    sids.append(sid)

    assert payload["session"]["workspace"] == str(workspace)
    assert payload["session"]["project_id"] == pid
    assert "workspace_fallback" not in payload

    # An explicit workspace in the request still wins over the project default.
    other = tmp_path / "explicit-ws"
    other.mkdir()
    _post_json("/api/workspaces/add", {"path": str(other)})
    payload2, status = _post_json(
        "/api/session/new",
        {"project_id": pid, "workspace": str(other), "worktree": False},
    )
    assert status == 200, payload2
    sids.append(payload2["session"]["session_id"])
    assert payload2["session"]["workspace"] == str(other)


def test_live_session_new_falls_back_when_project_workspace_missing(
    tmp_path, live_projects
):
    pids, sids = live_projects
    pid = _new_live_project(pids, name="live-fallback")
    workspace = tmp_path / "gone-soon"
    workspace.mkdir()
    _, status = _post_json(
        "/api/projects/update", {"project_id": pid, "workspace": str(workspace)}
    )
    assert status == 200

    # The directory disappears after being configured.
    workspace.rmdir()

    payload, status = _post_json(
        "/api/session/new", {"project_id": pid, "worktree": False}
    )

    assert status == 200, payload
    sids.append(payload["session"]["session_id"])
    assert "workspace_fallback" in payload
    assert payload["workspace_fallback"]["reason"]
    assert payload["workspace_fallback"]["used"]
    assert payload["session"]["workspace"] != str(workspace)


def test_live_shared_memory_write_read_and_cleanup(live_projects):
    pids, sids = live_projects
    pid = _new_live_project(pids, name="live-memory")
    payload, status = _post_json(
        "/api/session/new", {"project_id": pid, "worktree": False}
    )
    assert status == 200, payload
    sid = payload["session"]["session_id"]
    sids.append(sid)

    # Not written yet: the panel sees an empty (but addressed) field set.
    read, status = _get_json(f"/api/memory?session_id={urllib.parse.quote(sid)}")
    assert status == 200
    assert read["project_memory"] == ""
    assert read["project_memory_project_id"] == pid

    written, status = _post_json(
        "/api/memory/write",
        {"section": "project_memory", "project_id": pid, "content": "- 项目约定：中文回复"},
    )
    assert status == 200, written
    assert written["section"] == "project_memory"
    memory_file = TEST_STATE_DIR / "project_memory" / f"{pid}.md"
    assert memory_file.read_text(encoding="utf-8") == "- 项目约定：中文回复"

    read2, _ = _get_json(f"/api/memory?session_id={urllib.parse.quote(sid)}")
    assert read2["project_memory"] == "- 项目约定：中文回复"
    assert read2["project_memory_mtime"] is not None
    assert read2["project_memory_project_name"] == "live-memory"

    # Deleting the project removes the shared memory file (best-effort).
    _, status = _post_json("/api/projects/delete", {"project_id": pid})
    assert status == 200
    pids.remove(pid)
    assert not memory_file.exists()


def test_live_memory_write_rejects_unknown_and_foreign_projects(live_projects):
    pids, _ = live_projects
    pid = _new_live_project(pids, name="live-foreign")

    payload, status = _post_json(
        "/api/memory/write",
        {"section": "project_memory", "project_id": "../../etc/passwd", "content": "x"},
    )
    assert status == 400

    payload, status = _post_json(
        "/api/memory/write",
        {"section": "project_memory", "project_id": "aaaaaaaaaaaa", "content": "x"},
    )
    assert status == 404

    # Sanity: the project's own id is accepted.
    payload, status = _post_json(
        "/api/memory/write",
        {"section": "project_memory", "project_id": pid, "content": "ok"},
    )
    assert status == 200, payload
