"""Security regression tests for thread-scoped temporary Slack delegation."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.pairing import PairingStore, _merge_pairing_dir
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource



def _store(tmp_path, monkeypatch) -> PairingStore:
    import gateway.pairing as pairing_mod

    monkeypatch.setattr(pairing_mod, "PAIRING_DIR", tmp_path / "pairing")
    return PairingStore()


def _store_at(path) -> PairingStore:
    path.mkdir(parents=True, exist_ok=True)
    store = PairingStore.__new__(PairingStore)
    store._dir = path
    store._lock = threading.RLock()
    store._profile = None
    return store


def _slack_source(
    *,
    user_id: str = "U06EGAQ943V",
    channel_id: str = "C_RECIPES",
    thread_id: str = "1787129215.308139",
    workspace_id: str = "T_WORKSPACE",
) -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=channel_id,
        chat_type="group",
        user_id=user_id,
        user_name="Guest",
        thread_id=thread_id,
        scope_id=workspace_id,
    )


def _grant(store: PairingStore, *, now: float = 1_000.0) -> dict:
    return store.grant_temporary(
        platform="slack",
        user_id="U06EGAQ943V",
        user_name="Guest",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        granted_by="U04HAU25G",
        duration_seconds=3600,
        purpose="Qualify recipe migration feedback only",
        now=now,
    )


def test_temporary_grant_matches_only_exact_workspace_channel_thread_and_user(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    grant = _grant(store)

    assert store.get_active_delegation(_slack_source(), now=1_001.0) == grant
    assert store.get_active_delegation(
        _slack_source(user_id="U029M30PFS8"), now=1_001.0
    ) is None
    assert store.get_active_delegation(
        _slack_source(channel_id="C_OTHER"), now=1_001.0
    ) is None
    assert store.get_active_delegation(
        _slack_source(thread_id="999.000"), now=1_001.0
    ) is None
    assert store.get_active_delegation(
        _slack_source(workspace_id="T_OTHER"), now=1_001.0
    ) is None


def test_temporary_grant_expires_and_is_pruned(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    _grant(store, now=1_000.0)

    assert store.get_active_delegation(_slack_source(), now=4_599.9) is not None
    assert store.get_active_delegation(_slack_source(), now=4_600.0) is None
    assert store.list_temporary(
        platform="slack",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        now=4_600.0,
    ) == []

    payload = json.loads((store._dir / "slack-delegations.json").read_text())
    assert payload["version"] == 1
    assert payload["grants"] == {}
    assert payload["audit"][-1]["event"] == "expired"


def test_malformed_expiry_fails_closed_and_is_pruned(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    path = store._dir / "slack-delegations.json"
    payload = json.loads(path.read_text())
    next(iter(payload["grants"].values()))["expires_at"] = "not-a-number"
    path.write_text(json.dumps(payload))

    assert store.get_active_delegation(_slack_source(), now=1_001.0) is None
    cleaned = json.loads(path.read_text())
    assert cleaned["grants"] == {}
    assert cleaned["audit"][-1]["event"] == "invalid_record_pruned"


def test_whitespace_only_persisted_purpose_fails_closed(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    path = store._dir / "slack-delegations.json"
    payload = json.loads(path.read_text())
    next(iter(payload["grants"].values()))["purpose"] = "   "
    path.write_text(json.dumps(payload))

    assert store.get_active_delegation(_slack_source(), now=1_001.0) is None
    cleaned = json.loads(path.read_text())
    assert cleaned["grants"] == {}
    assert cleaned["audit"][-1]["event"] == "invalid_record_pruned"


def test_corrupt_security_state_fails_closed_without_resetting_file(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    path = store._dir / "slack-delegations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G")

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()

    assert runner._is_user_authorized(_slack_source()) is False
    assert path.read_text(encoding="utf-8") == "{broken"

    monkeypatch.setenv("SLACK_DELEGATION_ADMINS", "U04HAU25G")
    event = MessageEvent(
        text="/delegate status",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U04HAU25G"),
    )
    response = asyncio.run(runner._handle_delegate_command(event))
    assert "unavailable or corrupted" in response
    assert path.read_text(encoding="utf-8") == "{broken"


def test_non_utf8_and_unknown_top_level_state_fail_closed(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    path = store._dir / "slack-delegations.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(RuntimeError, match="unreadable"):
        store.get_active_delegation(_slack_source(), now=1_001.0)
    assert path.read_bytes() == b"\xff\xfe"

    path.write_text(
        json.dumps({"version": 1, "grants": {}, "audit": [], "unexpected": True})
    )
    with pytest.raises(RuntimeError, match="invalid schema"):
        store.get_active_delegation(_slack_source(), now=1_001.0)
    assert "unexpected" in path.read_text()

    for invalid_version in (True, 1.0):
        path.write_text(
            json.dumps({"version": invalid_version, "grants": {}, "audit": []})
        )
        with pytest.raises(RuntimeError, match="invalid schema"):
            store.get_active_delegation(_slack_source(), now=1_001.0)

    path.unlink()
    _grant(store)
    state = json.loads(path.read_text())
    next(iter(state["grants"].values()))["unexpected_admin"] = True
    path.write_text(json.dumps(state))
    assert store.get_active_delegation(_slack_source(), now=1_001.0) is None
    cleaned = json.loads(path.read_text())
    assert cleaned["grants"] == {}
    assert cleaned["audit"][-1]["event"] == "invalid_record_pruned"

    path.write_text(
        json.dumps({"version": 1, "grants": {}, "audit": [{"malformed": True}]})
    )
    with pytest.raises(RuntimeError, match="invalid schema"):
        store.get_active_delegation(_slack_source(), now=1_001.0)


def test_split_directory_migration_unions_delegations_and_audit(tmp_path):
    active_dir = tmp_path / "active"
    alternate_dir = tmp_path / "alternate"
    active = _store_at(active_dir)
    alternate = _store_at(alternate_dir)
    active.grant_temporary(
        platform="slack",
        user_id="U_ACTIVE",
        user_name="Active",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        granted_by="U04HAU25G",
        duration_seconds=3600,
        purpose="Active purpose",
        now=1_000.0,
    )
    alternate.grant_temporary(
        platform="slack",
        user_id="U_ALTERNATE",
        user_name="Alternate",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        granted_by="U04HAU25G",
        duration_seconds=3600,
        purpose="Alternate purpose",
        now=1_001.0,
    )

    _merge_pairing_dir(active_dir, alternate_dir)

    state = json.loads((active_dir / "slack-delegations.json").read_text())
    assert {grant["user_id"] for grant in state["grants"].values()} == {
        "U_ACTIVE",
        "U_ALTERNATE",
    }
    assert len(state["audit"]) == 2


@pytest.mark.parametrize("corruption", ["malformed", "oversized"])
def test_split_directory_migration_rejects_corrupt_audit(tmp_path, corruption):
    active_dir = tmp_path / "active"
    alternate_dir = tmp_path / "alternate"
    active = _store_at(active_dir)
    alternate = _store_at(alternate_dir)
    for store, user_id, now in (
        (active, "U_ACTIVE", 1_000.0),
        (alternate, "U_ALTERNATE", 1_001.0),
    ):
        store.grant_temporary(
            platform="slack",
            user_id=user_id,
            user_name=user_id,
            chat_id="C_RECIPES",
            thread_id="1787129215.308139",
            scope_id="T_WORKSPACE",
            granted_by="U04HAU25G",
            duration_seconds=3600,
            purpose=f"Purpose {user_id}",
            now=now,
        )
    alternate_path = alternate_dir / "slack-delegations.json"
    alternate_state = json.loads(alternate_path.read_text())
    if corruption == "malformed":
        alternate_state["audit"] = [{"malformed": True}]
    else:
        alternate_state["audit"] = alternate_state["audit"] * 10_001
    alternate_path.write_text(json.dumps(alternate_state))

    _merge_pairing_dir(active_dir, alternate_dir)

    active_state = json.loads((active_dir / "slack-delegations.json").read_text())
    assert {grant["user_id"] for grant in active_state["grants"].values()} == {
        "U_ACTIVE"
    }


def test_temporary_grant_file_is_private_and_revoke_is_thread_scoped(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    path = store._dir / "slack-delegations.json"

    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        lock_path = store._dir / ".slack-delegations.lock"
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600

    assert store.revoke_temporary(
        platform="slack",
        user_id="U06EGAQ943V",
        chat_id="C_OTHER",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        now=1_001.0,
    ) is False
    assert store.revoke_temporary(
        platform="slack",
        user_id="U06EGAQ943V",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        now=1_001.0,
    ) is True
    assert store.get_active_delegation(_slack_source(), now=1_001.0) is None


def test_authz_marks_temporary_delegate_without_persistently_approving_user(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_001.0)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G")

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()
    source = _slack_source()

    assert runner._is_user_authorized(source) is True
    assert source.temporary_delegated is True
    assert source.delegation_purpose == "Qualify recipe migration feedback only"
    assert store.is_approved("slack", "U06EGAQ943V") is False


def test_authz_honors_temporary_grant_without_any_environment_allowlist(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_001.0)
    for name in (
        "SLACK_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()
    source = _slack_source()

    assert runner._is_user_authorized(source) is True
    assert source.temporary_delegated is True


def test_multiplex_missing_profile_store_never_falls_back_to_global_grant(
    tmp_path, monkeypatch
):
    global_store = _store(tmp_path, monkeypatch)
    _grant(global_store)
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_001.0)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G")

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = global_store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()
    runner.config.multiplex_profiles = True
    source = _slack_source()
    source.profile = "secondary"

    assert runner._is_user_authorized(source) is False
    assert source.temporary_delegated is False


def test_baseline_allowlist_wins_over_overlapping_temporary_grant(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_001.0)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G,U06EGAQ943V")

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()
    source = _slack_source()
    source.temporary_delegated = True
    source.delegation_purpose = "stale"

    assert runner._is_user_authorized(source) is True
    assert source.temporary_delegated is False
    assert source.delegation_purpose is None


def test_deferred_event_is_reauthorized_after_revoke_and_internal_is_denied(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    _grant(store)
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_001.0)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G")
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = store
    runner.pairing_stores = {}
    runner.config = GatewayConfig()

    event = MessageEvent(
        text="queued feedback",
        message_type=MessageType.TEXT,
        source=_slack_source(),
    )
    assert runner._reauthorize_deferred_event(event) is True
    assert "TEMPORARY DELEGATION POLICY" in (event.channel_prompt or "")

    internal_event = MessageEvent(
        text="synthetic completion",
        message_type=MessageType.TEXT,
        source=_slack_source(),
        internal=True,
    )
    assert runner._reauthorize_deferred_event(internal_event) is False

    assert store.revoke_temporary(
        platform="slack",
        user_id="U06EGAQ943V",
        chat_id="C_RECIPES",
        thread_id="1787129215.308139",
        scope_id="T_WORKSPACE",
        now=1_001.0,
    ) is True
    revoked_event = MessageEvent(
        text="queued feedback",
        message_type=MessageType.TEXT,
        source=_slack_source(),
    )
    assert runner._reauthorize_deferred_event(revoked_event) is False


def test_delegated_user_cannot_run_admin_commands_when_slash_policy_is_disabled():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    source = _slack_source()
    source.temporary_delegated = True

    assert runner._check_slash_access(source, "restart") is not None
    assert runner._check_slash_access(source, "delegate") is not None
    assert runner._check_slash_access(source, "help") is None
    assert runner._check_slash_access(source, "whoami") is None


def test_delegate_cannot_answer_plain_text_dangerous_approval(monkeypatch):
    source = _slack_source()
    source.temporary_delegated = True
    event = MessageEvent(
        text="yes",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter = SimpleNamespace(
        _send_with_retry=AsyncMock(),
        _unwrap_ephemeral=lambda value: (value, None),
    )
    runner = object.__new__(GatewayRunner)
    setattr(runner, "_is_user_authorized", lambda source, **_kwargs: True)
    setattr(runner, "_effective_busy_input_mode", lambda source: "queue")
    runner._draining = False
    setattr(runner, "_adapter_for_source", lambda source: adapter)
    setattr(runner, "_reply_anchor_for_event", lambda event: None)
    setattr(
        runner,
        "_thread_metadata_for_source",
        lambda source, reply_anchor=None: {},
    )
    approve_handler = AsyncMock(return_value="must not run")
    deny_handler = AsyncMock(return_value="must not run")
    setattr(runner, "_handle_approve_command", approve_handler)
    setattr(runner, "_handle_deny_command", deny_handler)
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _key: True)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(event, "session-key")
    )

    assert handled is True
    approve_handler.assert_not_awaited()
    deny_handler.assert_not_awaited()
    adapter._send_with_retry.assert_awaited_once()
    assert "cannot approve" in adapter._send_with_retry.await_args.kwargs["content"]


def test_busy_delegate_is_queued_not_steered_or_interrupted(monkeypatch):
    source = _slack_source()
    source.temporary_delegated = True
    event = MessageEvent(
        text="run an owner-context side effect",
        message_type=MessageType.TEXT,
        source=source,
    )
    running_agent = SimpleNamespace(
        steer=MagicMock(return_value=True),
        redirect=MagicMock(return_value=True),
        interrupt=MagicMock(),
    )
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    queue_event = MagicMock()
    runner = object.__new__(GatewayRunner)
    setattr(runner, "_is_user_authorized", lambda source, **_kwargs: True)
    setattr(runner, "_effective_busy_input_mode", lambda source: "steer")
    runner._draining = False
    setattr(runner, "_adapter_for_source", lambda source: adapter)
    setattr(runner, "_queue_or_replace_pending_event", queue_event)
    setattr(
        runner,
        "_peek_session_state",
        lambda _key: SimpleNamespace(turn=SimpleNamespace(agent=running_agent)),
    )
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda _key: False)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(event, "session-key")
    )

    assert handled is True
    queue_event.assert_called_once_with("session-key", event)
    running_agent.steer.assert_not_called()
    running_agent.redirect.assert_not_called()
    running_agent.interrupt.assert_not_called()


def test_pending_media_never_merges_across_owner_and_delegate_identity():
    owner_event = MessageEvent(
        text="owner caption",
        message_type=MessageType.PHOTO,
        source=_slack_source(user_id="U04HAU25G"),
        media_urls=["/tmp/owner.jpg"],
        media_types=["image/jpeg"],
    )
    delegate_source = _slack_source()
    delegate_source.temporary_delegated = True
    delegate_source.delegation_purpose = "Recipe feedback only"
    delegate_source.delegation_expires_at = 4_600.0
    delegate_source.delegation_granted_by = "U04HAU25G"
    delegate_event = MessageEvent(
        text="delegate feedback",
        message_type=MessageType.TEXT,
        source=delegate_source,
    )
    adapter = SimpleNamespace(_pending_messages={"session-key": owner_event})
    enqueue = MagicMock()
    runner = object.__new__(GatewayRunner)
    setattr(runner, "_adapter_for_source", lambda source: adapter)
    setattr(runner, "_queue_depth", lambda key, adapter=None: 1)
    setattr(runner, "_enqueue_fifo", enqueue)

    runner._queue_or_replace_pending_event("session-key", delegate_event)

    assert owner_event.text == "owner caption"
    assert owner_event.source.user_id == "U04HAU25G"
    enqueue.assert_called_once_with("session-key", delegate_event, adapter)


def test_delegation_state_is_wire_invisible_and_builds_trusted_boundary():
    source = _slack_source()
    source.temporary_delegated = True
    source.delegation_purpose = "Recipe feedback only"
    source.delegation_granted_by = "U04HAU25G"
    source.delegation_expires_at = 4_600.0

    serialized = source.to_dict()
    assert "temporary_delegated" not in serialized
    assert "delegation_purpose" not in serialized

    restored = SessionSource.from_dict(serialized)
    assert restored.temporary_delegated is False
    prompt = GatewayRunner._temporary_delegation_prompt(source)
    assert prompt is not None
    assert "gateway verified" in prompt
    assert "Recipe feedback only" in prompt
    assert "non-admin temporary delegate" in prompt
    assert "Do not expose information from other chats" in prompt

    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text="feedback",
        message_type=MessageType.TEXT,
        source=source,
        channel_prompt="Existing channel policy",
    )
    assert runner._apply_temporary_delegation_prompt(event) is True
    assert event.channel_prompt is not None
    assert event.channel_prompt.startswith("Existing channel policy\n\n")
    assert "Recipe feedback only" in event.channel_prompt


def test_delegate_command_requires_explicit_admin_even_if_user_is_allowlisted(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U04HAU25G,U029M30PFS8")
    monkeypatch.setenv("SLACK_DELEGATION_ADMINS", "U04HAU25G")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = store
    runner.pairing_stores = {}
    event = MessageEvent(
        text="/delegate <@U06EGAQ943V> 1h migration feedback",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U029M30PFS8"),
    )

    response = asyncio.run(runner._handle_delegate_command(event))

    assert "Only configured delegation admins" in response
    assert store.list_temporary(platform="slack") == []


def test_delegate_command_rejects_unknown_deleted_or_bot_target(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setenv("SLACK_DELEGATION_ADMINS", "U04HAU25G")
    validator = AsyncMock(
        return_value=(False, "Slack bot accounts cannot receive delegation", "")
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = store
    runner.pairing_stores = {}
    setattr(
        runner,
        "_adapter_for_source",
        lambda source: SimpleNamespace(validate_delegation_target=validator),
    )
    event = MessageEvent(
        text="/delegate <@U06EGAQ943V> 1h migration feedback",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U04HAU25G"),
    )

    response = asyncio.run(runner._handle_delegate_command(event))

    assert "bot accounts" in response
    validator.assert_awaited_once_with("U06EGAQ943V", "T_WORKSPACE")
    assert store.list_temporary(platform="slack") == []


def test_delegate_command_grants_status_and_revoke_in_current_thread(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setenv("SLACK_DELEGATION_ADMINS", "U04HAU25G")
    monkeypatch.setattr("gateway.pairing.time.time", lambda: 1_000.0)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = store
    runner.pairing_stores = {}
    validator = AsyncMock(return_value=(True, "", "Léa Oustry"))
    setattr(
        runner,
        "_adapter_for_source",
        lambda source: SimpleNamespace(validate_delegation_target=validator),
    )
    source = _slack_source(user_id="U04HAU25G")

    grant_event = MessageEvent(
        text="/delegate <@U06EGAQ943V> 1h Qualify recipe migration feedback only",
        message_type=MessageType.COMMAND,
        source=source,
    )
    granted = asyncio.run(runner._handle_delegate_command(grant_event))
    assert "Temporary Slack delegation granted" in granted
    assert "1h" in granted

    status_event = MessageEvent(
        text="/delegate status",
        message_type=MessageType.COMMAND,
        source=source,
    )
    status_text = asyncio.run(runner._handle_delegate_command(status_event))
    assert "<@U06EGAQ943V>" in status_text
    assert "Qualify recipe migration feedback only" in status_text

    revoke_event = MessageEvent(
        text="/delegate revoke <@U06EGAQ943V>",
        message_type=MessageType.COMMAND,
        source=source,
    )
    revoked = asyncio.run(runner._handle_delegate_command(revoke_event))
    assert "revoked" in revoked.lower()
    assert store.get_active_delegation(_slack_source(), now=1_001.0) is None
    payload = json.loads((store._dir / "slack-delegations.json").read_text())
    assert payload["audit"][-1]["event"] == "revoked"
    assert payload["audit"][-1]["grant"]["revoked_by"] == "U04HAU25G"


def test_delegate_command_rejects_missing_thread_bad_duration_and_empty_purpose(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setenv("SLACK_DELEGATION_ADMINS", "U04HAU25G")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = store
    runner.pairing_stores = {}

    no_thread = MessageEvent(
        text="/delegate <@U06EGAQ943V> 1h purpose",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U04HAU25G", thread_id=""),
    )
    assert "thread" in asyncio.run(runner._handle_delegate_command(no_thread)).lower()

    bad_duration = MessageEvent(
        text="/delegate <@U06EGAQ943V> 25h purpose",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U04HAU25G"),
    )
    assert "duration" in asyncio.run(
        runner._handle_delegate_command(bad_duration)
    ).lower()

    no_purpose = MessageEvent(
        text="/delegate <@U06EGAQ943V> 1h",
        message_type=MessageType.COMMAND,
        source=_slack_source(user_id="U04HAU25G"),
    )
    assert "purpose" in asyncio.run(
        runner._handle_delegate_command(no_purpose)
    ).lower()


def test_temporary_grant_concurrent_refresh_keeps_valid_state_and_audit(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, monkeypatch)
    # Separate store instances have separate in-process RLocks. Only the
    # process-shared file lock prevents lost read-modify-write updates here.
    stores = [PairingStore() for _ in range(12)]

    def writer(index: int):
        stores[index].grant_temporary(
            platform="slack",
            user_id=f"U_{index}",
            user_name=f"Guest {index}",
            chat_id="C_RECIPES",
            thread_id="1787129215.308139",
            scope_id="T_WORKSPACE",
            granted_by="U04HAU25G",
            duration_seconds=3600,
            purpose=f"Purpose {index}",
            now=1_000.0 + index,
        )

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    payload = json.loads((store._dir / "slack-delegations.json").read_text())
    assert payload["version"] == 1
    assert len(payload["grants"]) == 12
    assert len(payload["audit"]) == 12
    assert {event["event"] for event in payload["audit"]} == {"granted"}
