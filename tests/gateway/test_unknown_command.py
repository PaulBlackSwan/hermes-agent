"""Tests for gateway warning when an unrecognized /command is dispatched.

Without this warning, unknown slash commands get forwarded to the LLM as plain
text, which often leads to silent failure (e.g. the model inventing a bogus
delegate_task call instead of telling the user the command doesn't exist).
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_voice_event(text: str = "voice_message_1.ogg") -> MessageEvent:
    source = _make_source()
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=source,
        message_id="m1",
        media_urls=["/tmp/voice_message_1.ogg"],
        media_types=["audio/ogg"],
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_unknown_slash_command_returns_guidance(monkeypatch):
    """A genuinely unknown /foobar should return user-facing guidance, not
    silently drop through to the LLM."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # If the LLM were called, this would fail: the guard must short-circuit
    # before _run_agent is invoked.
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/definitely-not-a-command"))

    assert result is not None
    assert "Unknown command" in result
    assert "/definitely-not-a-command" in result
    assert "/commands" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_known_slash_command_not_flagged_as_unknown(monkeypatch):
    """A real built-in like /status must NOT hit the unknown-command guard."""
    runner = _make_runner()
    # Make _handle_status_command exist via the normal path by running a real
    # dispatch. If the guard fires, the return string will mention "Unknown".
    runner._running_agents[build_session_key(_make_source())] = MagicMock()

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_egress_slash_command_reports_proxy_status(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        "hermes_cli.proxy_cli.format_status_text",
        lambda: "Egress proxy status\nEnabled: no",
    )

    result = await runner._handle_message(_make_event("/egress"))

    assert result is not None
    assert "Egress proxy status" in result
    assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_underscored_alias_for_hyphenated_builtin_not_flagged(monkeypatch):
    """Telegram autocomplete sends /reload_mcp for the /reload-mcp built-in.
    That must NOT be flagged as unknown."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # Prevent real MCP work; we only care that the unknown guard doesn't fire.
    async def _noop_reload(*_a, **_kw):
        return "mcp reloaded"

    runner._handle_reload_mcp_command = _noop_reload  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload_mcp"))

    # Whatever /reload_mcp returns, it must not be the unknown-command guard.
    if result is not None:
        assert "Unknown command" not in result


# ------------------------------------------------------------------
# command:<name> decision hook — deny / handled / rewrite
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_hook_rewrite_routes_to_plugin(monkeypatch):
    """A rewrite decision should re-resolve the command and route to the new one."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("rewritten command leaked to the agent")
    )

    call_log = []

    async def _emit_collect(event_type, ctx):
        call_log.append(event_type)
        if event_type == "command:status":
            return [
                {
                    "decision": "rewrite",
                    "command_name": "metricas",
                    "raw_args": "dias:7",
                }
            ]
        return []

    runner.hooks.emit_collect = AsyncMock(side_effect=_emit_collect)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"metricas": {"description": "Metrics", "args_hint": "dias:7"}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: (lambda args: f"metrics {args}") if name == "metricas" else None,
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "metrics dias:7"
    # First emit_collect fires on the original command; after rewrite the
    # dispatcher does NOT re-fire for the new command (one decision per turn).
    assert call_log == ["command:status"]


@pytest.mark.asyncio
async def test_command_hook_cannot_rewrite_delegate_floor_to_admin_command():
    runner = _make_runner()
    source = _make_source()
    source.temporary_delegated = True
    event = MessageEvent(text="/help", source=source, message_id="m1")
    restart_handler = AsyncMock(
        side_effect=AssertionError("delegate reached rewritten admin handler")
    )
    runner._handle_restart_command = restart_handler
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "restart",
                "raw_args": "",
            }
        ]
    )

    result = await runner._handle_message(event)

    assert result is not None
    assert "unavailable to temporary delegates" in result
    restart_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_temporary_delegate_unknown_or_skill_slash_is_denied_before_dispatch():
    runner = _make_runner()
    event = _make_event("/fake-skill")
    event.source.temporary_delegated = True
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("delegate slash leaked to skill/agent dispatch")
    )

    result = await runner._handle_message(event)

    assert result is not None
    assert "unavailable to temporary delegates" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_temporary_delegate_cannot_answer_pending_slash_confirmation(monkeypatch):
    from tools import slash_confirm

    runner = _make_runner()
    event = _make_event("/approve")
    event.source.temporary_delegated = True
    resolve = AsyncMock(return_value="must not resolve")
    monkeypatch.setattr(
        slash_confirm,
        "get_pending",
        lambda _key: {"confirm_id": "owner-confirm"},
    )
    monkeypatch.setattr(slash_confirm, "resolve", resolve)

    result = await runner._handle_message(event)

    assert result is not None
    assert "unavailable to temporary delegates" in result
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_busy_runner_queues_temporary_delegate_without_steering():
    runner = _make_runner()
    event = _make_event("delegate splice")
    event.source.temporary_delegated = True
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    running_agent.steer = MagicMock(return_value=True)
    runner._running_agents[session_key] = running_agent
    queue_event = MagicMock()
    runner._queue_or_replace_pending_event = queue_event

    result = await runner._handle_message(event)

    assert result is not None
    assert "queued as a separate turn" in result
    queue_event.assert_called_once_with(session_key, event)
    running_agent.steer.assert_not_called()
    running_agent.interrupt.assert_not_called()
