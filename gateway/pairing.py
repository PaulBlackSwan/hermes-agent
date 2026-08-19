"""
DM Pairing System

Code-based approval flow for authorizing new users on messaging platforms.
Instead of static allowlists with user IDs, unknown users receive a one-time
pairing code that the bot owner approves via the CLI.

Security features (based on OWASP + NIST SP 800-63-4 guidance):
  - 8-char codes from 32-char unambiguous alphabet (no 0/O/1/I)
  - Cryptographic randomness via secrets.choice()
  - 1-hour code expiry
  - Max 3 pending codes per platform
  - Rate limiting: 1 request per user per 10 minutes
  - Lockout after 5 failed approval attempts (1 hour)
  - File permissions: chmod 0600 on all data files
  - Codes are never logged to stdout

Storage: ~/.hermes/pairing/
"""

import hashlib
import json
import logging
import math
import os
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from gateway.whatsapp_identity import (
    expand_whatsapp_aliases,
    normalize_whatsapp_identifier,
)
from hermes_constants import (
    get_default_hermes_root,
    get_hermes_dir,
    get_hermes_home,
)
from utils import atomic_replace

logger = logging.getLogger(__name__)

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:  # pragma: no cover - platform-specific import
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

# Timing constants
CODE_TTL_SECONDS = 3600             # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600            # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600              # Lockout duration after too many failures

# Limits
MAX_PENDING_PER_PLATFORM = 3        # Max pending codes per platform
MAX_FAILED_ATTEMPTS = 5             # Failed approvals before lockout
MIN_TEMPORARY_DELEGATION_SECONDS = 60
MAX_TEMPORARY_DELEGATION_SECONDS = 24 * 60 * 60
MAX_TEMPORARY_DELEGATION_PURPOSE_CHARS = 500
TEMPORARY_DELEGATION_SCHEMA_VERSION = 1
MAX_TEMPORARY_DELEGATION_AUDIT_EVENTS = 10_000

PAIRING_DIR = get_hermes_dir("platforms/pairing", "pairing")


# Platform value -> its per-platform allowlist env var. When an operator has
# already configured an allowlist for a platform, approving a pairing code also
# writes the user into that allowlist (and revoking removes them), so the
# operator's own list stays the single visible/editable source of truth instead
# of drifting from an opaque approved.json (#23778 consolidation, option i).
# Platforms absent from this map (or with no allowlist configured) keep the
# pairing store as the sole grant record, honored by the authz union.
_PLATFORM_ALLOWLIST_ENV = {
    "telegram": "TELEGRAM_ALLOWED_USERS",
    "discord": "DISCORD_ALLOWED_USERS",
    "whatsapp": "WHATSAPP_ALLOWED_USERS",
    "whatsapp_cloud": "WHATSAPP_CLOUD_ALLOWED_USERS",
    "slack": "SLACK_ALLOWED_USERS",
    "signal": "SIGNAL_ALLOWED_USERS",
    "email": "EMAIL_ALLOWED_USERS",
    "sms": "SMS_ALLOWED_USERS",
    "mattermost": "MATTERMOST_ALLOWED_USERS",
    "matrix": "MATRIX_ALLOWED_USERS",
    "dingtalk": "DINGTALK_ALLOWED_USERS",
    "feishu": "FEISHU_ALLOWED_USERS",
    "wecom": "WECOM_ALLOWED_USERS",
    "wecom_callback": "WECOM_CALLBACK_ALLOWED_USERS",
    "weixin": "WEIXIN_ALLOWED_USERS",
    "bluebubbles": "BLUEBUBBLES_ALLOWED_USERS",
    "qqbot": "QQ_ALLOWED_USERS",
    "yuanbao": "YUANBAO_ALLOWED_USERS",
}


def _allowlist_env_for_platform(platform: str) -> Optional[str]:
    """Return the per-platform allowlist env var name, or None.

    Falls back to the platform registry for plugin platforms so a plugin's
    own ``allowed_users_env`` is honored too.
    """
    platform = (platform or "").lower().strip()
    env_var = _PLATFORM_ALLOWLIST_ENV.get(platform)
    if env_var:
        return env_var
    try:
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get(platform)
        if entry and entry.allowed_users_env:
            return entry.allowed_users_env
    except Exception:
        pass
    return None


def _split_allowlist(raw: str) -> list:
    return [uid.strip() for uid in raw.split(",") if uid.strip()]


def _platform_uses_whatsapp_identity(platform: str) -> bool:
    """True for Baileys WhatsApp and Meta Cloud — same phone/JID identity rules."""
    return (platform or "").strip().lower() in {"whatsapp", "whatsapp_cloud"}


def _normalize_user_id(platform: str, user_id: str) -> str:
    """Normalize platform-specific user IDs before persisting / comparing them."""
    raw_user_id = str(user_id or "").strip()
    if _platform_uses_whatsapp_identity(platform):
        return normalize_whatsapp_identifier(raw_user_id) or raw_user_id
    return raw_user_id


def _user_id_aliases(platform: str, user_id: str) -> set[str]:
    """Return all known equivalent user IDs for auth / allowlist matching."""
    raw_user_id = str(user_id or "").strip()
    if not raw_user_id:
        return set()

    aliases = {raw_user_id, _normalize_user_id(platform, raw_user_id)}
    if _platform_uses_whatsapp_identity(platform):
        aliases.update(expand_whatsapp_aliases(raw_user_id))
    aliases.discard("")
    return aliases


def _user_ids_match(platform: str, left: str, right: str) -> bool:
    """Return True when two user IDs represent the same principal."""
    left_aliases = _user_id_aliases(platform, left)
    right_aliases = _user_id_aliases(platform, right)
    return bool(left_aliases and right_aliases and (left_aliases & right_aliases))


def _read_allowlist_env(env_var: str) -> str:
    """Read a platform allowlist env var through the profile secret scope.

    Under multiplexing the process env may hold ANOTHER profile's allowlist
    (first-writer-wins YAML→env bridges), so reads must honor the installed
    scope's verdict — including a scoped miss returning empty rather than
    borrowing the process value.  Unscoped callers (single-profile CLI /
    admin endpoints) keep the legacy ``os.getenv`` read.

    TODO(profile-secrets): the grant mirror below still WRITES through
    ``hermes_cli.config.save_env_value`` / ``remove_env_value``, which target
    the root ``.env`` — those writes need a profile-aware counterpart before
    pairing grants can be mirrored correctly under multiplexing.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return (get_secret(env_var) or "").strip()
        except UnscopedSecretError:
            pass
    except Exception:
        pass
    return (os.getenv(env_var) or "").strip()


def _sync_allowlist_add(platform: str, user_id: str) -> None:
    """Add ``user_id`` to the platform allowlist env var IF one is configured.

    Option (i): only materialize the grant into the allowlist when the operator
    already runs an allowlist for this platform. On an open gateway (no
    allowlist) we do nothing — the pairing store remains the grant record and
    the authz union honors it, so we never silently convert an open gateway into
    a locked one on first pairing.
    """
    env_var = _allowlist_env_for_platform(platform)
    if not env_var:
        return
    current = _read_allowlist_env(env_var)
    if not current:
        return  # No allowlist configured — leave the gateway open (option i).
    ids = _split_allowlist(current)
    if "*" in ids or str(user_id) in ids:
        return  # Already covered.
    ids.append(str(user_id))
    try:
        from hermes_cli.config import save_env_value

        save_env_value(env_var, ",".join(ids))
    except Exception:
        # Best-effort: the pairing store grant still authorizes via the union,
        # so a failure here degrades to "grant recorded but not mirrored".
        pass


def _iter_live_gateway_adapters():
    """Yield adapters from the in-process GatewayRunner, if one is running."""
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    except Exception:
        return
    if runner is None:
        return
    adapters = getattr(runner, "adapters", None) or {}
    for adapter in adapters.values():
        if adapter is not None:
            yield adapter
    profile_adapters = getattr(runner, "_profile_adapters", None) or {}
    for mapping in profile_adapters.values():
        for adapter in (mapping or {}).values():
            if adapter is not None:
                yield adapter


def _adapter_platform_name(adapter) -> str:
    platform = getattr(adapter, "platform", None)
    if platform is not None:
        value = getattr(platform, "value", None)
        if value:
            return str(value).strip().lower()
    name = getattr(adapter, "name", None)
    return str(name or "").strip().lower()


def _purge_allowlist_entries(entries, platform: str, user_id: str):
    """Drop alias-equivalent allowlist entries while preserving ``*``."""
    if entries is None:
        return entries
    if isinstance(entries, str):
        parts = _split_allowlist(entries)
        remaining = [
            part for part in parts
            if part == "*" or not _user_ids_match(platform, part, str(user_id))
        ]
        return ",".join(remaining)
    if isinstance(entries, (set, frozenset)):
        return {
            entry for entry in entries
            if str(entry).strip() == "*"
            or not _user_ids_match(platform, str(entry), str(user_id))
        }
    if isinstance(entries, (list, tuple)):
        return [
            entry for entry in entries
            if str(entry).strip() == "*"
            or not _user_ids_match(platform, str(entry), str(user_id))
        ]
    return entries


def _sync_live_adapter_allowlist_remove(platform: str, user_id: str) -> None:
    """Clear revoked principals from in-process adapter allowlist snapshots.

    ``WhatsAppAdapter`` (and Cloud) snapshot ``_allow_from`` at construction.
    Pairing revoke updates ``WHATSAPP_ALLOWED_USERS`` / cloud env, but when the
    revoked principal was the sole entry the env key is removed entirely.
    Intake must not keep authorizing from the stale snapshot until restart.
    """
    platform_name = (platform or "").strip().lower()
    if not platform_name or not str(user_id or "").strip():
        return
    for adapter in _iter_live_gateway_adapters():
        if _adapter_platform_name(adapter) != platform_name:
            continue
        if hasattr(adapter, "_allow_from"):
            try:
                adapter._allow_from = _purge_allowlist_entries(
                    set(adapter._allow_from or ()), platform_name, user_id
                )
            except Exception:
                pass
        extra = getattr(getattr(adapter, "config", None), "extra", None)
        if isinstance(extra, dict) and "allow_from" in extra:
            try:
                extra["allow_from"] = _purge_allowlist_entries(
                    extra.get("allow_from"), platform_name, user_id
                )
            except Exception:
                pass


def _sync_allowlist_remove(platform: str, user_id: str) -> None:
    """Remove ``user_id`` (and WhatsApp alias equivalents) from the allowlist.

    Matching must mirror PairingStore / authz WhatsApp alias rules: approve
    mirrors a normalized phone into ``WHATSAPP_ALLOWED_USERS``, while revoke
    is often invoked with a JID or device-suffix form. Exact-string delete
    would leave the allowlist entry and keep the sender authorized.

    Also clears matching entries from any in-process platform adapter
    ``_allow_from`` snapshot so sole-entry revocation is effective without a
    gateway restart.
    """
    env_var = _allowlist_env_for_platform(platform)
    if not env_var:
        return
    current = _read_allowlist_env(env_var)
    if not current:
        return  # No allowlist configured — do not touch config-only snapshots.
    ids = _split_allowlist(current)
    # Never strip a wildcard grant; drop every entry that aliases-matches.
    remaining = [
        i for i in ids
        if i == "*" or not _user_ids_match(platform, i, str(user_id))
    ]
    if len(remaining) == len(ids):
        return  # Not present.
    try:
        from hermes_cli.config import save_env_value, remove_env_value

        if remaining:
            save_env_value(env_var, ",".join(remaining))
        else:
            remove_env_value(env_var)
    except Exception:
        pass
    _sync_live_adapter_allowlist_remove(platform, user_id)


def _load_json_file(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError, UnicodeError):
            return {}
    return {}


@contextmanager
def _exclusive_file_lock(path: Path):
    """Hold one private advisory file lock across processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            getattr(msvcrt, "locking")(
                handle.fileno(), getattr(msvcrt, "LK_LOCK"), 1
            )
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                getattr(msvcrt, "locking")(
                    handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                )
        finally:
            handle.close()


def _temporary_audit_event_is_valid(event: Any) -> bool:
    if not isinstance(event, dict) or set(event) != {
        "event",
        "occurred_at",
        "grant_id",
        "grant",
    }:
        return False
    if event.get("event") not in {
        "granted",
        "refreshed",
        "expired",
        "invalid_record_pruned",
        "revoked",
    }:
        return False
    occurred_at = event.get("occurred_at")
    if (
        isinstance(occurred_at, bool)
        or not isinstance(occurred_at, (int, float))
        or not math.isfinite(float(occurred_at))
    ):
        return False
    return (
        isinstance(event.get("grant_id"), str)
        and bool(event["grant_id"])
        and isinstance(event.get("grant"), dict)
    )


def _temporary_state_for_merge(path: Path) -> Optional[dict]:
    """Read structurally valid delegation state without authorizing records."""
    if not path.exists():
        return {
            "version": TEMPORARY_DELEGATION_SCHEMA_VERSION,
            "grants": {},
            "audit": [],
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "grants", "audit"}
        or type(state.get("version")) is not int
        or state.get("version") != TEMPORARY_DELEGATION_SCHEMA_VERSION
        or not isinstance(state.get("grants"), dict)
        or not isinstance(state.get("audit"), list)
        or len(state.get("audit", [])) > MAX_TEMPORARY_DELEGATION_AUDIT_EVENTS
        or any(
            not _temporary_audit_event_is_valid(event)
            for event in state.get("audit", [])
        )
    ):
        return None
    return state


def _merge_temporary_delegation_file(src: Path, dest: Path) -> None:
    """Merge exact grant/audit collections while locking both layouts."""
    platform = src.name.removesuffix("-delegations.json")
    lock_paths = sorted(
        {
            src.parent / f".{platform}-delegations.lock",
            dest.parent / f".{platform}-delegations.lock",
        },
        key=lambda value: str(value),
    )
    with _exclusive_file_lock(lock_paths[0]):
        with _exclusive_file_lock(lock_paths[1]):
            alternate = _temporary_state_for_merge(src)
            current = _temporary_state_for_merge(dest)
            if alternate is None or current is None:
                logger.warning(
                    "Skipping corrupt temporary delegation migration: %s -> %s",
                    src,
                    dest,
                )
                return
            grants = dict(alternate["grants"])
            grants.update(current["grants"])
            audit = list(alternate["audit"]) + list(current["audit"])
            deduped_audit = []
            seen = set()
            for event in audit:
                try:
                    marker = json.dumps(event, sort_keys=True, ensure_ascii=False)
                except (TypeError, ValueError):
                    continue
                if marker in seen:
                    continue
                seen.add(marker)
                deduped_audit.append(event)
            deduped_audit.sort(
                key=lambda event: (
                    float(event.get("occurred_at", 0))
                    if isinstance(event, dict)
                    and isinstance(event.get("occurred_at"), (int, float))
                    else 0
                )
            )
            merged = {
                "version": TEMPORARY_DELEGATION_SCHEMA_VERSION,
                "grants": grants,
                "audit": deduped_audit[-MAX_TEMPORARY_DELEGATION_AUDIT_EVENTS:],
            }
            if merged != current:
                _secure_write(dest, json.dumps(merged, indent=2, ensure_ascii=False))


def _merge_pairing_dir(active_dir: Path, alternate_dir: Path) -> None:
    """Merge split legacy/new pairing data into the active PairingStore dir.

    Older installs use ``{HERMES_HOME}/pairing`` while newer code/docs may
    write ``{HERMES_HOME}/platforms/pairing``. If both directories exist, the
    gateway must not silently ignore approved users sitting in the inactive
    location; otherwise already-paired Feishu users get asked for a fresh code.
    """
    if not alternate_dir.exists() or active_dir.resolve() == alternate_dir.resolve():
        return
    active_dir.mkdir(parents=True, exist_ok=True)
    for src in alternate_dir.glob("*.json"):
        if not src.is_file():
            continue
        dest = active_dir / src.name
        if src.name.endswith("-delegations.json"):
            _merge_temporary_delegation_file(src, dest)
            continue
        merged = _load_json_file(src)
        if not merged:
            continue
        current = _load_json_file(dest)
        before = dict(current)
        # Active data wins on key conflict; otherwise union the inactive data.
        merged.update(current)
        if merged != before:
            _secure_write(dest, json.dumps(merged, indent=2, ensure_ascii=False))


def _migrate_split_pairing_dirs(
    *,
    home: Optional[Path] = None,
    active: Optional[Path] = None,
) -> None:
    home = home or get_hermes_home()
    old_dir = home / "pairing"
    new_dir = home / "platforms" / "pairing"
    active = active or PAIRING_DIR
    alternate = new_dir if active.resolve() == old_dir.resolve() else old_dir
    _merge_pairing_dir(active, alternate)


def _secure_write(path: Path, data: str) -> None:
    """Write data to file with restrictive permissions (owner read/write only).

    Uses a temp-file + atomic rename so readers always see either the old
    complete file or the new one — never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows doesn't support chmod the same way
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class PairingStore:
    """
    Manages pairing codes and approved user lists.

    Data files per platform:
      - {platform}-pending.json   : pending pairing requests
      - {platform}-approved.json  : approved (paired) users
      - _rate_limits.json         : rate limit tracking

    When constructed with ``profile="<name>"``, storage resolves from that
    profile's own HERMES_HOME using the same legacy/consolidated layout rules
    as ``hermes -p <name> pairing ...``. This keeps multiplex gateways and
    profile-scoped CLI approvals on one whitelist. Without a profile, storage
    is the global pairing directory for the current HERMES_HOME.
    """

    def __init__(self, profile: Optional[str] = None):
        # Resolve storage directory lazily — tests use a temp HERMES_HOME
        # and PairingStore may be constructed before the env is set.
        if profile:
            root = get_default_hermes_root()
            profile_home = (
                root
                if profile == "default"
                else root / "profiles" / profile
            )
            self._dir = get_hermes_dir(
                "platforms/pairing",
                "pairing",
                home=profile_home,
            )
        else:
            self._dir = PAIRING_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        if profile:
            # Explicit stores must resolve exactly as a standalone
            # ``hermes -p <profile> pairing ...`` process does. Merge the
            # alternate old/new layout so upgrades cannot split approvals.
            _migrate_split_pairing_dirs(home=profile_home, active=self._dir)
        else:
            # Heal installs whose global pairing data ended up split across
            # the legacy and new directories.
            _migrate_split_pairing_dirs()
        # Protects all read-modify-write cycles. The gateway runs multiple
        # platform adapters concurrently in threads sharing one PairingStore.
        self._lock = threading.RLock()
        self._profile = profile  # for diagnostics / log lines

    @property
    def profile(self) -> Optional[str]:
        """Profile name this store is scoped to, or None for the global store."""
        return self._profile

    def _pending_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-pending.json"

    def _approved_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-approved.json"

    def _rate_limit_path(self) -> Path:
        return self._dir / "_rate_limits.json"

    def _delegations_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-delegations.json"

    def _delegations_lock_path(self, platform: str) -> Path:
        return self._dir / f".{platform}-delegations.lock"

    @contextmanager
    def _temporary_delegation_process_lock(self, platform: str):
        """Serialize delegation read-modify-write cycles across processes."""
        with _exclusive_file_lock(self._delegations_lock_path(platform)):
            yield

    @staticmethod
    def _empty_temporary_delegation_state() -> dict:
        return {
            "version": TEMPORARY_DELEGATION_SCHEMA_VERSION,
            "grants": {},
            "audit": [],
        }

    def _load_temporary_delegation_state(self, platform: str) -> dict:
        """Strictly load security state; corruption fails closed, never resets."""
        path = self._delegations_path(platform)
        if not path.exists():
            return self._empty_temporary_delegation_state()
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"temporary delegation state is unreadable: {path}"
            ) from exc
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "grants", "audit"}
            or type(state.get("version")) is not int
            or state.get("version") != TEMPORARY_DELEGATION_SCHEMA_VERSION
            or not isinstance(state.get("grants"), dict)
            or not isinstance(state.get("audit"), list)
            or len(state.get("audit", [])) > MAX_TEMPORARY_DELEGATION_AUDIT_EVENTS
            or any(
                not _temporary_audit_event_is_valid(event)
                for event in state.get("audit", [])
            )
        ):
            raise RuntimeError(
                f"temporary delegation state has an invalid schema: {path}"
            )
        return state

    def _save_temporary_delegation_state(self, platform: str, state: dict) -> None:
        _secure_write(
            self._delegations_path(platform),
            json.dumps(state, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _append_temporary_delegation_audit(
        state: dict,
        *,
        event: str,
        occurred_at: float,
        grant_id: str,
        grant: dict,
    ) -> None:
        state["audit"].append(
            {
                "event": event,
                "occurred_at": float(occurred_at),
                "grant_id": grant_id,
                "grant": dict(grant),
            }
        )
        overflow = len(state["audit"]) - MAX_TEMPORARY_DELEGATION_AUDIT_EVENTS
        if overflow > 0:
            del state["audit"][:overflow]

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except PermissionError as e:
                # Surface this loudly: a 0600 file owned by a different user
                # (classic Docker symptom: `docker exec` runs as root and writes
                # the file, then the gateway process — running as `hermes` after
                # gosu drop — can't read it) would otherwise be swallowed by
                # the generic OSError branch below, silently leaving the user
                # marked unauthorized. See issue #10270.
                try:
                    st = path.stat()
                    owner_info = f"owner_uid={st.st_uid} mode={oct(st.st_mode)[-4:]}"
                except OSError:
                    owner_info = "<stat failed>"
                # os.geteuid doesn't exist on Windows; the Docker scenario is
                # POSIX-only, but the gateway (and this fallback) runs anywhere.
                euid = os.geteuid() if hasattr(os, "geteuid") else "n/a"
                logger.warning(
                    "Pairing file %s exists but is not readable as uid=%s (%s; %s). "
                    "If you ran `docker exec <container> hermes pairing approve ...` as root, "
                    "re-run with `docker exec -u hermes <container> ...` and "
                    "chown the existing file to the hermes user, or restart the "
                    "container so the entrypoint can fix ownership.",
                    path, euid, owner_info, e,
                )
                return {}
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))

    def _normalize_user_id(self, platform: str, user_id: str) -> str:
        """Normalize platform-specific user IDs before persisting them."""
        return _normalize_user_id(platform, user_id)

    def _user_id_aliases(self, platform: str, user_id: str) -> set[str]:
        """Return all known equivalent user IDs for auth/rate-limit checks."""
        return _user_id_aliases(platform, user_id)

    def _user_ids_match(self, platform: str, left: str, right: str) -> bool:
        """Return True when two user IDs represent the same principal."""
        return _user_ids_match(platform, left, right)

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        approved = self._load_json(self._approved_path(platform))
        for approved_user_id in approved:
            if self._user_ids_match(platform, approved_user_id, user_id):
                return True
        return False

    def list_approved(self, platform: str = None) -> list:
        """List approved users, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("approved")
        for p in platforms:
            approved = self._load_json(self._approved_path(p))
            for uid, info in approved.items():
                results.append({"platform": p, "user_id": uid, **info})
        return results

    def _approve_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """Add a user to the approved list. Must be called under self._lock."""
        approved = self._load_json(self._approved_path(platform))
        normalized_user_id = self._normalize_user_id(platform, user_id)
        duplicate_ids = [
            approved_user_id
            for approved_user_id in approved
            if self._user_ids_match(platform, approved_user_id, normalized_user_id)
        ]
        for approved_user_id in duplicate_ids:
            del approved[approved_user_id]

        approved[normalized_user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

        # Mirror the grant into the operator's allowlist when one is configured
        # (option i), so the pairing store and the allowlist stay a single
        # visible source of truth. No-op on open gateways.
        _sync_allowlist_add(platform, normalized_user_id)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        with self._lock:
            approved = self._load_json(path)
            matching_ids = [
                approved_user_id
                for approved_user_id in approved
                if self._user_ids_match(platform, approved_user_id, user_id)
            ]
            if matching_ids:
                for approved_user_id in matching_ids:
                    del approved[approved_user_id]
                self._save_json(path, approved)
                # Keep the allowlist mirror in sync: revoking a paired user
                # also removes the entry the approval added (option i). No-op if
                # the user was added to the allowlist by other means.
                _sync_allowlist_remove(platform, user_id)
                return True
        return False

    # ----- Temporary thread-scoped delegations -----

    @staticmethod
    def _temporary_delegation_key(
        platform: str,
        scope_id: str,
        chat_id: str,
        thread_id: str,
        user_id: str,
    ) -> str:
        canonical = "\x00".join(
            (platform, scope_id, chat_id, thread_id, user_id)
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:24]

    def _temporary_delegation_record_is_valid(
        self,
        grant_id: str,
        grant: Any,
    ) -> bool:
        if not isinstance(grant, dict):
            return False
        if set(grant) != {
            "platform",
            "user_id",
            "user_name",
            "chat_id",
            "thread_id",
            "scope_id",
            "granted_by",
            "purpose",
            "created_at",
            "expires_at",
        }:
            return False
        required_strings = (
            "platform",
            "user_id",
            "chat_id",
            "thread_id",
            "scope_id",
            "granted_by",
            "purpose",
        )
        if any(
            not isinstance(grant.get(key), str) or not grant[key]
            for key in required_strings
        ):
            return False
        if not isinstance(grant.get("user_name"), str):
            return False
        created_at = grant.get("created_at")
        expires_at = grant.get("expires_at")
        if (
            isinstance(created_at, bool)
            or isinstance(expires_at, bool)
            or not isinstance(created_at, (int, float))
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(float(created_at))
            or not math.isfinite(float(expires_at))
        ):
            return False
        duration = float(expires_at) - float(created_at)
        if not (
            MIN_TEMPORARY_DELEGATION_SECONDS
            <= duration
            <= MAX_TEMPORARY_DELEGATION_SECONDS
        ):
            return False
        if (
            not grant["purpose"].strip()
            or len(grant["purpose"]) > MAX_TEMPORARY_DELEGATION_PURPOSE_CHARS
        ):
            return False
        expected_id = self._temporary_delegation_key(
            grant["platform"],
            grant["scope_id"],
            grant["chat_id"],
            grant["thread_id"],
            grant["user_id"],
        )
        return grant_id == expected_id

    def _cleanup_expired_temporary_locked(
        self,
        state: dict,
        *,
        now: float,
    ) -> bool:
        grants = state["grants"]
        expired = []
        for grant_id, grant in grants.items():
            invalid = not self._temporary_delegation_record_is_valid(
                grant_id, grant
            )
            expires_at = 0.0
            if not invalid:
                try:
                    expires_at = float(grant.get("expires_at") or 0)
                    invalid = not math.isfinite(expires_at)
                except (TypeError, ValueError):
                    invalid = True
            if invalid or expires_at <= now:
                expired.append((grant_id, grant, invalid))
        for grant_id, grant, invalid in expired:
            grants.pop(grant_id, None)
            self._append_temporary_delegation_audit(
                state,
                event="invalid_record_pruned" if invalid else "expired",
                occurred_at=now,
                grant_id=grant_id,
                grant=grant if isinstance(grant, dict) else {"raw_type": type(grant).__name__},
            )
        if expired:
            logger.info(
                "Pruned %d expired/invalid temporary delegation record(s)",
                len(expired),
            )
        return bool(expired)

    def grant_temporary(
        self,
        *,
        platform: str,
        user_id: str,
        user_name: str,
        chat_id: str,
        thread_id: str,
        scope_id: str,
        granted_by: str,
        duration_seconds: int,
        purpose: str,
        now: Optional[float] = None,
    ) -> dict:
        """Create or refresh an exact user/workspace/channel/thread grant."""
        platform = str(platform or "").strip().lower()
        user_id = self._normalize_user_id(platform, user_id)
        user_name = str(user_name or "").strip()
        chat_id = str(chat_id or "").strip()
        thread_id = str(thread_id or "").strip()
        scope_id = str(scope_id or "").strip()
        granted_by = self._normalize_user_id(platform, granted_by)
        purpose = " ".join(str(purpose or "").split())
        try:
            duration_seconds = int(duration_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("temporary delegation duration must be an integer") from exc

        if not all((platform, user_id, chat_id, thread_id, scope_id, granted_by)):
            raise ValueError(
                "temporary delegation requires platform, user, workspace, channel, "
                "thread, and grantor"
            )
        if not (
            MIN_TEMPORARY_DELEGATION_SECONDS
            <= duration_seconds
            <= MAX_TEMPORARY_DELEGATION_SECONDS
        ):
            raise ValueError("temporary delegation duration must be between 1m and 24h")
        if not purpose:
            raise ValueError("temporary delegation purpose is required")
        if len(purpose) > MAX_TEMPORARY_DELEGATION_PURPOSE_CHARS:
            raise ValueError(
                f"temporary delegation purpose exceeds "
                f"{MAX_TEMPORARY_DELEGATION_PURPOSE_CHARS} characters"
            )

        now_value = time.time() if now is None else float(now)
        if not math.isfinite(now_value):
            raise ValueError("temporary delegation clock must be finite")
        grant = {
            "platform": platform,
            "user_id": user_id,
            "user_name": user_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "scope_id": scope_id,
            "granted_by": granted_by,
            "purpose": purpose,
            "created_at": now_value,
            "expires_at": now_value + duration_seconds,
        }
        grant_id = self._temporary_delegation_key(
            platform, scope_id, chat_id, thread_id, user_id
        )
        with self._lock, self._temporary_delegation_process_lock(platform):
            state = self._load_temporary_delegation_state(platform)
            self._cleanup_expired_temporary_locked(state, now=now_value)
            event_name = "refreshed" if grant_id in state["grants"] else "granted"
            state["grants"][grant_id] = grant
            self._append_temporary_delegation_audit(
                state,
                event=event_name,
                occurred_at=now_value,
                grant_id=grant_id,
                grant=grant,
            )
            self._save_temporary_delegation_state(platform, state)
        logger.info(
            "Temporary delegation granted: platform=%s user=%s scope=%s "
            "chat=%s thread=%s grantor=%s expires_at=%.3f",
            platform,
            user_id,
            scope_id,
            chat_id,
            thread_id,
            granted_by,
            grant["expires_at"],
        )
        return dict(grant)

    def get_active_delegation(
        self,
        source: Any,
        *,
        now: Optional[float] = None,
    ) -> Optional[dict]:
        """Return the exact active delegation for a SessionSource-like object."""
        platform_obj = getattr(source, "platform", None)
        platform = str(getattr(platform_obj, "value", platform_obj) or "").strip().lower()
        user_id = self._normalize_user_id(platform, getattr(source, "user_id", ""))
        chat_id = str(getattr(source, "chat_id", "") or "").strip()
        thread_id = str(getattr(source, "thread_id", "") or "").strip()
        scope_id = str(
            getattr(source, "scope_id", None)
            or getattr(source, "guild_id", None)
            or ""
        ).strip()
        if not all((platform, user_id, chat_id, thread_id, scope_id)):
            return None

        now_value = time.time() if now is None else float(now)
        if not math.isfinite(now_value):
            return None
        grant_id = self._temporary_delegation_key(
            platform, scope_id, chat_id, thread_id, user_id
        )
        with self._lock, self._temporary_delegation_process_lock(platform):
            state = self._load_temporary_delegation_state(platform)
            if self._cleanup_expired_temporary_locked(state, now=now_value):
                self._save_temporary_delegation_state(platform, state)
            grant = state["grants"].get(grant_id)
            if not isinstance(grant, dict):
                return None
            expected = {
                "platform": platform,
                "user_id": user_id,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "scope_id": scope_id,
            }
            if any(str(grant.get(key) or "") != value for key, value in expected.items()):
                return None
            return dict(grant)

    def list_temporary(
        self,
        *,
        platform: str,
        chat_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> list[dict]:
        """List active grants, optionally narrowed to one exact Slack thread."""
        platform = str(platform or "").strip().lower()
        if not platform:
            return []
        now_value = time.time() if now is None else float(now)
        if not math.isfinite(now_value):
            return []
        with self._lock, self._temporary_delegation_process_lock(platform):
            state = self._load_temporary_delegation_state(platform)
            if self._cleanup_expired_temporary_locked(state, now=now_value):
                self._save_temporary_delegation_state(platform, state)
            result = []
            for grant in state["grants"].values():
                if not isinstance(grant, dict):
                    continue
                if chat_id is not None and grant.get("chat_id") != str(chat_id):
                    continue
                if thread_id is not None and grant.get("thread_id") != str(thread_id):
                    continue
                if scope_id is not None and grant.get("scope_id") != str(scope_id):
                    continue
                result.append(dict(grant))
        return sorted(result, key=lambda item: (item["expires_at"], item["user_id"]))

    def revoke_temporary(
        self,
        *,
        platform: str,
        user_id: str,
        chat_id: str,
        thread_id: str,
        scope_id: str,
        revoked_by: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Revoke only the target user's grant in the exact thread scope."""
        platform = str(platform or "").strip().lower()
        user_id = self._normalize_user_id(platform, user_id)
        chat_id = str(chat_id or "").strip()
        thread_id = str(thread_id or "").strip()
        scope_id = str(scope_id or "").strip()
        revoked_by = self._normalize_user_id(platform, revoked_by or "")
        if not all((platform, user_id, chat_id, thread_id, scope_id)):
            return False
        grant_id = self._temporary_delegation_key(
            platform, scope_id, chat_id, thread_id, user_id
        )
        now_value = time.time() if now is None else float(now)
        if not math.isfinite(now_value):
            return False
        with self._lock, self._temporary_delegation_process_lock(platform):
            state = self._load_temporary_delegation_state(platform)
            self._cleanup_expired_temporary_locked(state, now=now_value)
            grant = state["grants"].pop(grant_id, None)
            if grant is None:
                self._save_temporary_delegation_state(platform, state)
                return False
            self._append_temporary_delegation_audit(
                state,
                event="revoked",
                occurred_at=now_value,
                grant_id=grant_id,
                grant={**grant, "revoked_by": revoked_by or None},
            )
            self._save_temporary_delegation_state(platform, state)
        logger.info(
            "Temporary delegation revoked: platform=%s user=%s scope=%s "
            "chat=%s thread=%s revoked_by=%s",
            platform,
            user_id,
            scope_id,
            chat_id,
            thread_id,
            revoked_by or "unknown",
        )
        return True

    # ----- Pending codes -----

    @staticmethod
    def _hash_code(code: str, salt: bytes) -> str:
        """Hash a pairing code with the given salt using SHA-256."""
        return hashlib.sha256(salt + code.encode("utf-8")).hexdigest()

    def _finish_approval(
        self, platform: str, pending: dict, matched_key: str, matched_entry: dict
    ) -> dict:
        """Remove a pending request and approve its user. Must hold self._lock."""
        del pending[matched_key]
        self._save_json(self._pending_path(platform), pending)

        # A successful approval proves the requester is legitimate, so the
        # brute-force failure streak must not carry over. Without this,
        # isolated mistyped codes accumulate across the gateway's lifetime
        # (the counter is persisted in _rate_limits.json and only ever
        # reset when a lockout fires) and eventually trip a spurious
        # lockout on a single fresh typo — rejecting even a valid code.
        self._reset_failed_attempts(platform)

        self._approve_user(
            platform, matched_entry["user_id"], matched_entry.get("user_name", "")
        )

        return {
            "user_id": matched_entry["user_id"],
            "user_name": matched_entry.get("user_name", ""),
        }

    def generate_code(
        self, platform: str, user_id: str, user_name: str = ""
    ) -> Optional[str]:
        """
        Generate a pairing code for a new user.

        Returns the code string, or None if:
          - User is rate-limited (too recent request)
          - Max pending codes reached for this platform
          - User/platform is in lockout due to failed attempts

        The code is NOT stored in plaintext.  Only a salted SHA-256 hash is
        persisted so that reading the pending file does not reveal codes.
        """
        with self._lock:
            self._cleanup_expired(platform)
            normalized_user_id = self._normalize_user_id(platform, user_id)

            # Check lockout
            if self._is_locked_out(platform):
                return None

            # Check rate limit for this specific user
            if self._is_rate_limited(platform, user_id):
                return None

            # Check max pending
            pending = self._load_json(self._pending_path(platform))
            if len(pending) >= MAX_PENDING_PER_PLATFORM:
                return None

            # Generate cryptographically random code
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))

            # Hash the code with a random salt before storing
            salt = os.urandom(16)
            code_hash = self._hash_code(code, salt)

            # Use a unique entry id as the key (not the code itself)
            entry_id = secrets.token_hex(8)

            # Store pending request with hashed code
            pending[entry_id] = {
                "hash": code_hash,
                "salt": salt.hex(),
                "user_id": normalized_user_id,
                "user_name": user_name,
                "created_at": time.time(),
            }
            self._save_json(self._pending_path(platform), pending)

            # Record rate limit
            self._record_rate_limit(platform, user_id)

            return code

    def approve_code(self, platform: str, code: str) -> Optional[dict]:
        """
        Approve a pairing code. Adds the user to the approved list.

        Returns ``{user_id, user_name}`` on success, ``None`` if the code is
        invalid/expired OR the platform is currently locked out after
        ``MAX_FAILED_ATTEMPTS`` failed approvals (#10195). Callers can
        disambiguate with ``_is_locked_out(platform)``.

        Verification: the user-provided code is hashed with each stored
        entry's salt and compared to the stored hash using constant-time
        comparison. Pre-hash entries (legacy plaintext-key format from
        pre-upgrade pending.json files) are silently ignored — they get
        pruned at TTL by ``_cleanup_expired``.
        """
        with self._lock:
            self._cleanup_expired(platform)
            code = code.upper().strip()

            # Lockout check — must run before the pending lookup so a
            # valid code (e.g. one already sitting in pending) cannot be
            # accepted once the lockout fires. Without this, the lockout
            # only blocks `generate_code`, not `approve_code` — nullifying
            # the brute-force protection for any code already issued.
            if self._is_locked_out(platform):
                return None

            pending = self._load_json(self._pending_path(platform))

            # Find the entry whose hash matches the provided code.
            # Tolerate legacy plaintext-key entries (no salt/hash) and
            # malformed entries — skip them rather than KeyError, so an
            # in-place upgrade across an existing pending.json doesn't
            # crash on the first approve call. Legacy entries get pruned
            # at their TTL by _cleanup_expired.
            matched_key = None
            matched_entry = None
            for entry_id, entry in pending.items():
                if not isinstance(entry, dict):
                    continue
                if "salt" not in entry or "hash" not in entry:
                    continue
                try:
                    salt = bytes.fromhex(entry["salt"])
                except ValueError:
                    continue
                candidate_hash = self._hash_code(code, salt)
                if secrets.compare_digest(candidate_hash, entry["hash"]):
                    matched_key = entry_id
                    matched_entry = entry
                    break

            if matched_key is None:
                self._record_failed_attempt(platform)
                return None

            return self._finish_approval(platform, pending, matched_key, matched_entry)

    @staticmethod
    def looks_like_request_id(value: str) -> bool:
        """True when ``value`` has the shape of a ``list_pending`` request id.

        Request ids are ``secrets.token_hex(8)`` (16 lowercase hex chars);
        pairing codes are 8 chars from an unambiguous uppercase alphabet that
        excludes every hex letter's ambiguity partner. The two shapes cannot
        collide, so callers accepting either can dispatch on this.
        """
        value = str(value or "").strip()
        return len(value) == 16 and all(c in "0123456789abcdefABCDEF" for c in value)

    def approve_request(self, platform: str, request_id: str) -> Optional[dict]:
        """
        Approve a pending pairing request by its server-side request id.

        This is the grant path for authenticated admin surfaces (``hermes
        pairing list``, the dashboard/desktop approve buttons), which show
        pending requests but must never reveal the one-time code DM'd to the
        user. Returns ``{user_id, user_name}`` on success, ``None`` for an
        unknown/expired request id.

        Unlike :meth:`approve_code` this does NOT count a miss toward the
        brute-force lockout, and is not itself gated by one. The lockout
        protects the 8-char code space against guessing over a messaging
        channel; a request id is only ever obtained by an admin already
        authenticated to this store, so a stale id means "the row you clicked
        expired", not an attack. Counting it here let a few GUI clicks on a
        stale list lock the operator out of the CLI's code path too.
        """
        with self._lock:
            self._cleanup_expired(platform)
            request_id = str(request_id or "").strip().lower()
            if not request_id:
                return None

            pending = self._load_json(self._pending_path(platform))
            for entry_id, entry in pending.items():
                if not isinstance(entry, dict):
                    continue
                if "salt" not in entry or "hash" not in entry:
                    continue
                if secrets.compare_digest(str(entry_id).lower(), request_id):
                    return self._finish_approval(platform, pending, entry_id, entry)

            return None

    def list_pending(self, platform: str = None) -> list:
        """List pending pairing requests, optionally filtered by platform.

        Codes are stored hashed and are never returned. Each entry exposes a
        server-side ``request_id`` that an authenticated admin surface passes
        to :meth:`approve_request`. Legacy pre-hash entries have no approvable
        id — they report an empty ``request_id`` and age out at TTL.
        """
        results = []
        with self._lock:
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                self._cleanup_expired(p)
                pending = self._load_json(self._pending_path(p))
                for entry_id, info in pending.items():
                    if not isinstance(info, dict):
                        continue
                    created_at = info.get("created_at")
                    if not isinstance(created_at, (int, float)):
                        continue
                    age_min = int((time.time() - created_at) / 60)
                    is_modern = isinstance(info.get("hash"), str) and isinstance(
                        info.get("salt"), str
                    )
                    results.append({
                        "platform": p,
                        "request_id": str(entry_id) if is_modern else "",
                        "user_id": info.get("user_id", ""),
                        "user_name": info.get("user_name", ""),
                        "age_minutes": age_min,
                    })
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        with self._lock:
            count = 0
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                pending = self._load_json(self._pending_path(p))
                count += len(pending)
                self._save_json(self._pending_path(p), {})
        return count

    # ----- Rate limiting and lockout -----

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Check if a user has requested a code too recently."""
        limits = self._load_json(self._rate_limit_path())
        for alias in self._user_id_aliases(platform, user_id):
            key = f"{platform}:{alias}"
            last_request = limits.get(key, 0)
            if (time.time() - last_request) < RATE_LIMIT_SECONDS:
                return True
        return False

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        """Record the time of a pairing request for rate limiting."""
        limits = self._load_json(self._rate_limit_path())
        now = time.time()
        for alias in self._user_id_aliases(platform, user_id):
            key = f"{platform}:{alias}"
            limits[key] = now
        self._save_json(self._rate_limit_path(), limits)

    def _is_locked_out(self, platform: str) -> bool:
        """Check if a platform is in lockout due to failed approval attempts."""
        limits = self._load_json(self._rate_limit_path())
        lockout_key = f"_lockout:{platform}"
        lockout_until = limits.get(lockout_key, 0)
        return time.time() < lockout_until

    def _record_failed_attempt(self, platform: str) -> None:
        """Record a failed approval attempt. Triggers lockout after MAX_FAILED_ATTEMPTS."""
        limits = self._load_json(self._rate_limit_path())
        fail_key = f"_failures:{platform}"
        fails = limits.get(fail_key, 0) + 1
        limits[fail_key] = fails
        if fails >= MAX_FAILED_ATTEMPTS:
            lockout_key = f"_lockout:{platform}"
            limits[lockout_key] = time.time() + LOCKOUT_SECONDS
            limits[fail_key] = 0  # Reset counter
            print(f"[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s "
                  f"after {MAX_FAILED_ATTEMPTS} failed attempts", flush=True)
        self._save_json(self._rate_limit_path(), limits)

    def _reset_failed_attempts(self, platform: str) -> None:
        """Clear the accumulated failed-approval counter after a success.

        Called from the ``approve_code`` success path so that a legitimate
        approval resets the brute-force streak (standard lockout semantics:
        the counter tracks *consecutive* failures, not lifetime ones).
        """
        limits = self._load_json(self._rate_limit_path())
        fail_key = f"_failures:{platform}"
        if limits.get(fail_key):
            limits[fail_key] = 0
            self._save_json(self._rate_limit_path(), limits)

    # ----- Cleanup -----

    def _cleanup_expired(self, platform: str) -> None:
        """Remove expired pending codes.

        Tolerant of malformed / legacy entries — anything without a numeric
        ``created_at`` is treated as expired (it's effectively unusable
        with the new hash-keyed schema anyway).
        """
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        expired = []
        for entry_id, info in pending.items():
            if not isinstance(info, dict):
                expired.append(entry_id)
                continue
            created_at = info.get("created_at")
            if not isinstance(created_at, (int, float)):
                expired.append(entry_id)
                continue
            if (now - created_at) > CODE_TTL_SECONDS:
                expired.append(entry_id)
        if expired:
            for entry_id in expired:
                del pending[entry_id]
            self._save_json(path, pending)

    def _all_platforms(self, suffix: str) -> list:
        """List all platforms that have data files of a given suffix."""
        platforms = []
        for f in self._dir.iterdir():
            if f.name.endswith(f"-{suffix}.json"):
                platform = f.name.replace(f"-{suffix}.json", "")
                if not platform.startswith("_"):
                    platforms.append(platform)
        return platforms
