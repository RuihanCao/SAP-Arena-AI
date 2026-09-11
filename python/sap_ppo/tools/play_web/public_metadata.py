"""Public display metadata, without machine-specific locations."""

import re
from pathlib import PurePosixPath, PureWindowsPath

from .public_settings import public_mode

_LOCAL_KEYS = frozenset({
    "module_path", "repo_root", "head_ref", "archive_root", "snapshot_path",
    "baseline_id", "honest_driver_commit", "reference_execution_commit",
})
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|/(?:root|home|Users|tmp)/)[^\s\"'<>]*")
_LEGACY_ID = re.compile(r"^arm-c-(.+)$")


def public_text(value: str) -> str:
    return _LOCAL_PATH.sub("<local path>", value)


def public_payload(value):
    """Copy a response without changing the stored game or its numeric data."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _LOCAL_KEYS:
                continue
            if key == "path" and isinstance(item, str):
                item = PureWindowsPath(item).name if "\\" in item else PurePosixPath(item).name
            if key == "gear" and isinstance(item, str):
                item = public_mode(item)
            if key == "schema_version" and item == "exp16_value_log_v1":
                item = "value_log_v1"
            if key == "agent_id" and isinstance(item, str):
                item = _LEGACY_ID.sub(r"bc-value-\1", item)
            if key in {"agent_name", "agent"} and isinstance(item, str) and re.search(r"arm[ -]c|exp\d+", item, re.I):
                item = "BC + value search"
            result[key] = public_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [public_payload(item) for item in value]
    if isinstance(value, str):
        return public_text(value)
    return value
