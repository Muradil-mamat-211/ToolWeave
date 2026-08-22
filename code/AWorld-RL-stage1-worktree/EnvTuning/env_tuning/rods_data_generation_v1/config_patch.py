"""Official Config Patch prompt plus deterministic, type-safe application."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from .error_taxonomy import PATCHABLE_ERRORS
from .llm_backend import LLMBackend
from .metrics import GeneratorMetrics
from .models import ErrorRecord, PatchOperation
from .parsing import StructuredParseError, parse_config_patch_response
from .prompts import load_prompt


class UnsafePatchError(ValueError):
    pass


def _split_field_path(field_path: str) -> list[str]:
    r"""Split dot notation while supporting ``\.`` for literal key dots."""

    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for char in field_path:
        if escaped:
            if char not in {".", "\\"}:
                raise UnsafePatchError(f"unsupported field-path escape: \\{char}")
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            if not current:
                raise UnsafePatchError("patch field path has an empty segment")
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        raise UnsafePatchError("patch field path has a dangling escape")
    if not current:
        raise UnsafePatchError("patch field path has an empty segment")
    parts.append("".join(current))
    return parts


def _validate_gorilla_filesystem_config(config: Mapping[str, Any]) -> None:
    """Fail closed if a patch corrupts the official BFCL filesystem tree.

    Every child under ``root`` is an official BFCL node with a ``type``.  This
    catches the observed incident where ``report.txt`` was split into
    ``report -> txt`` and the next VM crashed with ``KeyError('type')``.
    """

    filesystem = config.get("GorillaFileSystem")
    if filesystem is None:
        return
    if not isinstance(filesystem, Mapping):
        raise UnsafePatchError("GorillaFileSystem config must be a mapping")
    root = filesystem.get("root")
    if not isinstance(root, Mapping):
        raise UnsafePatchError("GorillaFileSystem.root must be a mapping")

    def validate_node(node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            raise UnsafePatchError(f"filesystem node must be a mapping at {path}")
        node_type = node.get("type")
        if node_type == "file":
            if "content" not in node:
                raise UnsafePatchError(f"filesystem file has no content at {path}")
            return
        if node_type != "directory":
            raise UnsafePatchError(f"filesystem node has invalid/missing type at {path}")
        contents = node.get("contents")
        if not isinstance(contents, Mapping):
            raise UnsafePatchError(f"filesystem directory has invalid contents at {path}")
        for name, child in contents.items():
            validate_node(child, f"{path}/{name}")

    for name, node in root.items():
        validate_node(node, f"root/{name}")


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge; later leaves win only when container types agree."""

    output = copy.deepcopy(dict(base))
    for key, patch_value in patch.items():
        if key not in output:
            output[key] = copy.deepcopy(patch_value)
            continue
        base_value = output[key]
        if isinstance(base_value, Mapping):
            if not isinstance(patch_value, Mapping):
                raise UnsafePatchError(f"scalar cannot replace dict at {key}")
            output[key] = deep_merge(base_value, patch_value)
            continue
        if isinstance(patch_value, Mapping):
            raise UnsafePatchError(f"dict cannot replace scalar at {key}")
        if base_value is not None and patch_value is not None:
            # bool is an int subclass; compare exact concrete types so a model
            # cannot silently turn booleans into numeric flags.
            numeric = isinstance(base_value, (int, float)) and not isinstance(base_value, bool)
            patch_numeric = isinstance(patch_value, (int, float)) and not isinstance(patch_value, bool)
            if type(base_value) is not type(patch_value) and not (numeric and patch_numeric):
                raise UnsafePatchError(
                    f"type-incompatible leaf patch at {key}: "
                    f"{type(base_value).__name__} -> {type(patch_value).__name__}"
                )
        output[key] = copy.deepcopy(patch_value)
    return output


def operations_to_patch(operations: Sequence[PatchOperation]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for operation in operations:
        if not operation.class_name or not operation.field_path:
            raise UnsafePatchError("patch class and field path must be non-empty")
        parts = _split_field_path(operation.field_path)
        root = patch.setdefault(operation.class_name, {})
        if not isinstance(root, dict):
            raise UnsafePatchError("patch class root collision")
        cursor = root
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise UnsafePatchError(f"patch path collision at {part}")
            cursor = existing
        leaf = parts[-1]
        if leaf in cursor and isinstance(cursor[leaf], dict) != isinstance(operation.value, dict):
            raise UnsafePatchError(f"incompatible repeated patch for {operation.field_path}")
        cursor[leaf] = copy.deepcopy(operation.value)
    return patch


def apply_patch_operations(
    current_config: Mapping[str, Any], operations: Sequence[PatchOperation]
) -> tuple[dict[str, Any], dict[str, Any]]:
    patch = operations_to_patch(operations)
    merged = deep_merge(current_config, patch)
    _validate_gorilla_filesystem_config(merged)
    return merged, patch


class ConfigPatchAgent:
    def __init__(self, backend: LLMBackend, metrics: GeneratorMetrics):
        self.backend = backend
        self.metrics = metrics

    async def propose(
        self,
        error: ErrorRecord,
        *,
        current_config: Mapping[str, Any],
    ) -> tuple[str, list[PatchOperation]]:
        if error.error_type not in PATCHABLE_ERRORS:
            raise ValueError(f"non-patchable error cannot invoke Config Patch: {error.error_type}")
        system = load_prompt("official_rods/config_patch_system.txt")
        user = load_prompt(
            "official_rods/config_patch_user.txt",
            {
                "error_type": error.error_type.value,
                "error_function": ", ".join(error.function_names),
                "error_detail": error.detail,
                "config_str": json.dumps(current_config, ensure_ascii=False, indent=2, sort_keys=True),
            },
        )
        # Project transport rule around the unchanged official C.2 prompt.
        # Dot notation is ambiguous for filesystem keys such as report.txt;
        # deterministic application uses a backslash to escape literal dots.
        user += (
            "\n\n# Deterministic field-path transport\n"
            "An unescaped dot separates nested mapping fields. Use \\. for a "
            "literal dot inside one key (example: contents.report\\.txt)."
        )
        response = await self.backend.complete(
            role="config_patch",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            metadata={"seed_id": error.seed_id, "attempt_id": error.attempt_id},
        )
        self.metrics.increment("latency/config_patch_seconds_sum", response.latency_seconds)
        self.metrics.increment("latency/config_patch_count")
        return parse_config_patch_response(response.text)
