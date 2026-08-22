"""Typed records shared by the data-generation pipeline.

The hierarchy is seed -> pipeline attempt -> user turn -> individual function
call.  Generator data has no policy-token dimension and never constructs RL
advantages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from .error_taxonomy import ErrorType


SEED_SCHEMA_VERSION = "rods_boundary_seed.v1"
CANDIDATE_SCHEMA_VERSION = "rods_validated_candidate.v1"
SUPPORTED_DATA_TYPES = frozenset(
    {
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_builtin(value: Any) -> Any:
    """Convert common array/model containers into JSON-stable values."""

    if hasattr(value, "model_dump"):
        return to_builtin(value.model_dump())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_builtin(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


@dataclass(frozen=True)
class SeedRecord:
    schema_version: str
    sample_id: str
    data_type: str
    Q_old: list[Any]
    GT_old: list[Any]
    available_functions: list[dict[str, Any]]
    initial_config: dict[str, Any]
    mean_progress: float
    boundary_score_phi: float
    training_epoch_or_step: dict[str, int]
    generation_metadata: dict[str, Any]

    @property
    def source_epoch(self) -> int:
        return int(self.training_epoch_or_step["epoch"])

    @property
    def source_global_step(self) -> int:
        return int(self.training_epoch_or_step["global_step"])

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SeedRecord":
        value = to_builtin(raw)
        required = {
            "schema_version",
            "sample_id",
            "data_type",
            "Q_old",
            "GT_old",
            "available_functions",
            "initial_config",
            "mean_progress",
            "boundary_score_phi",
            "training_epoch_or_step",
            "generation_metadata",
        }
        if set(value) != required:
            raise ValueError(f"seed fields differ from contract: {set(value) ^ required}")
        if value["schema_version"] != SEED_SCHEMA_VERSION:
            raise ValueError("unsupported seed schema_version")
        if not isinstance(value["sample_id"], str) or not value["sample_id"]:
            raise ValueError("seed sample_id must be a non-empty string")
        if value["data_type"] not in SUPPORTED_DATA_TYPES:
            raise ValueError(f"unsupported BFCL data_type: {value['data_type']!r}")
        for key in ("Q_old", "GT_old", "available_functions"):
            if not isinstance(value[key], list):
                raise ValueError(f"seed {key} must be a list")
        if not all(isinstance(item, Mapping) for item in value["available_functions"]):
            raise ValueError("seed available_functions must contain objects")
        if not isinstance(value["initial_config"], Mapping):
            raise ValueError("seed initial_config must be an object")
        if not isinstance(value["generation_metadata"], Mapping):
            raise ValueError("seed generation_metadata must be an object")
        progress = float(value["mean_progress"])
        phi = float(value["boundary_score_phi"])
        if not 0.0 <= progress <= 1.0 or not 0.0 <= phi <= 1.0:
            raise ValueError("seed progress and phi must be in [0, 1]")
        step = value["training_epoch_or_step"]
        if not isinstance(step, Mapping) or set(("epoch", "global_step")) - set(step):
            raise ValueError("seed training_epoch_or_step is malformed")
        if int(step["epoch"]) < 0 or int(step["global_step"]) < 0:
            raise ValueError("seed epoch/global_step cannot be negative")
        return cls(
            schema_version=value["schema_version"],
            sample_id=value["sample_id"],
            data_type=value["data_type"],
            Q_old=list(value["Q_old"]),
            GT_old=list(value["GT_old"]),
            available_functions=[dict(item) for item in value["available_functions"]],
            initial_config=dict(value["initial_config"]),
            mean_progress=progress,
            boundary_score_phi=phi,
            training_epoch_or_step={"epoch": int(step["epoch"]), "global_step": int(step["global_step"])},
            generation_metadata=dict(value["generation_metadata"]),
        )


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    class_name: str
    schema: dict[str, Any]
    level: str = "BOTTOM_LEVEL"
    decomposition: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]
    class_name: str

    def canonical(self) -> str:
        arguments = ", ".join(
            f"{key}={self.arguments[key]!r}" for key in sorted(self.arguments)
        )
        return f"{self.name}({arguments})"


@dataclass(frozen=True)
class PlanTurn:
    turn_id: int
    class_name: str
    function_names: tuple[str, ...]


@dataclass(frozen=True)
class PlannerResult:
    reason: str
    narrative: str
    turns: tuple[PlanTurn, ...]


@dataclass(frozen=True)
class ExecutionRecord:
    turn_id: int
    call_id: int
    call: FunctionCall
    canonical_call: str
    pre_state: dict[str, Any]
    execution_result: Any
    post_state: dict[str, Any]
    dependency_provenance: dict[str, Any]
    success: bool
    semantic_outcome: str = "SUCCESS"
    semantic_detail: str = ""
    error_detail: str | None = None


@dataclass
class SynthesizedTurn:
    turn_id: int
    class_name: str
    calls: list[FunctionCall]
    execution_records: list[ExecutionRecord]
    raw_query: str
    query: str
    query_verification_reason: str
    recovery_tools: list[dict[str, Any]] = field(default_factory=list)
    is_intentional_missing: bool = False
    missing_kind: str | None = None

    @property
    def ground_truth(self) -> list[str]:
        if self.is_intentional_missing:
            return []
        return [call.canonical() for call in self.calls]


@dataclass
class ConversationDraft:
    narrative: str
    data_type: str
    initial_config: dict[str, Any]
    initial_tools: list[dict[str, Any]]
    involved_classes: list[str]
    turns: list[SynthesizedTurn]
    synthesis_environment_id: str
    structural_profile: dict[str, Any]


@dataclass(frozen=True)
class ErrorRecord:
    error_type: ErrorType
    seed_id: str
    attempt_id: int
    turn_id: int | None
    function_names: tuple[str, ...]
    detail: str
    patchable: bool
    timestamp: str = field(default_factory=utc_now)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["error_type"] = self.error_type.value
        value["function_names"] = list(self.function_names)
        return to_builtin(value)


@dataclass(frozen=True)
class PatchOperation:
    class_name: str
    field_path: str
    value: Any


@dataclass(frozen=True)
class JudgeResult:
    reason: str
    decision: str
    fail_reason: str

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendResponse:
    role: str
    text: str
    request_id: str
    raw_response: dict[str, Any]
    latency_seconds: float


@dataclass
class PipelineResult:
    seed_id: str
    status: str
    candidate: dict[str, Any] | None
    errors: list[ErrorRecord]
    attempts: int
    planner_calls: int
    blocklist_history: list[list[str]]
    config_patch_history: list[dict[str, Any]]
    metrics: dict[str, float]
    reason: str = ""
    checkpoint: dict[str, Any] = field(default_factory=dict)


def stable_id(prefix: str, payload: Any, *, length: int = 24) -> str:
    encoded = json.dumps(to_builtin(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:length]
    return f"{prefix}_{digest}"


class SeedStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    DROPPED = "DROPPED"
