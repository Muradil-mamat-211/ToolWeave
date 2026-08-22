"""RODS boundary output and validated-candidate replay interfaces.

This is the Training Branch boundary.  It emits seeds and ingests only already
validated BFCL candidates; it does not implement planning, rewriting, judging,
or environment synthesis.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import extract_available_functions, to_builtin


SEED_SCHEMA_VERSION = "rods_boundary_seed.v1"
CANDIDATE_SCHEMA_VERSION = "rods_validated_candidate.v1"
STATE_SCHEMA_VERSION = "rods_stage3_lifecycle_state.v2"
LEGACY_STATE_SCHEMA_VERSION = "rods_stage3_lifecycle_state.v1"

BFCL_DATA_TYPES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


@dataclass(frozen=True)
class BoundaryDecision:
    mean_progress: float
    classification: str
    boundary_score_phi: float

    @property
    def is_boundary(self) -> bool:
        return self.classification == "boundary"


def classify_progress(
    mean_progress: float,
    *,
    boundary_low: float = 0.20,
    boundary_high: float = 0.85,
) -> BoundaryDecision:
    """Classify using only mean RODS Progress Reward across rollouts."""

    value = float(mean_progress)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"mean Progress Reward must be in [0, 1], got {value}")
    if value < boundary_low:
        classification = "too_hard"
    elif value <= boundary_high:
        classification = "boundary"
    else:
        classification = "mastered"
    return BoundaryDecision(
        mean_progress=value,
        classification=classification,
        boundary_score_phi=4.0 * value * (1.0 - value),
    )


@dataclass(frozen=True)
class LifecycleConfig:
    enabled: bool = False
    seed_output_path: str = ""
    candidate_input_path: str = ""
    state_path: str = ""
    generated_pool_cap: int = 400
    trial_observation_count: int = 1
    trial_eviction_threshold: float = 0.20
    retire_mastered_threshold: float = 0.95
    retire_too_hard_threshold: float = 0.20
    boundary_low: float = 0.20
    boundary_high: float = 0.85
    # RODS specifies these mechanisms but does not publish unique values for M,
    # M_tau, or c.  All three remain None unless an experiment config supplies
    # them together; an incomplete selection policy fails closed.
    max_seeds_per_selection: int | None = None
    seed_type_quotas: Mapping[str, int] | None = None
    seed_cooldown_steps: int | None = None
    # Production Stage3 synthesis must fail fast until the paper-unspecified
    # project choices M, all M_tau, and c are supplied explicitly.  Stage1/2,
    # ingestion-only tests, and generator-disabled modes may leave this false.
    require_seed_selection_config: bool = False
    # Appendix B.1 explicitly fixes beta=20% of the active pool immediately
    # before epoch-boundary injection.
    injection_ratio: float = 0.20
    # The paper requires stale retirement but publishes no reproducible window.
    # None keeps the optional hook disabled rather than inventing a threshold.
    stale_after_steps: int | None = None

    @property
    def seed_selection_configured(self) -> bool:
        return (
            self.max_seeds_per_selection is not None
            and self.seed_type_quotas is not None
            and self.seed_cooldown_steps is not None
        )

    def __post_init__(self) -> None:
        if self.enabled and not all(
            (self.seed_output_path, self.candidate_input_path, self.state_path)
        ):
            raise ValueError("enabled Stage3 lifecycle requires seed, candidate, and state paths")
        if self.generated_pool_cap < 0:
            raise ValueError("generated_pool_cap cannot be negative")
        if self.trial_observation_count < 1:
            raise ValueError("trial_observation_count must be at least one")
        if not 0.0 <= self.injection_ratio <= 1.0:
            raise ValueError("injection_ratio must be in [0, 1]")
        if self.stale_after_steps is not None and self.stale_after_steps < 1:
            raise ValueError("stale_after_steps must be positive or null")
        selection_parts = (
            self.max_seeds_per_selection is not None,
            self.seed_type_quotas is not None,
            self.seed_cooldown_steps is not None,
        )
        if any(selection_parts) and not all(selection_parts):
            raise ValueError("M, all M_tau quotas, and cooldown c must be configured together")
        if self.enabled and self.require_seed_selection_config and not all(selection_parts):
            raise ValueError(
                "RODS Stage3 seed selection is not configured: M/M_tau/c are "
                "paper-unspecified project hyperparameters"
            )
        if self.seed_selection_configured:
            assert self.max_seeds_per_selection is not None
            assert self.seed_type_quotas is not None
            assert self.seed_cooldown_steps is not None
            if self.max_seeds_per_selection <= 0:
                raise ValueError("max_seeds_per_selection must be positive")
            if self.seed_cooldown_steps < 0:
                raise ValueError("seed_cooldown_steps cannot be negative")
            if set(self.seed_type_quotas) != set(BFCL_DATA_TYPES):
                raise ValueError(f"seed_type_quotas must contain exactly {BFCL_DATA_TYPES!r}")
            if any(quota < 0 for quota in self.seed_type_quotas.values()):
                raise ValueError("seed type quotas cannot be negative")
            if sum(self.seed_type_quotas.values()) != self.max_seeds_per_selection:
                raise ValueError("sum(seed_type_quotas) must equal max_seeds_per_selection")
        if not (
            0.0
            <= self.retire_too_hard_threshold
            <= self.boundary_low
            <= self.boundary_high
            <= self.retire_mastered_threshold
            <= 1.0
        ):
            raise ValueError("invalid RODS boundary/retirement thresholds")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LifecycleConfig":
        raw = raw or {}
        raw_quotas = raw.get("seed_type_quotas")
        seed_type_quotas: dict[str, int] | None
        if raw_quotas is None:
            seed_type_quotas = None
        elif not isinstance(raw_quotas, Mapping):
            raise ValueError("seed_type_quotas must be a mapping or null")
        elif all(value is None for value in raw_quotas.values()):
            seed_type_quotas = None
        elif any(value is None for value in raw_quotas.values()):
            raise ValueError("seed_type_quotas must be entirely configured or entirely null")
        else:
            seed_type_quotas = {str(key): int(value) for key, value in raw_quotas.items()}

        raw_max_seeds = raw.get("max_seeds_per_selection")
        raw_cooldown = raw.get("seed_cooldown_steps")
        raw_stale = raw.get("stale_after_steps")
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            seed_output_path=str(raw.get("seed_output_path", "")),
            candidate_input_path=str(raw.get("candidate_input_path", "")),
            state_path=str(raw.get("state_path", "")),
            generated_pool_cap=int(raw.get("generated_pool_cap", 400)),
            trial_observation_count=int(raw.get("trial_observation_count", 1)),
            trial_eviction_threshold=float(raw.get("trial_eviction_threshold", 0.20)),
            retire_mastered_threshold=float(raw.get("retire_mastered_threshold", 0.95)),
            retire_too_hard_threshold=float(raw.get("retire_too_hard_threshold", 0.20)),
            boundary_low=float(raw.get("boundary_low", 0.20)),
            boundary_high=float(raw.get("boundary_high", 0.85)),
            max_seeds_per_selection=(None if raw_max_seeds is None else int(raw_max_seeds)),
            seed_type_quotas=seed_type_quotas,
            seed_cooldown_steps=(None if raw_cooldown is None else int(raw_cooldown)),
            require_seed_selection_config=bool(
                raw.get("require_seed_selection_config", False)
            ),
            injection_ratio=float(raw.get("injection_ratio", 0.20)),
            stale_after_steps=(None if raw_stale is None else int(raw_stale)),
        )
        return config


def _sample_id(sample: Mapping[str, Any]) -> str:
    extra_info = sample.get("extra_info", {})
    kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, Mapping) else {}
    value = kwargs.get("id") if isinstance(kwargs, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("candidate sample requires extra_info.interaction_kwargs.id")
    return value


def validate_candidate_record(raw_record: Any) -> dict[str, Any]:
    """Fail closed unless the Generator marks and packages a valid BFCL row."""

    record = to_builtin(raw_record)
    if not isinstance(record, Mapping):
        raise ValueError("candidate record must be an object")
    if record.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema_version is not supported")
    if record.get("validated") is not True:
        raise ValueError("candidate validated must be exactly true")
    validation = record.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise ValueError("candidate requires validation.passed=true")
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    sample = record.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError("candidate sample must be an object")
    required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    if not required_columns.issubset(sample):
        raise ValueError(f"candidate sample is missing columns: {required_columns - set(sample)}")

    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("candidate prompt must be a non-empty message list")
    available_functions = extract_available_functions(prompt)
    if not available_functions:
        raise ValueError("candidate must expose at least one available function")
    system_messages = [
        message.get("content", "")
        for message in prompt
        if isinstance(message, Mapping) and message.get("role") == "system"
    ]
    if (
        len(system_messages) != 1
        or "<think>" not in system_messages[0]
        or "<answer>" not in system_messages[0]
        or "<thinking>" in system_messages[0]
    ):
        raise ValueError("candidate prompt does not use the executable <think>/<answer> protocol")

    reward_model = sample.get("reward_model")
    if not isinstance(reward_model, Mapping) or reward_model.get("style") != "interaction":
        raise ValueError("candidate reward_model.style must be interaction")
    extra_info = sample.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError("candidate extra_info must be an object")
    kwargs = extra_info.get("interaction_kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError("candidate interaction_kwargs must be an object")
    required_kwargs = {
        "name",
        "id",
        "initial_config",
        "involved_classes",
        "ground_truth",
        "processed_question",
        "question",
    }
    if not required_kwargs.issubset(kwargs):
        raise ValueError(f"candidate interaction kwargs missing: {required_kwargs - set(kwargs)}")
    question = to_builtin(kwargs["question"])
    ground_truth = to_builtin(kwargs["ground_truth"])
    if not isinstance(question, list) or not isinstance(ground_truth, list):
        raise ValueError("candidate question and ground_truth must be turn lists")
    if len(question) != len(ground_truth) or not question:
        raise ValueError("candidate question/ground_truth turn counts must agree and be non-zero")
    initial_config = kwargs["initial_config"]
    if isinstance(initial_config, str):
        parsed_config = json.loads(initial_config)
    elif isinstance(initial_config, Mapping):
        parsed_config = initial_config
    else:
        raise ValueError("candidate initial_config must be JSON or an object")
    if not isinstance(parsed_config, Mapping):
        raise ValueError("candidate initial_config must decode to an object")
    _sample_id(sample)
    return dict(record)


class JsonlQueue:
    """Single-driver append/read filesystem protocol with durable records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(to_builtin(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {self.path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"queue record at {self.path}:{line_number} is not an object")
                records.append(record)
        return records


class RODSStage3Lifecycle:
    """Driver-side boundary detector and generated-pool lifecycle manager."""

    def __init__(self, config: LifecycleConfig | Mapping[str, Any]):
        self.config = config if isinstance(config, LifecycleConfig) else LifecycleConfig.from_mapping(config)
        self.seed_queue = JsonlQueue(self.config.seed_output_path)
        self.candidate_queue = JsonlQueue(self.config.candidate_input_path)
        self.state_path = Path(self.config.state_path) if self.config.state_path else None
        self._original_rows: list[dict[str, Any]] | None = None
        self._original_ids: set[str] = set()
        self._state = self._load_state()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "active_candidates": {},
            "processed_candidate_ids": [],
            "rejected_candidates": {},
            # Cooldown is keyed by stable sample identity. A key containing the
            # current step cannot prevent re-emission on the next step.
            "last_emitted_step": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.is_file():
            return self._empty_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        schema_version = state.get("schema_version")
        if schema_version == LEGACY_STATE_SCHEMA_VERSION:
            last_emitted_step: dict[str, int] = {}
            for raw_key in state.pop("emitted_seed_keys", []):
                if not isinstance(raw_key, str) or ":" not in raw_key:
                    continue
                sample_id, raw_step = raw_key.rsplit(":", 1)
                try:
                    step = int(raw_step)
                except ValueError:
                    continue
                last_emitted_step[sample_id] = max(last_emitted_step.get(sample_id, step), step)
            state["last_emitted_step"] = last_emitted_step
            state["schema_version"] = STATE_SCHEMA_VERSION
        elif schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported lifecycle state at {self.state_path}")

        state.setdefault("active_candidates", {})
        state.setdefault("processed_candidate_ids", [])
        state.setdefault("rejected_candidates", {})
        state.setdefault("last_emitted_step", {})
        return state

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        temporary.write_text(
            json.dumps(to_builtin(self._state), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    @property
    def active_generated_count(self) -> int:
        return len(self._state["active_candidates"])

    def _initialize_original_rows(self, dataset: Any) -> None:
        if self._original_rows is not None:
            return
        active_sample_ids = {
            item["sample_id"] for item in self._state["active_candidates"].values()
        }
        original_rows: list[dict[str, Any]] = []
        for raw_row in dataset.dataframe:
            row = to_builtin(raw_row)
            if _sample_id(row) not in active_sample_ids:
                original_rows.append(row)
        self._original_rows = original_rows
        self._original_ids = {_sample_id(row) for row in original_rows}

    def _replace_dataset(self, dataset: Any) -> None:
        if self._original_rows is None:
            raise AssertionError("original dataset rows were not initialized")
        from datasets import Dataset

        generated_rows = [
            item["sample"]
            for _, item in sorted(
                self._state["active_candidates"].items(),
                key=lambda pair: (pair[1]["ingested_step"], pair[0]),
            )
        ]
        rows = [*self._original_rows, *generated_rows]
        # Reuse the original Arrow features: candidates cannot smuggle an
        # incompatible training schema into the active pool.
        dataset.dataframe = Dataset.from_list(rows, features=dataset.dataframe.features)

    @staticmethod
    def _latest_progress(item: Mapping[str, Any]) -> float | None:
        observations = item.get("progress_observations", [])
        if not isinstance(observations, list) or not observations:
            return None
        return float(observations[-1])

    @classmethod
    def _priority_phi(cls, item: Mapping[str, Any]) -> float | None:
        """Return the RODS variance proxy, or None before a trial observation."""

        progress = cls._latest_progress(item)
        if progress is None:
            return None
        return 4.0 * progress * (1.0 - progress)

    def _retire_generated(self, *, global_step: int) -> tuple[int, int, int]:
        """Apply generated-only burn-in, drift, and optional stale retirement."""

        retired_too_hard = 0
        retired_mastered = 0
        retired_stale = 0
        active = self._state["active_candidates"]
        for candidate_id, item in list(active.items()):
            observations = item.get("progress_observations", [])
            if len(observations) >= self.config.trial_observation_count:
                trial_progress = float(
                    observations[self.config.trial_observation_count - 1]
                )
                latest = float(observations[-1])
                if trial_progress < self.config.trial_eviction_threshold:
                    retired_too_hard += 1
                    del active[candidate_id]
                    continue
                if latest < self.config.retire_too_hard_threshold:
                    retired_too_hard += 1
                    del active[candidate_id]
                    continue
                if latest > self.config.retire_mastered_threshold:
                    retired_mastered += 1
                    del active[candidate_id]
                    continue

            if self.config.stale_after_steps is not None:
                reference_step = item.get("last_observed_step")
                if reference_step is None:
                    reference_step = item.get("ingested_step")
                if reference_step is not None and (
                    int(global_step) - int(reference_step) >= self.config.stale_after_steps
                ):
                    retired_stale += 1
                    del active[candidate_id]

        return retired_too_hard, retired_mastered, retired_stale

    def _prune_excess_generated(self) -> int:
        """Enforce P_max by removing observed generated rows with lowest phi.

        Unobserved rows have no defensible RODS priority and remain protected by
        the trial mechanism.  Normal ingestion never exceeds P_max; the error
        branch therefore only guards incompatible restored state/config pairs.
        """

        active = self._state["active_candidates"]
        excess = len(active) - self.config.generated_pool_cap
        if excess <= 0:
            return 0
        ranked: list[tuple[float, int, str]] = []
        for candidate_id, item in active.items():
            phi = self._priority_phi(item)
            if phi is None:
                continue
            ranked.append(
                (
                    phi,
                    int(item.get("last_observed_step", item.get("ingested_step", -1))),
                    str(candidate_id),
                )
            )
        ranked.sort()
        if len(ranked) < excess:
            raise RuntimeError(
                "cannot enforce generated_pool_cap without assigning fake priority "
                "or evicting unobserved trial candidates"
            )
        for _, _, candidate_id in ranked[:excess]:
            del active[candidate_id]
        return excess

    def _ingest_candidates(
        self,
        *,
        epoch: int,
        global_step: int,
        max_new_this_epoch: int,
    ) -> tuple[int, int, int]:
        """Ingest a FIFO prefix under both beta*|D_active| and P_max.

        Deferred records remain unprocessed in the append-only queue and are
        reconsidered at the next epoch boundary.
        """

        processed = set(self._state["processed_candidate_ids"])
        rejected: dict[str, str] = self._state["rejected_candidates"]
        active = self._state["active_candidates"]
        active_sample_ids = {item["sample_id"] for item in active.values()}
        pending_candidate_ids: set[str] = set()
        pending_sample_ids: set[str] = set()
        ready: list[tuple[dict[str, Any], str, str]] = []
        not_ready_count = 0
        rejected_count = 0

        for raw_record in self.candidate_queue.read():
            raw_id = raw_record.get("candidate_id")
            if isinstance(raw_id, str) and (
                raw_id in processed
                or raw_id in rejected
                or raw_id in pending_candidate_ids
            ):
                continue
            try:
                record = validate_candidate_record(raw_record)
                candidate_id = record["candidate_id"]
                sample_id = _sample_id(record["sample"])
                if candidate_id in active:
                    raise ValueError(f"duplicate active candidate id: {candidate_id}")
                if (
                    sample_id in self._original_ids
                    or sample_id in active_sample_ids
                    or sample_id in pending_sample_ids
                ):
                    raise ValueError(f"duplicate active/original/pending sample id: {sample_id}")

                generation_metadata = record.get("generation_metadata", {})
                if generation_metadata is None:
                    generation_metadata = {}
                if not isinstance(generation_metadata, Mapping):
                    raise ValueError("generation_metadata must be an object when provided")
                generated_epoch = generation_metadata.get("generated_epoch")
                if generated_epoch is not None and int(generated_epoch) >= int(epoch):
                    not_ready_count += 1
                else:
                    ready.append((record, candidate_id, sample_id))
                pending_candidate_ids.add(candidate_id)
                pending_sample_ids.add(sample_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                reject_id = str(raw_id or f"invalid_record_{len(rejected)}")
                rejected[reject_id] = str(exc)
                rejected_count += 1

        free_pool_slots = max(self.config.generated_pool_cap - len(active), 0)
        admission_limit = min(
            int(max_new_this_epoch),
            free_pool_slots,
            len(ready),
        )
        for record, candidate_id, sample_id in ready[:admission_limit]:
            active[candidate_id] = {
                "sample_id": sample_id,
                "sample": to_builtin(record["sample"]),
                "generation_metadata": to_builtin(record.get("generation_metadata", {})),
                "validation": to_builtin(record["validation"]),
                "ingested_epoch": int(epoch),
                "ingested_step": int(global_step),
                "last_observed_epoch": None,
                "last_observed_step": None,
                "progress_observations": [],
            }
            processed.add(candidate_id)

        self._state["processed_candidate_ids"] = sorted(processed)
        deferred = not_ready_count + len(ready) - admission_limit
        return admission_limit, rejected_count, deferred

    def on_epoch_boundary(self, dataset: Any, *, epoch: int, global_step: int) -> dict[str, float]:
        """Retire/prune generated rows and rate-limit validated injection."""

        if not self.config.enabled:
            return {}
        self._initialize_original_rows(dataset)
        retired_hard, retired_mastered, retired_stale = self._retire_generated(
            global_step=global_step
        )
        priority_pruned = self._prune_excess_generated()
        active_pool_before_injection = len(self._original_ids) + self.active_generated_count
        max_new_this_epoch = math.floor(
            self.config.injection_ratio * active_pool_before_injection
        )
        ingested, rejected, deferred = self._ingest_candidates(
            epoch=epoch,
            global_step=global_step,
            max_new_this_epoch=max_new_this_epoch,
        )
        if self.active_generated_count > self.config.generated_pool_cap:
            raise AssertionError("generated pool exceeded generated_pool_cap")
        self._replace_dataset(dataset)
        self._save_state()
        return {
            "rods_stage3_lifecycle/active_generated_count": float(self.active_generated_count),
            "rods_stage3_lifecycle/active_pool_before_injection": float(
                active_pool_before_injection
            ),
            "rods_stage3_lifecycle/max_new_this_epoch": float(max_new_this_epoch),
            "rods_stage3_lifecycle/injection_ratio": float(self.config.injection_ratio),
            "rods_stage3_lifecycle/ingested_candidate_count": float(ingested),
            "rods_stage3_lifecycle/rejected_candidate_count": float(rejected),
            "rods_stage3_lifecycle/deferred_candidate_count": float(deferred),
            "rods_stage3_lifecycle/retired_too_hard_count": float(retired_hard),
            "rods_stage3_lifecycle/retired_mastered_count": float(retired_mastered),
            "rods_stage3_lifecycle/retired_stale_count": float(retired_stale),
            "rods_stage3_lifecycle/priority_pruned_count": float(priority_pruned),
            "rods_stage3_lifecycle/original_protected_count": float(len(self._original_ids)),
        }

    def _record_generated_observation(
        self,
        sample_id: str,
        mean_progress: float,
        *,
        epoch: int,
        global_step: int,
    ) -> None:
        for item in self._state["active_candidates"].values():
            if item["sample_id"] == sample_id:
                item.setdefault("progress_observations", []).append(float(mean_progress))
                item["last_observed_epoch"] = int(epoch)
                item["last_observed_step"] = int(global_step)
                return

    def observe_and_emit(
        self,
        *,
        progress_rewards: Sequence[float],
        uids: Sequence[Any],
        rollout_provenance: Sequence[Any],
        epoch: int,
        global_step: int,
    ) -> dict[str, float]:
        """Observe K rollout outcomes and emit boundary seeds from R_P only."""

        if not self.config.enabled:
            return {}
        if not (len(progress_rewards) == len(uids) == len(rollout_provenance)):
            raise ValueError("progress, uid, and provenance batches must align")

        grouped_scores: dict[str, list[float]] = {}
        grouped_context: dict[str, Mapping[str, Any]] = {}
        for score, uid, raw_context in zip(progress_rewards, uids, rollout_provenance):
            key = str(uid)
            grouped_scores.setdefault(key, []).append(float(score))
            context = to_builtin(raw_context)
            if isinstance(context, Mapping):
                grouped_context.setdefault(key, context)

        eligible_by_type: dict[str, list[dict[str, Any]]] = {
            data_type: [] for data_type in BFCL_DATA_TYPES
        }
        seen_boundary_samples: set[str] = set()
        cooldown_filtered = 0
        unsupported_type_count = 0
        class_counts = {"too_hard": 0, "boundary": 0, "mastered": 0}
        for uid, scores in grouped_scores.items():
            mean_progress = sum(scores) / len(scores)
            decision = classify_progress(
                mean_progress,
                boundary_low=self.config.boundary_low,
                boundary_high=self.config.boundary_high,
            )
            class_counts[decision.classification] += 1
            context = grouped_context.get(uid)
            if context is None:
                continue
            sample_id = str(context.get("prompt_id", ""))
            if sample_id:
                self._record_generated_observation(
                    sample_id,
                    mean_progress,
                    epoch=epoch,
                    global_step=global_step,
                )
            if not decision.is_boundary or not bool(context.get("context_reliable", False)):
                continue
            if not sample_id or sample_id in seen_boundary_samples:
                continue
            seen_boundary_samples.add(sample_id)
            data_type = str(context.get("data_type", ""))
            if data_type not in eligible_by_type:
                unsupported_type_count += 1
                continue
            if self.config.seed_selection_configured:
                assert self.config.seed_cooldown_steps is not None
                last_step = self._state["last_emitted_step"].get(sample_id)
                if last_step is not None and (
                    int(global_step) - int(last_step) < self.config.seed_cooldown_steps
                ):
                    cooldown_filtered += 1
                    continue
            eligible_by_type[data_type].append(
                {
                    "uid": uid,
                    "scores": scores,
                    "sample_id": sample_id,
                    "data_type": data_type,
                    "context": context,
                    "decision": decision,
                }
            )

        selected: list[dict[str, Any]] = []
        quota_filtered = 0
        selected_by_type = {data_type: 0 for data_type in BFCL_DATA_TYPES}
        if self.config.seed_selection_configured:
            assert self.config.seed_type_quotas is not None
            assert self.config.max_seeds_per_selection is not None
            for data_type in BFCL_DATA_TYPES:
                ranked = sorted(
                    eligible_by_type[data_type],
                    key=lambda item: (
                        -item["decision"].boundary_score_phi,
                        item["sample_id"],
                        item["uid"],
                    ),
                )
                quota = self.config.seed_type_quotas[data_type]
                quota_filtered += max(len(ranked) - quota, 0)
                for type_rank, item in enumerate(ranked[:quota], 1):
                    item["type_rank"] = type_rank
                    selected.append(item)
            # The validated config requires sum(M_tau) == M.  This slice is a
            # second invariant guard and never redistributes unused type quota.
            selected = selected[: self.config.max_seeds_per_selection]

        emitted: list[dict[str, Any]] = []
        for item in selected:
            context = item["context"]
            decision = item["decision"]
            data_type = item["data_type"]
            seed = {
                "schema_version": SEED_SCHEMA_VERSION,
                "sample_id": item["sample_id"],
                "data_type": data_type,
                "Q_old": to_builtin(context.get("questions", [])),
                "GT_old": to_builtin(context.get("ground_truth", [])),
                "available_functions": to_builtin(context.get("available_functions", [])),
                "initial_config": to_builtin(context.get("initial_config", {})),
                "mean_progress": decision.mean_progress,
                "boundary_score_phi": decision.boundary_score_phi,
                "training_epoch_or_step": {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                },
                "generation_metadata": {
                    "source": "RODS Training Branch boundary detector",
                    "uid": item["uid"],
                    "rollout_count": len(item["scores"]),
                    "progress_source": "R_P_only",
                    "type_rank_by_phi": item["type_rank"],
                    "max_seeds_per_selection": self.config.max_seeds_per_selection,
                    "seed_type_quota": self.config.seed_type_quotas[data_type],
                    "seed_cooldown_steps": self.config.seed_cooldown_steps,
                },
            }
            emitted.append(seed)
            selected_by_type[data_type] += 1
            self._state["last_emitted_step"][item["sample_id"]] = int(global_step)

        self.seed_queue.append(emitted)
        self._save_state()
        total_groups = len(grouped_scores)
        metrics = {
            "rods_boundary/seed_emitted_count": float(len(emitted)),
            "rods_boundary/seed_selection_configured": float(
                self.config.seed_selection_configured
            ),
            "rods_boundary/cooldown_filtered_count": float(cooldown_filtered),
            "rods_boundary/quota_filtered_count": float(quota_filtered),
            "rods_boundary/unsupported_type_count": float(unsupported_type_count),
            "rods_boundary/too_hard_count": float(class_counts["too_hard"]),
            "rods_boundary/boundary_count": float(class_counts["boundary"]),
            "rods_boundary/mastered_count": float(class_counts["mastered"]),
            "rods_boundary/boundary_rate": class_counts["boundary"] / total_groups if total_groups else 0.0,
        }
        for data_type in BFCL_DATA_TYPES:
            metrics[f"rods_boundary/seed_emitted_{data_type}"] = float(
                selected_by_type[data_type]
            )
        return metrics
