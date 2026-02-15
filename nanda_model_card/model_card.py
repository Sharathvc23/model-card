"""Unified Model Card — single metadata schema for all model types.

Covers LLM LoRA adapters, edge ONNX models, federated models, and
heuristic fallbacks.  This is the canonical metadata record that flows
through training, registry publishing, and governance gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

MODEL_STATUSES = frozenset(
    {"shadow", "ready", "deprecated", "archived"}
)
MODEL_TYPES = frozenset(
    {"lora_adapter", "onnx_edge", "federated", "heuristic"}
)
RISK_LEVELS = frozenset({"low", "medium", "high"})


@dataclass
class ModelCard:
    """Unified metadata record for any model type.

    Attributes:
        card_id: Unique card identifier.
        model_id: The model this card describes.
        version: Monotonically increasing version number.
        model_type: One of ``lora_adapter``, ``onnx_edge``,
            ``federated``, ``heuristic``.
        owner: Owner of this model (team, org, or individual).
        base_model: Underlying foundation model name.
        architecture: Human-readable architecture string
            (e.g. ``llama-3.1-8b+lora``, ``onnx-linear``).
        profile_tags: Descriptive tags for training purpose.
        allowed_profiles: Which profiles may use this model.
        risk_level: Blast-radius assessment (``low``, ``medium``,
            ``high``).
        weights_hash: SHA-256 hex digest of the weight blob.
        dataset_id: Provenance link to training dataset.
        dataset_hash: SHA-256 hex digest of the training data.
        dataset_size: Number of training examples.
        metrics: Training / evaluation metrics.
        status: Lifecycle status (``shadow``, ``ready``,
            ``deprecated``, ``archived``).
        created_at: When the card was created.
        approved_by: Identifier of human approver
            (None = auto-approved).
        shadow_until: When shadow mode expires.
        correlation_id: Threading ID for lifecycle audit.
        base_manifest_id: Reference to base manifest.
    """

    card_id: str = field(
        default_factory=lambda: f"card:{uuid4().hex[:16]}"
    )
    model_id: str = ""
    version: int = 1
    model_type: str = "lora_adapter"
    owner: str = ""
    base_model: str = ""
    architecture: str = ""
    profile_tags: list[str] = field(default_factory=list)
    allowed_profiles: list[str] = field(
        default_factory=lambda: ["default"]
    )
    risk_level: str = "low"
    weights_hash: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    dataset_size: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "shadow"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    approved_by: str | None = None
    shadow_until: datetime | None = None
    correlation_id: str | None = None
    base_manifest_id: str | None = None

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'; "
                f"expected one of {sorted(MODEL_TYPES)}"
            )
        if self.status not in MODEL_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'; "
                f"expected one of {sorted(MODEL_STATUSES)}"
            )
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(
                f"Invalid risk_level '{self.risk_level}'; "
                f"expected one of {sorted(RISK_LEVELS)}"
            )

    @property
    def is_servable(self) -> bool:
        """Whether this model can serve live traffic."""
        return self.status in ("ready", "shadow")

    @property
    def is_shadow(self) -> bool:
        """Whether the model is in shadow mode."""
        return self.status == "shadow"

    @property
    def version_key(self) -> str:
        """Canonical key for registry lookups: ``model_id:vN``."""
        return f"{self.model_id}:v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "card_id": self.card_id,
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type,
            "owner": self.owner,
            "base_model": self.base_model,
            "architecture": self.architecture,
            "profile_tags": self.profile_tags,
            "allowed_profiles": self.allowed_profiles,
            "risk_level": self.risk_level,
            "weights_hash": self.weights_hash,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "dataset_size": self.dataset_size,
            "metrics": self.metrics,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "approved_by": self.approved_by,
            "shadow_until": (
                self.shadow_until.isoformat()
                if self.shadow_until
                else None
            ),
            "correlation_id": self.correlation_id,
            "base_manifest_id": self.base_manifest_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(timezone.utc)

        shadow_until = data.get("shadow_until")
        if isinstance(shadow_until, str):
            shadow_until = datetime.fromisoformat(shadow_until)

        return cls(
            card_id=data.get("card_id", f"card:{uuid4().hex[:16]}"),
            model_id=data.get("model_id", ""),
            version=data.get("version", 1),
            model_type=data.get("model_type", "lora_adapter"),
            owner=data.get("owner", ""),
            base_model=data.get("base_model", ""),
            architecture=data.get("architecture", ""),
            profile_tags=data.get("profile_tags", []),
            allowed_profiles=data.get("allowed_profiles", ["default"]),
            risk_level=data.get("risk_level", "low"),
            weights_hash=data.get("weights_hash"),
            dataset_id=data.get("dataset_id"),
            dataset_hash=data.get("dataset_hash"),
            dataset_size=data.get("dataset_size", 0),
            metrics=data.get("metrics", {}),
            status=data.get("status", "shadow"),
            created_at=created_at,
            approved_by=data.get("approved_by"),
            shadow_until=shadow_until,
            correlation_id=data.get("correlation_id"),
            base_manifest_id=data.get("base_manifest_id"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


def compute_dataset_hash(data: list[dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 hash of a training dataset.

    Sorts keys in each example for reproducibility.

    Args:
        data: List of training examples.

    Returns:
        Hex digest string.
    """
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "MODEL_STATUSES",
    "MODEL_TYPES",
    "ModelCard",
    "RISK_LEVELS",
    "compute_dataset_hash",
]
