"""Tests for ModelCard creation, validation, properties, and serialization.

# Step 1 — Assumption Audit
# - model_id can be any string (including empty); version >= 1; dataset_size >= 0
# - status/model_type/risk_level must come from the frozen sets
# - transition_to enforces a forward-only DAG; archived is terminal
# - to_dict/from_dict round-trips all 21 fields; from_dict ignores extra keys
# - MODEL_STATUSES / MODEL_TYPES / RISK_LEVELS are frozensets (immutable)
# - metrics dict values are arbitrary floats (including NaN/Inf?)
#
# Step 2 — Gap Analysis
# - No boundary tests for version=1 (minimum valid) or dataset_size=0
# - No stress test for very long model_id strings
# - No test for NaN / Inf in metrics and their JSON serialization
# - No test that from_dict silently ignores unknown keys
# - No test that transition_to same status raises
# - No immutability test for the frozenset constants
#
# Step 3 — Break It List
# - float('nan') in metrics — JSON serializes but is not valid JSON per spec
# - float('inf') in metrics — json.dumps raises ValueError by default
# - 10 000-char model_id — memory / version_key explosion?
# - shadow → shadow self-transition — should be rejected by the DAG
# - Mutating MODEL_STATUSES at runtime — must be impossible (frozenset)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from sm_model_card import (
    MODEL_STATUSES,
    MODEL_TYPES,
    RISK_LEVELS,
    ModelCard,
    compute_dataset_hash,
)

# -- Construction & defaults -------------------------------------------------


class TestModelCardCreation:
    def test_default_fields(self) -> None:
        card = ModelCard(model_id="test-model")
        assert card.model_id == "test-model"
        assert card.version == 1
        assert card.model_type == "lora_adapter"
        assert card.status == "shadow"
        assert card.risk_level == "low"
        assert card.allowed_profiles == ["default"]
        assert card.card_id.startswith("card:")

    def test_full_construction(self) -> None:
        now = datetime.now(timezone.utc)
        card = ModelCard(
            model_id="classifier-v3",
            version=3,
            model_type="onnx_edge",
            owner="ml-team",
            base_model="onnx-linear",
            architecture="onnx-linear-v2",
            profile_tags=["default", "defense"],
            allowed_profiles=["default"],
            risk_level="medium",
            weights_hash="abc123",
            dataset_id="ds:xyz",
            dataset_hash="def456",
            dataset_size=1000,
            metrics={"accuracy": 0.95, "training_loss": 0.02},
            status="ready",
            created_at=now,
            approved_by="operator-1",
            shadow_until=now,
        )
        assert card.model_id == "classifier-v3"
        assert card.version == 3
        assert card.owner == "ml-team"
        assert card.is_servable is True
        assert card.is_shadow is False
        assert card.version_key == "classifier-v3:v3"


# -- Validation --------------------------------------------------------------


class TestModelCardValidation:
    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelCard(model_type="invalid_type")

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            ModelCard(status="unknown")

    def test_invalid_risk_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid risk_level"):
            ModelCard(risk_level="critical")

    def test_all_valid_model_types(self) -> None:
        for mt in MODEL_TYPES:
            card = ModelCard(model_type=mt)
            assert card.model_type == mt

    def test_all_valid_statuses(self) -> None:
        for s in MODEL_STATUSES:
            card = ModelCard(status=s)
            assert card.status == s

    def test_all_valid_risk_levels(self) -> None:
        for r in RISK_LEVELS:
            card = ModelCard(risk_level=r)
            assert card.risk_level == r


# -- Properties ---------------------------------------------------------------


class TestModelCardProperties:
    def test_shadow_is_servable(self) -> None:
        card = ModelCard(status="shadow")
        assert card.is_servable is True
        assert card.is_shadow is True

    def test_ready_is_servable(self) -> None:
        card = ModelCard(status="ready")
        assert card.is_servable is True
        assert card.is_shadow is False

    def test_deprecated_not_servable(self) -> None:
        card = ModelCard(status="deprecated")
        assert card.is_servable is False

    def test_archived_not_servable(self) -> None:
        card = ModelCard(status="archived")
        assert card.is_servable is False

    def test_version_key_format(self) -> None:
        card = ModelCard(model_id="abc", version=5)
        assert card.version_key == "abc:v5"


# -- Serialization ------------------------------------------------------------


class TestModelCardSerialization:
    def test_round_trip(self) -> None:
        card = ModelCard(
            model_id="test-rt",
            version=2,
            model_type="federated",
            status="ready",
            risk_level="medium",
            metrics={"loss": 0.5},
            owner="team-a",
        )
        d = card.to_dict()
        restored = ModelCard.from_dict(d)
        assert restored.model_id == card.model_id
        assert restored.version == card.version
        assert restored.model_type == card.model_type
        assert restored.status == card.status
        assert restored.risk_level == card.risk_level
        assert restored.metrics == card.metrics
        assert restored.owner == card.owner

    def test_to_json(self) -> None:
        card = ModelCard(model_id="json-test")
        json_str = card.to_json()
        assert '"model_id": "json-test"' in json_str

    def test_from_dict_defaults(self) -> None:
        card = ModelCard.from_dict({})
        assert card.model_id == ""
        assert card.status == "shadow"
        assert card.owner == ""

    def test_shadow_until_round_trip(self) -> None:
        now = datetime.now(timezone.utc)
        card = ModelCard(
            model_id="shadow-test",
            shadow_until=now,
        )
        d = card.to_dict()
        restored = ModelCard.from_dict(d)
        assert restored.shadow_until is not None
        assert restored.shadow_until.isoformat() == now.isoformat()

    def test_to_dict_contains_all_fields(self) -> None:
        card = ModelCard(model_id="fields-test")
        d = card.to_dict()
        expected_keys = {
            "card_id",
            "model_id",
            "version",
            "model_type",
            "owner",
            "base_model",
            "architecture",
            "profile_tags",
            "allowed_profiles",
            "risk_level",
            "weights_hash",
            "dataset_id",
            "dataset_hash",
            "dataset_size",
            "metrics",
            "status",
            "created_at",
            "approved_by",
            "shadow_until",
            "correlation_id",
            "base_manifest_id",
        }
        assert set(d.keys()) == expected_keys


# -- Dataset hash -------------------------------------------------------------


class TestComputeDatasetHash:
    def test_deterministic(self) -> None:
        data = [{"instruction": "A", "output": "B"}]
        h1 = compute_dataset_hash(data)
        h2 = compute_dataset_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_data_different_hash(self) -> None:
        d1 = [{"instruction": "A"}]
        d2 = [{"instruction": "B"}]
        assert compute_dataset_hash(d1) != compute_dataset_hash(d2)

    def test_key_order_irrelevant(self) -> None:
        d1 = [{"a": 1, "b": 2}]
        d2 = [{"b": 2, "a": 1}]
        assert compute_dataset_hash(d1) == compute_dataset_hash(d2)

    def test_empty_dataset(self) -> None:
        h = compute_dataset_hash([])
        assert len(h) == 64


# -- State transitions --------------------------------------------------------


class TestModelCardTransitions:
    def test_shadow_to_ready(self) -> None:
        card = ModelCard(status="shadow")
        card.transition_to("ready")
        assert card.status == "ready"

    def test_ready_to_deprecated(self) -> None:
        card = ModelCard(status="ready")
        card.transition_to("deprecated")
        assert card.status == "deprecated"

    def test_deprecated_to_archived(self) -> None:
        card = ModelCard(status="deprecated")
        card.transition_to("archived")
        assert card.status == "archived"

    def test_archived_is_terminal(self) -> None:
        card = ModelCard(status="archived")
        with pytest.raises(ValueError, match="Cannot transition"):
            card.transition_to("ready")

    def test_backwards_transition_blocked(self) -> None:
        card = ModelCard(status="ready")
        with pytest.raises(ValueError, match="Cannot transition"):
            card.transition_to("shadow")

    def test_skip_transition_allowed(self) -> None:
        card = ModelCard(status="shadow")
        card.transition_to("archived")
        assert card.status == "archived"

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError, match="version must be >= 1"):
            ModelCard(version=0)
        with pytest.raises(ValueError, match="version must be >= 1"):
            ModelCard(version=-1)

    def test_negative_dataset_size_raises(self) -> None:
        with pytest.raises(ValueError, match="dataset_size must be >= 0"):
            ModelCard(dataset_size=-1)


# -- Additional edge-case tests -----------------------------------------------


class TestEdgeCases:
    """Edge-case tests for ModelCard."""

    def test_version_key_with_model_id(self) -> None:
        """version_key format is '{model_id}:v{version}'."""
        card = ModelCard(model_id="my-model", version=7)
        assert card.version_key == "my-model:v7"

    def test_from_dict_with_missing_fields_uses_defaults(self) -> None:
        """from_dict with minimal dict fills in sensible defaults."""
        card = ModelCard.from_dict({"model_id": "minimal"})
        assert card.model_id == "minimal"
        assert card.version == 1
        assert card.model_type == "lora_adapter"
        assert card.status == "shadow"
        assert card.risk_level == "low"
        assert card.owner == ""
        assert card.metrics == {}
        assert card.allowed_profiles == ["default"]

    def test_metrics_accepts_float_values(self) -> None:
        """metrics dict with float values works correctly."""
        card = ModelCard(
            model_id="float-metrics",
            metrics={"accuracy": 0.9523, "loss": 0.0012, "f1": 1.0},
        )
        assert card.metrics["accuracy"] == 0.9523
        assert card.metrics["loss"] == 0.0012
        assert card.metrics["f1"] == 1.0
        # Round-trip via dict
        d = card.to_dict()
        restored = ModelCard.from_dict(d)
        assert restored.metrics == card.metrics

    def test_all_model_types_are_servable_in_shadow(self) -> None:
        """Every model_type is servable when status is shadow."""
        for mt in MODEL_TYPES:
            card = ModelCard(model_type=mt, status="shadow")
            assert card.is_servable is True, f"{mt} should be servable in shadow"
            assert card.is_shadow is True


# -- R1-R10 adversarial / boundary tests -------------------------------------


LONG_MODEL_ID = "m" * 10_000


class TestAdversarialBoundary:
    """Boundary, failure, and sad-path tests (R1-R10 protocol)."""

    # -- boundary (R5) --------------------------------------------------------

    def test_version_exactly_one_is_valid(self) -> None:
        """version=1 is the minimum valid value (boundary)."""
        card = ModelCard(model_id="boundary", version=1)
        assert card.version == 1

    def test_dataset_size_exactly_zero_is_valid(self) -> None:
        """dataset_size=0 is the minimum valid value (boundary)."""
        card = ModelCard(model_id="boundary", dataset_size=0)
        assert card.dataset_size == 0

    # -- stress ---------------------------------------------------------------

    def test_extremely_long_model_id(self) -> None:
        """10 000-char model_id should be accepted and round-trip."""
        card = ModelCard(model_id=LONG_MODEL_ID)
        assert card.model_id == LONG_MODEL_ID
        assert card.version_key == f"{LONG_MODEL_ID}:v1"
        restored = ModelCard.from_dict(card.to_dict())
        assert restored.model_id == LONG_MODEL_ID

    # -- failure / sad path ---------------------------------------------------

    def test_metrics_with_nan_value(self) -> None:
        """float('nan') in metrics — to_json uses default=str so it won't crash."""
        card = ModelCard(model_id="nan-test", metrics={"loss": float("nan")})
        assert math.isnan(card.metrics["loss"])
        # to_json must not raise (default=str handles non-finite floats)
        json_str = card.to_json()
        assert "NaN" in json_str or "nan" in json_str.lower()

    def test_metrics_with_inf_value(self) -> None:
        """float('inf') in metrics — to_json uses default=str so it won't crash."""
        card = ModelCard(model_id="inf-test", metrics={"score": float("inf")})
        assert math.isinf(card.metrics["score"])
        json_str = card.to_json()
        assert "Infinity" in json_str or "inf" in json_str.lower()

    def test_from_dict_with_extra_keys_ignored(self) -> None:
        """Unknown keys in from_dict are silently ignored."""
        card = ModelCard.from_dict(
            {"model_id": "extra", "totally_unknown": 42, "another": "nope"}
        )
        assert card.model_id == "extra"
        assert not hasattr(card, "totally_unknown")

    def test_from_dict_with_none_status_uses_default(self) -> None:
        """Missing status key in from_dict defaults to 'shadow'."""
        card = ModelCard.from_dict({"model_id": "no-status"})
        assert card.status == "shadow"

    def test_transition_to_same_status_raises(self) -> None:
        """Transitioning shadow → shadow is a self-loop and must raise."""
        card = ModelCard(status="shadow")
        with pytest.raises(ValueError, match="Cannot transition"):
            card.transition_to("shadow")

    def test_frozen_model_statuses_immutable(self) -> None:
        """MODEL_STATUSES is a frozenset; .add() must raise AttributeError."""
        with pytest.raises(AttributeError):
            MODEL_STATUSES.add("rogue")  # type: ignore[attr-defined]
