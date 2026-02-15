"""Tests for ModelCard creation, validation, properties, and serialization."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nanda_model_card import (
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
            "card_id", "model_id", "version", "model_type",
            "owner", "base_model", "architecture",
            "profile_tags", "allowed_profiles", "risk_level",
            "weights_hash", "dataset_id", "dataset_hash",
            "dataset_size", "metrics", "status", "created_at",
            "approved_by", "shadow_until", "correlation_id",
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
