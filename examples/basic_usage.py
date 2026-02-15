"""Basic model card usage.

Usage::

    python examples/basic_usage.py
"""

from __future__ import annotations

from nanda_model_card import ModelCard, compute_dataset_hash


def main() -> None:
    # Create a model card for a LoRA adapter
    card = ModelCard(
        model_id="sentiment-v3",
        version=3,
        model_type="lora_adapter",
        owner="ml-team",
        base_model="llama-3.1-8b",
        architecture="llama-3.1-8b+lora",
        profile_tags=["sentiment", "classification"],
        risk_level="low",
        weights_hash="sha256:abcdef1234567890",
        dataset_id="ds:sentiment-2025",
        dataset_size=50000,
        metrics={"accuracy": 0.94, "training_loss": 0.28},
        status="ready",
        approved_by="governance-lead",
    )

    print(f"Card: {card.card_id}")
    print(f"Model: {card.model_id} v{card.version}")
    print(f"Servable: {card.is_servable}")
    print(f"Version key: {card.version_key}")
    print()

    # Serialize / deserialize
    d = card.to_dict()
    restored = ModelCard.from_dict(d)
    assert restored.model_id == card.model_id
    print(f"Round-trip OK: {restored.version_key}")
    print()

    # JSON output
    print(card.to_json())
    print()

    # Dataset hashing
    dataset = [
        {"instruction": "Is this positive?", "output": "yes"},
        {"instruction": "Is this negative?", "output": "no"},
    ]
    h = compute_dataset_hash(dataset)
    print(f"Dataset hash: {h}")


if __name__ == "__main__":
    main()
