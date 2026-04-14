# A Unified Model Card Schema for Multi-Paradigm ML Systems

**Authors:** StellarMinds ([stellarminds.ai](https://stellarminds.ai))
**Date:** April 2026
**Version:** 0.2.0

## Abstract

`sm-model-card` is a zero-dependency Python dataclass that provides a unified metadata schema for LoRA adapters, ONNX edge models, federated models, and heuristic fallbacks within a single type-safe structure. It solves the schema fragmentation problem that arises when autonomous agent registries must discover, evaluate, and route traffic to heterogeneous model types. The schema enforces a four-state lifecycle with explicit servability semantics, validates invariants at construction time, and supports deterministic dataset fingerprinting. The implementation uses only the Python standard library.

## Problem

Modern ML platforms manage heterogeneous model artifacts spanning fine-tuned LLM adapters, quantized edge models, federated aggregates, and heuristic fallbacks. Each paradigm carries distinct metadata requirements — adapter models reference a base model, edge models specify target hardware, and heuristic systems may lack traditional training metrics entirely. Existing model card formats treat these paradigms as separate concerns, leading to schema fragmentation. When an autonomous agent registry must discover, evaluate, and route traffic to any of these model types, it needs a single queryable schema with consistent lifecycle semantics rather than paradigm-specific card formats.

## What It Does

- Defines a single 20-field dataclass covering 4 model paradigms: `lora_adapter`, `onnx_edge`, `federated`, `heuristic`
- Enforces a 4-state lifecycle machine: `shadow` (canary traffic) -> `ready` (production) -> `deprecated` (no new traffic) -> `archived` (audit only)
- Validates `model_type`, `status`, and `risk_level` against frozenset constraints at construction time via `__post_init__`
- Exposes `is_servable` (True for shadow and ready) and `is_shadow` computed properties for automated traffic routing decisions
- Computes deterministic dataset fingerprints via `compute_dataset_hash()` using canonical JSON hashing (sorted keys, compact separators, UTF-8)
- Generates `version_key` patterns (`"{model_id}:v{version}"`) for O(1) registry lookups
- Supports lossless round-trip serialization with ISO 8601 datetime handling (`to_dict`, `from_dict`, `to_json`)
- Requires zero external dependencies — uses only Python's `dataclasses`, `hashlib`, `json`, and `uuid` modules

## Architecture

The `ModelCard` dataclass organizes 20 fields into five semantic groups:

| Field Group | Fields | Purpose |
|-------------|--------|---------|
| Identity | `card_id`, `model_id`, `version` | Unique identification and versioning |
| Paradigm | `model_type`, `base_model`, `architecture` | Model type discrimination and lineage |
| Classification | `owner`, `profile_tags`, `allowed_profiles`, `risk_level` | Access control and risk assessment |
| Provenance | `weights_hash`, `dataset_id`, `dataset_hash`, `dataset_size`, `metrics` | Training data and integrity linking |
| Lifecycle | `status`, `created_at`, `approved_by`, `shadow_until`, `correlation_id`, `base_manifest_id` | State management and audit tracing |

The four model types cover the common paradigms found in production ML platforms:

| Type | Description | Typical Use |
|------|-------------|-------------|
| `lora_adapter` | Low-Rank Adaptation fine-tuned model | LLM specialization (default) |
| `onnx_edge` | ONNX or TFLite model for edge inference | Mobile and IoT deployment |
| `federated` | Federated learning aggregate | Privacy-preserving training |
| `heuristic` | Rule-based or deterministic fallback | Baseline or safety net |

The lifecycle state machine defines servability semantics:

| Status | Servable | Shadow | Description |
|--------|:--------:|:------:|-------------|
| `shadow` | Yes | Yes | Deployed for canary traffic; not yet promoted |
| `ready` | Yes | No | Approved for live production traffic |
| `deprecated` | No | No | Superseded; no new traffic routed |
| `archived` | No | No | Retained for audit and compliance only |

The `shadow` state is a key design choice: by allowing shadow models to be servable, the schema supports canary deployment patterns where a new model receives a fraction of live traffic before full promotion.

The package exports five symbols: `ModelCard` (core schema), `MODEL_TYPES` (valid model types), `MODEL_STATUSES` (valid lifecycle states), `RISK_LEVELS` (valid risk assessments), and `compute_dataset_hash` (deterministic fingerprinting). The `compute_dataset_hash` function uses `json.dumps` with `sort_keys=True` and compact separators to produce reproducible SHA-256 digests regardless of dictionary key ordering or JSON formatter whitespace conventions.

## Key Design Decisions

- **Frozenset over Enum for validation:** `MODEL_TYPES`, `MODEL_STATUSES`, and `RISK_LEVELS` are `frozenset[str]` rather than Python `Enum` types. String values round-trip through JSON without custom encoders or special serialization logic while still providing O(1) membership validation. This is a deliberate trade-off: Enum provides IDE autocompletion, but frozenset avoids the serialization complexity that Enum introduces in network protocols where values must be plain strings.

- **`__post_init__` for fail-fast validation:** Invalid `model_type`, `status`, or `risk_level` values raise `ValueError` at construction time with descriptive error messages listing the complete set of allowed values. No `ModelCard` instance can be created with invalid enum-like values, preventing malformed records from ever entering a registry. This is stronger than downstream validation because the invariant is enforced at the data boundary.

- **Single dataclass (no paradigm fragmentation):** One schema type covers all four model paradigms rather than requiring paradigm-specific card formats. A LoRA adapter and a heuristic fallback share the same queryable structure, enabling uniform registry queries like "find all servable models with low risk" without paradigm-specific handling or type dispatch logic.

- **Transition guards via lifecycle properties:** `is_servable` returns `True` only for `shadow` and `ready` states, giving load balancers a single boolean to check rather than implementing status-awareness. `is_shadow` enables traffic splitters to apply shadow-specific sampling ratios. These properties encode routing logic in the schema itself rather than requiring consumers to understand the full status taxonomy.

- **Auto-generated card_id with namespace prefix:** The `card_id` defaults to `f"card:{uuid4().hex[:16]}"`, providing a practically unique 16-character hex identifier with a `card:` prefix that enables namespace-aware indexing in multi-entity registries without requiring callers to generate their own IDs.

## Ecosystem Integration

The `sm-model-card` package occupies the metadata layer in the NANDA ecosystem, providing the foundational schema that companion packages build upon for integrity verification and cryptographic governance.

| Package | Role | Question Answered |
|---------|------|-------------------|
| `sm-model-provenance` | Identity metadata | Where did this model come from? |
| **`sm-model-card`** | **Metadata schema** | **What is this model?** |
| `sm-model-integrity-layer` | Integrity verification | Does metadata meet policy? |
| `sm-model-governance` | Cryptographic governance | Has this model been approved? |
| `sm-bridge` | Transport layer | How is it exposed to the network? |

The integrity layer consumes `weights_hash`, `model_type`, `risk_level`, and `base_model` fields for hash verification, governance policy enforcement, and lineage chain reconstruction. The integrity layer's `ModelLineage.from_provenance()` method uses `model_id`, `base_model`, and `model_type` fields to automatically construct derivation chains. The governance layer accepts the full model card as an audit artifact during training completion and uses `weights_hash` as the integrity anchor linking governance decisions to specific trained weights.

In NANDA-compatible agent registries, a model card is typically embedded in agent metadata under the `x_model_provenance` vendor extension key. The `model_type`, `status`, and `risk_level` fields enable automated discovery queries such as "find all servable LoRA adapters with low risk" or "list deprecated models for cleanup." The `version_key` property provides the canonical lookup key for registry implementations, combining `model_id` with a monotonically increasing version number in a human-readable format.

Several model card fields map directly to provenance fields in the companion `sm-model-provenance` package: `model_id`, `model_type`, `base_model`, `weights_hash`, and `risk_level`. This field alignment is by design — provenance carries the identity subset of model card metadata, allowing systems that only need "who is this model" to avoid the full model card dependency while maintaining consistent field semantics across the ecosystem.

The `to_dict()` method serializes all 20 fields, converting `datetime` objects to ISO 8601 strings and preserving `None` values. The `from_dict()` class method handles ISO 8601 datetime parsing and supplies defaults for missing fields, enabling backward-compatible deserialization of older card formats. The `to_json()` method produces pretty-printed JSON with 2-space indentation via `json.dumps(self.to_dict(), indent=2, default=str)`. The round-trip property `ModelCard.from_dict(card.to_dict()) == card` holds for all valid instances, ensuring lossless persistence across serialization boundaries.

The `compute_dataset_hash()` function implements reproducible fingerprinting of training datasets. Three design choices ensure determinism: `sort_keys=True` normalizes dictionary key order so `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` produce identical hashes, compact separators `(",", ":")` eliminate whitespace variation from different JSON formatters, and UTF-8 encoding provides a canonical byte representation for the SHA-256 input. The resulting 64-character hex digest can be stored in the `dataset_hash` field, creating a verifiable link between a model card and its training data.

The `metrics` field accepts arbitrary key-value pairs (e.g., `{"accuracy": 0.94, "training_loss": 0.28}`), allowing paradigm-specific metrics without schema changes. The `correlation_id` field enables audit tracing across the governance pipeline, linking the model card to its approval record and deployment event.

## Future Work

- Schema evolution for additional model types (mixture-of-experts, multimodal) via backward-compatible optional fields
- Formalized state transition guards preventing invalid moves (e.g., `archived` to `ready`) with transition guard methods
- Hardware affinity metadata for edge deployment orchestrators to match models to available hardware
- Federated aggregation metadata capturing participant counts, aggregation rounds, and differential privacy parameters

## References

1. NANDA Protocol. "Network of AI Agents in Decentralized Architecture." https://projectnanda.org
2. HuggingFace. "Model Cards." https://huggingface.co/docs/hub/model-cards
