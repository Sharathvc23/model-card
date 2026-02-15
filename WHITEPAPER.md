# A Unified Model Card Schema for Multi-Paradigm ML Systems

**Authors:** StellarMinds ([stellarminds.ai](https://stellarminds.ai))
**Date:** February 2026
**Version:** 1.0

## Abstract

Machine learning operations increasingly span diverse model paradigms — from large language model adapters to edge-deployed inference engines, federated learning aggregates, and rule-based heuristic systems. Existing model card standards, while valuable for documentation, were not originally designed with the structural rigor needed for automated governance in decentralized agent registries. This paper presents `nanda-model-card`, a zero-dependency Python dataclass that provides a unified metadata schema covering LoRA adapters, ONNX edge models, federated models, and heuristic fallbacks within a single type-safe structure. The schema enforces lifecycle semantics through a four-state machine (shadow → ready → deprecated → archived), validates invariants at construction time via `__post_init__`, supports deterministic dataset fingerprinting through canonical JSON hashing, and provides a `version_key` pattern for registry lookups. The implementation requires no external dependencies, passes strict static analysis, and integrates with the broader NANDA ecosystem for integrity verification and cryptographic governance.

## 1. Introduction

### 1.1 Problem Statement

Modern ML platforms manage heterogeneous model artifacts. A single organization may simultaneously operate fine-tuned LLM adapters (LoRA), quantized models for edge deployment (ONNX/TFLite), federated learning aggregates, and deterministic heuristic fallbacks. Each paradigm carries distinct metadata requirements — adapter models reference a base model, edge models specify target hardware, and heuristic systems may lack traditional training metrics entirely.

Existing model card formats tend to treat these paradigms as separate concerns, which can lead to schema fragmentation. When an autonomous agent registry must discover, evaluate, and route traffic to any of these model types, it needs a single queryable schema with consistent lifecycle semantics.

### 1.2 Motivation

The NANDA (Network of AI Agents in Decentralized Architecture) ecosystem requires machine-readable model metadata that can be:

1. **Validated at construction time** — preventing malformed records from entering the registry.
2. **Queried uniformly** — regardless of whether the underlying model is a LoRA adapter or a heuristic fallback.
3. **Lifecycle-tracked** — with clear servability semantics at each state.
4. **Integrity-linked** — connecting metadata to verifiable weight hashes and dataset provenance.

### 1.3 Contributions

This paper makes the following contributions:

- A **single dataclass schema** with 20 typed fields covering four model paradigms, reducing the need for paradigm-specific card formats.
- A **four-state lifecycle machine** with explicit servability semantics, enabling automated traffic routing decisions.
- A **deterministic dataset hashing** algorithm using canonical JSON serialization for reproducible provenance fingerprinting.
- A **version_key registry pattern** providing O(1) lookup for versioned model identifiers.
- A **zero-dependency implementation** using only the Python standard library, suitable for constrained deployment environments.

## 2. Related Work

### 2.1 Google Model Cards for Model Reporting

Mitchell et al. (2019) introduced Model Cards as structured documentation for trained ML models, covering intended use, evaluation metrics, and ethical considerations. While foundational for transparency, Google Model Cards are primarily documentation artifacts — they do not currently provide machine-enforced validation, lifecycle state management, or the multi-paradigm type discrimination that automated agent routing may require.

### 2.2 HuggingFace Model Cards

The HuggingFace Hub extends the model card concept with YAML frontmatter for metadata (tags, datasets, metrics) and free-form Markdown for documentation. This approach excels for human-readable discoverability but does not currently enforce structural invariants at the schema level. A HuggingFace card for a LoRA adapter and one for an ONNX model may not share a common lifecycle schema, and construction-time validation of metadata consistency is not built in.

### 2.3 MLflow Model Registry

MLflow's model registry provides versioning and stage transitions (Staging → Production → Archived). However, MLflow's stages are relatively coarse-grained and coupled to its tracking server. The registry does not currently distinguish model paradigms at the schema level, nor does it provide built-in deterministic dataset fingerprinting.

### 2.4 Gaps Addressed

This work addresses three gaps in existing approaches:

1. **Multi-paradigm unification** — a single schema type for LoRA, ONNX edge, federated, and heuristic models.
2. **Construction-time validation** — invariant enforcement at object creation, not as a downstream lint step.
3. **Deterministic provenance hashing** — reproducible dataset fingerprints for integrity verification across distributed systems.

## 3. Design / Architecture

### 3.1 Core Data Model

The `ModelCard` is implemented as a Python `dataclass` with 20 fields organized into five semantic groups:

**Identity fields:** `card_id` (auto-generated 16-character hex UUID with `card:` prefix), `model_id`, `version` (monotonically increasing integer).

**Paradigm fields:** `model_type` (validated against `MODEL_TYPES` frozenset), `base_model`, `architecture` (human-readable descriptor such as `"llama-3.1-8b+lora"`).

**Classification fields:** `owner`, `profile_tags`, `allowed_profiles`, `risk_level` (validated against `RISK_LEVELS`: low, medium, high).

**Provenance fields:** `weights_hash` (SHA-256 hex digest), `dataset_id`, `dataset_hash`, `dataset_size`, `metrics` (arbitrary key-value pairs).

**Lifecycle fields:** `status` (validated against `MODEL_STATUSES`), `created_at` (UTC datetime), `approved_by`, `shadow_until`, `correlation_id`, `base_manifest_id`.

### 3.2 Model Type Taxonomy

The schema recognizes four model types, encoded as the `MODEL_TYPES` frozenset:

| Type | Description | Typical Use |
|------|-------------|-------------|
| `lora_adapter` | Low-Rank Adaptation fine-tuned model | LLM specialization |
| `onnx_edge` | ONNX or TFLite model for edge inference | Mobile / IoT deployment |
| `federated` | Federated learning aggregate | Privacy-preserving training |
| `heuristic` | Rule-based / deterministic fallback | Baseline or safety net |

The `lora_adapter` type serves as the default, reflecting the prevalence of adapter-based fine-tuning in contemporary LLM workflows.

### 3.3 Lifecycle State Machine

The `status` field implements a four-state lifecycle:

```
shadow ──→ ready ──→ deprecated ──→ archived
```

| Status | Servable | Shadow | Description |
|--------|:--------:|:------:|-------------|
| `shadow` | Yes | Yes | Deployed for shadow traffic; not yet promoted |
| `ready` | Yes | No | Approved for live production traffic |
| `deprecated` | No | No | Superseded; no new traffic routed |
| `archived` | No | No | Retained for audit and compliance only |

Two computed properties expose servability semantics:

- `is_servable → bool`: Returns `True` for `shadow` and `ready` states, enabling automated load balancers to include the model in routing pools.
- `is_shadow → bool`: Returns `True` only for `shadow`, allowing traffic splitters to apply shadow-specific sampling ratios.

The shadow state is a key design choice. By allowing shadow models to be servable, the schema supports canary deployment patterns where a new model receives a fraction of live traffic before full promotion.

### 3.4 Key Design Decisions

**Frozenset-based validation over Enum.** The `MODEL_TYPES`, `MODEL_STATUSES`, and `RISK_LEVELS` constraints are implemented as `frozenset[str]` rather than Python `Enum` types. This decision prioritizes serialization simplicity — string values round-trip through JSON without custom encoders — while still providing O(1) membership validation.

**Post-init validation.** The `__post_init__` method validates `model_type`, `status`, and `risk_level` against their respective frozensets at construction time. This fail-fast approach helps ensure that no `ModelCard` instance can be created with invalid enum-like values, reducing a common class of downstream errors.

**Auto-generated card_id.** The `card_id` defaults to `f"card:{uuid4().hex[:16]}"`, providing a practically unique identifier without requiring the caller to generate one. The `card:` prefix enables namespace-aware indexing in multi-entity registries.

## 4. Implementation

### 4.1 The ModelCard Dataclass

The core implementation resides in a single module (`model_card.py`). The `ModelCard` class uses Python's `@dataclass` decorator with carefully chosen defaults:

```python
@dataclass
class ModelCard:
    card_id: str = field(default_factory=lambda: f"card:{uuid4().hex[:16]}")
    model_id: str = ""
    version: int = 1
    model_type: str = "lora_adapter"
    # ... 16 additional fields
```

The `__post_init__` method performs three validation checks:

```python
def __post_init__(self) -> None:
    if self.model_type not in MODEL_TYPES:
        raise ValueError(...)
    if self.status not in MODEL_STATUSES:
        raise ValueError(...)
    if self.risk_level not in RISK_LEVELS:
        raise ValueError(...)
```

Each validation produces a descriptive error message listing the allowed values, aiding debuggability in automated pipelines.

### 4.2 Deterministic Dataset Hashing

The `compute_dataset_hash` function implements reproducible fingerprinting of training datasets:

```python
def compute_dataset_hash(data: list[dict[str, Any]]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Three design choices ensure determinism:

1. **`sort_keys=True`** — Dictionary key order is normalized, so `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` produce identical hashes.
2. **Compact separators `(",", ":")`** — Eliminates whitespace variation that could arise from different JSON formatters.
3. **UTF-8 encoding** — Provides a canonical byte representation for the SHA-256 input.

The resulting 64-character hex digest can be stored in the `dataset_hash` field, creating a verifiable link between a model card and its training data.

### 4.3 Version Key Registry Pattern

The `version_key` property generates canonical registry lookup keys:

```python
@property
def version_key(self) -> str:
    return f"{self.model_id}:v{self.version}"
```

This produces keys like `"sentiment-v3:v3"`, combining the model identifier with a monotonically increasing version number. The format supports O(1) dictionary lookups in registry implementations while remaining human-readable in audit logs.

### 4.4 Serialization

The `ModelCard` provides three serialization methods:

- **`to_dict()`** — Converts all 20 fields to a dictionary, serializing `datetime` objects to ISO 8601 strings and preserving `None` values.
- **`from_dict(data)`** — Class method that deserializes a dictionary back to a `ModelCard`, handling ISO 8601 datetime parsing and supplying defaults for missing fields.
- **`to_json()`** — Produces a pretty-printed JSON string with 2-space indentation via `json.dumps(self.to_dict(), indent=2, default=str)`.

The round-trip property `ModelCard.from_dict(card.to_dict()) == card` holds for all valid instances, ensuring lossless persistence.

### 4.5 Public API

The package exports five symbols from its `__init__.py`:

| Export | Type | Purpose |
|--------|------|---------|
| `ModelCard` | class | Core metadata schema |
| `MODEL_TYPES` | frozenset | Valid model type values |
| `MODEL_STATUSES` | frozenset | Valid lifecycle states |
| `RISK_LEVELS` | frozenset | Valid risk assessments |
| `compute_dataset_hash` | function | Deterministic dataset fingerprinting |

## 5. Integration

### 5.1 NANDA Ecosystem Context

The `nanda-model-card` package occupies the **metadata layer** in the NANDA ecosystem, answering the question: *"What is this model?"* It provides the foundational schema that two companion packages build upon:

| Package | Role | Question Answered |
|---------|------|-------------------|
| `nanda-model-card` | Metadata schema | What is this model? |
| `nanda-model-integrity-layer` | Integrity verification | Does this model's metadata meet policy? |
| `nanda-model-governance` | Cryptographic governance | Has this model been approved? |

### 5.2 Integration with the Integrity Layer

The `nanda-model-integrity-layer` package consumes model card fields through its `ModelProvenance` dataclass and governance policy engine. Key integration points include:

- **`weights_hash`** — The model card's weight hash is verified by the integrity layer's `verify_provenance_integrity()` function using pluggable hash providers (SHA-256, SHA-384, SHA-512, BLAKE2b).
- **`model_type`** — The integrity layer's `RequireBaseModel` governance policy checks that adapter-type models (as declared in the card's `model_type` field) have a `base_model` specified.
- **`dataset_hash`** — The deterministic dataset hash produced by `compute_dataset_hash()` can be included in provenance records for cross-verification.
- **Lineage reconstruction** — The integrity layer's `ModelLineage.from_provenance()` method uses `model_id`, `base_model`, and `model_type` fields to automatically construct derivation chains.

### 5.3 Integration with the Governance Layer

The `nanda-model-governance` package uses model card metadata at two critical points:

- **Training completion** — The `GovernanceCoordinator.complete_training()` method accepts a `card` dictionary parameter, allowing the full model card to travel through the governance pipeline as an audit artifact.
- **Promotion gates** — The governance layer's 5-gate promotion check indirectly depends on model card integrity, since the `weights_hash` field establishes the link between what was trained and what is being deployed.

### 5.4 Agent Registry Discovery

In NANDA-compatible agent registries, a model card is typically embedded in the agent's metadata under the `x_model_provenance` vendor extension key. The `model_type`, `status`, and `risk_level` fields enable automated agent discovery queries such as:

- *"Find all servable LoRA adapters with low risk"* — filters on `is_servable`, `model_type == "lora_adapter"`, `risk_level == "low"`.
- *"List deprecated models for cleanup"* — filters on `status == "deprecated"`.

## 6. Evaluation

### 6.1 Test Coverage

The test suite contains **28 test methods** organized across 5 test classes:

| Test Class | Tests | Coverage Area |
|------------|:-----:|---------------|
| `TestModelCardCreation` | 2 | Default values, full construction |
| `TestModelCardValidation` | 6 | Invalid inputs, exhaustive valid value iteration |
| `TestModelCardProperties` | 4 | Servability, shadow detection, version key format |
| `TestModelCardSerialization` | 6 | Dict/JSON round-trip, datetime handling, field completeness |
| `TestComputeDatasetHash` | 4 | Determinism, collision resistance, key order invariance, empty dataset |

All tests pass under `pytest` with strict mode. The codebase also passes `mypy --strict` static analysis and `ruff` linting with rule sets E, F, W, I, UP, B, SIM, and RUF.

### 6.2 Example: Creating and Querying a Model Card

```python
from nanda_model_card import ModelCard, compute_dataset_hash

card = ModelCard(
    model_id="sentiment-v3",
    version=3,
    model_type="lora_adapter",
    owner="ml-team",
    base_model="llama-3.1-8b",
    architecture="llama-3.1-8b+lora",
    risk_level="low",
    metrics={"accuracy": 0.94, "training_loss": 0.28},
    status="ready",
    approved_by="governance-lead",
)

assert card.is_servable is True
assert card.version_key == "sentiment-v3:v3"

# Deterministic dataset fingerprinting
dataset = [
    {"instruction": "Is this positive?", "output": "yes"},
    {"instruction": "Is this negative?", "output": "no"},
]
card_with_hash = ModelCard(
    model_id="sentiment-v3",
    dataset_hash=compute_dataset_hash(dataset),
    dataset_size=len(dataset),
)
assert len(card_with_hash.dataset_hash) == 64  # SHA-256 hex digest
```

### 6.3 Validation Behavior

Construction-time validation prevents invalid instances:

```python
try:
    ModelCard(model_type="transformer")  # Not in MODEL_TYPES
except ValueError as e:
    # "Invalid model_type 'transformer'; expected one of
    #  ['federated', 'heuristic', 'lora_adapter', 'onnx_edge']"
    pass
```

## 7. Conclusion

### 7.1 Summary

This paper presented `nanda-model-card`, a unified model card schema that bridges four ML paradigms under a single validated dataclass. The design encourages correctness through construction-time validation, provides clear lifecycle semantics through a four-state machine with explicit servability properties, and supports reproducible provenance through deterministic dataset hashing. The zero-dependency implementation aims to ensure deployment compatibility across constrained environments while maintaining strict type safety.

### 7.2 Future Work

Several directions merit further investigation:

- **Schema evolution** — Adding support for additional model types (e.g., mixture-of-experts, multimodal) while maintaining backward compatibility through optional fields.
- **Extended lifecycle transitions** — Formalizing allowed state transitions (e.g., preventing `archived → ready`) with transition guard methods.
- **Hardware affinity metadata** — Adding fields for target device specifications, enabling edge deployment orchestrators to match models to available hardware.
- **Federated aggregation metadata** — Capturing participant counts, aggregation rounds, and differential privacy parameters for federated model types.

## References

1. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., and Gebru, T. (2019). "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\*)*, pp. 220–229.

2. HuggingFace. "Model Cards." HuggingFace Hub Documentation. https://huggingface.co/docs/hub/model-cards

3. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S.A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., and Zuber, C. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Engineering Bulletin*, 41(4), pp. 39–45.

4. NANDA Protocol. "Network of AI Agents in Decentralized Architecture." https://projectnanda.org

5. Python Software Foundation. "dataclasses — Data Classes." Python 3.10+ Documentation. https://docs.python.org/3/library/dataclasses.html
