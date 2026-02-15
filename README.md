# nanda-model-card

Unified model card schema for [NANDA](https://projectnanda.org)-compatible agent registries.

A single dataclass that covers LoRA adapters, edge ONNX models, federated models, and heuristic fallbacks, with built-in validation, lifecycle status tracking, and serialization. Zero runtime dependencies.

## Installation

```bash
pip install nanda-model-card
```

## Quick Start

```python
from nanda_model_card import ModelCard

card = ModelCard(
    model_id="sentiment-v3",
    version=3,
    model_type="lora_adapter",
    owner="ml-team",
    base_model="llama-3.1-8b",
    architecture="llama-3.1-8b+lora",
    risk_level="low",
    weights_hash="sha256:abcdef1234567890",
    metrics={"accuracy": 0.94, "training_loss": 0.28},
    status="ready",
)

print(card.version_key)    # "sentiment-v3:v3"
print(card.is_servable)    # True
print(card.to_json())      # Full JSON representation
```

## Model Types

| Type | Description |
|------|-------------|
| `lora_adapter` | LLM LoRA fine-tuned adapters |
| `onnx_edge` | Edge ONNX / TFLite models |
| `federated` | Federated learning models |
| `heuristic` | Rule-based / heuristic fallbacks |

## Lifecycle Status

```
shadow  -->  ready  -->  deprecated  -->  archived
```

| Status | `is_servable` | Description |
|--------|:---:|-------------|
| `shadow` | Yes | Deployed but not yet promoted |
| `ready` | Yes | Approved for live traffic |
| `deprecated` | No | Superseded, no new traffic |
| `archived` | No | Retained for audit only |

## Risk Levels

| Level | Description |
|-------|-------------|
| `low` | Minimal blast radius |
| `medium` | Moderate impact on failure |
| `high` | Requires additional governance approval |

## API

### ModelCard

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `card_id` | `str` | auto | Unique card identifier |
| `model_id` | `str` | `""` | Model identifier |
| `version` | `int` | `1` | Version number |
| `model_type` | `str` | `"lora_adapter"` | One of the four model types |
| `owner` | `str` | `""` | Owner (team, org, individual) |
| `base_model` | `str` | `""` | Foundation model name |
| `architecture` | `str` | `""` | Architecture description |
| `risk_level` | `str` | `"low"` | Risk assessment |
| `weights_hash` | `str \| None` | `None` | SHA-256 of weights |
| `metrics` | `dict` | `{}` | Training/eval metrics |
| `status` | `str` | `"shadow"` | Lifecycle status |
| `approved_by` | `str \| None` | `None` | Approver identifier |

**Properties**: `is_servable`, `is_shadow`, `version_key`

**Methods**: `to_dict()`, `from_dict()`, `to_json()`

### compute_dataset_hash()

Deterministic SHA-256 hash of a training dataset (sorts keys for reproducibility).

```python
from nanda_model_card import compute_dataset_hash

h = compute_dataset_hash([{"input": "hello", "output": "world"}])
```

## Related Packages

| Package | Question it answers |
|---------|-------------------|
| [`nanda-model-provenance`](https://github.com/Sharathvc23/nanda-model-provenance) | "Where did this model come from?" (identity, versioning, provider, NANDA serialization) |
| `nanda-model-card` (this package) | "What is this model?" (unified metadata schema — type, status, risk level, metrics, weights hash) |
| [`nanda-model-integrity-layer`](https://github.com/Sharathvc23/nanda-model-integrity-layer) | "Does this model's metadata meet policy?" (rule-based checks) |
| [`nanda-model-governance`](https://github.com/Sharathvc23/nanda-governance) | "Has this model been cryptographically approved for deployment?" (approval flow with signatures, quorum, scoping, revocation) |
| [`nanda-bridge`](https://github.com/Sharathvc23/nanda-bridge) | "How do I expose this to the NANDA network?" (FastAPI router, AgentFacts models, delta sync) |

## Development

```bash
git clone https://github.com/Sharathvc23/nanda-model-card.git
cd nanda-model-card
pip install -e ".[dev]"
pytest tests/ -v
ruff check nanda_model_card/
mypy nanda_model_card/ --strict
```

## Credits

Developed by [stellarminds.ai](https://stellarminds.ai), open-sourced for [projectnanda.org](https://projectnanda.org).

## License

MIT
