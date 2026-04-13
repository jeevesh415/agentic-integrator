# Agentic Integrator

**Agent-S3 + World Model + Advanced Memory + Continuous Vision for Human-Like Screen Navigation**

> AI that operates computers like humans do — seeing continuously, navigating visually by appearance, learning and remembering through hierarchical, holographic, and hyper-dimensional memory systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgenticIntegrator                           │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐ │
│  │ Continuous    │──▶│ Enhanced Worker   │──▶│ World Model    │ │
│  │ Vision       │   │ (Agent-S3+)       │   │ Planner        │ │
│  │ Pipeline     │   │                   │   │                │ │
│  │ • Video      │   │ • Multi-candidate │   │ • LLM-based    │ │
│  │ • Change Det │   │ • Visual anchors  │   │ • Safety Gate  │ │
│  │ • Attention  │   │ • Appearance nav  │   │ • Adaptive     │ │
│  └──────┬───────┘   └────────┬──────────┘   └───────┬────────┘ │
│         │                    │                       │          │
│         ▼                    ▼                       ▼          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Unified Memory Controller                      ││
│  │                                                             ││
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐ ││
│  │  │ Hierarchical │  │ Holographic  │  │ Hyper-Dimensional │ ││
│  │  │ Memory       │  │ Memory (HRR) │  │ Memory (HDC)      │ ││
│  │  │              │  │              │  │                   │ ││
│  │  │ L1: Episodic │  │ Circular     │  │ 10,000D bipolar   │ ││
│  │  │ L2: Semantic │  │ convolution  │  │ vectors           │ ││
│  │  │ L3: Procedurl│  │ Content-addr │  │ One-shot learning │ ││
│  │  └─────────────┘  └──────────────┘  └───────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 🧠 Three Memory Systems (Not Just One)

| Memory | Inspired By | Encodes | Retrieves By |
|--------|-------------|---------|--------------|
| **Hierarchical** | Tulving's taxonomy / ACT-R | Raw experiences → abstract patterns → learned skills | Temporal/causal reasoning |
| **Holographic** | Plate (1995) HRR | Structured role-filler bindings via circular convolution | Partial cue (content-addressable) |
| **Hyper-Dimensional** | Kanerva (2009) HDC | Visual scenes as 10,000D bipolar vectors | One-shot pattern matching |

### 👁️ Continuous Vision (Not Screenshots)

The agent sees a **continuous video stream**, not static screenshots:
- **VisualChangeDetector** — detects meaningful state changes (ignores noise)
- **TemporalAttention** — focuses on regions with recent activity (like human gaze)
- **KeyframeExtraction** — captures significant moments automatically
- Waits for stability after actions (handles animations, loading states)

### 🎯 Navigate by Appearance (Not Coordinates)

Instead of "click at pixel (523, 417)", the agent thinks **"click the blue Submit button"**:
- **ColorMatcher** — find elements by color ("the red Delete button")
- **VisualAnchors** — remember elements by how they look, track across layout changes
- **VisualFeatureExtractor** — shape, size, position, color, edge density
- **AppearanceBasedGrounding** — locate elements by visual description

### 🌍 World Model Planning

Before every action, the agent **imagines what will happen**:
- Generates 3+ candidate actions per step
- Simulates each via LLM world model (WebDreamer-style)
- Safety Gate blocks dangerous/irreversible actions
- Adaptive confidence skips simulation for simple actions

## Module Map

```
agentic_integrator/
├── integrator.py              # Main orchestrator
├── cli.py                     # Command-line interface
│
├── memory/                    # 🧠 Advanced Memory Systems
│   ├── hierarchical_memory.py  # L1 Episodic → L2 Semantic → L3 Procedural
│   ├── holographic_memory.py   # HRR: circular convolution, content-addressable
│   ├── hyperdimensional_memory.py  # HDC: 10,000D bipolar vectors, one-shot learning
│   ├── unified_controller.py   # Orchestrates all three memory systems
│   └── prediction_memory.py    # Tracks world model accuracy
│
├── vision/                    # 👁️ Continuous Vision
│   ├── __init__.py             # ContinuousVisionPipeline, ChangeDetector, Attention
│   └── visual_grounding.py     # ColorMatcher, VisualAnchors, AppearanceGrounding
│
├── world_model/               # 🌍 World Model
│   ├── base.py                 # SimulationResult, PlanningResult
│   ├── llm_world_model.py      # LLM-based simulation
│   ├── planner.py              # Multi-candidate planning
│   └── safety_gate.py          # Dangerous action detection
│
├── enhanced_worker/           # ⚡ Enhanced Agent-S3 Worker
│   ├── worker.py               # EnhancedWorker with planning
│   └── verification.py         # Prediction vs reality comparison
│
└── utils/
    └── prompts.py              # LLM prompts
```

## Quick Start

```python
from agentic_integrator import AgenticIntegrator

agent = AgenticIntegrator(
    provider="openai",
    model="gpt-4o",
    ground_provider="huggingface",
    ground_url="http://localhost:8080",
    ground_model="ui-tars-1.5-7b",
    world_model_candidates=3,
    safety_gate_enabled=True,
    adaptive_confidence=True,
)

# The agent sees a continuous video stream
# and navigates by visual appearance, not coordinates
info, actions = agent.predict("Open Chrome and search for AI papers", observation)
```

### Using the Memory Systems Directly

```python
from agentic_integrator.memory import UnifiedMemoryController
import numpy as np

memory = UnifiedMemoryController()

# Store a full experience (writes to all 3 systems simultaneously)
ids = memory.store_experience(
    visual_embedding=np.random.randn(512),
    action="click_submit_button",
    outcome="form_submitted_successfully",
    task="checkout_flow",
    app="chrome",
    reward=0.9,
    ui_elements=[{"type": "button", "color": "blue", "label": "Submit"}],
    role_fillers={"action": "click", "target": "submit", "page": "checkout"},
)

# Recall by situation (queries all systems in parallel)
memories = memory.recall_by_situation(
    visual_embedding=current_embedding,
    task="checkout_flow",
    query_features={"color": "blue", "type": "button"},
)
```

### Using Visual Grounding

```python
from agentic_integrator.vision.visual_grounding import AppearanceBasedGrounding

grounding = AppearanceBasedGrounding()

# Create a visual anchor for a UI element
anchor = grounding.create_anchor("submit_btn", screenshot, bbox=(500, 400, 120, 40), text_content="Submit")

# Later, find it by appearance (not coordinates!)
element = grounding.find_element(
    {"anchor_name": "submit_btn"},   # by known anchor
    current_screenshot,
)
# OR
element = grounding.find_element(
    {"color": "blue", "shape": "rectangle", "text": "Submit"},  # by description
    current_screenshot,
    candidate_bboxes=detected_bboxes,
)
```

## Installation

```bash
git clone https://github.com/jeevesh415/agentic-integrator.git
cd agentic-integrator
pip install -e .
```

## Requirements

- Python 3.9+
- numpy (core — all memory/vision systems)
- Pillow (vision pipeline)
- openai or anthropic (world model LLM calls)
- gui-agents (Agent-S3 — only needed for full integration, not standalone modules)

## References

- **Agent-S3**: [simular-ai/Agent-S](https://github.com/simular-ai/Agent-S) — Autonomous GUI agent, first to surpass human on OSWorld
- **WebDreamer**: "Is Your LLM Secretly a World Model of the Internet?" (2024) — LLMs as implicit world models
- **Holographic Reduced Representations**: Plate (1995) — Distributed associative memory via circular convolution
- **Hyper-Dimensional Computing**: Kanerva (2009) — Computing with 10,000D vectors
- **ACT-R**: Anderson (2007) — Cognitive architecture with hierarchical memory
