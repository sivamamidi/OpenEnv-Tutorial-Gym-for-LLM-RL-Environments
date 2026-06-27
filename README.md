# 🌐 OpenEnv Tutorial — Gym for LLM RL Environments

> Build, share, and run reinforcement learning environments as easily as pushing models to Hugging Face.

---

## What is OpenEnv?

**OpenEnv** is a unified, openly governed framework for creating and distributing RL environments. It's backed by a technical committee that includes Meta-PyTorch, Hugging Face, NVIDIA, Unsloth, Modal, Prime Intellect, Reflection, Mercor, and Fleet AI.

**The goal:** Make environment creation as easy and standardized as model sharing on Hugging Face.

---

## Key Features

| Feature | Description |
|---|---|
| **Standardized API** | Gymnasium-style `reset()`, `step()`, `state()` |
| **Type-Safe** | Full IDE autocomplete and error checking via Pydantic |
| **Containerized** | Environments run in Docker for isolation and reproducibility |
| **Shareable** | Push to Hugging Face Hub with one command |
| **Language-Agnostic** | HTTP/WebSocket API works from any language |

---

## RL in 60 Seconds

Reinforcement Learning is just a loop:

```
┌─────────────────────────────────────────────────────────┐
│                     THE RL LOOP                         │
│                                                         │
│   ┌─────────┐          ┌─────────────┐                  │
│   │  AGENT  │──action─▶│ ENVIRONMENT │                  │
│   │         │◀─reward──│             │                  │
│   │         │◀──obs────│             │                  │
│   └─────────┘          └─────────────┘                  │
│                                                         │
│   1. Agent observes the environment                     │
│   2. Agent chooses an action                            │
│   3. Environment returns reward + new observation       │
│   4. Repeat until done                                  │
└─────────────────────────────────────────────────────────┘
```

In code:

```python
result = env.reset()                    # Start episode
while not result.done:
    action = agent.choose(result.observation)
    result = env.step(action)           # Take action, get reward
    agent.learn(result.reward)
```

That's it. That's RL.

---

## Installation

```bash
pip install openenv
```

---

## Three Ways to Connect

OpenEnv gives you three methods to instantiate an environment — all of them expose the same API:

```python
# 1. Auto-download from Hugging Face Hub (starts Docker automatically)
env = OpenSpielEnv.from_hub('openenv/openspiel-env')

# 2. Start from a local Docker image
env = OpenSpielEnv.from_docker_image('openspiel-env:latest')

# 3. Connect to an already-running server
env = OpenSpielEnv(base_url='http://localhost:8000')
```

---

## Playing an Episode — The Catch Game

The **Catch** game is the hello-world of RL environments: a ball falls from the top of a grid, and your agent (a paddle) must move left/right to catch it.

```python
from openspiel_env.client import OpenSpielEnv
from openspiel_env.models import OpenSpielAction

env = OpenSpielEnv(base_url='http://localhost:8000')

with env.sync() as client:
    result = client.reset()                          # Start episode

    while not result.done:
        # Pick a random legal action
        action_id = random.choice(result.observation.legal_actions)
        action = OpenSpielAction(action_id=action_id, game_name="catch")

        result = client.step(action)                 # Step the environment

    state = client.state()
    print(f"Steps: {state.step_count}, Reward: {result.reward}")
    print("CAUGHT!" if result.reward > 0 else "MISSED!")
```

### Action Space

| `action_id` | Meaning |
|---|---|
| `0` | Move LEFT |
| `1` | STAY |
| `2` | Move RIGHT |

---

## The Type System (Pydantic Models)

OpenEnv is fully type-safe. Here are the three core models:

### 1. `OpenSpielObservation` — what the agent *sees* after each step

```python
OpenSpielObservation(
    info_state=[0.0, 0.0, 1.0, 0.0, 0.0, ...],  # Flattened grid (10×5 = 50 floats)
    legal_actions=[0, 1, 2],                       # Valid moves right now
    game_phase="playing",
    current_player_id=0,
    opponent_last_action=None,
)
```

`info_state` is a flattened grid where `1.0` marks the ball position and the paddle position. Index `row * width + col` gives you the cell.

---

### 2. `OpenSpielAction` — what you *send* to `step()`

```python
OpenSpielAction(
    action_id=1,           # 0=LEFT, 1=STAY, 2=RIGHT
    game_name="catch",
    game_params={"rows": 10, "columns": 5},
)
```

Only send actions that appear in `legal_actions` — the environment will reject illegal moves.

---

### 3. `OpenSpielState` — the environment's internal state

```python
OpenSpielState(
    game_name="catch",
    agent_player=0,
    opponent_policy="random",
    game_params={"rows": 10, "columns": 5},
    num_players=1,
)
```

Call `client.state()` at any point to inspect what the server knows. Useful for debugging and logging.

---

## How `info_state` Encodes the Grid

The Catch game uses a **10 rows × 5 columns** grid. The `info_state` is a flat list of 50 floats (all zeros except two `1.0`s):

```
Grid position → flat index: row * 5 + col

Row 0, Col 2 (ball start) → index 2   → info_state[2] = 1.0
Row 9, Col 2 (paddle)     → index 47  → info_state[47] = 1.0
```

As the ball falls one row per step, its `1.0` moves down the list. The paddle's `1.0` shifts left or right based on your action.

---

## Running Locally vs. On Colab

The notebook auto-detects the environment:

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
```

- **Colab:** No Docker support — connect to a remote server or Hugging Face Space.
- **Local:** Start the server with `docker run -p 8000:8000 openenv/openspiel-env:latest` or `openenv serve`.

When no server is available, the notebook falls back to a **pure-Python local simulation** that mirrors the real game logic exactly — so you can study the mechanics without any infrastructure.

---

## Repo Structure

```
openenv-tutorial/
├── openenv_tutorials.ipynb   # Main tutorial notebook
└── README.md                 # This file
```

---

## Running the Notebook

```bash
# Clone and open
git clone https://github.com/<your-username>/openenv-tutorial
cd openenv-tutorial

pip install openenv
# Start a local server (requires Docker)
docker run -p 8000:8000 openenv/openspiel-env:latest

jupyter notebook openenv_tutorials.ipynb
```

---

## References

- [OpenEnv Docs](https://github.com/openenv)
- [Hugging Face Hub](https://huggingface.co)
- [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel)
- [`adithya-s-k/RL_Envs_101`](https://github.com/adithya-s-k/RL_Envs_101) — the repo this tutorial is part of
