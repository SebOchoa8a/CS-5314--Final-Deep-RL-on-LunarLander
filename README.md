# CS 5314 Final Project — DQN on LunarLander

Team 15: Albert Yoshimoto, Sebastian Ochoa, Vanessa Chavez De La Rosa

## What's here

A Deep Q-Network agent for `LunarLander-v3` with three variants for ablation:

- **Vanilla DQN** — required baseline (Mnih et al. 2015)
- **Double DQN** — decouples action selection from evaluation (van Hasselt et al. 2016)
- **Prioritized Experience Replay** — samples high-TD-error transitions more often (Schaul et al. 2016)

The required course techniques (function approximation via neural net, experience replay buffer) are present in every variant. Double DQN and PER are the "beyond basic methods" extensions.

## Files

| File | What it does |
|------|--------------|
| `network.py` | `QNetwork` — small MLP (8 → 64 → 64 → 4) |
| `replay.py`  | `UniformReplayBuffer` and `PrioritizedReplayBuffer` (sum-tree based) |
| `agent.py`   | `DQNAgent` — handles learning, supports vanilla and Double targets |
| `train.py`   | Training loop with CLI, CSV logging, checkpointing |
| `evaluate.py`| Loads a checkpoint, runs greedy episodes, optionally renders |
| `plot.py`    | Plots single runs or compares variants across seeds |

## Setup

```bash
pip install -r requirements.txt
```

Note: `gymnasium[box2d]` requires `swig`. On Ubuntu: `sudo apt-get install swig`.
On macOS: `brew install swig`.

## Running

Train a single variant:
```bash
python train.py --variant vanilla --seed 0 --episodes 1000
python train.py --variant double --seed 0 --episodes 1000
python train.py --variant per --seed 0 --episodes 1000
python train.py --variant double_per --seed 0 --episodes 1000
```

Run the full ablation across 3 seeds:
```bash
for v in vanilla double per double_per; do
  for s in 0 1 2; do
    python train.py --variant $v --seed $s --no-early-stop
  done
done
```

Plot the comparison:
```bash
python plot.py --variants vanilla double per double_per --seeds 0 1 2 --out plots/comparison.png
```

Evaluate a trained agent:
```bash
python evaluate.py --checkpoint checkpoints/double_per_seed0_best.pt --episodes 100
```

Watch it play (requires a display):
```bash
python evaluate.py --checkpoint checkpoints/double_per_seed0_best.pt --episodes 5 --render
```

## Hyperparameters

Defaults in `train.py` are tuned for LunarLander and should not need changing
for the basic ablation. Key values:

- Learning rate: 5e-4
- Discount γ: 0.99
- Soft target update τ: 1e-3
- Batch size: 64
- Buffer size: 100,000
- ε: 1.0 → 0.01 over 500 episodes
- PER α=0.6, β: 0.4 → 1.0 annealed over 100k frames

## Expected results

Vanilla DQN usually solves LunarLander (avg ≥ 200 over 100 eps) in ~400-700 episodes.
Double DQN tends to be more stable. PER tends to learn faster early.
Combining both is usually the strongest variant. Results vary across seeds — that's
why we run multiple seeds and report mean ± std.
