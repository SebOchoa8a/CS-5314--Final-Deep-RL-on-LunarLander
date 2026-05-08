# CS 5314 Final Project — DQN on LunarLander

Team 15: Albert Yoshimoto, Sebastian Ochoa, Vanessa Chavez De La Rosa

## What's here

A Deep Q-Network agent for `LunarLander-v3` with four base variants for ablation:

- **Vanilla DQN** — required baseline (Mnih et al. 2015)
- **Double DQN** — decouples action selection from evaluation (van Hasselt et al. 2016)
- **Prioritized Experience Replay (PER)** — samples high-TD-error transitions more often (Schaul et al. 2016)
- **Double DQN + PER** — combines both extensions

The required course techniques (function approximation via neural net and experience replay buffer) are present in every base variant. Double DQN and PER are the main “beyond basic methods” extensions.

In addition to the four base variants, this project also includes two optional extensions:

- **Intrinsic Curiosity Module (ICM)** — adds an intrinsic reward bonus based on forward-model prediction error to encourage exploration
- **Potential-Based Reward Shaping** — adds a policy-invariant shaping bonus to guide learning without changing the optimal policy

These optional extensions can be enabled on top of a base variant for additional experiments.

## Files

| File | What it does |
|------|--------------|
| `network.py` | `QNetwork` — small MLP (8 → 64 → 64 → 4) |
| `replay.py`  | `UniformReplayBuffer` and `PrioritizedReplayBuffer` (sum-tree based) |
| `agent.py`   | `DQNAgent` — handles learning, supports vanilla and Double DQN targets |
| `train.py`   | Training loop with CLI, CSV logging, checkpointing, and optional ICM / reward shaping |
| `evaluate.py`| Loads a checkpoint, runs greedy episodes, optionally renders |
| `plot.py`    | Plots single runs or compares variants across seeds |
| `icm.py`     | Implements the Intrinsic Curiosity Module |
| `reward_shaping.py` | Implements potential-based reward shaping for LunarLander-v3 |
| `gui.py`     | Optional interactive GUI for training, visualization, and experimentation |

## Setup

```bash
pip install -r requirements.txt
```

Note: `gymnasium[box2d]` requires `swig`.

On Ubuntu:
```bash
sudo apt-get install swig
```

On macOS:
```bash
brew install swig
```

On Windows:
```bash
python -m pip install swig
```

Then install the Python requirements:

```bash
pip install -r requirements.txt
```

## Running

### Train a single base variant

```bash
python train.py --variant vanilla --seed 0 --episodes 1000
python train.py --variant double --seed 0 --episodes 1000
python train.py --variant per --seed 0 --episodes 1000
python train.py --variant double_per --seed 0 --episodes 1000
```

### Train a variant with optional extensions

```bash
python train.py --variant double --seed 0 --episodes 1000 --icm
python train.py --variant double --seed 0 --episodes 1000 --shaping
python train.py --variant double --seed 0 --episodes 1000 --icm --shaping
```

### Run the full base ablation across 3 seeds

```bash
for v in vanilla double per double_per; do
  for s in 0 1 2; do
    python train.py --variant $v --seed $s --no-early-stop
  done
done
```

### Run the extension ablation across 3 seeds

```bash
for s in 0 1 2; do
  python train.py --variant double --seed $s --icm --no-early-stop
  python train.py --variant double --seed $s --shaping --no-early-stop
  python train.py --variant double --seed $s --icm --shaping --no-early-stop
done
```

### Plot the base comparison

```bash
python plot.py --variants vanilla double per double_per --seeds 0 1 2 --out plots/base_ablation.png
```

### Plot the full comparison including extensions

```bash
python plot.py --variants vanilla double per double_per double_icm double_shaping double_icm_shaping --seeds 0 1 2 --out plots/full_ablation.png
```

### Evaluate a trained agent

```bash
python evaluate.py --checkpoint checkpoints/double_per_seed0_best.pt --episodes 100
```

### Watch it play (requires a display)

```bash
python evaluate.py --checkpoint checkpoints/double_per_seed0_best.pt --episodes 5 --render
```

### Run the optional GUI

```bash
python gui.py
```

The GUI allows you to:

- choose the base variant
- enable or disable ICM
- enable or disable reward shaping
- tune hyperparameters
- watch live training charts
- optionally preview the simulation

## Logging and outputs

Training creates:

- CSV logs in `logs/`
- best checkpoints in `checkpoints/`

Run names automatically include extension suffixes when used. For example:

- `double_seed0`
- `double_icm_seed0`
- `double_shaping_seed0`
- `double_icm_shaping_seed0`

Each CSV log stores per-episode values including:

- `episode`
- `reward`
- `moving_avg_100`
- `epsilon`
- `loss`
- `steps`
- `intrinsic_sum`
- `shaping_sum`
- `icm_loss`

The main success metric is based on the **extrinsic** LunarLander reward. If ICM or reward shaping is enabled, the agent learns from the modified reward internally, but the logged episode reward and solved threshold still use the original environment reward.

## Hyperparameters

Defaults in `train.py` are tuned for LunarLander and should not need changing for the standard experiments. Key values:

- Learning rate: 5e-4
- Discount γ: 0.99
- Soft target update τ: 1e-3
- Batch size: 64
- Buffer size: 100,000
- ε: 1.0 → 0.01 over 500 episodes
- PER α = 0.6
- PER β: 0.4 → 1.0 annealed over 100k frames

Additional optional extension parameters in `train.py`:

- `--icm-eta` default: 0.01
- `--icm-beta` default: 0.2
- `--icm-lr` default: 1e-3
- `--shaping` enables potential-based reward shaping

## GUI

`gui.py` is optional and not required for the final experiment pipeline. It provides an interactive way to:

- pick a base variant
- enable or disable ICM
- enable or disable reward shaping
- tune hyperparameters
- watch live training charts
- view logs
- optionally preview the simulation

It is useful for quick experimentation or demonstrations, but the final results should still come from `train.py`, `plot.py`, and `evaluate.py`.

## Expected results

Vanilla DQN usually learns a reasonable policy on LunarLander, while Double DQN tends to be more stable. PER often improves early learning speed, and combining Double DQN with PER is often one of the strongest base variants. The optional ICM and reward shaping extensions can also be compared to see whether they improve exploration or training efficiency. Results vary across seeds, which is why we run multiple seeds and report mean ± std.
