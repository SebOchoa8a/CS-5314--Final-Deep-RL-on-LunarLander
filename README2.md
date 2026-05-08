# CS 4320 Final Project — Deep RL on LunarLander-v3

**Team 15**  
Team members withheld for anonymous review.

## Project Overview

This project studies Deep Reinforcement Learning on **Gymnasium LunarLander-v3**. The core agent is a **Deep Q-Network (DQN)** with a neural-network value function and an experience replay buffer. These two components satisfy the project’s required techniques: **function approximation** and **experience replay**. On top of the baseline, we evaluate several extensions to go beyond basic methods. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

### Base variants
- **Vanilla DQN** — baseline method
- **Double DQN** — reduces Q-value overestimation by separating action selection from action evaluation
- **Prioritized Experience Replay (PER)** — samples high-TD-error transitions more often
- **Double DQN + PER** — combines both extensions :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

### Optional add-on extensions
- **ICM (Intrinsic Curiosity Module)** — adds an intrinsic reward bonus based on forward-model prediction error
- **Potential-Based Reward Shaping** — adds a policy-invariant shaping bonus of the form  
  \(F(s,s') = \gamma \Phi(s') - \Phi(s)\) using proximity, speed, angle, and leg-contact terms for LunarLander :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

## Files

### Core files
- `network.py` — defines the Q-network used to estimate action values from the 8-dimensional LunarLander state
- `replay.py` — implements both `UniformReplayBuffer` and `PrioritizedReplayBuffer`
- `agent.py` — defines `DQNAgent`, including vanilla and Double DQN target logic
- `train.py` — main training script; supports all base variants and optional ICM / reward shaping
- `evaluate.py` — loads a checkpoint and runs greedy evaluation episodes
- `plot.py` — creates plots from saved CSV logs for individual runs or variant comparisons :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

### Extension files
- `icm.py` — implements the Intrinsic Curiosity Module
- `reward_shaping.py` — implements potential-based reward shaping for LunarLander-v3 :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

### Optional interface
- `gui.py` — interactive training GUI that lets you choose a variant, toggle ICM and reward shaping, tune hyperparameters, watch live training charts, view logs, and optionally preview the simulation :contentReference[oaicite:15]{index=15}

## Installation

Create and activate a Python virtual environment, then install the requirements.

### Install dependencies
```bash
pip install -r requirements.txt
