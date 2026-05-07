"""
Training script for DQN on LunarLander-v3.

Toggle flags to run different variants:
    python train.py --variant vanilla
    python train.py --variant double
    python train.py --variant double_per
    python train.py --variant per       # PER without Double, for ablation

Add --icm or --shaping for the curiosity / potential-shaping ablations:
    python train.py --variant double --icm
    python train.py --variant double --shaping
    python train.py --variant double --icm --shaping

When --icm or --shaping is set, the run name gets a suffix so logs don't
collide with the base ablation:
    double           -> logs/double_seed0.csv
    double + icm     -> logs/double_icm_seed0.csv
    double + shaping -> logs/double_shaping_seed0.csv

Logs per-episode reward to logs/<run_name>.csv so plot.py can pick it up.
"""
import argparse
import csv
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from icm import ICM
from reward_shaping import PotentialShaping


VARIANTS = {
    # (use_double, use_per)
    "vanilla":    (False, False),
    "double":     (True,  False),
    "per":        (False, True),
    "double_per": (True,  True),
}


def set_seed(seed: int, env):
    """Seed everything that has a chance of producing randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def make_run_name(variant: str, seed: int, use_icm: bool, use_shaping: bool) -> str:
    """Build a unique run name based on what's enabled."""
    suffix = ""
    if use_icm:
        suffix += "_icm"
    if use_shaping:
        suffix += "_shaping"
    return f"{variant}{suffix}_seed{seed}"


def train(args):
    use_double, use_per = VARIANTS[args.variant]
    use_icm = args.icm
    use_shaping = args.shaping

    run_name = make_run_name(args.variant, args.seed, use_icm, use_shaping)
    print(f"\n=== Run: {run_name} ===")
    print(f"  Double DQN:         {use_double}")
    print(f"  Prioritized Replay: {use_per}")
    print(f"  ICM (curiosity):    {use_icm}")
    print(f"  Reward shaping:     {use_shaping}\n")

    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    set_seed(args.seed, env)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"  Device: {device}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        double=use_double,
        device=device,
    )

    if use_per:
        buffer = PrioritizedReplayBuffer(
            capacity=args.buffer_size,
            state_dim=state_dim,
            alpha=args.per_alpha,
            beta_start=args.per_beta_start,
            beta_frames=args.per_beta_frames,
            device=device,
        )
    else:
        buffer = UniformReplayBuffer(
            capacity=args.buffer_size,
            state_dim=state_dim,
            device=device,
        )

    # Optional extensions: built only when their flag is set
    icm = ICM(state_dim=state_dim, action_dim=action_dim,
              eta=args.icm_eta, beta=args.icm_beta, lr=args.icm_lr,
              device=device) if use_icm else None
    shaper = PotentialShaping(gamma=args.gamma) if use_shaping else None

    # Epsilon schedule: linear decay from eps_start to eps_end over eps_decay_episodes.
    # We decay per episode rather than per step because episode lengths vary a lot
    # in LunarLander (a quick crash vs a careful landing) and per-step decay would
    # bias exploration toward longer episodes.
    epsilon = args.eps_start

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    csv_path = f"logs/{run_name}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    # Extra columns track intrinsic / shaping signals so we can analyse them later.
    # They'll always be present even when the feature is off (just zero) — keeps
    # the log schema consistent across runs.
    writer.writerow(["episode", "reward", "moving_avg_100", "epsilon",
                     "loss", "steps", "intrinsic_sum", "shaping_sum", "icm_loss"])

    reward_window = deque(maxlen=100)
    best_avg = -float("inf")
    total_steps = 0
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        # We track *extrinsic* episode reward for evaluation purposes — this is
        # what the LunarLander "solved" criterion is defined on. The agent may
        # internally see r_ext + r_intrinsic + F_shaping, but we only judge
        # success by the unmodified extrinsic signal.
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        episode_steps = 0
        episode_intrinsic = 0.0
        episode_shaping = 0.0
        episode_icm_loss = 0.0
        icm_update_count = 0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Build the reward the *agent* sees by layering on optional bonuses.
            # The original extrinsic reward is preserved separately for logging.
            agent_reward = reward
            if shaper is not None:
                F = shaper.shaping_bonus(state, next_state)
                agent_reward += F
                episode_shaping += F
            if icm is not None:
                r_i = icm.intrinsic_reward(state, action, next_state)
                agent_reward += r_i
                episode_intrinsic += r_i

            buffer.add(state, action, agent_reward, next_state, terminated)
            # Note: we store `terminated` (not `done`) so the bootstrap target
            # is only zeroed on real terminal states, not on time-limit truncations.

            state = next_state
            episode_reward += reward    # <-- extrinsic only, for the success metric
            episode_steps += 1
            total_steps += 1

            # Learn once per environment step, after we have enough data
            if len(buffer) >= args.batch_size and total_steps % args.learn_every == 0:
                batch = buffer.sample(args.batch_size)
                td_errors, loss = agent.learn(batch)
                episode_loss += loss
                loss_count += 1
                if use_per:
                    buffer.update_priorities(batch[6], td_errors)
                # Update ICM on the same batch (its own optimizer, separate loss).
                # batch is (states, actions, rewards, next_states, dones, weights, indices)
                if icm is not None:
                    icm_loss = icm.update(batch[0], batch[1], batch[3])
                    episode_icm_loss += icm_loss
                    icm_update_count += 1

        # Decay epsilon
        epsilon = max(args.eps_end, epsilon - (args.eps_start - args.eps_end) / args.eps_decay_episodes)

        reward_window.append(episode_reward)
        moving_avg = np.mean(reward_window)
        avg_loss = episode_loss / max(loss_count, 1)
        avg_icm_loss = episode_icm_loss / max(icm_update_count, 1)

        writer.writerow([episode, episode_reward, moving_avg, epsilon,
                         avg_loss, episode_steps,
                         episode_intrinsic, episode_shaping, avg_icm_loss])
        csv_file.flush()

        if episode % args.log_every == 0:
            elapsed = time.time() - start_time
            extra = ""
            if use_icm:
                extra += f" | r_int {episode_intrinsic:6.2f}"
            if use_shaping:
                extra += f" | F {episode_shaping:7.2f}"
            print(f"  Ep {episode:4d} | R {episode_reward:7.1f} | "
                  f"Avg100 {moving_avg:7.1f} | Eps {epsilon:.3f} | "
                  f"Loss {avg_loss:.4f}{extra} | t={elapsed:.0f}s")

        # Save checkpoint when we hit a new best (and we've seen at least 100 episodes)
        if len(reward_window) == 100 and moving_avg > best_avg:
            best_avg = moving_avg
            agent.save(f"checkpoints/{run_name}_best.pt")

        # LunarLander is "solved" at avg 200 over 100 episodes
        if len(reward_window) == 100 and moving_avg >= 200.0 and not args.no_early_stop:
            print(f"\n  *** Solved at episode {episode} with avg reward {moving_avg:.1f} ***")
            break

    csv_file.close()
    env.close()
    print(f"\n  Done. Log saved to {csv_path}")
    print(f"  Best 100-ep avg: {best_avg:.1f}")
    return run_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="vanilla", choices=list(VARIANTS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--cuda", action="store_true")

    # Hyperparameters — values chosen to be reasonable defaults for LunarLander.
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learn-every", type=int, default=4)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-episodes", type=int, default=500)

    # PER hyperparameters (only used when variant has PER)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-beta-frames", type=int, default=100_000)

    # ICM hyperparameters (only used when --icm is set)
    p.add_argument("--icm", action="store_true",
                   help="Augment reward with ICM curiosity bonus")
    p.add_argument("--icm-eta", type=float, default=0.01,
                   help="Intrinsic reward scale (small for dense-reward envs)")
    p.add_argument("--icm-beta", type=float, default=0.2,
                   help="Forward/inverse loss balance for ICM")
    p.add_argument("--icm-lr", type=float, default=1e-3)

    # Reward shaping (only used when --shaping is set)
    p.add_argument("--shaping", action="store_true",
                   help="Add potential-based shaping bonus to reward")

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-early-stop", action="store_true",
                   help="Keep training past 'solved' threshold (useful for fair comparison)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
