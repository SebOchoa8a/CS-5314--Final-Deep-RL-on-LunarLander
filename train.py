"""
Training script for DQN on LunarLander-v3.

Toggle flags to run different variants:
    python train.py --variant vanilla
    python train.py --variant double
    python train.py --variant double_per
    python train.py --variant per       # PER without Double, for ablation

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


def train(args):
    use_double, use_per = VARIANTS[args.variant]
    run_name = f"{args.variant}_seed{args.seed}"
    print(f"\n=== Run: {run_name} ===")
    print(f"  Double DQN: {use_double}")
    print(f"  Prioritized Replay: {use_per}\n")

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
    writer.writerow(["episode", "reward", "moving_avg_100", "epsilon", "loss", "steps"])

    reward_window = deque(maxlen=100)
    best_avg = -float("inf")
    total_steps = 0
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        episode_steps = 0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, terminated)
            # Note: we store `terminated` (not `done`) so the bootstrap target
            # is only zeroed on real terminal states, not on time-limit truncations.
            # This is a subtle but important correctness detail.

            state = next_state
            episode_reward += reward
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

        # Decay epsilon
        epsilon = max(args.eps_end, epsilon - (args.eps_start - args.eps_end) / args.eps_decay_episodes)

        reward_window.append(episode_reward)
        moving_avg = np.mean(reward_window)
        avg_loss = episode_loss / max(loss_count, 1)

        writer.writerow([episode, episode_reward, moving_avg, epsilon, avg_loss, episode_steps])
        csv_file.flush()

        if episode % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Ep {episode:4d} | R {episode_reward:7.1f} | "
                  f"Avg100 {moving_avg:7.1f} | Eps {epsilon:.3f} | "
                  f"Loss {avg_loss:.4f} | t={elapsed:.0f}s")

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
    # See the writeup for justification.
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

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-early-stop", action="store_true",
                   help="Keep training past 'solved' threshold (useful for fair comparison)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
