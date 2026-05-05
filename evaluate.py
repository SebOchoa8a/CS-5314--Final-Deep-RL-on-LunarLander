"""
Evaluate a trained agent for N episodes with epsilon=0 (greedy).

Usage:
    python evaluate.py --checkpoint checkpoints/double_per_seed0_best.pt --episodes 100
"""
import argparse
import numpy as np
import gymnasium as gym
import torch

from agent import DQNAgent


def evaluate(checkpoint: str, episodes: int = 100, render: bool = False, seed: int = 42):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(checkpoint)
    agent.q_online.eval()

    rewards = []
    successes = 0  # episode reward >= 200 counts as a successful landing

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)  # greedy
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if ep_reward >= 200:
            successes += 1
        print(f"  Ep {ep+1}: reward={ep_reward:.1f}")

    rewards = np.array(rewards)
    print(f"\n=== Evaluation over {episodes} episodes ===")
    print(f"  Mean reward: {rewards.mean():.1f} ± {rewards.std():.1f}")
    print(f"  Min / Max:   {rewards.min():.1f} / {rewards.max():.1f}")
    print(f"  Success rate (>=200): {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    env.close()
    return rewards


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render, args.seed)
