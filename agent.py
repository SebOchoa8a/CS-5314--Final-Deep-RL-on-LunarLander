"""
DQN agent.

Supports vanilla DQN, Double DQN, and Prioritized Experience Replay
via configuration flags. The class is deliberately small — most of
the heavy lifting lives in the network and replay modules.

Vanilla DQN target:    y = r + gamma * max_a' Q_target(s', a')
Double DQN target:     y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

The Double DQN change decouples action selection from action evaluation,
which reduces the maximization bias that causes vanilla DQN to overestimate
Q-values (van Hasselt et al. 2016).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 1e-3,            # soft target-update rate
        double: bool = False,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.double = double
        self.device = device

        # Online net is trained; target net is a slow-moving copy
        self.q_online = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()  # target net never trains directly

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.q_online(state_t)
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch):
        """
        Run one gradient step on a sampled batch.

        Returns the per-sample TD errors (numpy) so a prioritized buffer
        can use them to update its priorities. Returns the scalar loss too.
        """
        states, actions, rewards, next_states, dones, weights, indices = batch

        # Q(s, a) — gather the action that was actually taken
        q_pred = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * Q_target(s', a*)
        # where a* depends on whether we're using Double DQN
        with torch.no_grad():
            if self.double:
                # Online net picks the action; target net evaluates it.
                # This decoupling is the entire point of Double DQN.
                next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
                q_next = self.q_target(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla: target net both picks and evaluates -> overestimation bias
                q_next = self.q_target(next_states).max(dim=1)[0]

            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        td_errors = q_target - q_pred

        # Importance-sampling weights are 1.0 for uniform replay, so this
        # reduces to plain MSE in the non-PER case.
        # We use Huber (smooth_l1) for robustness to occasional large TD errors.
        loss = (weights * nn.functional.smooth_l1_loss(
            q_pred, q_target, reduction="none"
        )).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping is cheap insurance against the rare exploding update
        nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update the target network: theta_target <- tau*theta_online + (1-tau)*theta_target
        # This is smoother than the hard periodic copy used in the original DQN paper
        # and tends to stabilize training in continuous-state environments.
        with torch.no_grad():
            for p_target, p_online in zip(
                self.q_target.parameters(), self.q_online.parameters()
            ):
                p_target.data.mul_(1.0 - self.tau).add_(self.tau * p_online.data)

        return td_errors.detach().cpu().numpy(), float(loss.item())

    def save(self, path: str):
        torch.save(self.q_online.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(state)
        self.q_target.load_state_dict(state)
