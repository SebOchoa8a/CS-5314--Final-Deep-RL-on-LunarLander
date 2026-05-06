"""
Intrinsic Curiosity Module (ICM) for LunarLander-v3.

Pathak et al. 2017 — "Curiosity-driven Exploration by Self-Supervised Prediction"
https://arxiv.org/abs/1705.05363

Architecture (three sub-networks):

  FeatureEncoder  phi: s  -> z  (state -> latent feature)
  InverseModel    g:  [z_t, z_{t+1}] -> logits over actions
  ForwardModel    f:  [z_t, one_hot(a_t)] -> z_{t+1}_hat

Intrinsic reward (eq. 6):
  r^i_t = (eta / 2) * || f(z_t, a_t) - phi(s_{t+1}) ||^2

Joint ICM loss (eq. 7):
  L_ICM = (1 - beta) * L_inverse + beta * L_forward
  L_inverse = CrossEntropy(g(z_t, z_{t+1}), a_t)
  L_forward  = (1/2) * || f(z_t, a_t) - z_{t+1} ||^2   (mean over batch & features)

  beta in [0,1]: weight forward vs inverse loss
  eta  > 0     : scales intrinsic reward magnitude

Usage (standalone — does not require any changes to agent.py or train.py):

    from icm import ICM

    icm = ICM(state_dim=8, action_dim=4, device=device)

    # Inside the per-step training loop, after buffer.add():
    r_i = icm.intrinsic_reward(state, action, next_state)
    total_reward = extrinsic_reward + r_i          # optionally mix in

    # After agent.learn(), update ICM on the same batch:
    icm_loss = icm.update(states, actions, next_states)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """Maps raw state s to a latent feature vector phi(s)."""

    def __init__(self, state_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, feature_dim),
            nn.ELU(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class InverseModel(nn.Module):
    """
    Predicts a_t from (phi(s_t), phi(s_{t+1})).
    Output: logits over action_dim — use CrossEntropy loss.
    Training this forces phi to encode action-relevant state features.
    """

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_t1], dim=-1))


class ForwardModel(nn.Module):
    """
    Predicts phi(s_{t+1}) from (phi(s_t), one_hot(a_t)).
    Prediction error = intrinsic reward signal.
    """

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, z_t: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, a_onehot], dim=-1))


# ---------------------------------------------------------------------------
# ICM wrapper
# ---------------------------------------------------------------------------

class ICM:
    """
    Wraps the three ICM sub-networks into a single object.

    Parameters
    ----------
    state_dim   : dimensionality of the environment observation (8 for LunarLander)
    action_dim  : number of discrete actions (4 for LunarLander)
    feature_dim : size of the latent feature vector phi(s)
    eta         : intrinsic reward scale (eta in the paper)
    beta        : forward/inverse loss balance (0 = only inverse, 1 = only forward)
    lr          : learning rate for the ICM optimizer
    device      : "cpu" or "cuda"
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        feature_dim: int = 64,
        eta: float = 0.01,
        beta: float = 0.2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.eta = eta
        self.beta = beta
        self.device = device

        self.encoder = FeatureEncoder(state_dim, feature_dim).to(device)
        self.inverse = InverseModel(feature_dim, action_dim).to(device)
        self.forward_model = ForwardModel(feature_dim, action_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.forward_model.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        Compute a scalar intrinsic reward for a single transition.

        Call this right after env.step(), before adding to the replay buffer,
        so you can optionally augment the extrinsic reward:
            r_total = r_extrinsic + icm.intrinsic_reward(s, a, s_next)

        Returns a plain Python float (already scaled by eta).
        """
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        s1 = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        a_oh = self._onehot(torch.tensor([action], device=self.device))

        z_t = self.encoder(s)
        z_t1 = self.encoder(s1)
        z_t1_hat = self.forward_model(z_t, a_oh)

        # eq. 6: r^i = (eta/2) * ||z_hat - z||^2
        r_i = (self.eta / 2.0) * F.mse_loss(z_t1_hat, z_t1, reduction="sum")
        return float(r_i.item())

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """
        Run one gradient step on a batch of transitions.

        Inputs are the same tensors already on `device` that come out of the
        replay buffer sample — just pass states, actions, next_states directly.

        Returns the scalar ICM loss (for logging).
        """
        a_oh = self._onehot(actions)

        z_t = self.encoder(states)
        z_t1 = self.encoder(next_states)

        # Inverse loss: can we recover the action from the state transition?
        action_logits = self.inverse(z_t, z_t1.detach())
        L_inverse = F.cross_entropy(action_logits, actions)

        # Forward loss: can we predict the next feature from current feature + action?
        # Stop gradient through z_t1 so the forward model doesn't "cheat" by
        # pulling z_t1 toward its own prediction.
        z_t1_hat = self.forward_model(z_t, a_oh)
        L_forward = 0.5 * F.mse_loss(z_t1_hat, z_t1.detach())

        # eq. 7 (ICM-only terms — no policy gradient term here since we're off-policy)
        loss = (1.0 - self.beta) * L_inverse + self.beta * L_forward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "inverse": self.inverse.state_dict(),
            "forward": self.forward_model.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.inverse.load_state_dict(ckpt["inverse"])
        self.forward_model.load_state_dict(ckpt["forward"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _onehot(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert integer action tensor to one-hot float, shape (B, action_dim)."""
        return F.one_hot(actions, num_classes=self.action_dim).float()
