"""
Replay buffers for DQN.

Two implementations sharing the same sample/add interface:
  - UniformReplayBuffer: standard FIFO buffer with uniform sampling
  - PrioritizedReplayBuffer: sum-tree based, samples by TD-error magnitude
                             (Schaul et al. 2016)

Both return (states, actions, rewards, next_states, dones, weights, indices).
For uniform, weights=1 and indices=None so the training loop can be agnostic.
"""
import numpy as np
import torch


class UniformReplayBuffer:
    """Standard experience replay (Mnih et al. 2015)."""

    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays for speed
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, s_next, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
            torch.ones(batch_size, device=self.device),  # uniform weights
            None,  # no indices needed for uniform
        )
        return batch

    def update_priorities(self, indices, td_errors):
        """No-op for uniform buffer; kept for interface compatibility."""
        pass

    def __len__(self):
        return self.size


class SumTree:
    """
    Binary tree where each parent stores the sum of its children's priorities.
    Allows O(log n) sampling proportional to priority.
    Leaf nodes store the actual priorities; internal nodes store sums.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Total nodes: capacity leaves + (capacity - 1) internal nodes
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0  # next leaf index to write
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_idx: int, priority: float):
        """Update a leaf and propagate the delta up to the root."""
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Walk up to the root, adding delta to each ancestor
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def add(self, priority: float):
        """Insert with the given priority at the next leaf slot."""
        self.update(self.data_ptr, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_leaf(self, value: float):
        """
        Find the leaf such that the cumulative sum at that leaf >= value.
        Returns (leaf_index, priority).
        """
        idx = 0
        while idx < self.capacity - 1:  # while not a leaf
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay (Schaul et al. 2016).

    Sampling probability of transition i:  P(i) = p_i^alpha / sum_k p_k^alpha
    Importance-sampling weight:  w_i = (N * P(i))^(-beta), normalized by max w.

    alpha controls how much prioritization is used (0 = uniform, 1 = full).
    beta controls IS correction strength (annealed from beta_start to 1.0).
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon  # ensures nonzero probability for all transitions
        self.device = device
        self.frame = 0  # for beta annealing

        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # new transitions get max priority so they're seen at least once

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def _beta(self) -> float:
        """Linearly anneal beta from beta_start to 1.0 over beta_frames steps."""
        fraction = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(self, s, a, r, s_next, done):
        idx = self.tree.data_ptr
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_next
        self.dones[idx] = float(done)
        # New transitions enter with max priority -> guaranteed at least one sample
        self.tree.add(self.max_priority ** self.alpha)

    def sample(self, batch_size: int):
        self.frame += 1
        beta = self._beta()

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        # Stratified sampling: divide [0, total] into batch_size equal segments
        # and draw one sample from each. Reduces variance vs naive sampling.
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            leaf_idx, priority = self.tree.get_leaf(value)
            indices[i] = leaf_idx
            priorities[i] = priority

        # Compute IS weights. P(i) = p_i / sum(p), w_i = (N * P(i))^(-beta)
        N = self.tree.size
        probs = priorities / self.tree.total()
        weights = (N * probs) ** (-beta)
        weights /= weights.max()  # normalize so max weight = 1 (stability)

        batch = (
            torch.from_numpy(self.states[indices]).to(self.device),
            torch.from_numpy(self.actions[indices]).to(self.device),
            torch.from_numpy(self.rewards[indices]).to(self.device),
            torch.from_numpy(self.next_states[indices]).to(self.device),
            torch.from_numpy(self.dones[indices]).to(self.device),
            torch.from_numpy(weights.astype(np.float32)).to(self.device),
            indices,
        )
        return batch

    def update_priorities(self, indices, td_errors):
        """Called after a learning step to set p_i = |TD_error| + epsilon."""
        td_errors = np.abs(td_errors) + self.epsilon
        for idx, err in zip(indices, td_errors):
            priority = err ** self.alpha
            self.tree.update(idx, priority)
            # Track max for new transitions
            if err > self.max_priority:
                self.max_priority = float(err)

    def __len__(self):
        return self.tree.size
