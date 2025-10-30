# Minimal Prioritized Experience Replay (PER) with α, β, and optional normalization/clipping
# - SumTree for O(log N) sampling/updates
# - α controls prioritization sharpness
# - β (annealed) controls importance-sampling (IS) correction strength
# - Optional batch max-normalization of IS weights and/or explicit clipping
# - Stratified sampling for lower-variance draws
#
# Notes
# -----
# * We store priorities already raised to α inside the tree. That makes sampling P(i) simple: p_i^α / sum(p^α).
# * When updating priorities from TD-errors, we set priority = (|δ| + ε)^α.
# * IS weights use P(i); when normalizing we divide by max(w) within the batch.
# * For new transitions, we insert with the current max priority to encourage exploration.
# * This buffer is dtype-agnostic for obs/actions: pass shapes and dtypes; it will keep numpy arrays internally
#   and return torch tensors on sample.

from __future__ import annotations
import math
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Callable, Protocol
import c4



# class Transition(NamedTuple):
#    s: np.ndarray
#    a: int
#    r: float
#    s2: np.ndarray
#    mask: np.ndarray
#    mask2: np.ndarray

class SimpleTransitionReplayBuffer:
    def __init__(self, 
                 capacity : int, 
                 obs_shape : Tuple[int, ...],
                 num_actions : int,
                 *,
                 dtype_obs=np.float32,
                 dtype_r=np.float32,
                 dtype_mask=np.float32,
                 dtype_a=np.int64):
        self.capacity = capacity
        self.s  = np.empty((capacity, *obs_shape), dtype=dtype_obs)
        self.a  = np.empty((capacity,), dtype=dtype_a)
        self.r  = np.empty((capacity,), dtype=dtype_r)
        self.s2 = np.empty((capacity, *obs_shape), dtype=dtype_obs)
        self.mask = np.empty((capacity, num_actions), dtype=dtype_mask)
        self.mask2 = np.empty((capacity, num_actions), dtype=dtype_mask)
        self.done = np.empty((capacity,), dtype=bool)
        self.size = 0
        self.next = 0

    def add(self, tr: c4.Transition) -> None:
        i = self.next
        self.s[i]  = tr.s
        self.a[i]  = tr.a
        self.r[i]  = tr.r
        self.s2[i] = tr.s2
        self.mask[i]  = tr.mask
        self.mask2[i] = tr.mask2
        self.done[i] = tr.done
        self.next = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def num_entries(self) -> int:
        return self.size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "idxs": idxs,
            "s": self.s[idxs],
            "a": self.a[idxs],
            "r": self.r[idxs],
            "s2": self.s2[idxs],
            "mask": self.mask[idxs],
            "mask2": self.mask2[idxs],
            "done": self.done[idxs],
        }



class SumTree:
    """Binary indexed tree that stores non-negative priorities and supports
    prefix-sum search. Layout uses a flat array of size 2*capacity.
    Leaves at [size : size+capacity).
    """
    def __init__(self, capacity: int):
        assert capacity > 0
        # Round capacity up to next power of two for simpler indexing
        self.capacity_pow2 = 1
        while self.capacity_pow2 < capacity:
            self.capacity_pow2 <<= 1
        self.capacity = capacity
        self.size = self.capacity_pow2
        self.tree = np.zeros(2 * self.size, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def update(self, leaf_idx: int, value: float) -> None:
        """Set priority at leaf position (0-based within capacity) to value."""
        assert 0 <= leaf_idx < self.capacity
        i = self.size + leaf_idx
        delta = value - self.tree[i]
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def get(self, leaf_idx: int) -> float:
        return float(self.tree[self.size + leaf_idx])

    def find_prefixsum(self, mass: float) -> int:
        """Return leaf index such that cumulative sum up to that leaf > mass.
        mass ∈ [0, total()). If mass >= total-1e-12, last leaf is returned.
        """
        i = 1
        if mass >= self.tree[i]:
            return min(self.capacity - 1, self.capacity - 1)  # defensive
        while i < self.size:
            left = 2 * i
            if mass < self.tree[left]:
                i = left
            else:
                mass -= self.tree[left]
                i = left + 1
        leaf = i - self.size
        return min(leaf, self.capacity - 1)


# Fix PERBuffer to match the TransitionReplayBuffer Protocol
#  * dtype naming
#  * use numpy arrays instead of torch
#  * Anything else, naming, etc.
# Fix Trainer usages of buffer, including adding the update-losses step if PERbuffer
# Fix references to TransitionReplayBuffer to refer to the Protocol
# Fix anything (e.g. trainer loop) that instantiates Trainer so it creates the buffer and passes it in
# Fix config so that it specifies PERbuffer or simple buffer
class PERBuffer:
    def __init__(
        self,
        capacity : int, 
        obs_shape : Tuple[int, ...],
        num_actions : int,
        *,
        dtype_obs=np.float32,
        dtype_r=np.float32,
        dtype_mask=np.float32,
        dtype_a=np.int64,
        alpha: float = 0.6,
        beta: float | Callable[[int], float] = 1.0,
        eps: float = 1e-5,
        normalize_weights: bool = True,
        w_clip: Optional[float] = None):
    
        assert 0 <= alpha <= 1
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = beta
        self.eps = float(eps)
        self.normalize_weights = bool(normalize_weights)
        self.w_clip = None if w_clip is None else float(w_clip)

        # Storage
        self.s = np.zeros((self.capacity, *obs_shape), dtype=dtype_obs)
        self.a = np.zeros((self.capacity,), dtype=dtype_a)
        self.r = np.zeros((self.capacity,), dtype=dtype_r)
        self.s2 = np.zeros((self.capacity, *obs_shape), dtype=dtype_obs)
        self.mask = np.empty((capacity, num_actions), dtype=dtype_mask)
        self.mask2 = np.empty((capacity, num_actions), dtype=dtype_mask)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

        # Priorities and tree store p_i^α
        self.tree = SumTree(self.capacity)
        self.max_priority_alpha = 1.0  # max (|δ|+ε)^α observed; start at 1 to sample fresh data

        self.next = 0
        self.size = 0

    def num_entries(self) -> int:
        return self.size

    def add(self, tr: c4.Transition) -> None:
        i = self.next

        # Update the current buffer location
        self.s[i]  = tr.s
        self.a[i]  = tr.a
        self.r[i]  = tr.r
        self.s2[i] = tr.s2
        self.mask[i]  = tr.mask
        self.mask2[i] = tr.mask2
        self.done[i] = tr.done

        # Update the corresponding sumtree location
        # We have not computed a priority, so set to the max priority
        # This prioritizes novel data
        p_alpha = self.max_priority_alpha
        self.tree.update(i, p_alpha)

        # Increment next and size
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _beta(self, frame_idx: int) -> float:
        beta = self.beta(frame_idx) if callable(self.beta) else self.beta
        return float(np.clip(beta, 0.0, 1.0))

    def sample(self, batch_size: int, frame_idx: int) -> Dict[str, Any]:
        assert self.size > 0
        batch_size = int(batch_size)
        total = self.tree.total()
        assert total > 0, "SumTree total is zero; all priorities are zero."

        # Define indices: Sample (stratified sampling) from buffer
        seg = total / batch_size
        samples = []
        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            mass = np.random.uniform(a, b)
            idx = self.tree.find_prefixsum(mass)
            samples.append(idx)
        idxs = np.array(samples, dtype=np.int64)

        # Retrieve the priority^alpha values, and normalize to make a prob dist
        p_alpha = np.array([self.tree.get(i) for i in idxs], dtype=np.float64)
        P = p_alpha / (total + 1e-12)

        # Compute beta for this time step
        beta = self._beta(frame_idx)

        # Compute Importance Sampling Weights: weight for i is: (1 / (N * P(i)))^β
        N = self.size
        w = (N * P) ** (-beta)

        # Normalize weights by max-within-batch to cap dynamic range at 1
        if self.normalize_weights:
            w /= (w.max() + 1e-12)

        # Enforce minimum weight
        if self.w_clip is not None:
            w = np.minimum(w, self.w_clip)

        return {
            "idxs": idxs,
            "s": self.s[idxs],
            "a": self.a[idxs],
            "r": self.r[idxs],
            "s2": self.s2[idxs],
            "mask": self.mask[idxs],
            "mask2": self.mask2[idxs],
            "done": self.done[idxs],
            "weights": w,
            "sample_probs": P,
            "beta": beta,
        }

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(td_errors).astype(np.float64)
        p_alpha = (td + self.eps) ** self.alpha
        self.max_priority_alpha = max(self.max_priority_alpha, float(p_alpha.max()))
        for i, p in zip(idxs, p_alpha):
            self.tree.update(int(i), float(p))


# # -------------------------------
# # Example usage (sketch)
# # -------------------------------
# if __name__ == "__main__":
#     # Toy 1D obs/action shapes
#     buf = PERBuffer(
#         capacity=100_000,
#         obs_shape=(8,),
#         act_shape=(),  # scalar action
#         device="cpu",
#         alpha=0.6,
#         beta_start=0.4,
#         beta_end=1.0,
#         beta_frames=200_000,
#         eps=1e-5,
#         normalize_weights=True,
#         w_clip=5.0,  # cap IS weights if desired
#     )

#     # Push some fake data
#     for t in range(5000):
#         s = np.random.randn(8).astype(np.float32)
#         a = np.random.randint(0, 4)
#         r = np.random.randn()
#         s2 = np.random.randn(8).astype(np.float32)
#         d = np.random.rand() < 0.05
#         buf.push(s, a, r, s2, d)

#     # Sample a batch and pretend-train
#     batch = buf.sample(batch_size=64, frame_idx=10_000)
#     obs, acts, rews, next_obs, dones, w = (
#         batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"], batch["weights"]
#     )

#     # Fake TD errors from a model (here random)
#     fake_td = np.random.randn(len(batch["idxs"]))
#     buf.update_priorities(batch["idxs"], fake_td)

#     print("Sampled weights (min,max):", float(w.min()), float(w.max()))
