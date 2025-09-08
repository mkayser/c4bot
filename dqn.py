from __future__ import annotations
import numpy as np
import c4
from typing import Protocol, Optional, NamedTuple, Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


#class Observation(NamedTuple):
#    s: np.ndarray
#    mask: Optional[np.ndarray] = None
#    info: Optional[Any] = None

class Trainer(Protocol):
    def start_game(self) -> None: ...
    def end_game(self) -> None: ...
    def add_transition(self, tr: c4.Transition) -> None: ...
    def train(self) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class DQNVariants():
    DQN = 'DQN'
    DoubleDQN = 'DoubleDQN'
    PrioritizedReplayDQN = 'PrioritizedReplayDQN'
    CategoricalDQN = 'CategoricalDQN'
    RainbowDQN = 'RainbowDQN'

class QFunction(Protocol):
    def __call__(self, s: np.ndarray) -> np.ndarray: ...
    def clone(self) -> 'QFunction': ...


class ConvNetQFunction(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, device: torch.device | str = "cpu"):
        super().__init__()
        c, h, w = input_shape
        # Track init params so we can clone later
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device_ = torch.device(device)

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.to(self.device_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), float32
        return self.net(x)

    @torch.no_grad()
    def scores(self, s: np.ndarray) -> np.ndarray:
        # Convenience inference helper
        self.eval()
        x = torch.from_numpy(s).float().unsqueeze(0).to(self.device_)  # (1,C,H,W)
        q = self(x).squeeze(0).cpu().numpy()
        return q

    def clone(self) -> ConvNetQFunction:
        # Proper clone: same class and constructor, then copy weights
        m = type(self)(self.input_shape, self.num_actions, device=self.device_)
        m.load_state_dict(self.state_dict())
        return m
    

# class Transition(NamedTuple):
#    s: np.ndarray
#    a: int
#    r: float
#    s2: np.ndarray
#    mask: np.ndarray
#    mask2: np.ndarray

    
class TransitionReplayBuffer:
    def __init__(self, 
                 capacity : int, 
                 obs_shape : Tuple[int, ...],
                 num_actions : int,
                 dtype_obs=np.float32):
        self.capacity = capacity
        self.s  = np.empty((capacity, *obs_shape), dtype=dtype_obs)
        self.a  = np.empty((capacity,), dtype=np.int64)
        self.r  = np.empty((capacity,), dtype=np.float32)
        self.s2 = np.empty((capacity, *obs_shape), dtype=dtype_obs)
        self.mask = np.empty((capacity, num_actions), dtype=np.float32)
        self.mask2 = np.empty((capacity, num_actions), dtype=np.float32)
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
        self.next = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.mask[idx], self.mask2[idx]


class VanillaDQNTrainer(Trainer):
    def __init__(self, 
                 qfunction : QFunction, 
                 *, 
                 buffer_size: int = 10000,
                 train_every: int = 1,
                 min_buffer_to_train: int = 1000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 target_update_every: int = 100,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 learning_rate: float = 1e-3
                 ):  
        self.qfunction = qfunction
        self.target_qfunction = qfunction.clone()  # Assume qfunction has a copy method
        self.buffer_size = buffer_size
        self.train_every = train_every
        self.min_buffer_to_train = min_buffer_to_train
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_every = target_update_every
        self.step_count = 0
        self.replay_buffer = TransitionReplayBuffer(buffer_size, qfunction.input_shape, qfunction.num_actions)
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.qfunction.parameters(), lr=learning_rate)
        assert self.min_buffer_to_train <= self.buffer_size
        assert self.train_every > 0
        assert self.batch_size > 0
        assert self.gamma >= 0.0 and self.gamma <= 1.0
        assert self.learning_rate > 0.0
        assert self.target_update_every > 0
        assert self.batch_size <= self.min_buffer_to_train

    def start_game(self) -> None:
        pass

    def end_game(self) -> None:
        pass

    def add_transition(self, tr: c4.Transition) -> None:
        self.replay_buffer.add(tr)
        self.step_count += 1
        if self.step_count > self.min_buffer_to_train and self.step_count % self.train_every == 0:
            self.train()
        if self.step_count % self.target_update_every == 0:
            # Update target network
            pass

    def train(self) -> None:
        # Only train if we have enough samples
        if self.replay_buffer.size < self.min_buffer_to_train:
            return
                
        # Sample a batch
        s_batch, a_batch, r_batch, s2_batch, mask_batch, mask2_batch = self.replay_buffer.sample(self.batch_size)
        s_batch = torch.from_numpy(s_batch).float().to(self.qfunction.device_)
        a_batch = torch.from_numpy(a_batch).long().to(self.qfunction.device_)
        r_batch = torch.from_numpy(r_batch).float().to(self.qfunction.device_)
        s2_batch = torch.from_numpy(s2_batch).float().to(self.qfunction.device_)
        mask2_batch = torch.from_numpy(mask2_batch).float().to(self.qfunction.device_)
        
        # Compute Q(s,a)
        q_values = self.qfunction(s_batch)  # (B, num_actions)
        q_s_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute target Q values
        with torch.no_grad():
            q_values_next = self.target_qfunction(s2_batch)  # (B, num_actions)
            if mask2_batch is not None:
                q_values_next = q_values_next * mask2_batch + (1 - mask2_batch) * (-1e10)
            max_q_s2 = q_values_next.max(1)[0]  # (B,)
            target_q = r_batch + self.gamma * max_q_s2  # (B,)

        # Compute loss
        loss = nn.MSELoss()(q_s_a, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        