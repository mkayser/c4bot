from __future__ import annotations
import numpy as np
import c4
from typing import Protocol, Optional, NamedTuple, Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import copy


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


#class ToyQFunction:

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

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.mask[idx], self.mask2[idx], self.done[idx]


import numpy as np
import torch
import torch.nn.functional as F

def log_dqn_metrics(*, step: int,
                q_sa: torch.Tensor,        # [B]
                target_y: torch.Tensor,    # [B]
                q_values: torch.Tensor,    # [B, A] from online net
                mask: torch.Tensor | None, # [B, A] bool (True = legal) or None
                q_net: torch.nn.Module,
                writer: SummaryWriter,
                grad_clip_max: float | None = None,
                replay=None,               # optional: must expose sample(..., with_age=True)
                tag_prefix: str = "train"):
    # --- TD loss & error percentiles ---
    with torch.no_grad():
        td_err = (q_sa - target_y).detach().cpu().numpy()
        td_abs = np.abs(td_err)
        td_loss = float(F.smooth_l1_loss(q_sa, target_y).item())
        writer.add_scalar(f"{tag_prefix}/td_loss", td_loss, step)
        for p, val in zip([50, 90, 99], np.percentile(td_abs, [50, 90, 99])):
            writer.add_scalar(f"{tag_prefix}/td_error_p{p}", float(val), step)

    # --- Q-value stats over legal actions only ---
    with torch.no_grad():
        q = q_values
        if mask is not None:
            q = q.masked_fill(~mask, float("nan"))
        q_np = q.detach().cpu().numpy()
        writer.add_scalar(f"{tag_prefix}/q_mean", np.nanmean(q_np), step)
        writer.add_scalar(f"{tag_prefix}/q_std",  np.nanstd(q_np),  step)
        writer.add_scalar(f"{tag_prefix}/q_min",  np.nanmin(q_np),  step)
        writer.add_scalar(f"{tag_prefix}/q_max",  np.nanmax(q_np),  step)

        # Fraction of illegal greedy actions (pre-mask)
        greedy = q_values.argmax(dim=1)  # indices ignoring mask
        if mask is not None:
            rows = torch.arange(q_values.size(0), device=q_values.device)
            frac_illegal = (~mask[rows, greedy]).float().mean().item()
            writer.add_scalar(f"{tag_prefix}/frac_illegal_greedy", frac_illegal, step)

    # --- Gradient norm (optionally also clip here) ---
    total_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(),
                                                grad_clip_max) if grad_clip_max else \
                 torch.linalg.vector_norm(
                     torch.stack([p.grad.detach().norm() for p in q_net.parameters()
                                  if p.grad is not None]), ord=2).item()
    writer.add_scalar(f"{tag_prefix}/grad_norm", float(total_norm), step)

    # --- Replay age distribution (if available) ---
    if replay is not None and getattr(replay, "sample", None):
        try:
            _, ages = replay.sample(64, with_age=True)  # returns (batch, ages)
            ages = np.asarray(ages, dtype=np.int64)
            writer.add_scalar(f"{tag_prefix}/replay_age_mean", ages.mean(), step)
            for p, val in zip([10, 50, 90], np.percentile(ages, [10, 50, 90])):
                writer.add_scalar(f"{tag_prefix}/replay_age_p{p}", float(val), step)
        except TypeError:
            pass  # buffer doesn't support with_age

# helper function to compute L2 norm of model parameters for debugging
def l2_params(m): 
    return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters()))

class VanillaDQNTrainer(Trainer):
    def __init__(self, 
                 qfunction : QFunction, 
                 *, 
                 buffer_size: int = 10000,
                 train_every: int = 1,
                 min_buffer_to_train: int = 1000,
                 batch_size: int = 32,
                 step_length_distribution: Dict[int, float] = {1: 1.0},
                 gamma: float = 0.99,
                 target_update_every: int = 100,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 learning_rate: float = 1e-3,
                 max_gradient_norm: float = 10.0,
                 writer: Optional[SummaryWriter] = None,
                 log_every: int = 100
                 ):  
        self.qfunction = qfunction

        # Clone and freeze target network
        self.target_qfunction = qfunction.clone()
        self.target_qfunction.eval()
        for p in self.target_qfunction.parameters():
            p.requires_grad = False

        self.buffer_size = buffer_size
        self.train_every = train_every
        self.min_buffer_to_train = min_buffer_to_train
        self.batch_size = batch_size
        self.step_length_distribution = step_length_distribution
        self._validate_step_length_distribution()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.target_update_every = target_update_every
        self.step_count = 0
        self.train_step_count = 0
        self.replay_buffer = TransitionReplayBuffer(buffer_size, qfunction.input_shape, qfunction.num_actions)
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.qfunction.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss(beta=1.0, reduction='mean')
        self.writer = writer
        self.log_every = log_every

        assert self.min_buffer_to_train <= self.buffer_size
        assert self.train_every > 0
        assert self.batch_size > 0
        assert self.gamma >= 0.0 and self.gamma <= 1.0
        assert self.learning_rate > 0.0
        assert self.target_update_every > 0
        assert self.batch_size <= self.min_buffer_to_train

    def _validate_step_length_distribution(self):
        total = sum(self.step_length_distribution.values())
        if not np.isclose(total, 1.0):
            raise ValueError("step_length_distribution values must sum to 1.0")
        for k in self.step_length_distribution.keys():
            if not isinstance(k, int) or k < 1:
                raise ValueError("step_length_distribution keys must be integers >= 1")

    def start_game(self) -> None:
        pass

    def end_game(self) -> None:
        pass

    def add_episode(self, episode: List[c4.Transition]) -> None:
        # Pick a step length according to the distribution
        lengths, probs = zip(*self.step_length_distribution.items())
        step_length = np.random.choice(lengths, p=probs)
        for i in range(len(episode) - step_length + 1):
            tr = episode[i]
            if step_length == 1:
                tr = c4.Transition(
                    s=tr.s,
                    a=tr.a,
                    r=tr.r,
                    s2=tr.s2,
                    mask=tr.mask,
                    mask2=tr.mask2,
                    done=tr.done,
                    info={'step_length': 1}
                )
            else:
                # Multi-step transition
                # Reward = sum of discounted rewards
                # s2, mask2, done come from last transition in the sequence
                r = tr.r
                g = 1.0
                for j in range(1, step_length):
                    g *= self.gamma
                    r += g * episode[i + j].r
                s2 = episode[i + step_length - 1].s2
                mask2 = episode[i + step_length - 1].mask2
                done = episode[i + step_length - 1].done

                tr = c4.Transition(
                    s=tr.s,
                    a=tr.a,
                    r=r,
                    s2=s2,
                    mask=tr.mask,
                    mask2=mask2,
                    done=done,
                    info={'step_length': step_length}
                )
            self._add_transition(tr)

    def _add_transition(self, tr: c4.Transition) -> None:
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
        
        # Update train step count
        self.train_step_count += 1

                        
        # Sample a batch
        s_batch, a_batch, r_batch, s2_batch, mask_batch, mask2_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        s_batch = torch.from_numpy(s_batch).float().to(self.qfunction.device_)
        a_batch = torch.from_numpy(a_batch).long().to(self.qfunction.device_)
        r_batch = torch.from_numpy(r_batch).float().to(self.qfunction.device_)
        s2_batch = torch.from_numpy(s2_batch).float().to(self.qfunction.device_)
        mask2_batch = torch.from_numpy(mask2_batch).float().to(self.qfunction.device_)
        done_batch = torch.from_numpy(done_batch).bool().to(self.qfunction.device_)
        
        # Compute Q(s,a)
        q_values = self.qfunction(s_batch)  # (B, num_actions)
        q_s_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute target Q values

        with torch.no_grad():
            q_next = self.target_qfunction(s2_batch)             # (B, A)
            if mask2_batch is not None:
                mask2 = mask2_batch.to(q_next.device, dtype=q_next.dtype)
                q_next = q_next.masked_fill(mask2 < 0.5, torch.finfo(q_next.dtype).min)

            max_q_s2, _ = q_next.max(dim=1)                      # (B,)
            target_q = r_batch + self.gamma * torch.where(
                done_batch, torch.zeros_like(max_q_s2), max_q_s2
            )

        # Compute loss
        loss = self.criterion(q_s_a, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qfunction.parameters(), self.max_gradient_norm)
        self.optimizer.step()

        # Update target network
        if self.train_step_count % self.target_update_every == 0:
            self.target_qfunction.load_state_dict(self.qfunction.state_dict())  

        # Debug logging: currently every step
        if self.writer is not None:
            # Debug metrics: online and target L2 norm, norm of diff
            with torch.no_grad():
                diff = torch.sqrt(sum(((p1.detach()-p2.detach())**2).sum()
                                    for p1,p2 in zip(self.qfunction.parameters(), self.target_qfunction.parameters())))
                self.writer.add_scalar("debug/param_norm_online", l2_params(self.qfunction).item(), self.train_step_count)
                self.writer.add_scalar("debug/param_norm_target", l2_params(self.target_qfunction).item(), self.train_step_count)
                self.writer.add_scalar("debug/param_l2_diff",     diff.item(), self.train_step_count)

            # Debug metrics: done and reward mean in batch, tick for target hard update
            self.writer.add_scalar("debug/done_mean",  done_batch.float().mean().item(), self.train_step_count)
            self.writer.add_scalar("debug/r_mean",     r_batch.mean().item(), self.train_step_count)
            self.writer.add_scalar("debug/target_hard_update", int(self.train_step_count % self.target_update_every == 0), self.train_step_count)



        # Logging
        if self.writer is not None and self.train_step_count % self.log_every == 0:
            log_dqn_metrics(
                step=self.train_step_count,
                q_sa=q_s_a,
                target_y=target_q,
                q_values=q_values,
                mask=None, 
                q_net=self.qfunction,
                writer=self.writer,
                tag_prefix="train",
                replay=self.replay_buffer
            )

        