import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Protocol, Callable
from logger import HtmlQLLogger


class Agent(Protocol):
    def start_game(self, player_id: str) -> None: ...
    def act(self, s:np.ndarray, action_mask: np.ndarray) -> int: ...
    # No training inside the agent; pure selector.
    def end_game(self) -> None: ...
    def name(self) -> str:
        return self.__class__.__name__


# --- A trivial baseline agent ------------------------------------------------
class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def start_game(self, player_id: str) -> None:
        pass
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        # choose uniformly among mask-positive moves
        legal_moves = np.flatnonzero(action_mask)
        chosen_move = self.rng.choice(legal_moves)
        return int(chosen_move)
    
    def end_game(self) -> None:
        pass


class ActionPicker(Protocol):
    def __call__(self, scores: np.ndarray, *, mask: np.ndarray, step: int) -> int: ...


class EpsilonGreedyPicker:
    def __init__(self, 
                 epsilon: float | Callable[[int], float], 
                 rng: np.random.Generator,
                 writer: Optional[Any] = None,
                 writer_tag_prefix: str = "epsilon_greedy"):
        self.epsilon = epsilon
        self.rng = rng
        self.writer = writer
        self.writer_tag_prefix = writer_tag_prefix

    def _eps(self, step :int) -> float:
        e = self.epsilon(step) if callable(self.epsilon) else self.epsilon
        return float(np.clip(e, 0.0, 1.0))

    def __call__(self, scores, *, mask=None, step) -> int:
        scores = np.asarray(scores)
        legal_idx = np.flatnonzero(mask)
        eps = self._eps(step)
        if self.writer:
            self.writer.add_scalar(self.writer_tag_prefix + "/epsilon", eps, step)
        if not legal_idx.size:
            raise ValueError("No legal actions available.")
        if self.rng.random() < eps:
            return int(self.rng.choice(legal_idx))
        legal_scores = scores[legal_idx]
        best = np.flatnonzero(legal_scores == legal_scores.max())
        return int(legal_idx[self.rng.choice(best)])




class QAgent(Agent):
    def __init__(self, 
                 qfunction: Any, 
                 action_picker: Any,
                 logger: Optional[HtmlQLLogger] = None
                 ):
        self.qfunction = qfunction
        self.action_picker = action_picker
        self.global_step = 0
        self.game_step = 0
        self.logger = logger

    def start_game(self, player_id: str) -> None:
        if self.logger:
            self.logger.start_game()
            self.game_step = 0
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        self.game_step += 1
        self.global_step += 1
        scores : np.ndarray = self.qfunction.scores(obs)
        chosen_move = self.action_picker(scores, mask=action_mask, step=self.global_step)
        if self.logger:
            self.logger.add_row(self.game_step, scores, action_mask, obs, chosen_move)
        return int(chosen_move)
    
    def end_game(self) -> None:
        if self.logger:
            self.logger.end_game()