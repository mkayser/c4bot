import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Protocol


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


def make_epsilon_greedy(epsilon: float, rng: np.random.Generator):
    def _epsilon_greedy(scores, *, mask=None) -> int:
        scores = np.asarray(scores)
        legal_idx = np.flatnonzero(mask)
        if not legal_idx.size:
            raise ValueError("No legal actions available.")
        if rng.random() < epsilon:
            return int(rng.choice(legal_idx))
        legal_scores = scores[legal_idx]
        best = np.flatnonzero(legal_scores == legal_scores.max())
        return int(legal_idx[rng.choice(best)])
    return _epsilon_greedy


class QAgent(Agent):
    def __init__(self, qfunction: Any, action_picker: Any):
        self.qfunction = qfunction
        self.action_picker = action_picker

    def start_game(self, player_id: str) -> None:
        pass
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        scores : np.ndarray = self.qfunction.scores(obs)
        chosen_move = self.action_picker(scores, mask=action_mask)
        return int(chosen_move)
    
    def end_game(self) -> None:
        pass