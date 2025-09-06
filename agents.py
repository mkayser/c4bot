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


class ActionPickers():
    def epsilon_greedy(epsilon: float, rng: np.random.Generator):
        return lambda scores: (
            int(rng.choice(np.flatnonzero(scores == np.max(scores))))
            if rng.random() > epsilon
            else int(rng.choice(len(scores)))
        )
    
    def softmax(temperature: float, rng: np.random.Generator):
        return lambda scores: (
            int(rng.choice(len(scores), p=np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))))
        )

class QAgent(Agent):
    def __init__(self, qfunction: Any, action_picker: Any):
        self.qfunction = qfunction
        self.action_picker = action_picker

    def start_game(self, player_id: str) -> None:
        pass
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        scores = self.qfunction(obs)
        masked_scores = np.where(action_mask, scores, -np.inf)
        chosen_move = self.action_picker(masked_scores)
        return int(chosen_move)
    
    def end_game(self) -> None:
        pass