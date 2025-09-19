import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Protocol, Callable
from logger import HtmlQLLogger
import minimax


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


# --- Mostly random but plays forced wins -----
class RandomC3AgentWithForcedWins(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def start_game(self, player_id: str) -> None:
        pass

    def _is_winning_board(self, board: np.ndarray) -> bool:
        # Check horizontal, vertical, and diagonal for a win
        for r in range(6):
            for c in range(5):
                if np.all(board[0, r, c:c+3] == 1.0):
                    return True
        for c in range(7):
            for r in range(4):
                if np.all(board[0, r:r+3, c] == 1.0):
                    return True
        for r in range(4):
            for c in range(5):
                if np.all([board[0, r+i, c+i] == 1.0 for i in range(3)]):
                    return True
                if np.all([board[0, r+2-i, c+i] == 1.0 for i in range(3)]):
                    return True
        return False
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        # Check for immediate winning moves
        for move in np.flatnonzero(action_mask):
            temp_board = obs.copy()
            for r in range(5, -1, -1):
                if temp_board[0, r, move] == 0 and temp_board[1, r, move] == 0:
                    temp_board[0, r, move] = 1.0 
                    break
            if self._is_winning_board(temp_board):
                return int(move)
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


class NegamaxAgent(Agent):
    def __init__(self, h:int, w:int, search_depth = 4, connect_goal:int = 4):
        self.h = h
        self.w = w
        self.search_depth = search_depth
        self.connect_goal = connect_goal
        self.board_evaluator = minimax.SimpleEvaluator(h, w)

    def start_game(self, player_id):
        pass

    def end_game(self):
        pass

    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        assert obs.shape == (2, self.h, self.w)
        board_state = (obs[0] - obs[1]).astype(np.int8)
        c4env = minimax.C4Env(self.h, self.w, self.connect_goal, board_state)
        return minimax.negamax_best_action(c4env, self.search_depth, self.board_evaluator)
    
class HumanAgent(Agent):
    def __init__(self):
        pass

    def start_game(self, player_id):
        pass

    def end_game(self):
        pass

    def _display_board_ansi(self, obs: np.ndarray):
        ch,h,w = obs.shape
        print("\n")
        for r in range(h):
            row_chars = []
            for c in range(w):
                if obs[0,r,c] == 1:
                    row_chars.append("o")
                elif obs[1,r,c] == 1:
                    row_chars.append("x")
                else:
                    row_chars.append(".")
            print("".join(row_chars))
        print("\n")

    def _prompt_action(self, width):
        action_str = input(f"Enter an action [0 - {width-1}]: ")
        return int(action_str)

    def act(self, obs:np.ndarray, action_mask: np.ndarray):
        c,h,w = obs.shape
        self._display_board_ansi(obs)
        action = self._prompt_action(w)
        assert(action_mask[action])
        return action
        


class QAgent(Agent):
    def __init__(self, 
                 qfunction: Any, 
                 action_picker: ActionPicker,
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