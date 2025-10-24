import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Protocol, Callable, TextIO
from logger import HtmlQLLogger
import minimax
import utils
import time
import sys


class Agent(Protocol):
    def start_game(self, player_id: str) -> None: ...
    def act(self, s:np.ndarray, action_mask: np.ndarray) -> int: ...
    def observe_final_state(self, s:np.ndarray) -> None: ...
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
    
    def observe_final_state(self, s:np.ndarray) -> None:
        pass
    
    def end_game(self) -> None:
        pass


# --- A trivial baseline agent ------------------------------------------------
class AlwaysPlayFixedColumnAgent(Agent):
    def __init__(self, column: int):
        self.column = column

    def start_game(self, player_id: str) -> None:
        pass
          
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        assert self.column < action_mask.size
        if action_mask[self.column]:
            return self.column
        else:
            # backoff: choose leftmost legal move
            legal_moves = np.flatnonzero(action_mask)
            assert legal_moves.size > 0
            return int(legal_moves[0])
    
    def observe_final_state(self, s:np.ndarray) -> None:
        pass
    
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
    
    def observe_final_state(self, s:np.ndarray) -> None:
        pass
    
    def end_game(self) -> None:
        pass


class ActionPicker(Protocol):
    def __call__(self, scores: np.ndarray, *, mask: np.ndarray, step: int) -> int: ...


class EpsilonGreedyPicker:
    def __init__(self, 
                 epsilon: float | Callable[[int], float], 
                 rng: np.random.Generator,
                 writer: Optional[utils.SummaryWriterLike] = None):
        self.epsilon = epsilon
        self.rng = rng
        self.writer = writer

    def _eps(self, step :int) -> float:
        e = self.epsilon(step) if callable(self.epsilon) else self.epsilon
        return float(np.clip(e, 0.0, 1.0))

    def __call__(self, scores, *, mask=None, step) -> int:
        scores = np.asarray(scores)
        legal_idx = np.flatnonzero(mask)
        eps = self._eps(step)
        if self.writer:
            self.writer.add_scalar("epsilon", eps, step)
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

    def observe_final_state(self, s:np.ndarray) -> None:
        pass
    


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

    def observe_final_state(self, s:np.ndarray) -> None:
        pass



class QAgent(Agent):
    def __init__(self, 
                 qfunction: Any, 
                 action_picker: ActionPicker,
                 logger: Optional[HtmlQLLogger] = None,
                 debug_stream: Optional[TextIO] = None
                 ):
        self.qfunction = qfunction
        self.action_picker = action_picker
        self.global_step = 0
        self.game_step = 0
        self.logger = logger
        self.debug_stream = debug_stream

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
        if self.debug_stream:
            print(f"--------\nObs=\n{(obs[0]-obs[1]).astype(np.int8)}\nMask={action_mask}\n", file=self.debug_stream)
            with np.printoptions(precision=3, floatmode='fixed', linewidth=np.inf):
                print(f"QScores={scores}\n", file=self.debug_stream)
            print(f"Chosen={chosen_move}\n", file=self.debug_stream)
        return int(chosen_move)
    
    def observe_final_state(self, s:np.ndarray) -> None:
        pass
    
    def end_game(self) -> None:
        if self.logger:
            self.logger.end_game()


import os
import pygame
import numpy as np

class HumanPygameAgent(Agent):
    def __init__(self, cell=80, margin=12):
        self.cell = cell
        self.margin = margin
        self.screen = None
        self.clock = None
        self.player_id = None
        self.cols = None
        self.rows = None
        self.bg = (20, 24, 35)
        self.grid = (40, 46, 60)
        self.p1 = (255, 200, 0)   # o
        self.p2 = (255, 70, 70)   # x
        self.ghost = (180, 180, 180)

    def start_game(self, player_id):
        self.player_id = player_id
        # (re)initialized lazily on first .act() when we know board size

    def end_game(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
        self.cols = None
        self.rows = None

    # ------------- minimal UI helpers -------------
    def _ensure_window(self, h, w):
        if self.screen is not None:
            return
        # If running headless, caller must set SDL_VIDEODRIVER appropriately.
        pygame.init()
        self.clock = pygame.time.Clock()
        width_px  = w * self.cell + 2 * self.margin
        height_px = (h + 1) * self.cell + 2 * self.margin  # +1 row for hover
        self.screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Connect-4 (Human)")

    def _draw(self, obs, hover_col=None, mask=None):
        ch, h, w = obs.shape
        self.screen.fill(self.bg)

        # board rect
        board_top = self.margin + self.cell  # leave one row on top for hover
        pygame.draw.rect(
            self.screen, self.grid,
            pygame.Rect(self.margin, board_top, w*self.cell, h*self.cell),
            border_radius=12
        )

        # cells
        for r in range(h):
            for c in range(w):
                cx = self.margin + c*self.cell + self.cell//2
                cy = board_top + r*self.cell + self.cell//2
                # empty hole
                color = (230, 235, 245)
                # piece?
                if obs[0, r, c] == 1:
                    color = self.p1
                elif obs[1, r, c] == 1:
                    color = self.p2
                pygame.draw.circle(self.screen, color, (cx, cy), int(self.cell*0.38))

        # hover indicator (top row)
        if hover_col is not None:
            cx = self.margin + hover_col*self.cell + self.cell//2
            cy = self.margin + self.cell//2
            valid = True if (mask is None or (0 <= hover_col < len(mask) and mask[hover_col])) else False
            pygame.draw.circle(self.screen, self.ghost if valid else (120,120,120), (cx, cy), int(self.cell*0.35), 3)

        pygame.display.flip()

    def _col_from_mouse(self, x):
        x_rel = x - self.margin
        if x_rel < 0:
            return None
        col = x_rel // self.cell
        return int(col)
    
    def _display_and_get_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray], wait_for_legal_action: bool) -> int:
        """
        Blocking: opens (or reuses) a tiny pygame window, shows board,
        and returns a column index chosen by the user (respecting action_mask).
        """
        ch, h, w = obs.shape
        self.cols, self.rows = w, h
        self._ensure_window(h, w)

        hover_col = None
        while True:
            # event pump
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # let caller handle this however they prefer
                    raise SystemExit("HumanPygameAgent window closed")
                elif event.type == pygame.MOUSEMOTION:
                    col = self._col_from_mouse(event.pos[0])
                    hover_col = col if (col is not None and 0 <= col < w) else None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if wait_for_legal_action:
                        col = self._col_from_mouse(event.pos[0])
                        if col is not None and 0 <= col < w and action_mask[col]:
                            self._draw(obs, hover_col=None, mask=action_mask)  # quick redraw
                            return col  # ✔ chosen legal action
                    else:
                        return -1

            # draw (with hover)
            self._draw(obs, hover_col=hover_col, mask=action_mask)
            self.clock.tick(60)

    # ------------- main API -------------
    def act(self, obs: np.ndarray, action_mask: np.ndarray):
        return self._display_and_get_action(obs, action_mask, wait_for_legal_action=True)

    def observe_final_state(self, s:np.ndarray) -> None:
        self._display_and_get_action(s, action_mask=None, wait_for_legal_action=False)
