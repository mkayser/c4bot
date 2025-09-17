import numpy as np
from typing import Callable, Optional, Tuple, Iterable
from typing import Protocol



# Reimplement a minimalist C4 environment. TODO: integrate with gym/pettingzoo

class C4EnvState:
    P0_turn = 0
    P1_turn = 1
    Draw = 2
    P0_won = 3
    P1_won = 4

def convert_pettingzoo_board_to_c4env(board: np.ndarray) -> np.ndarray:
    # PettingZoo uses 0=empty, 1=player0, 2=player1
    # We want -1=player1, 0=empty, +1=player0
    return np.where(board == 1, 1, np.where(board == 2, -1, 0)).astype(np.int8)

class C4Env:
    def __init__(self, height=6, width=7, connect=4, initial_board: Optional[np.ndarray] = None, p0_starts: bool = True):
        self.h = height
        self.w = width
        self.connect = connect
        if initial_board is not None:
            assert initial_board.shape == (self.h, self.w)
            self.board = initial_board.copy()
            assert np.all(np.isin(self.board, [-1, 0, 1])), "Board values must be -1, 0, or +1"
        else:
            self.board = np.zeros((self.h, self.w), dtype=np.int8)  # h rows, w columns
        self.all_moves = []
        self.state = C4EnvState.P0_turn if p0_starts else C4EnvState.P1_turn

    def reset(self):
        self.board.fill(0)
        self.all_moves = []
        self.state = C4EnvState.P0_turn
        return self._get_obs()
    
    def is_done(self):
        return self.state in (C4EnvState.Draw, C4EnvState.P0_won, C4EnvState.P1_won)
    
    def step(self, action: int):
        # Validate action
        if self.is_done():
            raise ValueError("Game is over. Please reset the environment.")
        
        if action < 0 or action >= self.w or not (self.board[0, action] == 0):
            raise ValueError(f"Illegal action: {action}")

        # Determine current player
        assert self.state in (C4EnvState.P0_turn, C4EnvState.P1_turn)
        current_player = 0 if self.state == C4EnvState.P0_turn else 1
        current_piece = 1 if self.state == C4EnvState.P0_turn else -1

        # Place piece, record move
        row = self._get_next_open_row(action)
        self.board[row, action] = current_piece
        self.all_moves.append((current_player, action, row))

        # Advance state
        if self._check_winner(row, action):
            #print(f"SETTING WINNER = {current_player}")
            #print(self.board.astype(np.int8))
            self.state = C4EnvState.P0_won if current_player == 0 else C4EnvState.P1_won
        elif self._check_draw():
            self.state = C4EnvState.Draw
        else:
            self.state = C4EnvState.P1_turn if current_player == 0 else C4EnvState.P0_turn

    def rewind_one_step(self):
        # Ensure there are moves to undo
        if not self.all_moves:
            raise ValueError("No moves to rewind.")

        # Pop last move
        last_player, last_action, last_row = self.all_moves.pop()

        # Remove piece from board
        self.board[last_row, last_action] = 0

        # Restore state
        state_before_rollback = self.state
        self.state = C4EnvState.P0_turn if last_player == 0 else C4EnvState.P1_turn
        #print(f"State was {state_before_rollback}, player of popped move was {last_player}, state after rollback is {self.state}")

    def get_state(self):
        return self.state
    
    def _get_next_open_row(self, col: int) -> int:
        for r in range(self.h-1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise ValueError(f"Column {col} is full. Checked an illegal move.")
    
    def _check_draw(self) -> bool:
        return np.all(self.board != 0)

    def _check_winner(self, row: int, col: int) -> bool:
        piece = self.board[row, col]
        for rstep, cstep in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1
            for direction in [1, -1]:
                r, c = row, col
                while True:
                    r += direction * rstep
                    c += direction * cstep
                    if 0 <= r < self.h and 0 <= c < self.w and self.board[r, c] == piece:
                        count += 1
                        if count >= self.connect:
                            return True
                    else:
                        break
        return False


    def get_legal_moves(self):
        return [c for c in range(self.w) if self.board[0, c] == 0]        


class EvaluationFunction(Protocol):
    def __call__(self, c4env: C4Env) -> float: ...

class SimpleEvaluator:
    class SequenceElement:
        SAME = 1
        EMPTY = 0
        OPPONENT = -1

    def __init__(self):
        SAME, EMPTY, OPPONENT = self.SequenceElement.SAME, self.SequenceElement.EMPTY, self.SequenceElement.OPPONENT
        self.sequences = [
            [EMPTY, SAME, SAME, SAME],  # open 3
            [SAME, SAME, SAME, EMPTY],  # open 3
            [SAME, SAME, EMPTY, SAME],   # 3 with gap
            [SAME, EMPTY, SAME, SAME],   # 3 with gap
            [EMPTY, SAME, SAME],         # 2 with gap
            [SAME, EMPTY, SAME],         # 2 with gap
            [SAME, SAME, EMPTY],         # 2 with gap
        ]

        self.directions = [
            (1, 0),  # vertical
            (0, 1),  # horizontal
            (1, 1),  # diagonal /
            (1, -1), # diagonal \
        ]

    def _count_sequences(self, board: np.ndarray, player: int) -> int:
        count = 0
        h, w = board.shape
        for r in range(h):
            for c in range(w):
                for dr, dc in self.directions:
                    for seq in self.sequences:
                        if self._matches_sequence(board, r, c, dr, dc, seq, player):
                            count += 1
        return count
    
    def _matches_sequence(self, board: np.ndarray, r: int, c: int, dr: int, dc: int, seq: list[int], player: int) -> bool:
        h, w = board.shape
        for elem in seq:
            if not (0 <= r < h and 0 <= c < w):
                return False
            cell = board[r, c]
            if elem == self.SequenceElement.SAME and cell != player:
                return False
            elif elem == self.SequenceElement.OPPONENT and cell != -player:
                return False
            elif elem == self.SequenceElement.EMPTY and cell != 0:
                return False
            r += dr
            c += dc
        return True
    
    def _raw_score(self, board: np.ndarray) -> float:
        p0_count = self._count_sequences(board, player=1)
        p1_count = self._count_sequences(board, player=-1)
        return float(p0_count - p1_count)
    
    def _clamped_scaled_score(self, board: np.ndarray) -> float:
        # Scale raw score to [-.9, +.9] range with clamping
        # Note that scale is arbitrary
        # We just want something monotonic and bounded away from -1/+1
        raw_score = self._raw_score(board)
        max_abs = 20.0
        clamped = max(-max_abs, min(max_abs, raw_score))
        return 0.9 * (clamped / max_abs)

    def __call__(self, c4env: C4Env) -> float:
        state = c4env.get_state()
        if state == C4EnvState.P0_won:
            return 1.0
        elif state == C4EnvState.P1_won:
            return -1.0
        elif state == C4EnvState.Draw:
            return 0.0
        else:
            return self._clamped_scaled_score(c4env.board)

# Negamax action picker with alpha-beta pruning
INF = float("inf")

SPACE = "    "
LOG_DEPTH=0

def negamax_best_action(
    c4env: C4Env,
    depth: int,
    evaluate: Callable[[object], float],
) -> Optional[int]:

    #print("Logging negamax...")
    def negamax(c4env: C4Env, 
                d: int, 
                alpha: float, 
                beta: float, 
                color: int) -> Tuple[float, Optional[int]]:
        global SPACE
        global LOG_DEPTH

        if d == 0 or c4env.is_done():
            #print(f"{SPACE*LOG_DEPTH} Color: {color} DONE at depth={d}")
            return color * evaluate(c4env), None

        best_score = -INF
        best_move: Optional[int] = None

        legal_moves = c4env.get_legal_moves()
        #print(f"{SPACE*LOG_DEPTH} Color: {color} Moves: {legal_moves}")
        # This should never happen since we check is_done() above
        assert len(legal_moves) > 0, "No legal moves available"
        
        for a in legal_moves:
            #print(f"{SPACE*LOG_DEPTH} Player={c4env.get_state()} Action={a} {{")
            c4env.step(a)
            LOG_DEPTH += 1
            # Flip window and sign for the opponent
            score, _ = negamax(c4env, d - 1, -beta, -alpha, -color)
            LOG_DEPTH -= 1
            #print(f"{SPACE*LOG_DEPTH} score of above path: {-score:.4f}}}")
            c4env.rewind_one_step()
            score = -score

            if score > best_score:
                best_score = score
                best_move = a

            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                #print(f"{SPACE*LOG_DEPTH} SKIPPING REMAINING CHILDREN BECAUSE {alpha:.4f} >= {beta:.4f}}}")
                break  # beta cutoff
        return best_score, best_move


    # color = +1 if Player0 to move, -1 if Player1 to move
    root_color = +1 if c4env.get_state() == C4EnvState.P0_turn else -1
    root_alpha, root_beta = -INF, INF
    best_score = -INF
    best_action: Optional[int] = None

    best_score, best_action = negamax(c4env, depth, root_alpha, root_beta, root_color)
    assert best_action is not None
    return best_action