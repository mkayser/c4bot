from agents import Agent
import numpy as np
import ctypes as C



_lib = C.CDLL("./libnegamaxbb.so")
_best_move = _lib.best_move_no_log
_best_move.argtypes = (C.c_uint64, C.c_uint64, C.c_int)
_best_move.restype  = C.c_int

_SENTINEL_MASK = 0
for _col in range(7):
    _SENTINEL_MASK |= 1 << (_col * 7 + 6)


def _to_bitboards(board_state: np.ndarray) -> tuple[int,int]:
    """
    board_state: (6,7) with values {+1 (me), 0, -1 (opp)}. Row 0 = top.
    Bit layout: for column c, bits [c*7 + r] where r=0 is bottom row, r=6 sentinel.
    """
    h, w = board_state.shape
    assert (h, w) == (6, 7)
    me = opp = 0
    for c in range(7):
        for r_top in range(6):
            v = int(board_state[r_top, c])
            if not v: continue
            r = 5 - r_top                      # bottom = 0
            bit = 1 << (c * 7 + r)
            if v > 0: me  |= bit
            else:     opp |= bit
    # Sentinel bits must stay clear; a set sentinel would fabricate vertical wins.
    me  &= ~_SENTINEL_MASK
    opp &= ~_SENTINEL_MASK
    assert (me & _SENTINEL_MASK) == 0
    assert (opp & _SENTINEL_MASK) == 0
    return me, opp


def debug_dump_board(board: np.ndarray):
    assert board.shape == (6,7)
    print("x = current player")
    for r in range(6):
        for c in range(7):
            char = '.'
            if board[r,c] == 1:
                char = 'x'
            elif board[r,c] == -1:
                char = 'o'
            print(f"{char}", end="")
        print("")


class NegamaxBBAgent(Agent):
    def __init__(self, h:int, w:int, search_depth=4, connect_goal:int=4):
        assert h==6 and w==7, "NegamaxBBAgent assumes 6x7"
        assert connect_goal==4, "NegamaxBBAgent assumes connect-4"
        self.h, self.w, self.search_depth = h, w, search_depth

    def start_game(self, player_id): pass
    def end_game(self): pass

    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        # obs: (2,6,7) one-hot planes. board_state ∈ {-1,0,1} with +1 = me
        assert obs.shape == (2, self.h, self.w)
        board_state = (obs[0] - obs[1]).astype(np.int8)

        #print("About to move in BBAgent")
        #debug_dump_board(board_state)

        me, opp = _to_bitboards(board_state)

        #print(f"me: {me:016x}   opp: {opp:016x}")

        col = int(_best_move(C.c_uint64(me), C.c_uint64(opp), int(self.search_depth)))

        #print(f"Best move = {col}")

        # Safety: honor mask (engine should already avoid full cols)
        legal = np.flatnonzero(action_mask).tolist()
        assert col in legal, f"Illegal column for board state: {obs}"
        return col


class RandomizedNegamaxBBAgent(Agent):
    def __init__(self, h:int, 
                 w:int, 
                 search_depth=4, 
                 connect_goal:int=4, 
                 seed:int=42, 
                 prob_of_random_move:float = 0.5,
                 switch_to_deterministic_after = None):
        assert h==6 and w==7, "RandomizedNegamaxBBAgent assumes 6x7"
        assert connect_goal==4, "RandomizedNegamaxBBAgent assumes connect-4"
        self.h, self.w, self.search_depth, self.prob_of_random_move = h, w, search_depth, prob_of_random_move
        self.rng = np.random.default_rng(seed)
        self.switch_to_deterministic_after = switch_to_deterministic_after
        assert prob_of_random_move >= 0.0 and prob_of_random_move <= 1.0

    def start_game(self, player_id): 
        self.move_count = 0

    def end_game(self): pass

    def act_randomly(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        # choose uniformly among mask-positive moves
        legal_moves = np.flatnonzero(action_mask)
        chosen_move = self.rng.choice(legal_moves)
        return int(chosen_move)

    def act_negamax(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        # obs: (2,6,7) one-hot planes. board_state ∈ {-1,0,1} with +1 = me
        assert obs.shape == (2, self.h, self.w)
        board_state = (obs[0] - obs[1]).astype(np.int8)

        me, opp = _to_bitboards(board_state)
        col = int(_best_move(C.c_uint64(me), C.c_uint64(opp), int(self.search_depth)))

        # Safety: honor mask (engine should already avoid full cols)
        legal = np.flatnonzero(action_mask).tolist()
        assert col in legal, f"Illegal column for board state: {obs}"
        return col
    
    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        self.move_count += 1

        if (self.switch_to_deterministic_after is None) or self.move_count <= self.switch_to_deterministic_after:
            choose_randomly = (self.rng.random() <= self.prob_of_random_move)
        else:
            choose_randomly = False

        if choose_randomly:
            return self.act_randomly(obs, action_mask)
        else:
            return self.act_negamax(obs, action_mask)
