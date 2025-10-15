from agents import Agent
import numpy as np
import ctypes as C



_lib = C.CDLL("./libnegamaxbb.so")
_best_move = _lib.best_move
_best_move.argtypes = (C.c_uint64, C.c_uint64, C.c_int)
_best_move.restype  = C.c_int

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
    return me, opp

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

        me, opp = _to_bitboards(board_state)
        col = int(_best_move(C.c_uint64(me), C.c_uint64(opp), int(self.search_depth)))

        # Safety: honor mask (engine should already avoid full cols)
        legal = np.flatnonzero(action_mask).tolist()
        if col not in legal:
            col = legal[0] if legal else -1
        return col
