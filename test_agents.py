import c4
import bbagents
import numpy as np
from typing import List



def drop(board: np.ndarray, player: int, col: int):
    assert player in [-1, 1]
    assert board.shape == (6,7)
    assert np.isin(board, [-1,0,1]).all()
    assert col in range(7)
    rtop:int = 0
    while rtop < 6 and board[rtop, col] == 0:
        rtop += 1
    assert rtop != 0
    board[rtop-1, col] = player

def drop_sequence(board: np.ndarray, start_player: int, moves: List[int]):
    player = start_player
    for m in moves:
        drop(board, player, m)
        player *= -1
    return player

def dump_board(board: np.ndarray):
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


def probe_bb_negamax_agent():
    try:
        agent = bbagents.NegamaxBBAgent(6, 7, 2, 4)
        while True:
            board:np.ndarray = np.zeros((6,7), dtype=np.int8)
            s = input("Input move sequence:  ")
            moves = [int(c) for c in s]
            player = drop_sequence(board, 1, moves)
            dump_board(board * player)
            me, opp = bbagents._to_bitboards(board)
            print(f"me: {me:016x}   opp: {opp:016x}")
            agent_board = np.zeros((2,6,7), dtype=np.float32)
            agent_board[0] = (board == player)
            agent_board[1] = (board == (player * -1))
            best_move = agent.act(agent_board, (board[0]==0).astype(np.bool))
            print(f"Best move: {best_move}")


    except KeyboardInterrupt:
        print("Terminating...")
        exit(0)


def main():
    probe_bb_negamax_agent()


if __name__ == "__main__":
    main()