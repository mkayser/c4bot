class C4GameRoles():
    P0 = 'player_0'
    P1 = 'player_1'

class C4GameRecord():    
    def __init__(self, metadata = {}):
        self.moves = []
        self.metadata = metadata
        self.winner = None

    def add_move(self, move_id):
        self.moves.append(move_id)

    def set_winner(self, player_id):
        if player_id == C4GameRoles.P0 or player_id == C4GameRoles.P1:
            self.winner = player_id
        else:
            self.winner = 'draw'
        
