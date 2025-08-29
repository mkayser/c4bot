import numpy as np


class C4Agent(ABC):
    """Minimal interface for a PettingZoo Connect 4 agent."""

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None):
        self.name = name or self.__class__.__name__
        self.rng = np.random.default_rng(seed)
        self.player_id: Optional[str] = None  # "player_0" or "player_1"

    def start_game(self, player_id: str):
        """Called once at the start of each game to (re)initialize agent state."""
        self.player_id = player_id

    @abstractmethod
    def act(
        self,
        obs: Any,                 # PettingZoo observation (usually Dict)
        legal_moves: List[int],   # computed from action_mask
        reward: float,            # reward since this agent last acted
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Optional[int]:
        """Return the chosen action (column 0..6). Return None if terminated/truncated."""
        raise NotImplementedError

    def end_game(self, final_rewards: Dict[str, float], info: Dict[str, Any]):
        """Optional: inspect outcome, update learning, etc."""
        pass


# --- A trivial baseline agent ------------------------------------------------
class RandomC4Agent(C4Agent):
    def act(self, obs, legal_moves, reward, terminated, truncated, info):
        return None if (terminated or truncated) else int(self.rng.choice(legal_moves))
