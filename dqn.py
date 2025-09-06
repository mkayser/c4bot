import numpy as np
import c4
from typing import Protocol, Optional, NamedTuple, Dict, List, Any


#class Observation(NamedTuple):
#    s: np.ndarray
#    mask: Optional[np.ndarray] = None
#    info: Optional[Any] = None

class Trainer(Protocol):
    def start_game(self) -> None: ...
    def end_game(self) -> None: ...
    def add_transition(self, tr: c4.Transition) -> None: ...
    def maybe_train(self) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class DQNVariants():
    DQN = 'DQN'
    DoubleDQN = 'DoubleDQN'
    PrioritizedReplayDQN = 'PrioritizedReplayDQN'
    CategoricalDQN = 'CategoricalDQN'
    RainbowDQN = 'RainbowDQN'


class DQNTrainer(Trainer):
    def __init__(self, qfunction, action_picker, variant: str = DQNVariants.DQN):
        self.qfunction = qfunction
        self.action_picker = action_picker
        self.variant = variant

    def start_game(self) -> None:
        pass

    def end_game(self) -> None:
        pass

    def add_transition(self, tr: c4.Transition) -> None:
        pass  # TODO    

    def maybe_train(self) -> None:
        pass

        