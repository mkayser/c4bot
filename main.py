import c4
import agents
from dqn import ConvNetQFunction, QFunction, VanillaDQNTrainer
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch


def print_transition(tr: c4.Transition) -> None:
    print(f"STATE {tr.a} {tr.r} STATE2 {tr.mask} {tr.mask2}")


# Determine if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize environment
env = c4.initialize_env()
env.reset()  # Need to reset before querying spaces

# Initialize Q-function and trainer
qfunc = ConvNetQFunction(input_shape=env.observation_space(env.agents[0])['observation'].shape, 
                         num_actions=env.action_space(env.agents[0]).n,
                         device=device)
action_picker = agents.make_epsilon_greedy(epsilon=0.1, rng=np.random.default_rng(42))
trainer = VanillaDQNTrainer(qfunction=qfunc, 
                            buffer_size=10000,
                            batch_size=32,
                            min_buffer_to_train=1000,
                            train_every=1)

a0 = agents.RandomAgent()
a1 = agents.QAgent(qfunction=qfunc, action_picker=action_picker)


num_games = 1000
for game_idx in range(num_games):
    print(f"Starting game {game_idx+1}/{num_games}: ", end="")
    result: c4.C4GameRecord
    trans1: c4.Transition
    trans2: c4.Transition
    result, trans1, trans2 = c4.play_game(env, a0, a1, render=False, verbose=False)
    print(f"Winner: {result.winner}")
    # Feed transitions to trainer
    for tr in trans2:
        assert isinstance(tr, c4.Transition)
        trainer.add_transition(tr)

