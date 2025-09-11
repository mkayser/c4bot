import c4
import agents
from dqn import ConvNetQFunction, QFunction, VanillaDQNTrainer
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import utils
from torch.utils.tensorboard import SummaryWriter


class RecentScoreRingBuffer:
    def __init__(self, size: int = 100):
        self.size = size
        self.buffer: List[float] = []
        self.index = 0
        self.sum = 0.0

    def add(self, score: float) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append(score)
        else:
            self.sum -= self.buffer[self.index]
            self.sum += score
            self.buffer[self.index] = score
            self.index = (self.index + 1) % self.size

    def average_if_full(self) -> float:
        if len(self.buffer) < self.size:
            return 0.0
        return self.sum / self.size

def print_transition(tr: c4.Transition) -> None:
    print(f"STATE {tr.a} {tr.r} STATE2 {tr.mask} {tr.mask2}")


# Determine if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize environment
env = c4.initialize_env()
env.reset()  # Need to reset before querying spaces

# Initialize Q-function and trainer
writer = SummaryWriter()  # TensorBoard writer
qfunc = ConvNetQFunction(input_shape=env.observation_space(env.agents[0])['observation'].shape, 
                         num_actions=env.action_space(env.agents[0]).n,
                         device=device)

action_picker_epsilon = utils.linear_sched(start=1.0, end=0.1, steps=100000)
action_picker = agents.EpsilonGreedyPicker(epsilon=action_picker_epsilon, 
                                           rng=np.random.default_rng(42), 
                                           writer=writer, 
                                           writer_tag_prefix="a2/epsilon")

trainer = VanillaDQNTrainer(qfunction=qfunc, 
                            buffer_size=10000,
                            batch_size=128,
                            step_length_distribution={1: 0.5, 2: 0.3, 3: 0.2},
                            min_buffer_to_train=1000,
                            train_every=1,
                            learning_rate=1e-3,
                            target_update_every=500,
                            max_gradient_norm=5.0,
                            gamma=0.99,
                            writer=writer)


with agents.HtmlQLLogger("game_log.html", append=False, max_games_to_write=100, game_write_interval=100) as logger:
    a1 = agents.RandomAgent()
    a2 = agents.QAgent(qfunction=qfunc, action_picker=action_picker, logger=logger)


    num_games = 300000
    recent_scores = RecentScoreRingBuffer(size=500)
    log_winrate_every = 100

    for game_idx in range(num_games):
        #print(f"Starting game {game_idx+1}/{num_games}: ", end="")
        result: c4.C4GameRecord
        trans1: List[c4.Transition]
        trans2: List[c4.Transition]
        result, trans1, trans2 = c4.play_game(env, a1, a2, render=False, verbose=False)

        # Track recent win rate for a1
        a2_score = 1.0 if result.winner == c4.C4GameRoles.P1 else 0.0
        recent_scores.add(a2_score)

        # Print game result
        if (game_idx % log_winrate_every) == 0 and game_idx > 0:
            avg_score = recent_scores.average_if_full()
            writer.add_scalar("game/recent_avg_score_a1", avg_score, game_idx)
            print(f"\rGame {game_idx}/{num_games}  recent avg score a1: {avg_score:.3f}           ", end="")


        # Feed transitions to trainer
        trainer.add_episode(trans2)

