import c4
import agents
from dqn import ConvNetQFunction, QFunction, VanillaDQNTrainer
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import utils
from torch.utils.tensorboard import SummaryWriter


class RecentValuesRingBuffer:
    def __init__(self, size: int = 100):
        self.size = size
        self.buffer: List[float] = []
        self.index = 0
        self.sum = 0.0

    def add(self, score: float) -> None:
        if len(self.buffer) < self.size:
            self.buffer.append(score)
            self.sum += score
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


def main():
    # Determine if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Initialize environment
    env = c4.initialize_env()
    env.reset()  # Need to reset before querying spaces

    # Initialize Q-function
    writer = SummaryWriter()  # TensorBoard writer
    qfunc = ConvNetQFunction(input_shape=env.observation_space(env.agents[0])['observation'].shape, 
                            num_actions=env.action_space(env.agents[0]).n,
                            device=device)

    # Create action picker with epsilon schedule
    action_picker_epsilon = utils.linear_sched(start=1.0, end=0.1, steps=70000)
    action_picker = agents.EpsilonGreedyPicker(epsilon=action_picker_epsilon, 
                                            rng=np.random.default_rng(42), 
                                            writer=writer, 
                                            writer_tag_prefix="a2/epsilon")

    # Create purely greedy action picker for evaluation
    greedy_action_picker = agents.EpsilonGreedyPicker(epsilon=0.0, 
                                                    rng=np.random.default_rng(43))

    # Initialize trainer
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



    # Disposable code for playing as human against an agent
    #a1 = agents.NegamaxAgent(6,7,2)
    #a2 = agents.HumanAgent()
    #record: c4.C4GameRecord
    #record, trans1, trans2 = c4.play_game(env, a1, a2, render=False, verbose=False)
    #print(f"Winner: {record.winner}")
    #print(f"Last move: {record.moves[-1]}")
    #return

    with agents.HtmlQLLogger("game_log_greedy.html", append=False, max_games_to_write=1000, game_write_interval=100) as logger:
        #a1 = agents.RandomAgent()
        a1 = agents.NegamaxAgent(6,7,2)
        a2 = agents.QAgent(qfunction=qfunc, action_picker=action_picker, logger=None)
        a2_greedy = agents.QAgent(qfunction=qfunc, action_picker=greedy_action_picker, logger=logger)

        num_games = 100000
        recent_scores = RecentValuesRingBuffer(size=500)
        recent_scores_greedy = RecentValuesRingBuffer(size=100)
        recent_game_lengths = RecentValuesRingBuffer(size=100)
        log_gamestats_every = 100
        play_greedy_every = 5

        for game_idx in range(num_games):
            #print(f"Starting game {game_idx+1}/{num_games}: ", end="")
            result: c4.C4GameRecord
            trans1: List[c4.Transition]
            trans2: List[c4.Transition]
            result, trans1, trans2 = c4.play_game(env, a1, a2, render=False, verbose=False)

            # Track recent win rate for a1
            a2_score = 1.0 if result.winner == c4.C4GameRoles.P1 else 0.0
            recent_scores.add(a2_score)
            recent_game_lengths.add(float(result.game_length()))

            # Play game with greedy a2 every log_winrate_every games
            if (game_idx % play_greedy_every) == 0 and game_idx > 0:
                result_greedy, _, _ = c4.play_game(env, a1, a2_greedy, render=False, verbose=False)
                a2_score_greedy = 1.0 if result_greedy.winner == c4.C4GameRoles.P1 else 0.0
                recent_scores_greedy.add(a2_score_greedy)
                avg_score_greedy = recent_scores_greedy.average_if_full()
                writer.add_scalar("game/recent_avg_score_a2_greedy", avg_score_greedy, game_idx)

            # Print game result
            if (game_idx % log_gamestats_every) == 0 and game_idx > 0:
                avg_score = recent_scores.average_if_full()
                avg_game_length = recent_game_lengths.average_if_full()
                writer.add_scalar("game/recent_avg_score_a1", avg_score, game_idx)
                writer.add_scalar("game/recent_avg_game_length", avg_game_length, game_idx)
                print(f"\rGame {game_idx}/{num_games}  recent avg score a1: {avg_score:.3f}      recent avg gamelen: {avg_game_length:.3f}        ", end="")


            # Feed transitions to trainer
            trainer.add_episode(trans2)


import cProfile, pstats, sys

profiler = cProfile.Profile()
profiler.enable()
try:
    main()  # your training loop
except KeyboardInterrupt:
    print("Interrupted, writing profile...")
finally:
    profiler.disable()
    profiler.dump_stats("profile.out")