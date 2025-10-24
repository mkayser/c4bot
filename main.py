import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import c4
import agents
import bbagents
from dqn import ConvNetQFunction, DeepConvNetQFunction, QFunction, VanillaDQNTrainer
from typing import Any, Dict, List, Tuple, Union, Callable, Literal, Optional
import numpy as np
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Process, Queue, Event
from multiprocessing.synchronize import Event as EventType
import queue
import time
from enum import Enum
import signal
from urllib.parse import urlparse
import os
import json


# Hydra config classes

@dataclass
class GameRunnerCfg:
    play_every: int
    pinned_player: str
    pinned_player_plays_first: Optional[bool]
    opponent_pool: List[str]
    dump_games_every: Optional[int]
    dump_games_location: Optional[str]
    log_every: int
    writer_prefix: str
    export_episodes: bool
    export_opponent_episodes: bool

@dataclass 
class RandomAgentCfg:
    rng_seed: int

@dataclass 
class HumanAgentCfg:
    pass

@dataclass 
class HumanPygameAgentCfg:
    pass

@dataclass 
class AlwaysPlayFixedColumnAgentCfg:
    column: int


@dataclass 
class NegamaxAgentCfg:
    h: int
    w: int
    search_depth: int

@dataclass 
class NegamaxBBAgentCfg:
    h: int
    w: int
    search_depth: int

@dataclass 
class RandomizedNegamaxBBAgentCfg:
    h: int
    w: int
    search_depth: int
    rng_seed: int
    prob_of_random_move: float

class QFunctionType(str, Enum):
    ConvNetQFunction="ConvNetQFunction"
    DeepConvNetQFunction="DeepConvNetQFunction"

@dataclass
class QFunctionCfg:
    input_shape: List[int]
    num_actions: int
    device: str
    load_from: Optional[str]
    type: str

class ActionPickerType(str, Enum):
    EpsilonGreedyPicker="EpsilonGreedyPicker"

@dataclass 
class EpsilonGreedyPickerCfg:
    epsilon: Dict[str, Any]
    rng_seed: int
    writer_prefix: str
    type: ActionPickerType = ActionPickerType.EpsilonGreedyPicker

# Later can make this a union 
ActionPickerCfg = EpsilonGreedyPickerCfg

@dataclass
class QAgentCfg:
    update_from_queue: bool
    html_log_file: Optional[str]
    html_log_max_games: Optional[int]
    html_log_game_write_interval: Optional[int]
    qfunction: QFunctionCfg
    action_picker: ActionPickerCfg
    debug_stream: Optional[str]

PlayerCfg = Union[HumanAgentCfg, HumanPygameAgentCfg, RandomAgentCfg, NegamaxAgentCfg, NegamaxBBAgentCfg, RandomizedNegamaxBBAgentCfg, QAgentCfg, AlwaysPlayFixedColumnAgentCfg]

@dataclass
class GamePlayLoopCfg:
    start_tick: int  # Starting tick for the game loop
    max_ticks: int  # Maximum ticks to run the game loop
    rng_seed: int
    game_runners: List[GameRunnerCfg]
    players: Dict[str, Any]

@dataclass
class LearnerCfg:
    start_tick: int
    max_ticks: int
    qfunction: QFunctionCfg
    step_lengths: Dict[int,float]
    replay_buffer_size: int
    replay_buffer_min_to_train: int
    max_ratio_of_train_steps_to_transitions: float
    max_idle_training_steps: int
    batch_size: int
    learning_rate: float
    train_every: int
    save_every: Optional[int]
    save_location: Optional[str]
    target_update_every: int
    update_player_every: int
    max_gradient_norm: float
    gamma: float



@dataclass
class LoggerCfg:
    writer_prefix: Optional[str]

@dataclass
class AppCfg:
    game_play_loop: GamePlayLoopCfg
    learner: LearnerCfg
    logger: LoggerCfg

####
# Loader functions

def normalize_player_configs(players_dc: DictConfig) -> DictConfig:
    schema = {
        "RandomAgent": RandomAgentCfg,
        "HumanPygameAgent": HumanPygameAgentCfg,
        "HumanAgent": HumanAgentCfg,
        "AlwaysPlayFixedColumnAgent": AlwaysPlayFixedColumnAgentCfg,
        "NegamaxAgent": NegamaxAgentCfg,
        "NegamaxBBAgent": NegamaxBBAgentCfg,
        "RandomizedNegamaxBBAgent": RandomizedNegamaxBBAgentCfg,
        "QAgent": QAgentCfg,
    }
    out = {}
    for name, node in players_dc.items():
        t = node.get("type")
        cls = schema.get(t)
        if not cls:
            raise ValueError(f"unknown player type {t!r} at players.{name}")
        raw = OmegaConf.to_container(node, resolve=False) or {}
        raw.pop("type", None)
        out[name] = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(raw))  # validates & fixes types
    return OmegaConf.create(out)

def load_qfunction(c: QFunctionCfg):
    if c.type == QFunctionType.ConvNetQFunction: 
        qfunction = ConvNetQFunction(c.input_shape, c.num_actions, c.device)
        if c.load_from is not None:
            state_dict = utils.load_model_state_dict(c.load_from, c.device)
            qfunction.load_state_dict(state_dict)
        return qfunction
    if c.type == QFunctionType.DeepConvNetQFunction:
        qfunction = DeepConvNetQFunction(c.input_shape, c.num_actions, c.device)
        if c.load_from is not None:
            state_dict = utils.load_model_state_dict(c.load_from, c.device)
            qfunction.load_state_dict(state_dict)
        return qfunction
    raise TypeError(f"Unsupported QFunctionCfg: {type(c).__name__}")

def load_epsilon_schedule(d: Dict[str, Any]):
    if d["type"] == "linear_schedule":
        return utils.linear_sched(float(d["start"]), float(d["end"]), int(d["steps"]))
    elif d["type"] == "spaced_half_sine_schedule":
        return utils.spaced_half_sine_sched(float(d["min_val"]), float(d["max_val"]), int(d["n_steps_curve"]), int(d["n_steps_flat"]))
    elif d["type"] == "fixed":
        return float(d["value"])
    raise TypeError(f"Unsupported epsilon schedule type: {d["type"]}")

def load_action_picker(c: ActionPickerCfg, writer: utils.SummaryWriterLike):
    if isinstance(c, EpsilonGreedyPickerCfg):
        prefixed_writer = writer.with_tag_prefix("action_picker")
        return agents.EpsilonGreedyPicker(load_epsilon_schedule(c.epsilon), np.random.default_rng(c.rng_seed), writer=prefixed_writer)
    raise NotImplementedError(f"{c} configures an action picker that is not implemented")

def load_player(c: PlayerCfg, writer: utils.SummaryWriterLike):
    if isinstance(c, RandomAgentCfg): 
        return agents.RandomAgent(c.rng_seed)
    if isinstance(c, HumanAgentCfg): 
        return agents.HumanAgent()
    if isinstance(c, HumanPygameAgentCfg): 
        return agents.HumanPygameAgent()
    if isinstance(c, AlwaysPlayFixedColumnAgentCfg): 
        return agents.AlwaysPlayFixedColumnAgent(c.column)
    if isinstance(c, NegamaxAgentCfg): 
        return agents.NegamaxAgent(c.h, c.w, c.search_depth)
    if isinstance(c, NegamaxBBAgentCfg): 
        return bbagents.NegamaxBBAgent(c.h, c.w, c.search_depth)
    if isinstance(c, RandomizedNegamaxBBAgentCfg): 
        return bbagents.RandomizedNegamaxBBAgent(c.h, c.w, c.search_depth, 4, c.rng_seed, c.prob_of_random_move)
    if isinstance(c, QAgentCfg): 
        if c.html_log_file:
            html_logger = agents.HtmlQLLogger(c.html_log_file, 
                                              max_games_to_write=c.html_log_max_games, 
                                              game_write_interval=c.html_log_game_write_interval)
            html_logger.__enter__()
        else:
            html_logger = None
        debug_stream = None
        if c.debug_stream is not None:
            assert c.debug_stream in ["stdout", "stderr"]
            debug_stream = sys.stdout if c.debug_stream=="stdout" else sys.stderr
        return agents.QAgent(load_qfunction(c.qfunction), load_action_picker(c.action_picker, writer), html_logger, debug_stream)
    raise NotImplementedError(f"{c} configures an agent that is not implemented")
    
def load_game_runner(c: GameRunnerCfg):
    return GameRunnerState(c, 
                           0, 
                           utils.RecentValuesRingBuffer(), 
                           utils.RecentValuesRingBuffer(), 
                           utils.RingBuffer(), 
                           utils.RecentValuesRingBuffer(), 0.0)

####

@dataclass 
class GameRunnerState:
    cfg: GameRunnerCfg
    num_games_played: int
    recent_scores: utils.RecentValuesRingBuffer
    recent_game_lengths: utils.RecentValuesRingBuffer
    recent_games: utils.RingBuffer
    recent_nonzero_transition_rewards: utils.RecentValuesRingBuffer
    total_reward: float


def install_child_sigint_ignorer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def game_play_loop(cfg: GamePlayLoopCfg, 
                   params_q: Queue, 
                   episodes_q: Queue, 
                   logs_q: Queue, 
                   stop_event: EventType):
    # instantiate env
    # create players
    # create game runners
    # initialize game tick, set max ticks
    # enter loop (while not stop event): 
    #   * Update weights from queue if present
    #   * tick game counter
    #   * iterate through game runners:
    #     * if time to run a game:
    #       * run a game
    #       * update stats and html log if necessary
    #       * export episodes to outqueue if necessary
    
    def play_one_game(gr: GameRunnerState, 
                      players: Dict[str, agents.Agent], 
                      episodes_q: Queue, 
                      rng: np.random.Generator,
                      writer: utils.SummaryWriterLike):
        
        # Pick players
        assert gr.cfg.pinned_player in players
        main_player_id = gr.cfg.pinned_player
        opponent_id = rng.choice(gr.cfg.opponent_pool)

        main_player = players[main_player_id]
        opponent = players[opponent_id]
        bool_array = np.array([False,True], dtype=np.bool)
        pinned_is_first = rng.choice(bool_array) if (gr.cfg.pinned_player_plays_first is None) else gr.cfg.pinned_player_plays_first

        # Play game
        result: c4.C4GameRecord
        episode: List[c4.Transition]
        opponent_episode: List[c4.Transition]
        score: float
        game_length: int
        game_record: Tuple[str, str, int, List[int]]

        if pinned_is_first:
            result, episode, opponent_episode = c4.play_game(env, main_player, opponent)
            score = 1.0 if result.winner == c4.C4GameRoles.P0 else 0.0
            #reward_sum = 0.5 * (1.0 + sum(item.r for item in episode))
            #assert score == reward_sum, f"Mismatched score: {score} expected but sum is {reward_sum} "
            game_length = float(result.game_length())
            game_record = [main_player_id, opponent_id, game_length, result.moves]
        else:
            result, opponent_episode, episode = c4.play_game(env, opponent, main_player)
            score = 1.0 if result.winner == c4.C4GameRoles.P1 else 0.0
            #reward_sum = 0.5 * (1.0 + sum(item.r for item in episode))
            #assert score == reward_sum, f"Mismatched score: {score} expected but sum is {reward_sum} "
            game_length = float(result.game_length())
            game_record = [opponent_id, main_player_id, game_length, result.moves]
        
        # Export episode
        if gr.cfg.export_episodes:
            try:
                episodes_q.put_nowait(episode)
            except queue.Full:
                print("Episodes queue full")

        # Export episode
        if gr.cfg.export_opponent_episodes:
            try:
                episodes_q.put_nowait(opponent_episode)
            except queue.Full:
                print("Episodes queue full")

        # Update stats
        gr.recent_scores.add(score)
        gr.recent_game_lengths.add(game_length)
        gr.recent_games.add(game_record)
        for tr in episode:
            if tr.r != 0.0:
                gr.recent_nonzero_transition_rewards.add(tr.r)
        gr.total_reward += sum(tr.r for tr in episode)
        gr.num_games_played += 1

        # Log result
        if (gr.num_games_played % gr.cfg.log_every) == 0:
            avg_score = gr.recent_scores.average_if_full()
            avg_game_length = gr.recent_game_lengths.average_if_full()
            avg_recent_nz_tr_rw = gr.recent_nonzero_transition_rewards.average_if_full()
            writer.add_scalar(f"{gr.cfg.writer_prefix}/recent_avg_score", avg_score, gr.num_games_played)
            writer.add_scalar(f"{gr.cfg.writer_prefix}/recent_avg_game_length", avg_game_length, gr.num_games_played)
            writer.add_scalar(f"{gr.cfg.writer_prefix}/recent_avg_nz_tr_rw", avg_recent_nz_tr_rw, gr.num_games_played)
            writer.add_scalar(f"{gr.cfg.writer_prefix}/total_reward", gr.total_reward, gr.num_games_played)
        
        if (gr.cfg.dump_games_every is not None) and (gr.cfg.dump_games_location is not None):
            if (gr.num_games_played % gr.cfg.dump_games_every) == 0:
                output_file_name = f"{gr.cfg.dump_games_location}/games.{gr.num_games_played:07d}.jsonl"
                os.makedirs(gr.cfg.dump_games_location, exist_ok=True)
                with open(output_file_name, "w") as fout:
                    game_records = gr.recent_games.get_all()
                    for game_record in game_records:
                        json.dump(game_record, fout)
                        fout.write("\n")
            


    try:
        # Ignore Ctrl+c
        install_child_sigint_ignorer()

        # Initialize environment
        env = c4.initialize_env()
        env.reset()  # Need to reset before querying spaces

        # Create SummaryWriterProxy
        writer: utils.SummaryWriterLike = utils.SummaryWriterProxy(logs_q)

        # Create players
        players: Dict[str, agents.Agent] = {}
        player_modules_to_update_from_queue: List[torch.nn.Module] = []
        for player_name, player_cfg in cfg.players.items():
            writer_prefix = f"players/{player_name}"
            players[player_name] = load_player(player_cfg, writer.with_tag_prefix(writer_prefix))
            if isinstance(player_cfg, QAgentCfg) and player_cfg.update_from_queue:
                assert isinstance(players[player_name], agents.QAgent)
                assert isinstance(players[player_name].qfunction, torch.nn.Module)
                player_modules_to_update_from_queue.append(players[player_name].qfunction)

        # Create game runners
        game_runners: List[GameRunnerState] = []
        for game_runner_cfg in cfg.game_runners:
            game_runners.append(load_game_runner(game_runner_cfg))

        # Initialize state
        max_ticks = cfg.max_ticks
        rng = np.random.default_rng(cfg.rng_seed)

        # Main loop
        game_tick = 0
        while (not stop_event.is_set()) and game_tick < max_ticks:
            # Drain param updates (keep only last)
            last_params = None
            try:
                while True:
                    last_params = params_q.get_nowait()
            except queue.Empty:
                pass

            # Update params if needed
            if last_params is not None:
                for player_module in player_modules_to_update_from_queue:
                    player_module.load_state_dict(last_params)

            # Increment game tick
            game_tick += 1

            # Play games if necessary
            n_games_played = 0
            
            for gr in game_runners:
                if game_tick % gr.cfg.play_every == 0:
                    play_one_game(gr, players, episodes_q, rng, writer)
                    n_games_played += 1
            
            # Idle briefly if nothing to do
            if last_params is None and n_games_played == 0:
                time.sleep(0.001)
    finally:
        # If we're done, cleanup queues we were producing to
        try:
            logs_q.cancel_join_thread()
            episodes_q.cancel_join_thread()
        except Exception:
            pass
    

def logger(cfg: LoggerCfg, logs_q: Queue, stop_event: EventType):

    # Ignore Ctrl+c
    install_child_sigint_ignorer()

    # Initialize
    writer = SummaryWriter() 

    # Main loop
    while not stop_event.is_set():
        # Drain updates
        updates = []
        max_drain = 1000
        try:
            for i in range(max_drain):
                updates.append(logs_q.get_nowait())
        except queue.Empty:
            pass

        # Submit entries to summarywriter 
        for method_name, args, kwargs in updates:
            if method_name == "add_scalar":
                writer.add_scalar(*args, **kwargs)
            else:
                raise NotImplementedError(f"Non-implemented logging function requested: {method_name}")

        # Idle briefly if nothing to do
        if len(updates) == 0:
            time.sleep(0.001)

def learner(cfg: LearnerCfg, 
            params_q: Queue, 
            episodes_q: Queue, 
            logs_q: Queue,
            stop_event: EventType):

    try:
        # Ignore Ctrl+c
        install_child_sigint_ignorer()

        # Create SummaryWriterProxy
        writer: utils.SummaryWriterLike = utils.SummaryWriterProxy(logs_q)

        # Initialize qfunction 
        qfunction = load_qfunction(cfg.qfunction)

        # Initialize model saver if needed
        saver = None
        if cfg.save_every is not None and cfg.save_location is not None:
            result = urlparse(cfg.save_location)
            assert result.scheme == ""
            saver = utils.LocalModelSaver(cfg.save_location)

        # Initialize trainer
        trainer = VanillaDQNTrainer(qfunction=qfunction, 
                                    buffer_size=cfg.replay_buffer_size,
                                    batch_size=cfg.batch_size,
                                    step_length_distribution=cfg.step_lengths,
                                    min_buffer_to_train=cfg.replay_buffer_min_to_train,
                                    train_every=cfg.train_every,
                                    save_every=cfg.save_every,
                                    model_saver=saver,
                                    learning_rate=cfg.learning_rate,
                                    target_update_every=cfg.target_update_every,
                                    max_gradient_norm=cfg.max_gradient_norm,
                                    gamma=cfg.gamma,
                                    start_tick = cfg.start_tick,
                                    writer=writer)

        # Assert values in range
        assert 1.0 <= cfg.max_ratio_of_train_steps_to_transitions 

        # Initialize tick counting variables
        n_transitions_till_player_update = cfg.update_player_every
        n_transitions = cfg.start_tick
        max_transitions = cfg.max_ticks
        max_ratio = cfg.max_ratio_of_train_steps_to_transitions
        max_idle_training_steps = cfg.max_idle_training_steps

        total_reward = 0.0
        write_debug_every = 100
        n_transitions_till_write_debug = 100

        # We set n_training_steps assuming no "backlog" of idle training steps
        n_training_steps = n_transitions * max_ratio


        # Main loop: 
        while not stop_event.is_set() and n_transitions < max_transitions:
            # Drain updates
            updates: List[List[c4.Transition]] = []
            max_drain = 50
            writer.add_scalar("debug/learner_queue_length", episodes_q.qsize(), n_transitions)
            try:
                for i in range(max_drain):
                    updates.append(episodes_q.get_nowait())
            except queue.Empty:
                pass

            # Process entries by pushing episodes into trainer
            for episode in updates:
                total_reward += sum(tr.r for tr in episode)
                trainer.add_episode(episode)
                # This is not 100% accurate because of multi-step transitions
                # But it doesn't matter, it's just for rough counting
                n_transitions += len(episode)
                n_training_steps += len(episode)
                n_transitions_till_player_update -= len(episode)
                n_transitions_till_write_debug -= len(episode)

                if n_transitions_till_write_debug <= 0:
                    writer.add_scalar("debug/learner_total_reward", total_reward, n_transitions)
                    while n_transitions_till_write_debug < 0:
                        n_transitions_till_write_debug += write_debug_every

                if n_transitions_till_player_update <= 0:
                    try:
                        assert isinstance(trainer.qfunction, torch.nn.Module)
                        params_q.put_nowait(trainer.qfunction.state_dict())
                    except queue.Full:
                        print("Params queue is full. Skipping.")
                    while n_transitions_till_player_update < 0:
                        n_transitions_till_player_update += cfg.update_player_every

            # If nothing else to do, do "idle training"
            if len(updates) == 0:
                if n_training_steps < n_transitions * max_ratio:
                    for i in range(max_idle_training_steps):
                        if n_training_steps < n_transitions * max_ratio:
                            trainer.train()
                            n_training_steps += 1

                else:
                    # If no idle training left, sleep
                    time.sleep(0.001)
    finally:
        # If we're done, cleanup queues we were producing to
        try:
            logs_q.cancel_join_thread()
            params_q.cancel_join_thread()
        except Exception:
            pass




@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Create typed config object
    typed_cfg = OmegaConf.merge(OmegaConf.structured(AppCfg), cfg)
    typed_cfg.game_play_loop.players = normalize_player_configs(typed_cfg.game_play_loop.players)
    appcfg: AppCfg = OmegaConf.to_object(typed_cfg)

    # Set up three processes (learner, player, logger) and three queues (params, episodes, logs)
    # Start all processes
    # Join and send graceful termination signal via Event if ctrl+c captured
    episodes, params, logs = Queue(), Queue(), Queue()
    stop = Event()
    A = Process(name="learner", target=learner, args=(appcfg.learner, params, episodes, logs, stop))
    B = Process(name="game_play_loop", target=game_play_loop, args=(appcfg.game_play_loop, params, episodes, logs, stop))
    C = Process(name="logger", target=logger, args=(appcfg.logger, logs, stop))

    procs = [A, B, C]
    finite_runtime_procs = [A, B]
    finished_procs = []

    for p in procs: p.start()
    try:
        while len(finished_procs) < len(finite_runtime_procs):
            finished_procs = [p for p in procs if p.exitcode is not None]
            # if any died abnormally, stop the rest
            for p in finished_procs:
                if p.exitcode != 0:
                    print(f"Process '{p._start_method}' has exited with errors.")
                raise Exception("Child process terminated with errors.")
            else:
                time.sleep(0.1)
    finally:
        stop.set()
        print("Shutting down...")
        for p in procs: p.join(timeout=5)
        # If any are still alive, terminate hard
        for p in procs:
            if p.is_alive(): 
                print(f"Timeout expired for process '{p.name}'. Killing...")
                p.terminate()





import cProfile, pstats, sys

# Profiler initialization
profiler = cProfile.Profile()
profiler.enable()


try:
    main()  
except KeyboardInterrupt:
    print("Interrupted, writing profile...")
finally:
    profiler.disable()
    profiler.dump_stats("profile.out")