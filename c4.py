#from pettingzoo.classic import connect_four_v3
import c4env
from supersuit import dtype_v0, normalize_obs_v0, observation_lambda_v0
from gymnasium import spaces
import numpy as np
import utils
from agents import Agent
from typing import Optional, NamedTuple, Dict, Tuple, Any, List, Union


class IncompleteTransition(NamedTuple):
    s: np.ndarray
    a: int
    mask: np.ndarray

# Transition tuple for experience replay
class Transition(NamedTuple):
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    mask: np.ndarray
    mask2: np.ndarray
    done: bool
    info: Dict = {}  # metadata for logging/debugging


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

    def game_length(self):
        return len(self.moves)
        



### Supersuit stuff for properly presenting the game state
def change_obs(obs, obs_space, _param):
    # obs is a dict; cast just the board tensor to float32
    obs = dict(obs)
    obs["observation"] = obs["observation"].astype(np.float32, copy=False)
    return obs

def change_space(old_space, _param):
    # Rebuild the Dict space but set the board Box to float32
    assert isinstance(old_space, spaces.Dict)
    board: spaces.Box = old_space["observation"]
    mask: spaces.Box  = old_space["action_mask"]
    new_board = spaces.Box(low=board.low, high=board.high, shape=board.shape, dtype=np.float32)
    return spaces.Dict({"observation": new_board, "action_mask": mask})
### End supersuit stuff


def initialize_env():
    env = c4env.env(render_mode="ansi")  # text render, no pygame window
    env = observation_lambda_v0(env, change_obs, change_space)

    print("render_mode seen by wrapper:", getattr(env, "render_mode", None))
    try:
        print("base render_mode:", env.unwrapped.render_mode)
        print("supported modes:", getattr(env.unwrapped, "metadata", {}).get("render_modes"))
    except Exception:
        pass
    return env

def c4_ansi_from_obs(obs) -> str:
    arr = obs["observation"]
    if arr.shape == (2, 6, 7):
        me, opp = arr[0], arr[1]
    elif arr.shape == (6, 7, 2):
        me, opp = arr[...,0], arr[...,1]
    else:
        raise ValueError(f"Unexpected obs shape {arr.shape}")

    grid = me.astype(np.int8) - opp.astype(np.int8)  # +1 me, -1 opp
    sym = {0: ".", 1: "X", -1: "O"}
    lines = []
    for r in range(6):  # show top row first
        lines.append("| " + " ".join(sym[int(grid[r, c])] for c in range(7)) + " |")
    return "\n".join(lines + ["  0 1 2 3 4 5 6"])

def get_normalized_state(last) -> tuple[np.ndarray, np.ndarray, float, bool, bool, Dict]:
    obs, reward, terminated, truncated, info = last
    assert isinstance(obs, dict) and "action_mask" in obs, "unexpected obs format"
    state = obs["observation"]
    action_mask = obs["action_mask"]
    return state, action_mask, reward, terminated, truncated, info

def play_game(
    env,
    agent0: Agent,
    agent1: Agent,
    *,
    render: bool = False,
    verbose: bool = False,
) -> Tuple[Any]:
    """
    Runs one game in the given PettingZoo env (already constructed; we just reset it).
    Returns a record of the game, as well as a list of Transition's from each agent perspective.
    Agents must implement the Agent interface.
    """

    env.reset()

    if verbose:
        print(f"=== New game: {agent0.name} (player_0) vs {agent1.name} (player_1) ===")
        print(c4_ansi_from_obs(env.observe(env.agent_selection)))
        print()

    # Map PettingZoo agent ids to our agents
    P0 = C4GameRoles.P0
    P1 = C4GameRoles.P1
    
    id_to_agent = {P0: agent0, P1: agent1}

    # Start recording
    record = C4GameRecord({P0: agent0.name, P1: agent1.name})

    # Notify agents of game start
    agent0.start_game(P0)
    agent1.start_game(P1)

    rewards = {P0: 0.0, P1: 0.0}

    transitions: Dict[str, List[Union[IncompleteTransition, Transition]]] = {P0: [], P1: []}


    for agent_id in env.agent_iter():
        current_agent = id_to_agent[agent_id]

        # Get the last observation/reward/termination for this agent
        # (the first time, this is just the initial observation and 0.0 reward
        s, action_mask, r, term, trunc, info = get_normalized_state(env.last())

        # Accumulate rewards
        rewards[agent_id] += r

        if len(transitions[agent_id]) > 0:
            # Complete the last transition for this agent
            last_tr = transitions[agent_id][-1]
            assert isinstance(last_tr, IncompleteTransition)
            tr = Transition(
                s=last_tr.s,
                a=last_tr.a,
                r=r,
                s2=s,
                mask=last_tr.mask,
                mask2=action_mask,
                done=term or trunc
            )
            transitions[agent_id][-1] = tr

        # Log
        if verbose:
            print(f"[turn] {agent_id} reward={r} term={term} trunc={trunc}")

        # Stop if done
        if term or trunc:
            env.step(None)  # must pass None when an agent is done
            continue

        # Act, check legality
        action = current_agent.act(s, action_mask)
        assert action_mask[action], f"{agent_id} chose illegal move {action}"

        # Save incomplete transition for this agent
        transitions[agent_id].append(IncompleteTransition(s=s, a=action, mask=action_mask))

        # Step
        env.step(int(action))
        
        # Record
        record.add_move(int(action))

        # Render
        if render:
            next_agent = env.agent_selection               # who moves next
            next_obs = env.observe(next_agent)             # AEC API: get obs for any agent
            print(c4_ansi_from_obs(next_obs))
            print(f"\nAction: {action}\n")


    # Game over; determine winner
    r0, r1 = rewards[P0], rewards[P1]
    if r0 > r1:
        record.set_winner(P0)
    elif r1 > r0:
        record.set_winner(P1)
    else:
        record.set_winner("draw")
        
    # Notify agents of game end
    agent0.end_game()
    agent1.end_game()

    return record, transitions[P0], transitions[P1]
