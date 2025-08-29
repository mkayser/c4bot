from pettingzoo.classic import connect_four_v3
from supersuit import dtype_v0, normalize_obs_v0, observation_lambda_v0
from gymnasium import spaces
import numpy as np
import utils
import agents


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


### Helper function to get legal moves
def get_legal_moves(obs, info=None, action_space_n=7):
    if info and "legal_moves" in info:
        return list(info["legal_moves"])  # works for some envs, not C4
    if isinstance(obs, dict) and "action_mask" in obs:
        return [i for i, ok in enumerate(obs["action_mask"]) if ok]
    return list(range(action_space_n))


def initialize_env():
    env = connect_four_v3.env(render_mode="ansi")  # text render, no pygame window
    env = observation_lambda_v0(env, change_obs, change_space)
    return env


def play_game(
    env,
    agent0: C4Agent,
    agent1: C4Agent,
    *,
    seed: Optional[int] = None,
    render: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Runs one game in the given PettingZoo env (already constructed; we just reset it).
    Returns a dict with rewards, winner, and move count.
    """

    env.reset(seed=seed)

    # Map PettingZoo agent ids to our agents
    P0 = "player_0"
    P1 = "player_1"
    
    id_to_agent = {P0: agent0, P1: agent1}
    agent0.start_game(P0)
    agent1.start_game(P1)

    move_count = 0
    if render:
        s = env.render()
        if isinstance(s, str):
            print(s)

    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        if verbose:
            print(f"[turn] {agent_id} reward={reward} term={term} trunc={trunc}")

        if term or trunc:
            env.step(None)  # must pass None when an agent is done
            continue

        legal = legal_from_obs(obs)
        action = id_to_agent[agent_id].act(obs, legal, reward, term, trunc, info)
        assert action in legal, f"{id_to_agent[agent_id].name} chose illegal move {action}"
        env.step(int(action))
        move_count += 1

        if render:
            s = env.render()
            if isinstance(s, str):
                print(s)

    # After the loop, PettingZoo has final per-agent rewards in env.rewards
    rewards = dict(env.rewards)
    if rewards.get("player_0", 0) > rewards.get("player_1", 0):
        winner = "player_0"
    elif rewards.get("player_1", 0) > rewards.get("player_0", 0):
        winner = "player_1"
    else:
        winner = "draw"

    # Notify agents (optional)
    agent0.end_game(rewards, info={})
    agent1.end_game(rewards, info={})

    return {"rewards": rewards, "winner": winner, "moves": move_count}


