from pettingzoo.classic import connect_four_v3
from supersuit import dtype_v0, normalize_obs_v0, observation_lambda_v0
from gymnasium import spaces
import numpy as np
import utils


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


env = connect_four_v3.env(render_mode="ansi")  # text render, no pygame window
env = observation_lambda_v0(env, change_obs, change_space)


env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    utils.jprint("obs", obs)
    utils.jprint("reward", reward)
    if term or trunc:
        action = None
    else:
        print("Env info: {}".format(env.infos[agent]))
        legal = env.infos[agent]["legal_moves"]
        action = legal[0]  # replace with your policy / DQN / PPO action
    env.step(action)
