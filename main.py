import c4
import agents



env = c4.initialize_env()

a0 = agents.RandomC4Agent()
a1 = agents.RandomC4Agent()

c4.play_game(env, a0, a1, render=True)
