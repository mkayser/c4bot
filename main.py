import c4
import agents


def print_transition(tr: c4.Transition) -> None:
    print(f"STATE {tr.a} {tr.r} STATE2 {tr.mask} {tr.mask2}")

env = c4.initialize_env()

a0 = agents.RandomAgent()
a1 = agents.RandomAgent()


results, trans1, trans2 = c4.play_game(env, a0, a1, render=True, verbose=True)

print("Transitions of agent 1:")
for t in trans1:
    print_transition(t)

print("Transitions of agent 2:")
for t in trans2:
    print_transition(t)
