from RLC.real_chess.agent import NanoActorCritic
import torch
import matplotlib.pyplot as plt
import numpy as np

from RLC.real_chess.environment import ChessEnv


agent = NanoActorCritic()
best_state = torch.load("agent_best.pth")
env = ChessEnv()


agent.actor.load_state_dict(best_state['actor_state_dict'])
agent.critic.load_state_dict(best_state['critic_state_dict'])


episode_active = 1

while episode_active > 0:
    state = env.layer_board
    logits = agent.actor(state)
    action_probas = agent.get_action_probabilities(env=env)
    q_values = agent.get_q_values(env)
    move, move_proba = agent.select_action(env, greedy=False)

    env.step(move)
    print((action_probas*100).round().view(8, 8))
    print(q_values.round().view(8, 8))


assert True
