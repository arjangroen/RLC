from RLC.real_chess.agent import NanoActorCritic
import torch
import matplotlib.pyplot as plt
import numpy as np

from RLC.real_chess.environment import Board


agent = NanoActorCritic()
best_state = torch.load("agent_320.pth")
board = Board()


agent.actor.load_state_dict(best_state['actor_state_dict'])
agent.critic.load_state_dict(best_state['critic_state_dict'])

states = torch.from_numpy(np.expand_dims(
    board.layer_board, axis=0)).float()


logits = agent.actor(states)
action_space = torch.from_numpy(np.expand_dims(board.project_legal_moves(),
                                               axis=0)).float()


action_probas = torch.nn.functional.softmax(logits).reshape(8, 8)

action_probas_legal = torch.nn.functional.softmax(
    agent.legalize_action_logits(logits, action_space_tensor=action_space)).reshape(8, 8)


assert True
