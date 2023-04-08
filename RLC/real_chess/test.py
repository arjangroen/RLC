import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
from chess.pgn import Game
import torch

load_best = False
gamma = .8


env = environment.Board(gamma=gamma)
player = agent.NanoActorCritic()
if load_best:
    best_state = torch.load("mc_agent_best.pth")
    player.actor.load_state_dict(best_state['actor_state_dict'])
    player.critic.load_state_dict(best_state['critic_state_dict'])

learner = learn.ReinforcementLearning(env, player, gamma=gamma, search_time=2)
learner.learn(iters=1000, timelimit_seconds=3600*6, test_at_zero=False, c=10)

# reward_smooth = pd.DataFrame(learner.reward_trace)
# reward_smooth.rolling(window=500, min_periods=0).mean().plot(figsize=(16, 9)

pgn = Game.from_board(learner.env.board)
with open("rlc_pgn", "w") as log:
    log.write(str(pgn))
