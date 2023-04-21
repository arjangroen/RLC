import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
from chess.pgn import Game
import torch

load_best = True

env = environment.ChessEnv()
player = agent.NanoActorCritic()
if load_best:
    best_state = torch.load("agent_best.pth")
    player.actor.load_state_dict(best_state['actor_state_dict'])
    player.critic.load_state_dict(best_state['critic_state_dict'])

learner = learn.ReinforcementLearning(
    env, player, search_time=1e-3)
# learner.best_so_far = 2.
final_game = learner.learn(
    iters=2500, timelimit_seconds=3600*6, test_at_zero=False, c=10)

# reward_smooth = pd.DataFrame(learner.reward_trace)
# reward_smooth.rolling(window=500, min_periods=0).mean().plot(figsize=(16, 9)

pgn = Game.from_board(final_game)
with open("rlc_pgn", "w") as log:
    log.write(str(pgn))
