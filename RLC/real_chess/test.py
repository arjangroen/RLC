import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
from chess.pgn import Game


env = environment.Board()
player = agent.ActorCritic()
learner = learn.ReinforcementLearning(env, player, gamma=0.5, search_time=2)
node = tree.Node(learner.env.board, gamma=learner.gamma)
learner.learn(iters=1000, timelimit_seconds=3600*6, test_at_zero=False, c=10)

# reward_smooth = pd.DataFrame(learner.reward_trace)
# reward_smooth.rolling(window=500, min_periods=0).mean().plot(figsize=(16, 9)

pgn = Game.from_board(learner.env.board)
with open("rlc_pgn", "w") as log:
    log.write(str(pgn))
