import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.01, network='big')
learner = learn.TD_search(env, player, gamma=0.8, search_time=2)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()

learner.learn(iters=1000, timelimit_seconds=3600)

reward_smooth = pd.DataFrame(learner.reward_trace)
reward_smooth.rolling(window=500, min_periods=0).mean().plot(figsize=(16, 9),
                                                             title='average performance over the last 3 episodes')
plt.show()

reward_smooth = pd.DataFrame(learner.piece_balance_trace)
reward_smooth.rolling(window=100, min_periods=0).mean().plot(figsize=(16, 9),
                                                             title='average piece balance over the last 3 episodes')
plt.show()

pgn = Game.from_board(learner.env.board)
with open("rlc_pgn", "w") as log:
    log.write(str(pgn))
