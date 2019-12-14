import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.001, network='big')
player.fix_model()
learner = learn.TD_search(env, player, gamma=0.8, search_time=1.5)
node = tree.Node(learner.env.board, gamma=learner.gamma)

w_before = learner.agent.model.get_weights()
n_iters = 105




print(opponent.predict(np.expand_dims(env.layer_board, axis=0)))
learner.search_time = 60
learner.play_game(n_iters)
pgn = Game.from_board(learner.env.board)
with open("rlc_pgn","w") as log:
    log.write(str(pgn))