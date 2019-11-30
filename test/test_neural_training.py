import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.001, network='')
learner = learn.TD_search(env, player, gamma=0.8, search_time=1.5)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()

w_before = learner.agent.model.get_weights()


def test_train():
    learner.learn(iters=11, timelimit_seconds=900)


test_train()

w_after = learner.agent.model.get_weights()

print("done")
