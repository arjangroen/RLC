import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game

opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN='4k3/8/8/8/8/8/8/R3K2R w Q - 0 1')
player = agent.Agent(lr=0.01,network='')
learner = learn.TD_search(env, player,gamma=0.8,search_time=2)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()

learner.learn(iters=10)

