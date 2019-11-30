import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import RLC

from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Q_learning
from RLC.capture_chess.agent import Agent

FEN = "k7/p1p1p1p1/1p1p1p1p/8/8/8/8/RNBQKBNR"

board = Board(FEN=FEN)

agent = Agent()

R = Q_learning(agent, board)
R.learn(iters=2)
