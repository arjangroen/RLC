from keras.models import load_model
import os
os.chdir('..')
from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game


opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.ActorCritic(lr=0.0005, network='big')
learner = learn.ReinforcementLearning(env, player, gamma=0.9, search_time=0.9)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()
n_iters = 3  # maximum number of iterations
timelimit = 25000 # maximum time for learning
network_replacement_interval = 10  # For the stability of the nearal network updates, the network is not continuously replaced
learner.learn(iters=n_iters,timelimit_seconds=timelimit,c=network_replacement_interval)