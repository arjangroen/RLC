from RLC.real_chess.agent import ActorCritic
from RLC.real_chess.environment import Board
from RLC.real_chess.learn import ReinforcementLearning

learning_agent = ActorCritic()
opposing_agent = ActorCritic()
board = Board()

R = ReinforcementLearning(board, learning_agent)
R.learn()



