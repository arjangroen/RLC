from RLC.real_chess.agent import Agent
from RLC.real_chess.environment import Board
from RLC.real_chess.learn import ReinforcementLearning

learning_agent = Agent()
opposing_agent = Agent()
board = Board()

R = ReinforcementLearning(board, learning_agent)
R.learn()



