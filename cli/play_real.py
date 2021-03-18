from RLC.real_chess.agent import ActorCritic
from RLC.real_chess.environment import Board
from RLC.real_chess.learn import ReinforcementLearning
from chess.pgn import Game
import datetime

learning_agent = ActorCritic()
opposing_agent = ActorCritic()
board = Board()

R = ReinforcementLearning(board, learning_agent)
board = R.learn(iters=1)
pgn = Game.from_board(board)

with open(f'game_{datetime.datetime.now().isoformat(timespec="seconds")}.pgn', 'w') as file:
    file.write(pgn)




