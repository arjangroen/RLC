from RLC.walking.agent import Piece
from RLC.walking.environment import Board
import pprint


def solve_with_policy_iteration(piece='king'):
    board = Board()
    board.render()
    King = Piece(board,piece=piece,k_max=32,synchronous=True)
    King.policy_iteration()

def solve_with_policy_iteration_async(piece='king'):
    board = Board()
    board.render()
    King = Piece(board,piece=piece,k_max=32,synchronous=False)
    King.policy_iteration()

def solve_with_value_iteration(piece='king'):
    board = Board()
    board.render()
    King = Piece(board,piece=piece,k_max=1,synchronous=True)
    King.policy_iteration()




if __name__ == '__main__':
    print("loaded walk_solve")