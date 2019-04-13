from RLC.shortest_path.agent import Piece
from RLC.shortest_path.environment import Board
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
    print("solutions to shortest path algorithm:")
    print("solve_with_policy_iteration")
    print("solve_with_policy_iteration_async")
    print("solve_with_value_iteration")
    print("\nwith possible pieces:")
    print("king, rook, bishop, knight")