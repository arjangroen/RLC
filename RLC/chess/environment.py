import chess
import numpy as np
from RLC.real_chess.tree import Node

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5


class Board(object):

    def __init__(self):
        """
        Chess Board Environment
        """
        self.board = chess.Board()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()
        self.node = Node()

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.turn:
            self.layer_board[6, 0, :] = 1
        else:
            self.layer_board[6, 0, :] = -1
        self.layer_board[7, :, :] = 1

    def update_layer_board(self, move=None):
        self._prev_layer_board = self.layer_board.copy()
        self.init_layer_board()

    def pop_layer_board(self):
        self.layer_board = self._prev_layer_board.copy()
        self._prev_layer_board = None

    def step(self, move):
        """
        Run a step
        Args:
            move: python chess move
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: float
                Difference in material value after the move
        """
        self.board.push(move)
        self.update_layer_board(move)
        if move not in self.node.children.keys():
            self.node.add_child(move)
        self.node = self.node.get_down(move)
        result = self.board.result()
        if result == "*":
            reward = 0
            episode_end = False
        elif result == "1-0":
            reward = 1
            episode_end = True
        elif result == "0-1":
            reward = -1
            episode_end = True
        elif result == "1/2-1/2":
            reward = 0
            episode_end = True

        return episode_end, reward

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board()
        self.init_layer_board()
        self.node = self.node.get_root()

    def reverse(self, reverse_layer_board=False):
        self.board.pop()
        if reverse_layer_board:
            self.init_layer_board()
        self.node = self.node.get_up()

    def backprop(self, value, gamma):
        root = self.node
        move_stack = []
        root.values.append(value)
        while root.parent:
            root = root.parent
            move_stack.append(self.board.pop())
            value = value * gamma
            root.values.append(value)


        # Return to checkpoint
        self.node = root
        while not self.node.checkpoint:
            self.step(move_stack.pop())

