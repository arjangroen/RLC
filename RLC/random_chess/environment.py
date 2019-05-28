import chess
import numpy as np

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

    def __init__(self,FEN="k7/p1p1p1p1/1p1p1p1p/8/8/8/8/RNBQKBNR"):
        if FEN:
            self.board = chess.Board(FEN)
        else:
            self.board = chess.Board()
        self.FEN = FEN
        self.init_action_space()
        self.init_layer_board()

    def init_action_space(self):
        old_squares = list(range(64))
        new_squares = list(range(64))
        self.action_space = np.zeros(shape=(64,64))

    def init_layer_board(self):
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
        if self.board.turn:
            self.layer_board[6, :, :] = 1
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1

    def step(self,action):
        self.board.push(action)
        if self.board.result() == "*":
            opponent_move = self.get_random_action()
            self.board.push(opponent_move)
            self.init_layer_board()
            if self.board.result() == "*":
                reward = 0
                episode_end = False
            else:
                reward = -100
                episode_end = True
        else:
            self.init_layer_board()
            reward = 100
            episode_end = True
        if self.board.is_game_over(claim_draw=True):
            reward = 0
            episode_end = True
        return episode_end, reward

    def get_random_action(self):
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves


    def project_legal_moves(self):
        self.action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0],move[1]] = 1
        return self.action_space

    def get_material_value(self):
        pawns = 1 * np.sum(self.layer_board[0,:])
        rooks = 5 * np.sum(self.layer_board[1,:])
        minor = 3 * np.sum(self.layer_board[2:4,:])
        queen = 9 * np.sum(self.layer_board[4,:])
        return pawns + rooks + minor + queen



    #def update_layer_board(self,move):
    #    row_from = move[0] // 8
    #    col_from = move[0] % 8
    #    row_to = move[1] // 8
    #    col_to = move[1] % 8
    #    plane = np.argmax(self.layer_board[:,row_from,col_from])
    #    self.layer_board[plane,row_to,col_to] = self.layer_board[plane,row_from,col_from]
    #    self.layer_board[plane, row_from, col_from] = 0
    #    self.layer_board[6, :, :] = 1 if self.board.turn else 0
    #    self.layer_board[7, :, :] = 1 if self.board.can_claim_draw() else 0

    def reset(self):
        self.board = chess.Board(self.FEN)
        self.init_layer_board()
        self.init_action_space()
