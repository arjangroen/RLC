import chess

class Node(object):
    
    def __init__(self,board):
        self.fen = board.fen()
        self.children = []
        self.n_visits = 0
        self.value = 0
        
    def add_node(self,board):
        self.children.append(Node(board))
