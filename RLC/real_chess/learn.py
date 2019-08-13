import numpy as np
import time

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class DummyModel(object):

    def __init__(self):
        pass

    def update(self):
        pass

    def predict(self):
        return np.random.randint(-100, 100) / 100.


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9):
        self.children = {}
        self.board = board
        self.parent = parent
        self.mean_value = None
        self.std_value = None
        self.upper_bound = None
        self.visits = 0
        self.balance = 0
        self.value_iters = 5
        self.values = []
        self.gamma = gamma

    def estimate_child_values(self, model):
        value_predictions = []
        for child in self.children.values():
            for i in range(self.value_iters):
                child.values.append(model.predict())
            child.mean_value = np.mean(child.values)
            child.std_value = np.std(child.values)
            child.upper_bound = child.mean_value + 2 * child.std_value

    def update_child(self, move, result):
        child = self.children[move]
        child.values.append(result)
        child.mean_value = np.mean(child.values)
        child.std_value = np.std(child.values)
        child.upper_bound = child.mean_value + 2 * child.std_value

    def update(self,result=None):
        if result:
            self.values.append(result)
        self.mean_value = np.mean(self.values)
        self.std_value = np.std(self.values)
        self.upper_bound = self.mean_value + 2 * self.std_value


    def add_children(self):
        for move in self.board.generate_legal_moves():
            self.board.push(move)
            self.children[move] = Node(self.board.copy(), parent=self)
            self.board.pop()

    def backprop(self, result):
        self.parent.values.append(result)

    def select(self):
        if self.children:
            max_upper = self.upper_bound if self.upper_bound else 0
            max_move = None
            for move, child in self.children.items():
                if child.upper_bound > max_upper:
                    max_upper = child.upper_bound
                    max_move = move
            if max_move:
                return self.children[max_move]
            else:
                return self
        else:
            return self

    def simulate(self, model, board, depth=0):
        if board.is_game_over() or depth > 50:
            result = 0
            return result
        if board.turn:
            successor_values = []
            for move in board.generate_legal_moves():
                board.push(move)
                if board.result == "1-0":
                    result = 1
                    return result
                successor_values.append(model.predict())
                board.pop()
            move_probas = softmax(np.array(successor_values))
            move = np.random.choice([x for x in board.generate_legal_moves()], p=move_probas)
            board.push(move)
        else:
            for move in board.generate_legal_moves():
                board.push(move)
                if board.result == "0-1":
                    result = -1
                    return result
                board.pop()
            move = np.random.choice([x for x in board.generate_legal_moves()])
            board.push(move)

        if depth == 0:
            print(board)
            board_copy = board.copy()

        result = self.gamma * self.simulate(model, board, depth=depth + 1)

        if depth == 0:
            return result, board_copy, move
        else:
            return result

def mcts(node,board, model,timelimit=30):
    """
    Return best node
    :param node:
    :param board:
    :param model:
    :return:
    """
    node.add_children()
    node.estimate_child_values(model)
    starttime = time.time()
    timelimit = 30
    while starttime + timelimit > time.time():
        while node.children:
            new_node = node.select()
            if new_node == node:
                node = new_node
                break
            node = new_node
        result, board_copy, move = node.simulate(model,node.board.copy())
        if move not in node.children.keys():
            node.children[move] = Node(board_copy.copy(),parent=node)

        node.update_child(move,result)

        node = node.children[move]
        while node.parent:
            node.backprop(result)
            node = node.parent
            node.update()
    return node.select()