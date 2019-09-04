import numpy as np

def softmax(x,temperature=1):
    return np.exp(x/temperature) / np.sum(np.exp(x/temperature))


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
        """Remove?"""
        value_predictions = []
        for child in self.children.values():
            for i in range(self.value_iters):
                child.values.append(model.predict())
            child.mean_value = np.mean(child.values)
            child.std_value = np.clip(np.std(child.values),0,0.5)
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
        print("adding children full width")
        for move in self.board.generate_legal_moves():
            self.board.push(move)
            self.children[move] = Node(self.board, parent=self)
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
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def simulate(self, model, env, depth=0):

        # Gradually reduce the temperature
        max_depth = 4  # Even for final move for black
        temperature = 1 + max_depth - depth
        if env.board.is_game_over() or depth > max_depth:
            if env.board.is_game_over(claim_draw=True):
                result = 0
            else:
                result = np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0)))
            return result
        if env.board.turn:
            successor_values = []
            for move in env.board.generate_legal_moves():
                env.board.push(move)
                env.update_layer_board(move)
                if env.board.result() == "1-0":
                    env.board.pop()
                    result = 1
                    if depth > 0:
                        return result
                    else:
                        return result, move
                successor_values.append(np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0))))
                env.board.pop()
                env.pop_layer_board()
            move_probas = softmax(np.array(successor_values),temperature=temperature)
            moves = [x for x in env.board.generate_legal_moves()]
            if len(moves) == 1:
                move = moves[0]
            else:
                move = np.random.choice(moves, p=np.squeeze(move_probas))
            env.step(move)
        else:
            successor_values = []
            for move in env.board.generate_legal_moves():
                env.board.push(move)
                env.update_layer_board(move)
                if env.board.result() == "0-1":
                    env.board.pop()
                    result = -1
                    if depth > 0:
                        return result
                    else:
                        return result, move
                successor_values.append(np.squeeze(env.opposing_agent.predict(np.expand_dims(env.layer_board, axis=0))))
                env.board.pop()
                env.pop_layer_board()
            move_probas = np.zeros(len(successor_values))
            move_probas[np.argmax(successor_values)] = 1
            moves = [x for x in env.board.generate_legal_moves()]
            if len(moves) == 1:
                move = moves[0]
            else:
                move = np.random.choice(moves, p=np.squeeze(move_probas))
            env.step(move)

        result = self.gamma * self.simulate(model, env, depth=depth + 1)
        env.board.pop()


        if depth == 0:
            # restore environment
            return result, move
        else:
            noise = np.random.randn()/1e3
            return result + noise
