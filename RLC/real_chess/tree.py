import numpy as np


def softmax(x,temperature=1):
    return np.exp(x/temperature) / np.sum(np.exp(x/temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9,stop_criterium=0.1):
        self.children = {}
        self.board = board
        self.parent = parent
        self.stop_criterium = stop_criterium
        self.visits = 0
        self.balance = 0
        self.value_iters = 5
        self.values = []
        self.gamma = gamma
        self.epsilon = 0.05
        self.starting_value = 0

    def update_child(self, move, result):
        child = self.children[move]
        child.values.append(result)

    def update(self,result=None):
        if result:
            self.values.append(result)

    def backprop(self, result):
        self.parent.values.append(self.gamma*result)

    def select(self):
        """Thompson sampling"""
        if self.children:
            max_sample = np.random.choice(self.values)
            max_move = None
            for move, child in self.children.items():
                child_sample = np.random.choice(child.values)
                if child_sample > max_sample:
                    max_sample = child_sample
                    max_move = move
            if max_move:
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def simulate(self, model, env, max_depth, depth=0):

        temperature = 1

        if depth == 0:
            self.starting_value = np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0)))

        if env.board.turn:
            successor_values = []
            for move in env.board.generate_legal_moves():
                episode_end, reward = env.step(move)

                # Winning moves are greedy
                if env.board.result() == "1-0":
                    env.board.pop()
                    result = 1
                    if depth > 0:
                        return result
                    else:
                        return result, move
                successor_values.append(reward + self.gamma * np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0))))
                env.board.pop()
                env.pop_layer_board()
            move_probas = softmax(np.array(successor_values),temperature=temperature)
            moves = [x for x in env.board.generate_legal_moves()]
            if len(moves) == 1:
                move = moves[0]
            else:
                move = np.random.choice(moves, p=np.squeeze(move_probas))
            episode_end, reward = env.step(move)
        else:
            successor_values = []
            for move in env.board.generate_legal_moves():
                env.board.push(move)
                env.update_layer_board(move)

                # Winning moves are get hardcoded result
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
            episode_end, reward = env.step(move)

            V = np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0)))
        if episode_end:
            result = reward
        elif depth == max_depth or np.abs(V-self.starting_value) > self.mcts_stop_tresh:
            result = V
        else:
            result = reward + self.gamma * self.simulate(model, env, max_depth, depth=depth+1)

        env.board.pop()


        if depth == 0:
            # restore environment
            return result, move
        else:
            noise = np.random.randn()/1e3
            return result + noise
