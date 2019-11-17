import numpy as np


def softmax(x,temperature=1):
    return np.exp(x/temperature) / np.sum(np.exp(x/temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9,stop_criterium=(-0.05,0.1)):
        self.children = {}
        self.board = board
        self.parent = parent
        self.stop_criterium = stop_criterium
        self.visits = 0
        self.balance = 0
        self.value_iters = 5
        self.values = []  # reward + Returns
        self.gamma = gamma
        self.epsilon = 0.05
        self.starting_value = 0
        self.FEN = board.fen()

    def update_child(self, move, result):
        child = self.children[move]
        child.values.append(result)

    def update(self,result=None):
        if result:
            self.values.append(result)

    def backprop(self, result):
        self.parent.values.append(self.gamma*result)

    def select(self, color=1):
        """Thompson sampling"""
        assert color == 1 or color == -1, "color has to be white (1) or black (-1)"
        if self.children:
            max_sample = np.random.choice(color * np.array(self.values))
            max_move = None
            for move, child in self.children.items():
                child_sample = np.random.choice(color * np.array(child.values))
                if child_sample > max_sample:
                    max_sample = child_sample
                    max_move = move
            if max_move:
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def simulate(self, model, env, depth=0, random=True):

        max_depth = 4

        temperature = 1

        if depth == 0:
            self.starting_value = np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0)))

        assert env.board.turn, f"turn out of sync in depth {depth} and result {env.board.result()}"

        if not random:
            successor_values = []
            for move in env.board.generate_legal_moves():
                episode_end, reward = env.step(move)

                # Winning moves are greedy
                if env.board.result() == "1-0":
                    env.board.pop()
                    Returns = 0
                    if depth > 0:
                        return reward + self.gamma * Returns
                    else:
                        return reward + self.gamma * Returns, move
                elif episode_end:
                    successor_values.append(reward)
                else:
                    successor_values.append(reward + self.gamma * np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0))))
                for _ in ['move, opponent_move']:
                    env.board.pop()
                env.init_layer_board()
            move_probas = softmax(np.array(successor_values),temperature=temperature)
            moves = [x for x in env.board.generate_legal_moves()]
            if len(moves) == 1:
                move = moves[0]
            else:
                move = np.random.choice(moves, p=np.squeeze(move_probas))
        elif random:
            move = np.random.choice([x for x in env.board.generate_legal_moves()])


        episode_end, reward = env.step(move)

        if episode_end:
            Returns = reward
        elif depth >= max_depth: #  or \
            # V * self.gamma**depth - self.starting_value > self.stop_criterium[1] or \
            # V * self.gamma**depth - self.starting_value < self.stop_criterium[0]:
            Returns = reward + self.gamma * np.squeeze(model.predict(np.expand_dims(env.layer_board,axis=0)))
            #Returns = reward + self.gamma * 0
        else:
            Returns = reward + self.gamma * self.simulate(model, env, depth=depth+1)

        env.board.pop()
        env.board.pop()


        if depth == 0:
            # restore environment
            return Returns, move
        else:
            noise = np.random.randn()/1e6
            return Returns + noise
