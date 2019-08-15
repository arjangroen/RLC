import numpy as np
import time

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class RandomModel(object):

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
        """Remove?"""
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

    def simulate(self, model, env, depth=0):
        if env.board.is_game_over() or depth > 50:
            result = 0
            return result
        if env.board.turn:
            successor_values = []
            for move in env.board.generate_legal_moves():
                env.board.push(move)
                if env.board.result == "1-0":
                    result = 1
                    return result
                successor_values.append(model.predict(env.board_layer))
                env.board.pop()
            move_probas = softmax(np.array(successor_values))
            move = np.random.choice([x for x in env.board.generate_legal_moves()], p=move_probas)
            env.board.push(move)
        else:
            for move in env.board.generate_legal_moves():
                env.board.push(move)
                if env.board.result == "0-1":
                    result = -1
                    return result
                env.board.pop()
            move = np.random.choice([x for x in env.board.generate_legal_moves()])
            env.board.push(move)

        if depth == 0:
            board_copy = board.copy()

        result = self.gamma * self.simulate(model, board, depth=depth + 1)

        if depth == 0:
            return result, board_copy, move
        else:
            return result

class lambda_search(object):

    def __init__(self,env,agent,lamb=0.9, gamma=0.9):
        self.env = env
        self.agent = agent
        self.lamb = lamb
        self.gamma = gamma
        self.E = np.array()

    def play_game(self,maxiter=50):
        """
        Play a game of capture chess
        Args:
            maxiter: int
                Maximum amount of steps per game

        Returns:

        """
        episode_end = False
        turncount = 0
        node = Node(self.env.board)
        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board

            action_values = self.agent.get_action_values(np.expand_dims(state,axis=0))
            action_values = np.reshape(np.squeeze(action_values),(64,64))
            action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
            action_values = np.multiply(action_values,action_space)
            move_from = np.argmax(action_values,axis=None) // 64
            move_to = np.argmax(action_values,axis=None) % 64
            moves = [x for x in self.env.board.generate_legal_moves() if\
                    x.from_square == move_from and x.to_square == move_to]
            if len(moves) == 0:  # If all legal moves have negative action value, explore.
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state = new_state * 0
            self.memory.append([state,(move_from, move_to),reward,new_state])
            self.sampling_probs.append(1)


            self.reward_trace.append(reward)

            self.update_agent(turncount)

        return self.env.board




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
        return node.select(), result