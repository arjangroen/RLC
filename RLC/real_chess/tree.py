import numpy as np


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9):
        """
        Game Node for Monte Carlo Tree Search
        Args:
            board: the chess board
            parent: the parent node
            gamma: the discount factor
        """
        self.children = {}  # Child nodes
        self.board = board  # Chess board
        self.parent = parent
        self.values = []  # reward + Returns
        self.gamma = gamma
        self.starting_value = 0

    def update_child(self, move, Returns):
        """
        Update a child with a simulation result
        Args:
            move: The move that leads to the child
            Returns: the reward of the move and subsequent returns

        Returns:

        """
        child = self.children[move]
        child.values.append(Returns)

    def update(self, Returns=None):
        """
        Update a node with observed Returns
        Args:
            Returns: Future returns

        Returns:

        """
        if Returns:
            self.values.append(Returns)

    def select(self, color=1):
        """
        Use Thompson sampling to select the best child node
        Args:
            color: Whether to select for white or black

        Returns:
            (node, move)
            node: the selected node
            move: the selected move
        """
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

    def simulate(self, model, env, depth=0, max_depth=4, random=False, temperature=1):
        """
        Recursive Monte Carlo Playout
        Args:
            model: The model used for bootstrap estimation
            env: the chess environment
            depth: The recursion depth
            max_depth: How deep to search
            temperature: softmax temperature

        Returns:
            Playout result.
        """
        board_in = env.board.fen()
        if env.board.turn and random:
            move = np.random.choice([x for x in env.board.generate_legal_moves()])
        else:
            successor_values = []
            for move in env.board.generate_legal_moves():
                episode_end, reward = env.step(move)
                result = env.board.result()

                if (result == "1-0" and env.board.turn) or (
                        result == "0-1" and not env.board.turn):
                    env.board.pop()
                    env.init_layer_board()
                    break
                else:
                    if env.board.turn:
                        sucval = reward + self.gamma * np.squeeze(
                            model.predict(np.expand_dims(env.layer_board, axis=0)))
                    else:
                        sucval = np.squeeze(env.opposing_agent.predict(np.expand_dims(env.layer_board, axis=0)))
                    successor_values.append(sucval)
                    env.board.pop()
                    env.init_layer_board()

            if not episode_end:
                if env.board.turn:
                    move_probas = softmax(np.array(successor_values), temperature=temperature)
                    moves = [x for x in env.board.generate_legal_moves()]
                else:
                    move_probas = np.zeros(len(successor_values))
                    move_probas[np.argmax(successor_values)] = 1
                    moves = [x for x in env.board.generate_legal_moves()]
                if len(moves) == 1:
                    move = moves[0]
                else:
                    move = np.random.choice(moves, p=np.squeeze(move_probas))

        episode_end, reward = env.step(move)

        if episode_end:
            Returns = reward
        elif depth >= max_depth:  # Bootstrap the Monte Carlo Playout
            Returns = reward + self.gamma * np.squeeze(model.predict(np.expand_dims(env.layer_board, axis=0)))
        else:  # Recursively continue
            Returns = reward + self.gamma * self.simulate(model, env, depth=depth + 1,temperature=temperature)

        env.board.pop()
        env.init_layer_board()

        board_out = env.board.fen()
        assert board_in == board_out

        if depth == 0:
            return Returns, move
        else:
            noise = np.random.randn() / 1e6
            return Returns + noise
