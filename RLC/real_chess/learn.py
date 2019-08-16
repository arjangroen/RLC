import numpy as np
import time
from RLC.real_chess.tree import Node

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))



class TD_search(object):

    def __init__(self,env,agent,lamb=0.9, gamma=0.9):
        self.env = env
        self.agent = agent
        self.tree = Node(self.env)
        self.lamb = lamb
        self.gamma = gamma

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
        tree = Node(self.env.board)

        # Play a game of chess
        while not episode_end:

            # Populate the tree with Monte Carlo Tree Search
            tree = self.mcts(tree)

            # Predict the current value
            state_value = self.agent.predict(self.env.layer_board)

            # Step the best move
            max_move = None
            max_value = -1
            for move, child in tree.children.items():
                if child.mean_value > max_value:
                    max_value = child.mean_value
                    max_move = move

            episode_end, reward = self.env.step(max_move)

            # Predict the successor value
            successor_state_value = self.agent.predict(self.env.layer_board)

            # Target = reward + successor value


            # construct training sample state, prediction, error

            #


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


    def mcts(self,node,timelimit=30):
        """
        Return best node
        :param node:
        :return:
        """
        starttime = time.time()
        timelimit = 30
        while starttime + timelimit > time.time():
            while node.children:
                new_node = node.select()
                if new_node == node:
                    node = new_node
                    break
                node = new_node
            result, board_copy, move = node.simulate(self.agent,node)
            if move not in node.children.keys():
                node.children[move] = Node(board_copy.copy(),parent=node)

            node.update_child(move,result)

            node = node.children[move]
            while node.parent:
                node.backprop(result)
                node = node.parent
                node.update()
        return node.select(), result