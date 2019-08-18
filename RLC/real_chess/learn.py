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
        self.memory = []
        self.memsize = 5000
        self.batch_size = 128

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
            state = self.env.layer_board.copy()
            state_value = self.agent.predict(state)

            # White's turn
            if self.env.board.turn():
                tree = self.mcts(tree)

                # Step the best move
                max_move = None
                max_value = -1
                for move, child in tree.children.items():
                    if child.mean_value > max_value:
                        max_value = child.mean_value
                        max_move = move

            # Black's turn
            else:
                max_move = None
                max_value = -1
                for move in self.env.board.generate_legal_moves():
                    self.env.step(move)
                    successor_state_value_opponent = self.env.oppossing_agent.predict(self.env.layer_board)
                    if successor_state_value_opponent > max_value:
                        max_move = move
                        max_value = successor_state_value_opponent


            episode_end, reward = self.env.step(max_move)

            # Predict the successor value
            successor_state_value = self.agent.predict(self.env.layer_board)

            # Target = reward + successor value
            target = reward + successor_state_value

            error = target - state_value

            # construct training sample state, prediction, error
            self.memory.append([state,target,error])

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0

            self.update_agent()

        return self.env.board

    def update_agent(self):
        batch = self.get_minibatch()



    def get_minibatch(self):
        sampling_priorities = np.abs(np.array([xp[2] for xp in self.memory]))
        sampling_probs = sampling_priorities / np.sum(sampling_priorities)
        sample_indices = [x for x in range(len(self.memory))]
        choice_indices = np.random.choice(sample_indices,min(len(self.memory),128),p=sampling_probs)
        minibatch = self.memory[choice_indices]




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