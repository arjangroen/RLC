import numpy as np
import time
from RLC.real_chess.tree import Node





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
        self.result_trace = []

    def learn(self,iters=100,c=5):
        for k in range(iters):
            self.env.reset()
            if k % c == 0:
                self.agent.fix_model()
            self.play_game()
        return self.env.board


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
        tree = Node(self.env.board,gamma=self.gamma)

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board.copy()
            state_value = self.agent.predict(np.expand_dims(state,axis=0))

            # White's turn
            if self.env.board.turn:
                print(self.env.board)
                tree = self.mcts(tree)
                print(self.env.board)

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

            new_state_value = self.agent.predict(np.expand_dims(self.env.layer_board,axis=0))
            error = new_state_value - state_value

            # construct training sample state, prediction, error
            self.memory.append([state,reward,self.env.layer_board,error])

            if len(self.memory) > self.memsize:
                self.memory.pop(0)
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0

            self.update_agent()

        self.result_trace.append(reward)

        return self.env.board

    def update_agent(self):
        choice_indices, minibatch = self.get_minibatch()
        td_errors = self.agent.network_update(minibatch,gamma=self.gamma)
        for index, error in zip(choice_indices,td_errors):
            self.memory[index][2] = error

    def get_minibatch(self):
        if len(self.memory) == 1:
            return 0, self.memory[0]
        else:
            sampling_priorities = np.abs(np.array([xp[3] for xp in self.memory]))
            sampling_probs = sampling_priorities / np.sum(sampling_priorities)
            print(sampling_probs.shape)
            sample_indices = [x for x in range(len(self.memory))]
            choice_indices = np.random.choice(sample_indices,min(len(self.memory),128),p=np.squeeze(sampling_probs))
            minibatch = [self.memory[idx] for idx in choice_indices]
        return choice_indices, minibatch


    def mcts(self,node,timelimit=15):
        """
        Return best node
        :param node:
        :return:
        """
        starttime = time.time()
        timelimit = 30
        board_copy = self.env.board.copy()
        while starttime + timelimit > time.time():
            while node.children:
                new_node = node.select()
                if new_node == node:
                    node = new_node
                    break
                node = new_node
            result, move = node.simulate(self.agent.model,self.env)
            if move not in node.children.keys():
                node.children[move] = Node(board_copy,parent=node)

            node.update_child(move,result)

            node = node.children[move]
            while node.parent:
                node.backprop(result)
                node = node.parent
                node.update()
        return node.select()