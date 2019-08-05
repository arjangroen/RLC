class Node(object):

    def __init__(self, board):
        self.fen = board.fen()
        self.children = {}
        self.n_visits = 0
        self.value = 0

    def add_node(self, move):
        self.children.append(Node(board))

class TreeSearch(object):

    def __init__(self,agent,env,gamma,lamb):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.lamb = lamb
        self.root_state = Node(env.board)

    def search_tree(self,node,board,iters=10):
        """
        Search Monte Carlo Style
        Returns:

        """
        actions = [x for x in board.generate_legal_moves()]
        if len(self.node.children) < len(actions)


        board_layers = []
        values = []
        for action in actions:
            board.push(action)
            #get layer board
            # add predictions

        for i in range(iters):
            # select UCB choice
            # search tree
            # update UCB







            reward = self.gamma * self.lamb * self.search_tree(board)







        if board.result()=="*":

            reward = self.gamma * self.lamb * self.search_tree(board)
        elif board.result() == "1-0":
            reward = 1
        elif board.result() == "1/2-1/2":
            reward = 0.5
        elif board.result() == "0-1":
            reward=0
        else:
            print("invalid board")
        return reward






