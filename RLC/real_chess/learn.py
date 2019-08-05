
class GameNode(object):

    def __init__(self,board):
        self.node = board.fen()
        self.children = []
        self.n_visits = 0

    def add_child(self,board):
        self.children.append(GameNode(board))



class  MCTS(object):

    def __init__(self,agent,env,gamma,lamb):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.lamb = lamb
        self.

    def search_tree(self,board,breadth=5):
        """
        Search Monte Carlo Style
        Returns:

        """
        #get action values




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






