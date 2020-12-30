class PlayOut(object):

    def __init__(self, gamma=0.9):
        self.gamma = gamma

    def sim(self, env, white, black, depth=0):
        """
        Recursive playout simulation
        :param env: environment
        :param white: white agent
        :param black: black agent
        :param depth: current depth
        :return: Returns
        """

        successor_values = []
        moves = []
        for move in env.board.generate_legal_moves():
            current_player = white if env.board.turn else black
            successor_values.append(current_player.evaluate(move, env))
            moves.append(move)
        selected_move = current_player.select_move_from_values(moves, successor_values)
        episode_end, reward = env.step(selected_move)

        if episode_end or depth > 50:
            return reward
        else:
            returns = reward + self.gamma * self.sim(env, white, black, depth=depth + 1)

        env.reverse()
        return returns


