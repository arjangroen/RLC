import numpy as np

class PlayOut(object):

    def __init__(self, gamma=0.9, max_depth=2):
        self.gamma = gamma
        self.max_depth=max_depth

    def sim(self, env, white, black, learning_agent_color, depth=0):
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
        if len(moves) == 0 or len(successor_values)==0:
            print("No moves")
            print(env.board)

        try:
            selected_move = current_player.select_move_from_values(moves, successor_values)
        except:
            pass
        episode_end, reward = env.step(selected_move)

        if episode_end:
            env.reverse()
            return reward
        elif depth > self.max_depth:  # Even number
            learning_player = white if learning_agent_color == 1 else black
            state_value = learning_player.predict(np.expand_dims(env.layer_board, axis=0))
            env.reverse()
            return reward + np.squeeze(state_value)
        else:
            returns = reward + self.gamma * self.sim(env, white, black, learning_agent_color, depth=depth + 1)
            env.reverse()
            return returns


