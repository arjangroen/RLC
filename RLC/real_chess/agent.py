from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, Dropout
from keras.losses import mean_squared_error
from keras.models import Model, clone_model, load_model
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np


class RandomAgent(object):

    def __init__(self, color=1):
        self.color = color

    def predict(self, board_layer):
        return np.random.randint(-5, 5) / 5

    def select_move(self, board):
        moves = [x for x in board.generate_legal_moves()]
        return np.random.choice(moves)


class GreedyAgent(object):

    def __init__(self, color=-1):
        self.color = color

    def predict(self, layer_board, noise=True):
        layer_board1 = layer_board[0, :, :, :]
        pawns = 1 * np.sum(layer_board1[0, :, :])
        rooks = 5 * np.sum(layer_board1[1, :, :])
        minor = 3 * np.sum(layer_board1[2:4, :, :])
        queen = 9 * np.sum(layer_board1[4, :, :])

        maxscore = 40
        material = pawns + rooks + minor + queen
        board_value = self.color * material / maxscore
        if noise:
            added_noise = np.random.randn() / 1e3
        return board_value + added_noise


class Agent(object):

    def __init__(self, lr=0.003, network='big'):
        self.optimizer = RMSprop(lr=lr)
        self.model = Model()
        self.proportional_error = False
        if network == 'simple':
            self.init_simple_network()
        elif network == 'super_simple':
            self.init_super_simple_network()
        elif network == 'alt':
            self.init_altnet()
        elif network == 'big':
            self.init_bignet()
        else:
            self.init_network()

    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """

        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        self.fixed_model.set_weights(self.model.get_weights())

    def init_network(self):
        layer_state = Input(shape=(8, 8, 8), name='state')

        openfile = Conv2D(3, (8, 1), padding='valid', activation='relu', name='fileconv')(layer_state)  # 3,8,1
        openrank = Conv2D(3, (1, 8), padding='valid', activation='relu', name='rankconv')(layer_state)  # 3,1,8
        quarters = Conv2D(3, (4, 4), padding='valid', activation='relu', name='quarterconv', strides=(4, 4))(
            layer_state)  # 3,2,2
        large = Conv2D(8, (6, 6), padding='valid', activation='relu', name='largeconv')(layer_state)  # 8,2,2

        board1 = Conv2D(16, (3, 3), padding='valid', activation='relu', name='board1')(layer_state)  # 16,6,6
        board2 = Conv2D(20, (3, 3), padding='valid', activation='relu', name='board2')(board1)  # 20,4,4
        board3 = Conv2D(24, (3, 3), padding='valid', activation='relu', name='board3')(board2)  # 24,2,2

        flat_file = Flatten()(openfile)
        flat_rank = Flatten()(openrank)
        flat_quarters = Flatten()(quarters)
        flat_large = Flatten()(large)

        flat_board = Flatten()(board1)
        flat_board3 = Flatten()(board3)

        dense1 = Concatenate(name='dense_bass')(
            [flat_file, flat_rank, flat_quarters, flat_large, flat_board, flat_board3])
        dropout1 = Dropout(rate=0.1)(dense1)
        dense2 = Dense(128, activation='sigmoid')(dropout1)
        dense3 = Dense(64, activation='sigmoid')(dense2)
        dropout3 = Dropout(rate=0.1)(dense3, training=True)
        dense4 = Dense(32, activation='sigmoid')(dropout3)
        dropout4 = Dropout(rate=0.1)(dense4, training=True)

        value_head = Dense(1)(dropout4)
        self.model = Model(inputs=layer_state,
                           outputs=[value_head])
        self.model.compile(optimizer=self.optimizer,
                           loss=[mean_squared_error]
                           )

    def init_simple_network(self):

        layer_state = Input(shape=(8, 8, 8), name='state')
        conv1 = Conv2D(8, (3, 3), activation='sigmoid')(layer_state)
        conv2 = Conv2D(6, (3, 3), activation='sigmoid')(conv1)
        conv3 = Conv2D(4, (3, 3), activation='sigmoid')(conv2)
        flat4 = Flatten()(conv3)
        dense5 = Dense(24, activation='sigmoid')(flat4)
        dense6 = Dense(8, activation='sigmoid')(dense5)
        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def init_super_simple_network(self):
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv1 = Conv2D(8, (3, 3), activation='sigmoid')(layer_state)
        flat4 = Flatten()(conv1)
        dense5 = Dense(10, activation='sigmoid')(flat4)
        value_head = Dense(1)(dense5)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def init_altnet(self):
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv1 = Conv2D(6, (1, 1), activation='sigmoid')(layer_state)
        flat2 = Flatten()(conv1)
        dense3 = Dense(128, activation='sigmoid')(flat2)

        value_head = Dense(1)(dense3)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def init_bignet(self):
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv_xs = Conv2D(4, (1, 1), activation='relu')(layer_state)
        conv_s = Conv2D(8, (2, 2), strides=(1, 1), activation='relu')(layer_state)
        conv_m = Conv2D(12, (3, 3), strides=(2, 2), activation='relu')(layer_state)
        conv_l = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(layer_state)
        conv_xl = Conv2D(20, (8, 8), activation='relu')(layer_state)
        conv_rank = Conv2D(3, (1, 8), activation='relu')(layer_state)
        conv_file = Conv2D(3, (8, 1), activation='relu')(layer_state)

        f_xs = Flatten()(conv_xs)
        f_s = Flatten()(conv_s)
        f_m = Flatten()(conv_m)
        f_l = Flatten()(conv_l)
        f_xl = Flatten()(conv_xl)
        f_r = Flatten()(conv_rank)
        f_f = Flatten()(conv_file)

        dense1 = Concatenate(name='dense_bass')([f_xs, f_s, f_m, f_l, f_xl, f_r, f_f])
        dense2 = Dense(256, activation='sigmoid')(dense1)
        dense3 = Dense(128, activation='sigmoid')(dense2)
        dense4 = Dense(56, activation='sigmoid')(dense3)
        dense5 = Dense(64, activation='sigmoid')(dense4)
        dense6 = Dense(32, activation='sigmoid')(dense5)

        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def predict_distribution(self, states, batch_size=256):
        """
        :param states: list of distinct states
        :param n:  each state is predicted n times
        :return:
        """
        predictions_per_state = int(batch_size / len(states))
        state_batch = []
        for state in states:
            state_batch = state_batch + [state for x in range(predictions_per_state)]

        state_batch = np.stack(state_batch, axis=0)
        predictions = self.model.predict(state_batch)
        predictions = predictions.reshape(len(states), predictions_per_state)
        mean_pred = np.mean(predictions, axis=1)
        std_pred = np.std(predictions, axis=1)
        upper_bound = mean_pred + 2 * std_pred

        return mean_pred, std_pred, upper_bound

    def predict(self, board_layer):
        return self.model.predict(board_layer)

    def TD_update(self, states, rewards, sucstates, episode_active, gamma=0.9):
        """
        Update the SARSA-network using samples from the minibatch
        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards and new states.

        Returns:
            td_errors: np.array
                array of temporal difference errors

        """
        suc_state_values = self.fixed_model.predict(sucstates)
        V_target = np.array(rewards) + np.array(episode_active) * gamma * np.squeeze(suc_state_values)
        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=states, y=V_target, epochs=1, verbose=0)

        V_state = self.model.predict(states)  # the expected future returns
        td_errors = V_target - np.squeeze(V_state)

        return td_errors

    def MC_update(self, states, returns):
        """
        Update network using a monte carlo playout
        Args:
            states: starting states
            returns: discounted future rewards

        Returns:
            td_errors: np.array
                array of temporal difference errors
        """
        self.model.fit(x=states, y=returns, epochs=0, verbose=0)
        V_state = np.squeeze(self.model.predict(states))
        td_errors = returns - V_state

        return td_errors
