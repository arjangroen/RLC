import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def policy_gradient_loss(Returns):
    def modified_crossentropy(action, action_probs):
        cost = (K.categorical_crossentropy(action, action_probs, from_logits=False, axis=1) * Returns)
        return K.mean(cost)

    return modified_crossentropy


class ActorCritic(nn.Module):

    def __init__(self, gamma=0.5, lr=0.01, verbose=0):
        """
        Agent that plays the white pieces in capture chess
        Args:
            gamma: float
                Temporal discount factor
            network: str
                'linear' or 'conv'
            lr: float
                Learning rate, ideally around 0.1
        """
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.init_actorcritic()
        self.weight_memory = []
        self.long_term_mean = []
        self.action_value_mem = []

    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """
        pass

    def init_actorcritic(self):
        """
        Convnet net for policy gradients
        Returns:

        """
        self.model_base = nn.Conv2d(in_channels=8,
                                    out_channels=4,
                                    kernel_size=(1, 1)
                                    )

        # Critics learns the value function
        self.critic_0 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1))
        self.critic_1 = nn.Flatten(start_dim=2)
        self.critic_out = nn.Flatten(start_dim=2)

        self.actor_0 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 1))
        self.actor_1 = nn.Flatten(start_dim=2)
        self.actor_2 = nn.Flatten(start_dim=2)
        self.actor_out = nn.Softmax(dim=1)

    def forward(self, state, actionspace):
        base = self.model_base(state)

        critic_0 = self.critic_0(base)
        critic_1a = torch.reshape(critic_0,
                                  shape=(critic_0.shape[0], critic_0.shape[2] * critic_0.shape[3], critic_0.shape[1]))
        critic_1b = torch.reshape(critic_0,
                                  shape=(critic_0.shape[0], critic_0.shape[1], critic_0.shape[2] * critic_0.shape[3]))
        critic_dot = torch.matmul(critic_1a, critic_1b)
        critic_out = self.critic_out(critic_dot)

        actor_0 = self.actor_0(base)
        actor_1a = torch.reshape(actor_0,
                                 shape=(actor_0.shape[0], actor_0.shape[2] * actor_0.shape[3], actor_0.shape[1]))
        actor_1b = torch.reshape(actor_0,
                                 shape=(actor_0.shape[0], actor_0.shape[1], actor_0.shape[2] * actor_0.shape[3]))
        actor_dot = torch.matmul(actor_1a, actor_1b)
        actor_2 = self.actor_2(actor_dot)
        actor_out = actionspace.mul(self.actor_out(actor_2))

        return actor_out, critic_out

    def select_action(self, env):
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
        action_probs, q_value_pred = self(state, action_space)
        action_probs = action_probs / action_probs.sum()
        action_probs = action_probs.reshape(4096,).detach().numpy()
        self.action_value_mem.append(action_probs)
        move = np.random.choice(range(4096), p=np.squeeze(action_probs))
        move_from = move // 64
        move_to = move % 64
        moves = [x for x in env.board.generate_legal_moves() if \
                 x.from_square == move_from and x.to_square == move_to]
        move = moves[0]  # When promoting a pawn, multiple moves can have the same from-to squares
        return move

    def network_update(self, states, moves, move_probas, rewards, successor_states):
        """
        Update the Q-network using samples from the minibatch
        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards and new states.

        Returns:
            td_errors: np.array
                array of temporal difference errors

        """
        pass

    def policy_gradient_update(self, states, actions, rewards, action_spaces, actor_critic=False):
        """
        Update parameters with Monte Carlo Policy Gradient algorithm
        Args:
            states: (list of tuples) state sequence in episode
            actions: action sequence in episode
            rewards: rewards sequence in episode

        Returns:

        """
        n_steps = len(states)
        Returns = []
        targets = np.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = actions[t]
            targets[t, action[0], action[1]] = 1
            if actor_critic:
                R = rewards[t, action[0] * 64 + action[1]]
            else:
                R = np.sum([r * self.gamma ** i for i, r in enumerate(rewards[t:])])
            Returns.append(R)

        if not actor_critic:
            mean_return = np.mean(Returns)
            self.long_term_mean.append(mean_return)
            train_returns = np.stack(Returns, axis=0) - np.mean(self.long_term_mean)
        else:
            train_returns = np.stack(Returns, axis=0)
        # print(train_returns.shape)
        targets = targets.reshape((n_steps, 4096))
        self.weight_memory.append(self.model.get_weights())
        self.model.fit(x=[np.stack(states, axis=0),
                          train_returns,
                          np.concatenate(action_spaces, axis=0)
                          ],
                       y=[np.stack(targets, axis=0)],
                       verbose=self.verbose
                       )

    class RandomAgent(object):

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
            self.init_network()
            self.fix_model()

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
