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
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
        action_probs, q_value_pred = self(state, action_space)
        action_probs = action_probs / action_probs.sum()
        action_probs = action_probs.reshape(4096, ).detach().numpy()
        self.action_value_mem.append(action_probs)
        move = np.random.choice(range(4096), p=action_probs)
        move_proba = action_probs[move]
        move_from = move // 64
        move_to = move % 64
        moves = [x for x in env.board.generate_legal_moves() if \
                 x.from_square == move_from and x.to_square == move_to]
        move = moves[0]  # When promoting a pawn, multiple moves can have the same from-to squares
        return move, move_proba

    def network_update(self, fixed_model, states, moves, move_probas, rewards, successor_states, successor_action):
        """
        :param fixed_model, stationary ActorCritic model
        :param states: Tensor of shape (n, 8, 8, 8)
        :param moves: Tensor one-hot encoding of shape (n, 64, 64)
        :param move_probas: Scalar between [0,1]
        :param rewards: Scalar between [-1, 1]
        :param successor_states: Tensor of shape (n, 8, 8, 8)
        :return:
        """
        q_values = fixed_model(states)[1]
        # q_values[]
        q_target = rewards + self.gamma * fixed_model(successor_states)[1][successor_action]

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
