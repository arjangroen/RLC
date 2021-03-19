import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, gamma=0.9, lr=0.001, verbose=0):
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
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """
        return self

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

    def forward(self, state):
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
        actor_out = self.actor_out(actor_2)

        return actor_out, critic_out

    def select_action(self, env):
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
        action_probs, q_value_pred = self(state)
        action_probs = action_probs * action_space
        action_probs = action_probs / action_probs.sum()
        action_probs = action_probs.detach().numpy().reshape(1, 4096)
        move = np.random.choice(range(4096), p=action_probs[0])
        move_proba = action_probs[0][move]
        move_from = move // 64
        move_to = move % 64
        moves = [x for x in env.board.generate_legal_moves() if \
                 x.from_square == move_from and x.to_square == move_to]
        move = moves[0]  # When promoting a pawn, multiple moves can have the same from-to squares
        return move, move_proba

    def network_update(self, fixed_model, episode_active, state, actions, reward, successor_state, successor_actions):
        """
        self.fixed_agent, episode_actives, states, moves, rewards, successor_states, successor_actions
        :param fixed_model, stationary ActorCritic model
        :param states: Tensor of shape (n_samples, 8, 8, 8)
        :param actions: list[chess.move]
        :param rewards: Tensor of shape (n_samples, 1)
        :param successor_states: Tensor of shape (n_samples, 8, 8, 8)
        :param successor_actions: list[chess.move]
        :return:
        """

        bootstrapped_q_values = reward.unsqueeze(dim=2) + episode_active.unsqueeze(dim=2) * torch.tensor(self.gamma) * \
                                fixed_model(successor_state)[1]
        action_probs, q_values = self(state)
        q_values_target = q_values.clone().detach()
        action_probs_list = []
        bootstrapped_q_values_a_list = []
        # Q VALUE LOSS
        for i in range(len(actions)):
            bootstrapped_q_value_a = bootstrapped_q_values[[i], successor_actions[i].from_square,
                                                           successor_actions[i].to_square].float()
            if successor_state[0, 6, 0, :].detach().numpy().sum() == 8.0:
                bootstrapped_q_values_a_list.append(bootstrapped_q_value_a)
            else:
                bootstrapped_q_values_a_list.append(-bootstrapped_q_value_a)
            q_values_target[[i], actions[i].from_square, actions[i].to_square] = bootstrapped_q_value_a.detach()
            action_probs_list.append(action_probs[[i], actions[i].from_square, actions[i].to_square])

        q_value_loss = F.mse_loss(q_values, q_values_target)

        # POLICY GRADIENT
        action_probs = torch.cat(action_probs_list, dim=0).unsqueeze(dim=1)
        advantages = torch.cat(bootstrapped_q_values_a_list, dim=0).unsqueeze(dim=1).detach()
        policy_gradient_loss = F.binary_cross_entropy(action_probs, torch.ones_like(action_probs.detach())) * advantages

        # GRADIENT DESCENT
        total_loss = q_value_loss + policy_gradient_loss.sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print("updated AC")


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
