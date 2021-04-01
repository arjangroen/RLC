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
                                    kernel_size=(1, 1),
                                    )

        # Critics learns the value function
        self.critic_0_a = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1))
        self.critic_0_b = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1))
        self.critic_0_a_flat = nn.Flatten(start_dim=1)
        self.critic_0_b_flat = nn.Flatten(start_dim=1)
        self.critic_2a = nn.Linear(in_features=128, out_features=64)
        self.critic_2b = nn.Linear(in_features=128, out_features=64)
        self.critic_out = nn.Flatten(start_dim=2)

        self.actor_0_a = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1))
        self.actor_0_b = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1))
        self.actor_0_a_flat = nn.Flatten(start_dim=1)
        self.actor_0_b_flat = nn.Flatten(start_dim=1)
        self.actor_2a = nn.Linear(in_features=128, out_features=64)
        self.actor_2b = nn.Linear(in_features=128, out_features=64)
        self.actor_out = nn.Softmax(2)

    def forward(self, state):
        base = self.model_base(state)
        base_activation = F.sigmoid(base)
        #base = state

        critic_0_a = F.sigmoid(self.critic_0_a(base_activation))
        critic_0_b = F.sigmoid(self.critic_0_b(base_activation))
        critic_1_a_flat = self.critic_0_a_flat(critic_0_a)
        critic_1_b_flat = self.critic_0_b_flat(critic_0_b)
        critic_2a = self.critic_2a(critic_1_a_flat)
        critic_2b = self.critic_2b(critic_1_b_flat)

        critic_dot = torch.bmm(critic_2a.unsqueeze(-1), critic_2b.unsqueeze(-2))
        critic_out = self.critic_out(critic_dot)

        actor_0_a = F.sigmoid(self.actor_0_a(base_activation))
        actor_0_b = F.sigmoid(self.actor_0_b(base_activation))
        actor_1a = self.actor_0_a_flat(actor_0_a)
        actor_1b = self.actor_0_b_flat(actor_0_b)
        actor_2a = self.actor_2a(actor_1a)
        actor_2b = self.actor_2b(actor_1b)
        actor_dot = torch.bmm(actor_2a.unsqueeze(-1), actor_2b.unsqueeze(-2)).unsqueeze(dim=1)
        actor_out = self.actor_out(actor_dot.view(*actor_dot.size()[:2], -1)).view_as(actor_dot).squeeze(dim=1)

        return actor_out, critic_out

    def select_action(self, env, greedy=False):
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
        action_probs, q_value_pred = self(state)
        action_probs = action_probs * action_space
        action_probs = action_probs / action_probs.sum()
        action_probs = action_probs.detach().numpy().reshape(1, 4096)
        if greedy:
            move = np.argmax(action_probs[0])
        else:
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

        target_q_values = reward.unsqueeze(dim=2) + episode_active.unsqueeze(dim=2) * torch.tensor(self.gamma) * \
                          fixed_model(successor_state)[1]
        target_q_values = target_q_values.detach()
        target_action_probs = torch.zeros_like(target_q_values, dtype=torch.float).detach()
        action_probs, q_values = self(state)
        q_values_target = q_values.clone().detach()
        colors = []
        advantages = []

        # Q VALUE LOSS
        for i in range(len(actions)):
            bootstrapped_q_value_a = target_q_values[[i], successor_actions[i].from_square,
                                                     successor_actions[i].to_square].float()
            if successor_state[0, 6, 0, :].detach().numpy().sum() == 8.0:
                colors.append(torch.Tensor([-1.]).float())
            else:
                colors.append(torch.Tensor([1.]).float())
            q_values_target[[i], actions[i].from_square, actions[i].to_square] = bootstrapped_q_value_a.detach()
            target_action_probs[[i], actions[i].from_square, actions[i].to_square] = torch.Tensor([1.]).float()
            advantages.append(bootstrapped_q_value_a)

        q_value_loss = F.mse_loss(q_values, q_values_target)

        # POLICY GRADIENT
        colors_vec = torch.cat(colors, dim=0).unsqueeze(dim=1).unsqueeze(dim=2).detach()
        advantages_vec = torch.cat(advantages, dim=0).unsqueeze(dim=1).unsqueeze(dim=2).detach() * colors_vec
        #advantages_vec = (advantages_vec - advantages_vec.mean()) / advantages_vec.std()
        # q_baseline = q_values.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        # advantages = (q_values_target - q_baseline) * colors_vec
        action_probs_flat = torch.flatten(action_probs, start_dim=1)
        target_action_probs_flat = torch.flatten(target_action_probs, start_dim=1)

        policy_gradient_loss = (F.cross_entropy(action_probs_flat, target_action_probs_flat.argmax(dim=1),
                                                reduction='none') * advantages_vec[:, 0, 0]).mean()
        print(policy_gradient_loss)

        # GRADIENT DESCENT
        total_loss = q_value_loss + policy_gradient_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print("updated AC")
        self.verify_update(state, actions, q_values_target, q_values, advantages_vec, action_probs)

    def verify_update(self, state, actions, q_values_target, q_values_before, advantages, action_probs_before):
        action_probs_after, q_values_after = self(state)
        q_loss_pre_batch = 0
        q_loss_post_batch = 0
        for i in range(len(actions)):
            print("\n_____DIAGNOSTICS_____")
            q_value_after = q_values_after[[i], actions[i].from_square,
                                           actions[i].to_square]
            q_value_before = q_values_before[[i], actions[i].from_square,
                                             actions[i].to_square]
            q_value_target = q_values_target[[i], actions[i].from_square, actions[i].to_square]

            action_prob_after = action_probs_after[[i], actions[i].from_square, actions[i].to_square]
            action_prob_before = action_probs_before[[i], actions[i].from_square, actions[i].to_square]

            q_loss_pre = (q_value_target - q_value_before) ** 2
            q_loss_post = (q_value_target - q_value_after) ** 2

            q_loss_pre_batch += q_loss_pre
            q_loss_post_batch += q_loss_post

            q_improvement = q_loss_post - q_loss_pre
            print("\nQ VALUE BEFORE UPDATE:", q_value_before)
            print("Q VALUE AFTER TARGET:", q_value_target)
            print("Q VALUE_AFTER UPDATE", q_value_after)
            print("Q IMPROVEMENT", q_improvement)
            print("Q SUCCESS:", q_improvement < 0)

            print("\nMOVE PROBA BEFORE:", action_prob_before)
            print("ADVANTAGE", advantages[[i]])
            print("MOVE PROBA AFTER:", action_prob_after)

        print("\nBATCH Q-LOSS BEFORE", q_loss_pre_batch)
        print("BATCH Q-LOSS AFTER", q_loss_post_batch)

    def MonteCarlo_update(self, state, actions, returns):
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
        raise NotImplementedError


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
