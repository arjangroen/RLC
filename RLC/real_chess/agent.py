import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class ActorCritic(nn.Module):

    def __init__(self, gamma=0.8, lr=.0005, verbose=0):
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.)

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
                                    out_channels=2,
                                    kernel_size=(1, 1),
                                    )
        self.model_base_b = nn.Conv2d(in_channels=8,
                                      out_channels=2,
                                      kernel_size=(1, 1)
                                      )

        # Critics learns the value function
        self.critic_0_a = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.critic_0_b = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.critic_0_a_flat = nn.Flatten(start_dim=1)
        self.critic_0_b_flat = nn.Flatten(start_dim=1)
        self.critic_2a = nn.Linear(in_features=64, out_features=64)
        self.critic_2b = nn.Linear(in_features=64, out_features=64)
        self.critic_out = nn.Flatten(start_dim=2)

        self.actor_0_a = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.actor_0_b = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1))
        self.actor_0_a_flat = nn.Flatten(start_dim=1)
        self.actor_0_b_flat = nn.Flatten(start_dim=1)
        self.actor_2a = nn.Linear(in_features=64, out_features=64)
        self.actor_2b = nn.Linear(in_features=64, out_features=64)
        self.actor_out = nn.Softmax(2)

    def forward(self, state):
        state_b = state.clone()

        base = self.model_base(state)
        base_activation = torch.sigmoid(base)
        # base = state

        base_b = self.model_base_b(state_b)
        base_activation_b = torch.sigmoid(base_b)

        critic_0_a = torch.sigmoid(self.critic_0_a(base_activation))
        critic_0_b = torch.sigmoid(self.critic_0_b(base_activation))
        critic_1_a_flat = self.critic_0_a_flat(critic_0_a)
        critic_1_b_flat = self.critic_0_b_flat(critic_0_b)
        critic_2a = torch.tanh(self.critic_2a(critic_1_a_flat))
        critic_2b = self.critic_2b(critic_1_b_flat)

        critic_dot = torch.bmm(critic_2a.unsqueeze(-1), critic_2b.unsqueeze(-2))
        critic_out = self.critic_out(critic_dot)

        actor_0_a = torch.sigmoid(self.actor_0_a(base_activation_b))
        actor_0_b = torch.sigmoid(self.actor_0_b(base_activation_b))
        actor_1a = self.actor_0_a_flat(actor_0_a)
        actor_1b = self.actor_0_b_flat(actor_0_b)
        actor_2a = torch.tanh(self.actor_2a(actor_1a))
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
        target_q_values_a = []
        predicted_q_values_a = []

        # Q VALUE LOSS
        for i in range(len(actions)):
            bootstrapped_q_value_a = target_q_values[[i], successor_actions[i].from_square,
                                                     successor_actions[i].to_square].float()
            q_value_a = q_values[[i], actions[i].from_square,
                                 actions[i].to_square].float()
            if successor_state[0, 6, 0, :].detach().numpy().sum() == 8.0:
                colors.append(torch.Tensor([-1.]).float())
            else:
                colors.append(torch.Tensor([1.]).float())
            # q_values_target[[i], actions[i].from_square, actions[i].to_square] = bootstrapped_q_value_a.detach()
            target_action_probs[[i], actions[i].from_square, actions[i].to_square] = torch.Tensor([1.]).float()
            target_q_values_a.append(bootstrapped_q_value_a)
            predicted_q_values_a.append(q_value_a)

        q_pred_vec = torch.cat(predicted_q_values_a, dim=0)
        q_target_vec = torch.cat(target_q_values_a, dim=0)

        q_value_loss = F.mse_loss(q_pred_vec, q_target_vec)

        # POLICY GRADIENT
        colors_vec = torch.cat(colors, dim=0).unsqueeze(dim=1).unsqueeze(dim=2).detach()
        advantages_vec = q_target_vec.unsqueeze(dim=1).unsqueeze(dim=2).detach() * colors_vec
        # advantages_vec = advantages_vec / advantages_vec.std()
        # q_baseline = q_values.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        # advantages = (q_values_target - q_baseline) * colors_vec
        action_probs_flat = torch.flatten(action_probs, start_dim=1)
        target_action_probs_flat = torch.flatten(target_action_probs, start_dim=1)

        policy_gradient_loss = -((target_action_probs_flat * torch.log(action_probs_flat)).sum(dim=1) * advantages_vec[:,
                                                                                                      0,
                                                                                                      0]).mean()
        loss_balance = policy_gradient_loss / q_value_loss

        print("LOSS RATIO PG_LOSS / Q_LOSS:", loss_balance)

        # GRADIENT DESCENT
        total_loss = q_value_loss + policy_gradient_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print("updated AC")
        self.verify_update(state, actions, q_target_vec, q_pred_vec, advantages_vec, action_probs)

    def legalize_action_probs(self, action_probs, action_space_tensor):
        action_probs_legal = action_probs * action_space_tensor
        action_probs_legal_normalized = action_probs_legal / action_probs_legal.sum()
        return action_probs_legal_normalized


    def network_update_mc(self, state, move, Returns, action_space):
        Returns_tensor = torch.tensor([Returns]).float()
        state_tensor = torch.from_numpy(state).unsqueeze(dim=0).float()
        action_space_tensor = torch.from_numpy(action_space).unsqueeze(dim=0).float()
        if state_tensor[0, 6, 0, :].detach().numpy().sum() == -8.0:
            color = torch.Tensor([-1.])
        else:
            color = torch.Tensor([1.])
        action_probs, q_values = self(state_tensor)
        action_probs_legal = self.legalize_action_probs(action_probs, action_space_tensor)

        predicted_returns = q_values[[0], move.from_square, move.to_square]
        proba = action_probs_legal[[0], move.from_square, move.to_square]
        q_value_loss = F.mse_loss(Returns_tensor, predicted_returns)
        policy_gradient_loss = -torch.log(proba) * Returns_tensor * color
        total_loss = q_value_loss + policy_gradient_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


        # VERIFY
        print("\nupdated AC")
        action_probs_post, q_values_post = self(state_tensor)
        action_probs_post_legal = self.legalize_action_probs(action_probs_post, action_space_tensor)
        predicted_returns_post = q_values_post[[0], move.from_square, move.to_square]
        proba_post = action_probs_post_legal[[0], move.from_square, move.to_square]

        print("\nLEARNED POLICY")
        print("COLOR: ", color)
        print("Returns ", Returns)
        print("Advantage", Returns_tensor*color)
        print("expected ", predicted_returns)
        print("new expected ", predicted_returns_post)
        print("proba ", proba)
        print("new proba ", proba_post)
        print("Change in proba", proba_post - proba)
        print("Change in Q", predicted_returns_post - predicted_returns)
        print("Loss reduction Q", F.mse_loss(Returns_tensor, predicted_returns_post) - q_value_loss)



    def verify_update(self, state, actions, q_values_target, q_values_before, advantages, action_probs_before):
        action_probs_after, q_values_after = self(state)
        q_loss_pre_batch = 0
        q_loss_post_batch = 0

        advantages_numpy = advantages.numpy().squeeze()
        proba_changes_numpy = []

        for i in range(len(actions)):
            print("\n_____DIAGNOSTICS_____")
            q_value_after = q_values_after[[i], actions[i].from_square,
                                           actions[i].to_square]
            q_value_before = q_values_before[[i]]
            q_value_target = q_values_target[[i]]

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

            print("COLOR:", state[0, 6, 0, :].detach().numpy().sum() == 8.0)
            print("\nMOVE PROBA BEFORE:", action_prob_before)
            print("ADVANTAGE", advantages[[i]])
            print("MOVE PROBA AFTER:", action_prob_after)
            proba_change = (action_prob_after - action_prob_before).detach().numpy()
            print("MOVE PROBA CHANGE:", proba_change)
            proba_changes_numpy.append(proba_change)

        proba_changes_numpy = np.array(proba_changes_numpy).squeeze()

        print("\nBATCH Q-LOSS BEFORE", q_loss_pre_batch / q_loss_post_batch.shape[0])
        print("BATCH Q-LOSS AFTER", q_loss_post_batch / q_loss_post_batch.shape[0])
        print("ADVANTAGE_PROBA_CORREL",np.corrcoef(x=advantages_numpy, y=proba_changes_numpy)[0,1])

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
