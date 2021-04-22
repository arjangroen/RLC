import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class ActorCritic(nn.Module):

    def __init__(self, gamma=0.8, lr=.001, verbose=0):
        """
        Agent that plays the white pieces in capture chess.
        The action space is defined as a combination of a "source square" and a "target square".
        The actor gives an action probability for each square combination.
        The critic gives an action value for each square combination.
        Args:
            gamma: float
                Temporal discount factor
            network: str
                'linear' or 'conv'
            lr: float
                Learning rate, ideally around 0.001
        """
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.init_actor()
        self.init_critic()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def init_actor(self):
        """
        Defines the actor/policy network, who calculates the action probabilities.
        :return:
        """
        self.identity_l = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.05)
        )  # The identiy serves as a 'score per square'
        self.identity_r = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.05)
        )  # The identiy serves as a 'score per square'
        self.convolutions = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.LeakyReLU(negative_slope=.05),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.LeakyReLU(negative_slope=.05)
        )
        self.convolutions_left = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.05)
        )
        self.convolutions_right = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.05)
        )
        self.actor_output_left = nn.LogSoftmax(dim=1)
        self.actor_output_right = nn.LogSoftmax(dim=1)

    def init_critic(self):
        """
        Initialize the critic, who calculates the action values.
        :return:
        """
        self.critic_identity_l = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.Tanh()
        )
        self.critic_identity_r = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.Tanh()
        )
        self.critic_convolutions = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.LeakyReLU(negative_slope=.1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.LeakyReLU(negative_slope=.1),
        )
        self.critic_convolutions_left = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.1)
        )
        self.critic_convolutions_right = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=.1)
        )

    def forward(self, state):
        """
        A forward pass through the neural network
        :param state: torch.Tensor of (1, 8, 8, 8)
        :return: (action_probabilites, q_values)
        """

        actor_identity_l = self.identity_l(state)
        actor_identity_r = self.identity_r(state)
        actor_convolutions = self.convolutions(state.clone())
        actor_convolutions_left = self.convolutions_left(actor_convolutions)
        actor_convolutions_right = self.convolutions_right(actor_convolutions)
        actor_left = torch.exp(self.actor_output_left(actor_identity_l + actor_convolutions_left))
        actor_right = torch.exp(self.actor_output_right(actor_identity_r + actor_convolutions_right))

        a_dot = torch.bmm(actor_left.unsqueeze(-1), actor_right.unsqueeze(-2))

        critic_identity_l = self.critic_identity_l(state.clone())
        critic_identity_r = self.critic_identity_r(state.clone())
        critic_convolutions = self.critic_convolutions(state.clone())
        critic_convolutions_left = self.critic_convolutions_left(critic_convolutions)
        critic_convolutions_right = self.critic_convolutions_right(critic_convolutions)
        critic_left = critic_convolutions_left + critic_identity_l
        critic_right = critic_convolutions_right + critic_identity_r

        critic_dot = torch.bmm(critic_left.unsqueeze(-1), critic_right.unsqueeze(-2))

        return a_dot, critic_dot


    def select_action(self, env, greedy=False):
        """
        Select an action using the action probabilites supplied by the actor
        :param env: Python chess environment
        :param bool greedy: Whether to always pick the most probable action
        :return: a move and its probability
        """
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
        action_probs, q_value_pred = self(state)
        action_probs = self.legalize_action_probs(action_probs, action_space)
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

    def td_update(self, fixed_model, episode_active, state, actions, reward, successor_state, successor_actions):
        """
        Performs a Temporal Difference Update on the network
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

        policy_gradient_loss = -(
                (target_action_probs_flat * torch.log(action_probs_flat)).sum(dim=1) * advantages_vec[:,
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
        self.verify_td_update(state, actions, q_target_vec, q_pred_vec, advantages_vec, action_probs)

    def legalize_action_probs(self, action_probs, action_space_tensor, eps=1e-6):
        """
        Sets the probability of illegal moves to 0 and re-normalizes the probabilities over the action space
        :param action_probs: torch.tensor = Action probabilites gives by the policy
        :param action_space_tensor: torch.tensor = Action Space
        :param eps: float = Small number for numerical stability
        :return: torch.tensor = action_probs_legal_normalized
        """
        action_probs_legal = (action_probs + torch.tensor([eps]).float()) * action_space_tensor
        action_probs_legal_normalized = action_probs_legal / action_probs_legal.sum()
        return action_probs_legal_normalized

    def network_update_mc(self, state, move, Returns, action_space):
        """
        Update the actor and the critic based on observed Returns in Monte Carlo simulations
        :param torch.tensor state: The root state of the board
        :param move: The move that was taken
        :param Returns: The observed return in the Monte Carlo simulations
        :param action_space: The action space
        :return:
        """
        Returns_tensor = torch.tensor([Returns]).float()
        state_tensor = torch.from_numpy(state).unsqueeze(dim=0).float()
        action_space_tensor = torch.from_numpy(action_space).unsqueeze(dim=0).float()
        if state_tensor[0, 6, 0, :].detach().numpy().sum() == -8.0:
            color = torch.Tensor([-1.])
        else:
            color = torch.Tensor([1.])
        action_probs, q_values = self(state_tensor)
        action_probs_legal = self.legalize_action_probs(action_probs, action_space_tensor)

        # Adjust advantage for mean advantage
        q_copy = q_values.clone().detach()
        advantage = (Returns_tensor - (q_copy * action_space_tensor).mean())

        predicted_returns = q_values[[0], move.from_square, move.to_square]
        proba = action_probs_legal[[0], move.from_square, move.to_square]
        q_value_loss = F.mse_loss(Returns_tensor, predicted_returns)
        policy_gradient_loss = -torch.log(proba) * advantage * color
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
        print("Advantage", advantage * color)
        print("proba ", proba)
        print("new proba ", proba_post)
        print("Change in proba", proba_post - proba)

        print("expected ", predicted_returns)
        print("new expected ", predicted_returns_post)
        print("Change in Q", predicted_returns_post - predicted_returns)
        print("Loss reduction Q", F.mse_loss(Returns_tensor, predicted_returns_post) - q_value_loss)

    def verify_td_update(self, state, actions, q_values_target, q_values_before, advantages, action_probs_before):
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
        print("ADVANTAGE_PROBA_CORREL", np.corrcoef(x=advantages_numpy, y=proba_changes_numpy)[0, 1])


class RandomAgent(object):

    def predict(self, board_layer):
        return np.random.randint(-5, 5) / 5

    def select_move(self, board):
        moves = [x for x in board.generate_legal_moves()]
        return np.random.choice(moves)
