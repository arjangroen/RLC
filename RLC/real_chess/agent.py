import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)

torch.autograd.set_detect_anomaly(True)

NUM_FEATURES = 1


class ReadState(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 1x1 convolutions, 2x2, 3x3 etc.
        self.convolution = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        conv_out = self.convolution(x)
        out = self.tanh(conv_out)
        return out


class MiniHead(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=64)

    def forward(self, x):
        x_from = self.flatten(x)
        x_to = self.linear(x_from)
        return x_from, x_to


class Block(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(num_features=4),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=4, out_channels=4,
                                             kernel_size=3, padding='same'),
                                   nn.BatchNorm2d(num_features=4)
                                   )

    def forward(self, x):
        x = x + self.block(x)
        return x


class Head(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.head = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1),
                                  nn.BatchNorm2d(num_features=16),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=8,
                                            kernel_size=3, padding='same'),
                                  nn.BatchNorm2d(num_features=8),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels=8, out_channels=4,
                                            kernel_size=3, padding='same'),
                                  nn.BatchNorm2d(num_features=4),
                                  nn.LeakyReLU(),
                                  # nn.Dropout2d(p=.25),
                                  nn.Flatten(),
                                  nn.Linear(in_features=8*8*4,
                                            out_features=8*8*2),
                                  # nn.Dropout(p=0.2),
                                  nn.LeakyReLU(),
                                  nn.Linear(in_features=8*8*2,
                                            out_features=8*8*1),
                                  )

    def forward(self, x):
        x = self.head(x)
        return x


class ChessNetwork(nn.Module):

    def __init__(self, is_actor=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_actor = is_actor
        self.input_module = ReadState()
        self.blocks = nn.Sequential(*[Block() for _ in range(3)])
        self.head_from = Head()
        self.head_to = Head()

    def forward(self, x):
        logging.info("torch.shape: %s", x.shape)
        color = x[:, 6, :, :].mean() if self.is_actor else torch.Tensor([1.])
        x = self.input_module(x)
        x = self.blocks(x)
        x_from = self.head_from(x)
        x_to = self.head_to(x)
        x = torch.bmm(x_from.unsqueeze(-1), x_to.unsqueeze(-2)) * color
        # batch_dim = x.shape[0]
        # board_dim = x.shape[1] * x.shape[2]
        # x = x.reshape(batch_dim, board_dim)
        return x


class MiniChessNetwork(nn.Module):

    def __init__(self, is_actor=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_actor = is_actor
        self.layer_1 = ReadState()
        self.head = MiniHead()

    def forward(self, x):
        color = x[:, 6, :, :].mean() if self.is_actor else torch.Tensor([1.])
        x = self.layer_1(x)
        x_from, x_to = self.head(x)
        x = torch.bmm(x_from.unsqueeze(-1), x_to.unsqueeze(-2)) * color
        return x


class NanoChessNetwork(nn.Module):

    def __init__(self, is_actor=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_actor = is_actor
        self.conv_1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.flat = nn.Flatten()

    def forward(self, x):
        color = x[:, 6, :, :].mean() if self.is_actor else torch.Tensor([1.])
        x = color * x
        x = self.conv_1(x)
        x = self.flat(x)
        return x


class NanoActorCritic(nn.Module):

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
        self.actor = NanoChessNetwork(is_actor=True)
        self.critic = NanoChessNetwork()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4, betas=(0.1, 0.11))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-5)
        self.historical_pg_losses = torch.tensor([0.]).float()
        self.replay_importance = []

    def forward(self, state):
        """
        A forward pass through the neural network
        :param state: torch.Tensor of (1, 8, 8, 8)
        :return: (action_probabilites, q_values)
        """

        action_logits = self.actor(state)
        action_values = self.critic(state)

        return action_logits, action_values

    def select_action(self, env, greedy=False):
        """
        Select an action using the action probabilites supplied by the actor
        :param env: Python chess environment
        :param bool greedy: Whether to always pick the most probable action
        :return: a move and its probability
        """
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(
            env.layer_board, axis=0)).float()
        action_logits, action_values = self(state)
        action_logits = self.legalize_action_logits(
            action_logits, action_space)
        action_logits = action_logits.reshape(1, 4096)
        action_probs = F.softmax(
            action_logits, dim=1).detach()
        if greedy:
            move = np.argmax(action_probs[0])
        else:
            move = np.random.choice(range(4096), p=action_probs[0].numpy())
        move_proba = action_probs[0][move]
        move_from = move // 64
        move_to = move % 64
        moves = [x for x in env.board.generate_legal_moves() if
                 x.from_square == move_from and x.to_square == move_to]
        # When promoting a pawn, multiple moves can have the same from-to squares
        move = moves[0]
        return move, move_proba

    def td_update(self, fixed_model, episode_active, state, actions, reward, successor_state, successor_actions,
                  action_spaces):
        """
        Performs a Temporal Difference Update on the network
        :param fixed_model, stationary ActorCritic model
        :param states: Tensor of shape (n_samples, 8, 8, 8)
        :param actions: list[chess.move]
        :param rewards: Tensor of shape (n_samples, 1)
        :param successor_states: Tensor of shape (n_samples, 8, 8, 8)
        :param successor_actions: list[chess.move]
        :param action_spaces: Tensor for shape (n_samples, 64, 64)
        :return:
        """
        target_q_values = reward.unsqueeze(dim=2) + episode_active.unsqueeze(dim=2) * torch.tensor(self.gamma) * \
            fixed_model(successor_state)[1]
        target_q_values = target_q_values.detach()

        target_action_probs = torch.zeros_like(
            target_q_values, dtype=torch.float).detach()
        action_logits, q_values = self(state)
        colors = []
        target_q_values_a = []
        predicted_q_values_a = []

        # Q VALUE LOSS
        for i in range(len(actions)):
            bootstrapped_q_value_a = target_q_values[[
                i], successor_actions[i].from_square, successor_actions[i].to_square].float()
            q_value_a = q_values[[i], actions[i].from_square,
                                 actions[i].to_square].float()
            if successor_state[0, 6, 0, :].detach().numpy().sum() == 8.0:
                colors.append(torch.Tensor([-1.]).float())
            else:
                colors.append(torch.Tensor([1.]).float())
            # q_values_target[[i], actions[i].from_square, actions[i].to_square] = bootstrapped_q_value_a.detach()
            target_action_probs[[i], actions[i].from_square,
                                actions[i].to_square] = torch.Tensor([1.]).float()
            target_q_values_a.append(bootstrapped_q_value_a)
            predicted_q_values_a.append(q_value_a)

        q_pred_vec = torch.cat(predicted_q_values_a, dim=0)
        q_target_vec = torch.cat(target_q_values_a, dim=0)

        q_value_loss = F.mse_loss(q_pred_vec, q_target_vec)

        # POLICY GRADIENT
        colors_vec = torch.cat(colors, dim=0).unsqueeze(
            dim=1).unsqueeze(dim=2).detach()
        advantages_vec = q_target_vec.unsqueeze(
            dim=1).unsqueeze(dim=2).detach() * colors_vec

        action_spaces_flat = torch.flatten(action_spaces, start_dim=1)
        action_probs_flat = torch.flatten(
            action_probs, start_dim=1) + torch.Tensor([1e-6]).float()
        action_probs_flat_legal = self.legalize_action_probs(
            action_probs_flat, action_spaces_flat)

        target_action_probs_flat = torch.flatten(
            target_action_probs, start_dim=1)

        crossentropy = self.cross_entropy_loss(
            action_probs_flat_legal, target_action_probs_flat.argmax(dim=1))

        policy_gradient_loss = crossentropy * advantages_vec[:, 0, 0].mean()
        loss_balance = policy_gradient_loss / q_value_loss

        print("LOSS RATIO PG_LOSS / Q_LOSS:", loss_balance)

        # GRADIENT DESCENT
        total_loss = q_value_loss + policy_gradient_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print("updated AC")
        self.verify_td_update(state, actions, q_target_vec,
                              q_pred_vec, advantages_vec, action_probs)

    def get_proba_for_move(self, move, action_logits, action_space_tensor):
        probas = F.softmax(self.legalize_action_logits(
            action_logits, action_space_tensor))
        proba = probas.view(probas.shape[0], 64, 64)[
            [0], move.from_square, move.to_square]
        return proba

    def flatten_for_softmax(self, action_logits):
        batch_dim = action_logits.shape[0]
        if len(action_logits.shape) == 3:
            action_logits = action_logits.reshape(batch_dim, 64*64)
        return action_logits

    def legalize_action_logits(self, action_logits, action_space_tensor):
        """
        Sets the probability of illegal moves to 0 and re-normalizes the probabilities over the action space
        :param action_probs: torch.tensor = Action probabilites gives by the policy
        :param action_space_tensor: torch.tensor = Action Space
        :param eps: float = Small number for numerical stability
        :return: torch.tensor = action_probs_legal_normalized
        """
        batch_dim = action_space_tensor.shape[0]

        if len(action_space_tensor.shape) == 3:
            action_space_tensor = action_space_tensor.reshape(batch_dim, 64*64)

        if len(action_logits.shape) == 3:
            action_logits = action_logits.reshape(batch_dim, 64*64)

        action_logits_legal = action_logits.masked_fill(
            action_space_tensor == 0, float('-inf'))
        return action_logits_legal

    def mc_update_agent(self, state, move, Returns, action_space, fixed_model):
        """
        Update the actor and the critic based on observed Returns in Monte Carlo simulations
        :param torch.tensor state: The root state of the board
        :param move: The move that was taken
        :param Returns: The observed return in the Monte Carlo simulations
        :param action_space: The action space
        :return:
        """
        n_legal_actions = action_space.sum()
        if n_legal_actions == 1:
            return

        Returns_tensor = torch.tensor(Returns).float()
        state_tensor = torch.from_numpy(state).unsqueeze(dim=0).float()
        action_space_tensor = torch.from_numpy(
            action_space).unsqueeze(dim=0).float()

        if state_tensor[0, 6, 0, :].detach().numpy().sum() == -8.0:
            color = torch.Tensor([-1.])
        else:
            color = torch.Tensor([1.])

        action_logits, q_values = self(state_tensor)
        action_logits_old, _ = fixed_model(state_tensor)
        action_logits_old = action_logits_old.detach()
        action_prob = self.get_proba_for_move(
            move, action_logits, action_space_tensor)
        action_prob_fixed = self.get_proba_for_move(
            move, action_logits_old, action_space_tensor)

        # entropy = torch.ones_like(action_space_tensor) / torch.tensor([4096.]).float()
        # alpha = torch.tensor([1e3])
        # entropy_bonus = alpha * - \
        #    (torch.log(action_probs) * action_probs).mean().mean()

        # Adjust advantage for mean advantage
        q_copy = q_values.clone().detach()
        # advantage = (Returns_tensor - (q_copy *
        #             action_space_tensor).sum()/n_legal_actions) * color
        advantage = (Returns_tensor * color) - torch.tensor([0.001])
        # advantage = Returns_tensor  # Use returns a advantage for now

        predicted_returns = q_values[[0], move.from_square, move.to_square]

        proba_rt_clipped = torch.clamp(
            action_prob/action_prob_fixed, min=0.9, max=1.1)
        proba_rt_unclipped = action_prob/action_prob_fixed

        ppo_loss = -torch.min(proba_rt_clipped*advantage,
                              proba_rt_unclipped*advantage)

        q_value_loss = F.mse_loss(Returns_tensor, predicted_returns)

        # The chance updating to proportional to the relative size of the loss.
        self.historical_pg_losses = torch.cat(
            (self.historical_pg_losses, ppo_loss.clone().detach()))
        if len(self.historical_pg_losses) > 1000:
            self.historical_pg_losses = self.historical_pg_losses[1:]

        action_space_tensor_backup = action_space_tensor.detach().numpy()

        self.critic_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        ppo_loss.backward()
        self.actor_optimizer.step()

        action_space_tensor_backup_2 = action_space_tensor.detach().numpy()

        # VERIFY successfull optimization step
        action_logits_post, q_values_post = self(state_tensor)
        predicted_returns_post = q_values_post[[
            0], move.from_square, move.to_square]
        proba_post = self.get_proba_for_move(
            move, action_logits_post, action_space_tensor)

        logging.info("\nLEARNED POLICY")
        logging.info("Q LOSS %s", q_value_loss)
        logging.info("POLICY GRADIENT LOSS %s", ppo_loss)
        logging.info("COLOR: %s", color)
        logging.info("Returns %s", Returns)
        logging.info("Advantage %s", advantage)
        logging.info("proba %s", action_prob)
        logging.info("new proba %s", proba_post)
        logging.info("Change in proba %s", proba_post - action_prob)
        logging.info("right policy direction %s", advantage *
                     (proba_post - action_prob) >= 0)
        logging.info("action space tensor unchanged %s",
                     (action_space_tensor_backup == action_space_tensor_backup_2).all())
        logging.info("proba_rt %s", proba_rt_unclipped)
        logging.info("new proba_rt %s", proba_post/action_prob_fixed)

        logging.info("expected %s", predicted_returns)
        logging.info("new expected %s", predicted_returns_post)
        logging.info("Change in Q %s",
                     predicted_returns_post - predicted_returns)
        logging.info("Loss reduction Q %s", F.mse_loss(
            Returns_tensor, predicted_returns_post) - q_value_loss)


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
        self.actor = MiniChessNetwork(is_actor=True)
        self.critic = MiniChessNetwork()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4, betas=(0.1, 0.11))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-5)
        self.historical_pg_losses = torch.tensor([0.]).float()
        self.replay_importance = []

    def forward(self, state):
        """
        A forward pass through the neural network
        :param state: torch.Tensor of (1, 8, 8, 8)
        :return: (action_probabilites, q_values)
        """

        action_logits = self.actor(state)
        action_values = self.critic(state)

        return action_logits, action_values

    def select_action(self, env, greedy=False):
        """
        Select an action using the action probabilites supplied by the actor
        :param env: Python chess environment
        :param bool greedy: Whether to always pick the most probable action
        :return: a move and its probability
        """
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(
            env.layer_board, axis=0)).float()
        action_logits, action_values = self(state)
        action_logits = self.legalize_action_logits(
            action_logits, action_space)
        action_logits = action_logits.reshape(1, 4096)
        action_probs = F.softmax(
            action_logits, dim=1).detach()
        if greedy:
            move = np.argmax(action_probs[0])
        else:
            move = np.random.choice(range(4096), p=action_probs[0].numpy())
        move_proba = action_probs[0][move]
        move_from = move // 64
        move_to = move % 64
        moves = [x for x in env.board.generate_legal_moves() if
                 x.from_square == move_from and x.to_square == move_to]
        # When promoting a pawn, multiple moves can have the same from-to squares
        move = moves[0]
        return move, move_proba

    def td_update(self, fixed_model, episode_active, state, actions, reward, successor_state, successor_actions,
                  action_spaces):
        """
        Performs a Temporal Difference Update on the network
        :param fixed_model, stationary ActorCritic model
        :param states: Tensor of shape (n_samples, 8, 8, 8)
        :param actions: list[chess.move]
        :param rewards: Tensor of shape (n_samples, 1)
        :param successor_states: Tensor of shape (n_samples, 8, 8, 8)
        :param successor_actions: list[chess.move]
        :param action_spaces: Tensor for shape (n_samples, 64, 64)
        :return:
        """
        target_q_values = reward.unsqueeze(dim=2) + episode_active.unsqueeze(dim=2) * torch.tensor(self.gamma) * \
            fixed_model(successor_state)[1]
        target_q_values = target_q_values.detach()

        target_action_probs = torch.zeros_like(
            target_q_values, dtype=torch.float).detach()
        action_logits, q_values = self(state)
        colors = []
        target_q_values_a = []
        predicted_q_values_a = []

        # Q VALUE LOSS
        for i in range(len(actions)):
            bootstrapped_q_value_a = target_q_values[[
                i], successor_actions[i].from_square, successor_actions[i].to_square].float()
            q_value_a = q_values[[i], actions[i].from_square,
                                 actions[i].to_square].float()
            if successor_state[0, 6, 0, :].detach().numpy().sum() == 8.0:
                colors.append(torch.Tensor([-1.]).float())
            else:
                colors.append(torch.Tensor([1.]).float())
            # q_values_target[[i], actions[i].from_square, actions[i].to_square] = bootstrapped_q_value_a.detach()
            target_action_probs[[i], actions[i].from_square,
                                actions[i].to_square] = torch.Tensor([1.]).float()
            target_q_values_a.append(bootstrapped_q_value_a)
            predicted_q_values_a.append(q_value_a)

        q_pred_vec = torch.cat(predicted_q_values_a, dim=0)
        q_target_vec = torch.cat(target_q_values_a, dim=0)

        q_value_loss = F.mse_loss(q_pred_vec, q_target_vec)

        # POLICY GRADIENT
        colors_vec = torch.cat(colors, dim=0).unsqueeze(
            dim=1).unsqueeze(dim=2).detach()
        advantages_vec = q_target_vec.unsqueeze(
            dim=1).unsqueeze(dim=2).detach() * colors_vec

        action_spaces_flat = torch.flatten(action_spaces, start_dim=1)
        action_probs_flat = torch.flatten(
            action_probs, start_dim=1) + torch.Tensor([1e-6]).float()
        action_probs_flat_legal = self.legalize_action_probs(
            action_probs_flat, action_spaces_flat)

        target_action_probs_flat = torch.flatten(
            target_action_probs, start_dim=1)

        crossentropy = self.cross_entropy_loss(
            action_probs_flat_legal, target_action_probs_flat.argmax(dim=1))

        policy_gradient_loss = crossentropy * advantages_vec[:, 0, 0].mean()
        loss_balance = policy_gradient_loss / q_value_loss

        print("LOSS RATIO PG_LOSS / Q_LOSS:", loss_balance)

        # GRADIENT DESCENT
        total_loss = q_value_loss + policy_gradient_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print("updated AC")
        self.verify_td_update(state, actions, q_target_vec,
                              q_pred_vec, advantages_vec, action_probs)

    def get_proba_for_move(self, move, action_logits, action_space_tensor):
        probas = F.softmax(self.legalize_action_logits(
            action_logits, action_space_tensor))
        proba = probas.view(probas.shape[0], 64, 64)[
            [0], move.from_square, move.to_square]
        return proba

    def flatten_for_softmax(self, action_logits):
        batch_dim = action_logits.shape[0]
        if len(action_logits.shape) == 3:
            action_logits = action_logits.reshape(batch_dim, 64*64)
        return action_logits

    def legalize_action_logits(self, action_logits, action_space_tensor):
        """
        Sets the probability of illegal moves to 0 and re-normalizes the probabilities over the action space
        :param action_probs: torch.tensor = Action probabilites gives by the policy
        :param action_space_tensor: torch.tensor = Action Space
        :param eps: float = Small number for numerical stability
        :return: torch.tensor = action_probs_legal_normalized
        """
        batch_dim = action_space_tensor.shape[0]

        if len(action_space_tensor.shape) == 3:
            action_space_tensor = action_space_tensor.reshape(batch_dim, 64*64)

        if len(action_logits.shape) == 3:
            action_logits = action_logits.reshape(batch_dim, 64*64)

        action_logits_legal = action_logits.masked_fill(
            action_space_tensor == 0, float('-inf'))
        return action_logits_legal

    def mc_update_agent(self, state, move, Returns, action_space, fixed_model):
        """
        Update the actor and the critic based on observed Returns in Monte Carlo simulations
        :param torch.tensor state: The root state of the board
        :param move: The move that was taken
        :param Returns: The observed return in the Monte Carlo simulations
        :param action_space: The action space
        :return:
        """
        n_legal_actions = action_space.sum()
        if n_legal_actions == 1:
            return

        Returns_tensor = torch.tensor(Returns).float()
        state_tensor = torch.from_numpy(state).unsqueeze(dim=0).float()
        action_space_tensor = torch.from_numpy(
            action_space).unsqueeze(dim=0).float()

        if state_tensor[0, 6, 0, :].detach().numpy().sum() == -8.0:
            color = torch.Tensor([-1.])
        else:
            color = torch.Tensor([1.])

        action_logits, q_values = self(state_tensor)
        action_logits_old, _ = fixed_model(state_tensor)
        action_logits_old = action_logits_old.detach()
        action_prob = self.get_proba_for_move(
            move, action_logits, action_space_tensor)
        action_prob_fixed = self.get_proba_for_move(
            move, action_logits_old, action_space_tensor)

        # entropy = torch.ones_like(action_space_tensor) / torch.tensor([4096.]).float()
        # alpha = torch.tensor([1e3])
        # entropy_bonus = alpha * - \
        #    (torch.log(action_probs) * action_probs).mean().mean()

        # Adjust advantage for mean advantage
        q_copy = q_values.clone().detach()
        # advantage = (Returns_tensor - (q_copy *
        #             action_space_tensor).sum()/n_legal_actions) * color
        advantage = (Returns_tensor * color) - torch.tensor([0.001])
        # advantage = Returns_tensor  # Use returns a advantage for now

        predicted_returns = q_values[[0], move.from_square, move.to_square]

        proba_rt_clipped = torch.clamp(
            action_prob/action_prob_fixed, min=0.9, max=1.1)
        proba_rt_unclipped = action_prob/action_prob_fixed

        ppo_loss = -torch.min(proba_rt_clipped*advantage,
                              proba_rt_unclipped*advantage)

        q_value_loss = F.mse_loss(Returns_tensor, predicted_returns)

        # The chance updating to proportional to the relative size of the loss.
        self.historical_pg_losses = torch.cat(
            (self.historical_pg_losses, ppo_loss.clone().detach()))
        if len(self.historical_pg_losses) > 1000:
            self.historical_pg_losses = self.historical_pg_losses[1:]

        action_space_tensor_backup = action_space_tensor.detach().numpy()

        self.critic_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        ppo_loss.backward()
        self.actor_optimizer.step()

        action_space_tensor_backup_2 = action_space_tensor.detach().numpy()

        # VERIFY successfull optimization step
        action_logits_post, q_values_post = self(state_tensor)
        predicted_returns_post = q_values_post[[
            0], move.from_square, move.to_square]
        proba_post = self.get_proba_for_move(
            move, action_logits_post, action_space_tensor)

        logging.info("\nLEARNED POLICY")
        logging.info("Q LOSS %s", q_value_loss)
        logging.info("POLICY GRADIENT LOSS %s", ppo_loss)
        logging.info("COLOR: %s", color)
        logging.info("Returns %s", Returns)
        logging.info("Advantage %s", advantage)
        logging.info("proba %s", action_prob)
        logging.info("new proba %s", proba_post)
        logging.info("Change in proba %s", proba_post - action_prob)
        logging.info("right policy direction %s", advantage *
                     (proba_post - action_prob) >= 0)
        logging.info("action space unchanged %s",
                     (action_space_tensor_backup == action_space_tensor_backup_2).all())
        logging.info("proba_rt %s", proba_rt_unclipped)
        logging.info("new proba_rt %s", proba_post/action_prob_fixed)

        logging.info("expected %s", predicted_returns)
        logging.info("new expected %s", predicted_returns_post)
        logging.info("Change in Q %s",
                     predicted_returns_post - predicted_returns)
        logging.info("Loss reduction Q %s", F.mse_loss(
            Returns_tensor, predicted_returns_post) - q_value_loss)

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

            action_prob_after = action_probs_after[[
                i], actions[i].from_square, actions[i].to_square]
            action_prob_before = action_probs_before[[
                i], actions[i].from_square, actions[i].to_square]

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
            proba_change = (action_prob_after -
                            action_prob_before).detach().numpy()
            print("MOVE PROBA CHANGE:", proba_change)
            proba_changes_numpy.append(proba_change)

        proba_changes_numpy = np.array(proba_changes_numpy).squeeze()

        print("\nBATCH Q-LOSS BEFORE", q_loss_pre_batch /
              q_loss_post_batch.shape[0])
        print("BATCH Q-LOSS AFTER", q_loss_post_batch /
              q_loss_post_batch.shape[0])
        print("ADVANTAGE_PROBA_CORREL", np.corrcoef(
            x=advantages_numpy, y=proba_changes_numpy)[0, 1])


class RandomAgent(object):

    def predict(self, board_layer):
        return np.random.randint(-5, 5) / 5

    def select_move(self, board):
        moves = [x for x in board.generate_legal_moves()]
        return np.random.choice(moves)
