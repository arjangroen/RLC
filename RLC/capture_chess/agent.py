from keras.models import Model, clone_model
from keras.layers import Input, Conv2D, Dense, Reshape, Dot, Add
from keras.optimizers import SGD
import numpy as np

class Agent(object):

    def __init__(self,alpha=0.05,lamb=0.9,gamma=0.5,epsilon=0.5,network='linear',lr=0.01):
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon
        self.network = network
        self.lr = lr
        self.init_network()


    def init_network(self):
        if self.network == 'linear':
            self.init_linear_network()
        elif self.network == 'conv':
            self.init_conv_network()

    def fix_model(self):
        optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
        self.fixed_model.set_weights(self.model.get_weights())


    def init_linear_network(self):
        optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)

        input_layer = Input(shape=(8,8,8),name='board_layer')
        reshape_input = Reshape((512,))(input_layer)
        #intermediate = Dense(1028)(reshape_input)
        output_layer = Dense(4096)(reshape_input)
        self.model = Model(inputs=[input_layer],outputs=[output_layer])
        self.model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

    def init_conv_network(self):
        optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
        #optimizer = Adagrad()
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8

        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)

        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def network_update(self,minibatch):
        states, moves, rewards, new_states = [],[],[],[]
        td_errors = []
        episode_ends = []
        for sample in minibatch:
            states.append(sample[0])
            moves.append(sample[1])
            rewards.append(sample[2])
            new_states.append(sample[3])
            if np.array_equal(sample[3], sample[3] * 0):
                episode_ends.append(0)
                print('episode end')
            else:
                episode_ends.append(1)

        q_target = np.array(rewards) + np.array(episode_ends) * self.gamma * np.max(self.fixed_model.predict(np.stack(new_states,axis=0)),axis=1)  # Max value of actions in new state
        q_state = self.model.predict(np.stack(states,axis=0))  # batch x 64 x 64
        q_state = np.reshape(q_state,(len(minibatch),64,64))
        for idx, move in enumerate(moves):
            td_errors.append(np.abs(q_state[idx,move[0],move[1]] - q_target[idx]))
            q_state[idx,move[0],move[1]] = q_target[idx]
        q_state = np.reshape(q_state,(len(minibatch),4096))
        self.model.fit(x=np.stack(states,axis=0),y=q_state,epochs=1)
        return td_errors

    def get_action_values(self,state):
        return self.fixed_model.predict(state)
















