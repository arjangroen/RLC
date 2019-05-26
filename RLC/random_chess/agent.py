from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Reshape, Flatten, Dot
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf

class Agent(object):

    def __init__(self,alpha=0.05,lamb=0.9,gamma=0.9,epsilon=0.5):
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_network()


    def init_network(self):
        self.init_conv_network()

    def init_naive_network(self):
        optimizer = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        input_layer = Input(shape=(8,8,8),name='board_layer')
        reshape_input = Reshape((512,))(input_layer)
        #intermediate = Dense(1028)(reshape_input)
        output_layer = Dense(4096)(reshape_input)
        self.model = Model(inputs=[input_layer],outputs=[output_layer])
        self.model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

    def init_conv_network(self):
        optimizer = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8

        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)

        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)

        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def network_update(self,minibatch,gamma=0.9):
        states, moves, rewards, new_states = [],[],[],[]
        for sample in minibatch:
            states.append(sample[0])
            moves.append(sample[1])
            rewards.append(sample[2])
            new_states.append(sample[3])

        q_targets = np.array(rewards) + gamma * np.max(self.model.predict(np.stack(new_states,axis=0)),axis=1)  # Max value of actions in new state
        q_state = self.model.predict(np.stack(states,axis=0))  # batch x 64 x 64
        q_state = np.reshape(q_state,(len(minibatch),64,64))
        for idx, move in enumerate(moves):
            q_state[idx,move[0],move[1]] = q_targets[idx]
        q_state = np.reshape(q_state,(len(minibatch),4096))
        self.model.fit(x=np.stack(states,axis=0),y=q_state,epochs=1)

    def get_action_values(self,state):
        return self.model.predict(state)
















