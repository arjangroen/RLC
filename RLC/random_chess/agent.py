from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Reshape
from keras.optimizers import SGD
import numpy as np

class Agent(object):

    def __init__(self,alpha=0.05,lamb=0.9,gamma=0.9,epsilon=0.5,memory=[]):
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_network()


    def init_network(self):
        self.init_naive_network()

    def init_naive_network(self):
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        input_layer = Input(shape=(8,8,8),name='board_layer')
        reshape_input = Reshape((512,))(input_layer)
        intermediate = Dense(1028)(reshape_input)
        output_layer = Dense(4096)(intermediate)
        self.model = Model(inputs=[input_layer],outputs=[output_layer])
        self.model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

    def init_conv_network(self):
        self.model = Model()
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        intermediate = Conv2D(16,(2,2))(input_layer)  # 16,7,7
        intermediate = Conv2D(32,(2,2))(intermediate)  # 32,6,6
        intermediate = Conv2D(64,(2,2))(intermediate)  # 64,5,5
        intermediate = Conv2D(128,(2,2))(intermediate)  # 128,4,4
        intermediate = Conv2D(256, (2, 2))(intermediate)  # 256,3,3
        intermediate = Dense(1024)(intermediate)
        output = Dense(4096)(intermediate)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

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
















