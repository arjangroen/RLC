from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, Dropout
from keras.losses import mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agent(object):

    def __init__(self):
        self.network = Model()
        self.init_network()

    def init_network(self):
        optimizer=Adam()
        layer_state = Input(shape=(8,8,8),name='state')

        openfile = Conv2D(3,(8,1),padding='valid',activation='relu',name='fileconv')(layer_state)  # 3,8,1
        openrank = Conv2D(3,(1,8),padding='valid',activation='relu',name='rankconv')(layer_state)  # 3,1,8
        quarters = Conv2D(3,(4,4),padding='valid',activation='relu',name='quarterconv',strides=(4,4))(layer_state)  # 3,2,2
        large = Conv2D(8,(6,6),padding='valid',activation='relu',name='largeconv')(layer_state)  # 8,2,2

        board1 = Conv2D(16, (3, 3), padding='valid', activation='relu', name='board1')(layer_state)   # 16,6,6
        board2 = Conv2D(20, (3, 3), padding='valid', activation='relu', name='board2')(board1)   # 20,4,4
        board3 = Conv2D(24, (3, 3), padding='valid',activation='relu',name='board3')(board2)  # 24,2,2

        flat_file = Flatten()(openfile)
        flat_rank = Flatten()(openrank)
        flat_quarters = Flatten()(quarters)
        flat_large = Flatten()(large)

        flat_board = Flatten()(board1)
        flat_board3 = Flatten()(board3)

        dense1 = Concatenate(name='dense_bass')([flat_file,flat_rank,flat_quarters,flat_large,flat_board,flat_board3])
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(128)(dropout1)
        dense3 = Dense(64)(dense2)
        dropout3 = Dropout(0.2)(dense3,training=True)
        dense4 = Dense(32)(dropout3)
        dropout4 = Dropout(0.2)(dense4,training=True)

        value_head = Dense(1,activation='tanh')(dropout4)
        self.network = Model(inputs=layer_state,
                               outputs=[value_head])
        self.network.compile(optimizer=optimizer,
                             loss=[mean_squared_error]
                            )

    def predict_distribution(self,states,batch_size=256):
        """
        :param states: list of distinct states
        :param n:  each state is predicted n times
        :return:
        """
        predictions_per_state = int(batch_size / len(states))
        state_batch = []
        for state in states:
            state_batch = state_batch + [state for x in range(predictions_per_state)]

        state_batch = np.stack(state_batch,axis=0)
        predictions = self.network.predict(state_batch)
        predictions = predictions.reshape(len(states),predictions_per_state)
        mean_pred = np.mean(predictions_per_state,axis=1)
        std_pred = np.std(predictions_per_state,axis=1)
        upper_bound = mean_pred + 2*std_pred

        return mean_pred, std_pred, upper_bound


















