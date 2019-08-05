from keras.models import Model, clone_model
from keras.layers import Input, Conv2D, Dense, Flatten, Concatenate
from keras.optimizers import SGD
from keras.losses import  mean_squared_error
import keras.backend as K

def policy_gradient_loss(Returns):
    def modified_crossentropy(action,action_probs):
        cost = (K.categorical_crossentropy(action,action_probs,from_logits=False,axis=1) * Returns)
        return cost
    return modified_crossentropy

class Agent(object):

    def __init__(self):
        self.network = Model()
        self.init_network()

    def init_network(self):
        optimizer=SGD()
        layer_state = Input(shape=(8,8,8),name='state')
        V = Input(shape=(8,8,8),name='value_estimate')
        c1 = Conv2D(8,(1,1),padding='valid',activation='relu',name='conv1')(layer_state)
        c2 = Conv2D(8,(4,4),padding='valid',activation='relu',name='conv2')(c1)
        flat1 = Flatten(name='conv1_flat')(c1)
        flat2 = Flatten(name='conv2_flat')(c2)
        concat3 = Concatenate(name='final_dense')([flat1,flat2])
        value_head = Dense(1)(concat3)
        action_probs = Dense(4096,activation='softmax')(concat3)
        self.network = Model(inputs=layer_state,
                               outputs=[value_head,action_probs])
        self.network.compile(optimizer=optimizer,
                             loss=[mean_squared_error,self.policy_gradient_loss(V)]
                            )











