import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model


class Actor:
    def __init__(self, state_dim, action_dim, action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        # state = Input(shape=self.state_dim, dtype='float32')
        state = Input(shape=self.state_dim)
        # x = Dense(40, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim)))(
        #     state)
        x = Dense(400, activation='relu')(state)
        # x = Dense(300, activation='relu', kernel_initializer=RU(-1/np.sqrt(400), 1/np.sqrt(400)))(x)
        x = Dense(400, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=RU(-0.003, 0.003))(x)
        return Model(inputs=state, outputs=out)


class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        # state = Input(shape=self.state_dim, name='state_input', dtype='float32')
        state = Input(shape=self.state_dim, name='state_input')
        state_out = Dense(40, activation='relu')(state)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation="relu")(state_out)
        state_out = BatchNormalization()(state_out)

        action = Input(shape=(self.action_dim,), name='action_input')
        action_out = Dense(32, activation="relu")(action)
        action_out = BatchNormalization()(action_out)

        x = concatenate([state_out, action_out])
        out = BatchNormalization()(x)
        out = Dense(512, activation="relu")(out)
        out = BatchNormalization()(out)
        out = Dense(1, activation='linear')(out)
        return Model(inputs=[state, action], outputs=out)
