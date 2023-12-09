import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import HeNormal

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Change size as needed
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Deep Q-Network."""
        model = Sequential()


        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=self.state_size,
                         kernel_initializer=HeNormal(), activation='elu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), kernel_initializer=HeNormal(), activation='elu'))
        model.add(Conv2D(64, (3, 3), kernel_initializer=HeNormal(), activation='elu'))
        
        model.add(Flatten())
        model.add(Dense(512, activation='elu', kernel_initializer=HeNormal()))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=HeNormal()))
        model.add(Dense(256, activation='elu', kernel_initializer=HeNormal()))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer=HeNormal()))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon = True):
        """Returns actions for given state as per current policy."""
        if(use_epsilon):
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        else:
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
    def replay(self, batch_size):
        """Trains the agent with experiences sampled from memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon: " + str(self.epsilon))

