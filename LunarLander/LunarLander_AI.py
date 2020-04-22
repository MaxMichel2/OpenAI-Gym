# IMPORTS
import gym
import random
from collections import deque

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear

import numpy as np

from LunarLanderConfig import *

# GLOBAL VARIABLES
env = gym.make(ENVIRONMENT)

class LunarLanderDQNAgent:
    # Initialise Agent
    def __init__(self, action_space, observation_space):

        self.action_space = action_space
        self.observation_space = observation_space

        self.number_of_actions = self.action_space.n
        self.number_of_observations = self.observation_space.shape[0]

        self.epsilon = EPSILON # Exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA # Discount factor
        self.alpha = ALPHA # Learning rate
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self.build_model()

    # Initialise Neural Network model 
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.number_of_observations, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(128, activation=relu))
        model.add(Dense(self.number_of_actions, activation=linear))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.alpha), metrics=['accuracy'])
        return model

    # Append the properties of a given state and step in the future
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action based on the exploration rate and current state of the network
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        # Calling i[x] in the following lines triggers a false-positive pylint unsubscriptable error
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        index = np.array([i for i in range(min(len(self.memory), self.batch_size))])
        targets_full[[index], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        

    def preprocess_state(self, state):
        return np.reshape(state, (1, self.number_of_observations))

def train_DQN(episodes=EPISODES):
    scores = deque(maxlen=CONSECUTIVE_EPISODES_TO_SOLVE)
    average_score = 0
    agent = LunarLanderDQNAgent(env.action_space, env.observation_space)

    for episode in range(episodes):
        state = agent.preprocess_state(env.reset())
        score = 0

        for _ in range(MAX_TICKS):
            action = agent.choose_action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = agent.preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                scores.append(score)
                average_score = np.mean(scores)

                text = "[Episode {} of {}] - Score time this episode was {} with epsilon = {}".format(episode, episodes, score, agent.epsilon)
                text2 = "- Over last {} episodes: Min = {:.2f}, Mean = {:.2f}, Max = {:.2f}".format(CONSECUTIVE_EPISODES_TO_SOLVE, min(scores), average_score, max(scores))
                print(text + "\n" + (15 + len(str(episode)) + len(str(episodes)))*' '+text2)

                if average_score >= POINTS_TO_SOLVE:
                    print("Lunar Lander solved in {} episodes with an average of {} points".format((episode-CONSECUTIVE_EPISODES_TO_SOLVE), average_score))
                    env.close()
                    exit()
                break
        agent.replay()


if __name__ == "__main__":
    train_DQN()