# Based off github.com/gsurma
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from CartPoleConfig import *

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.epsilon = EPSILON

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(EPSILON_MIN, self.epsilon)


def cartpole():
    env = gym.make(ENVIRONMENT)
    env.seed(0)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    scores = deque(maxlen=CONSECUTIVE_EPISODES_TO_SOLVE)

    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        tick = 0

        while True:
            tick += 1
            env.render()
            action = dqn_solver.choose_action(state)
            state_next, reward, done, _ = env.step(action)
            if not done:
                reward = reward
            else:
                reward = -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next

            if done or tick >= 200:
                scores.append(tick)
                mean_score = np.mean(scores)
                text = "[Episode {}] - Survival time this episode was {} ticks with epsilon = {}".format(episode, tick, dqn_solver.epsilon)
                text2 = "- Over last {} episodes: Min = {}, Mean = {:.2f}, Max = {}".format(CONSECUTIVE_EPISODES_TO_SOLVE, min(scores), mean_score, max(scores))
                print(text + "\n" + (11+len(str(episode)))*' '+text2)
                if mean_score >= SOLVE_TICKS and len(scores) >= CONSECUTIVE_EPISODES_TO_SOLVE:
                    solve_score = episode - CONSECUTIVE_EPISODES_TO_SOLVE
                    print("Solved in {} episodes, with {} total episodes.".format(solve_score, episode))
                    env.close()
                    exit()
                break

            dqn_solver.experience_replay()

if __name__ == "__main__":
    cartpole()