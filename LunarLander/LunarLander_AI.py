# IMPORTS
import gym
import random
import sys
import os
from collections import deque

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
from keras.callbacks import CSVLogger

import numpy as np

from LunarLanderConfig import *

# Save the state of the network to a .h5 file
import h5py
import argparse
import csv


class LunarLanderDQNAgent:
    # Initialise Agent
    def __init__(self):
        render, test_model = self._args()
        self.env = gym.make(ENVIRONMENT)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render = render

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
        
        self.save_name = ENVIRONMENT+'/'+ENVIRONMENT
        self.history = [('Episode', 'Score', 'Average score', 'Steps', 'Total steps')]
        self.csv_loss_logger = CSVLogger(ENVIRONMENT + '/' + ENVIRONMENT + '_loss.csv', append=True, separator=',')

        if test_model:
            self.load_model(test_model)
            self.test_agent()
        else:
            try:
                os.mkdir(ENVIRONMENT)
            except FileExistsError:
                pass
            self.train_agent()

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

        self.model.fit(states, targets_full, epochs=1, verbose=0, callbacks=[self.csv_loss_logger])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        

    def preprocess_state(self, state):
        return np.reshape(state, (1, self.number_of_observations))

    # Will probably add the train function here
    def train_agent(self):
        try:
            score_history = deque(maxlen=CONSECUTIVE_EPISODES_TO_SOLVE)
            total_steps = 0

            for episode in range(EPISODES):
                # Reset state at the beginning of game
                state = self.preprocess_state(self.env.reset())
                steps = 0
                score = 0

                while True: # Could also iterate to a maximum number of steps/ticks/frames in LunarLanderConfig
                    # Increment step count at each frame
                    steps += 1
                    total_steps += 1

                    # Render or not
                    if self.render:
                        self.env.render()

                    # Choose an action
                    action = self.choose_action(state)

                    # Take action and move to next step
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)

                    # Adjust score
                    score += reward

                    # Add to memory
                    self.remember(state, action, reward, next_state, done)

                    # Train with the experience
                    self.replay()

                    if done:
                        score_history.append(score)
                        average_score = np.mean(score_history)

                        text = "[Episode {} of {}] - Score time this episode was {} with epsilon = {}".format(episode, episodes, score, self.epsilon)
                        text2 = "- Over last {} episodes: Min = {:.2f}, Mean = {:.2f}, Max = {:.2f}".format(CONSECUTIVE_EPISODES_TO_SOLVE, min(score_history), average_score, max(score_history))
                        text3 = "- Steps this episode: {}, Total steps: {}".format(steps, total_steps)
                        print(text + "\n" + (15 + len(str(episode)) + len(str(episodes)))*' '+ text2 + "\n" + (15 + len(str(episode)) + len(str(episodes)))*' '+ text3)

                        # Check if the goal has been reached
                        if average_score >= POINTS_TO_SOLVE:
                            print("Lunar Lander solved in {} episodes with an average of {} points".format((episode-CONSECUTIVE_EPISODES_TO_SOLVE), average_score))
                            filename = self.save_name + '_final.h5'
                            print("Saving model to {}".format(filename))
                            self.save_model(filename)
                            self.exit()
                        break
                    
                    # If not done, advance to the next state for the following iteration
                    state = next_state

                # Save weights every 100 episodes
                if episode % 100 == 0:
                    filename = self.save_name + '_' + str(episode) + '.h5'
                    self.save_model(filename)
            
            self.exit()

        except KeyboardInterrupt:
            # Catch Ctrl+C and end the game correctly
            filename = self.save_name + '_final.h5'
            print("Saving model to {}".format(filename))
            self.save_model(filename)
            self.exit()
        except:
            self.env.close()
            sys.exit()

    def test_agent(self):
        try:
            score_history = deque(maxlen=CONSECUTIVE_EPISODES_TO_SOLVE)
            total_steps = 0

            for episode in range(CONSECUTIVE_EPISODES_TO_SOLVE):
                # Reset state at the beginning of game
                state = self.preprocess_state(self.env.reset())
                steps = 0
                score = 0

                while True: # Could also iterate to a maximum number of steps/ticks/frames in LunarLanderConfig
                    # Increment step count at each frame
                    steps += 1
                    total_steps += 1

                    # Render or not
                    if self.render:
                        self.env.render()

                    # Choose an action
                    action = self.choose_action(state)

                    # Take action and move to next step
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)

                    # Adjust score
                    score += reward

                    if done:
                        score_history.append(score)
                        average_score = np.mean(score_history)

                        text = "[Episode {} of 99] - Score time this episode was {} with epsilon = {}".format(episode, score, self.epsilon)
                        text2 = "- Over last {} episodes: Min = {:.2f}, Mean = {:.2f}, Max = {:.2f}".format(CONSECUTIVE_EPISODES_TO_SOLVE, min(score_history), average_score, max(score_history))
                        text3 = "- Steps this episode: {}, Total steps: {}".format(steps, total_steps)
                        print(text + "\n" + (17 + len(str(episode)))*' '+ text2 + "\n" + (17 + len(str(episode)))*' '+ text3)
                        break
                    
                    # If not done, advance to the next state for the following iteration
                    state = next_state
            
            self.env.close()

        except :
            print("Killing game")
            self.env.close()
            sys.exit()

    def exit(self):
        filename = self.save_name + '_history.csv'
        print("Saving training history to {}".format(filename))
        with open(filename, "w") as out:
            csv_out = csv.writer(out)
            for row in self.history:
                csv_out.writerow(row)
        
        print("Killing game")
        self.env.close()
        sys.exit()

    def save_model(self, filename):
        self.model.save_weights(filename)
    
    def load_model(self, filename):
        self.model.load_weights(filename)

    # Argument parser for agent options
    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--render', help="Render the game or not", default=True, type=bool)
        parser.add_argument('-tm', '--test_model', help="Filename of model of weights to test the performance of", default=None)
        args = parser.parse_args()
        render = args.render
        test_model = args.test_model

        return render, test_model

if __name__ == "__main__":
    LunarLanderDQNAgent()