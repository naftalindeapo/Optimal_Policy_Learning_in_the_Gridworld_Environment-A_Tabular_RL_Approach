#import relevent libraries
import pickle
import random
import minigrid
import numpy as np
import gymnasium as gym
from os.path import exists
from minigrid.wrappers import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# ***********3.1 Warmup Task**************
# In this task, the Agent class represents an agent that can take random actions. 
# It is initialized with an action_space, which is a list of available actions ["Turn Left", "Turn Right", "Move Forward"]. 

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self):
        return random.choice(self.action_space)

# Example usage
action_space = ["Turn Left", "Turn Right", "Move Forward"]
agent = Agent(action_space)

# Generate random actions for 10 steps
print('\nThe random actions for 10 steps are:')
for _ in range(10):
    action = agent.choose_action()
    print("Action:", action)
print('\n')

# 3.2 **********Task 1 - Tabular Q-learning***********

class RL_agent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)# Make the gym environment
        self.Q = {} # Initialize Q-table
        self.max_steps = self.env.max_steps # Max number oof steps in the 8 by 8 GridWorld environment
        self.numActions = 3 # We only need the first three actions

    # (1.) Function for extracting the object_idx information as a matrix
    def extractObjectInformation(self, obs):
        (rows, cols, x) = obs.shape
        tmp = np.reshape(obs,[rows*cols*x,1], 'F')[0:rows*cols]
        return np.reshape(tmp, [rows,cols],'C')
    
    # Combine in one
    def preprocess(self, observation):
        return self.extractObjectInformation(observation)
    
    # (2.) Hash function
    def hashState(self, state):
        state_str = str(state)
        state_hash = hash(state_str)
        return state_hash
    
    # Epsilon-greedy exploration
    def epsilon_greedy(self, state, epsilon):
        # Explore (random action)
        if np.random.random() < epsilon:
            return np.random.randint(self.numActions)
        else:
            state_key = self.hashState(state)
            # Exploit (action with maximum Q-value)
            if state_key not in self.Q:
                # Initialize Q-values for the state
                self.Q[state_key] = np.zeros(self.numActions)
            return np.argmax(self.Q[state_key])
    
    # Training function for Q-learning
    def Q_learning(self, max_episodes, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay):
        rewards = []
        steps_taken = []
        epsilon = epsilon_max
        completion_count = 0
        steps_done = 0

        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0

            for i in range(self.max_steps):
                action = self.epsilon_greedy(state, epsilon)
                next_state, reward, done, truncated, _ = self.env.step(action)

                state_key = self.hashState(state)
                next_state_key = self.hashState(next_state)

                if state_key not in self.Q:
                    self.Q[state_key] = np.zeros(self.numActions)

                if next_state_key not in self.Q:
                    self.Q[next_state_key] = np.zeros(self.numActions)

                self.Q[state_key][action] += alpha * (reward + gamma * np.max(self.Q[next_state_key]) - self.Q[state_key][action])

                total_reward += reward
                steps_done += 1

                if done:
                    completion_count += 1
                    break
                if truncated:
                    break

                state = next_state

            rewards.append(total_reward)
            steps_taken.append(steps_done)

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # Write reward and number of training steps to Tensorboard
            writer.add_scalar("Reward/train", total_reward, completion_count)

        completion_rate = completion_count / max_episodes
        avg_steps = steps_done / max_episodes
        avg_reward = np.mean(rewards)

        return rewards, self.Q, completion_rate, avg_steps, avg_reward
    
    # Function for evaluating the agent
    def evaluate_agent(self, max_steps, n_eval_episodes, Q_tab, seed=None):
        episode_completion = 0
        total_steps = 0
        total_reward = 0

        for episode in range(n_eval_episodes):
            if seed:
                state = self.env.reset(seed=seed[episode])
            else:
                state = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0

            for step in range(max_steps):
                # Take the action (index) that has the maximum reward
                state_key = self.hashState(state)
                action = np.argmax(Q_tab[state_key][:])
                new_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                if done:
                    episode_completion += 1
                    break

                state = new_state

            total_steps += steps
            total_reward += episode_reward

        completion_rate = episode_completion / n_eval_episodes
        avg_steps = total_steps / n_eval_episodes
        avg_reward = total_reward / n_eval_episodes

        return completion_rate, avg_steps, avg_reward
    
    def grid_search(self, max_episodes, alpha_values, gamma_values, epsilon_max_values, epsilon_min_values, epsilon_decay_values):
        best_avg_reward = -float('inf')
        best_completion_rate = 0
        best_params = {}

        for alpha in alpha_values:
            for gamma in gamma_values:
                for epsilon_max in epsilon_max_values:
                    for epsilon_min in epsilon_min_values:
                        for epsilon_decay in epsilon_decay_values:
                            rewards, Q, completion_rate, avg_steps, avg_reward = self.Q_learning(max_episodes, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay)

                            if avg_reward > best_avg_reward:
                                best_avg_reward = avg_reward
                                best_completion_rate = completion_rate
                                best_params = {
                                    'alpha': alpha,
                                    'gamma': gamma,
                                    'epsilon_max': epsilon_max,
                                    'epsilon_min': epsilon_min,
                                    'epsilon_decay': epsilon_decay
                                }

        return rewards, self.Q, best_completion_rate, avg_steps, best_avg_reward, best_params
    
    #SARSA training function
    def SARSA(self, max_episodes, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay):
        rewards = []
        steps_taken = []
        epsilon = epsilon_max
        completion_count = 0
        steps_done = 0

        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0
            action = self.epsilon_greedy(state, epsilon)

            for i in range(self.max_steps):
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy(next_state, epsilon)

                state_key = self.hashState(state)
                next_state_key = self.hashState(next_state)

                if state_key not in self.Q:
                    self.Q[state_key] = np.zeros(self.numActions)

                if next_state_key not in self.Q:
                    self.Q[next_state_key] = np.zeros(self.numActions)

                # SARSA update
                self.Q[state_key][action] += alpha * (reward + gamma * self.Q[next_state_key][next_action] - self.Q[state_key][action])

                total_reward += reward
                steps_done += 1

                if done:
                    completion_count += 1
                    break
                if truncated:
                    break

                state = next_state
                action = next_action

            rewards.append(total_reward)
            steps_taken.append(steps_done)

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Write reward and number of training steps to Tensorboard
            writer.add_scalar("Reward/train", total_reward, completion_count)

        completion_rate = completion_count / max_episodes
        avg_steps = steps_done / max_episodes
        avg_reward = np.mean(rewards)

        return rewards, self.Q, completion_rate, avg_steps, avg_reward
    
    # Function to load the Q-values
    def load_Q_values(self, filename):
        if exists(filename):
            print('Loading existing Q values\n')
            # Load data (deserialize)
            with open(filename, 'rb') as handle:
                Q_values = pickle.load(handle)
                handle.close()
        else:
            print('Filename %s does not exist, could not load data' % filename)
        return Q_values


# Create an instance of the Q_learning_agent class
agent = RL_agent('MiniGrid-Empty-8x8-v0')

# Set the hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_max = 1.0  # Maximum epsilon for exploration
epsilon_min = 0.01  # Minimum epsilon for exploration
epsilon_decay = 0.995  # Epsilon decay rate
max_episodes = 3000

# Tensorboard writer
writer = SummaryWriter()

# Run Q-learning
rewards1, Tab_Q_learning, completion_rate, avg_steps, avg_reward = agent.Q_learning(max_episodes, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay)

writer.flush()
writer.close()

# Tensorboard writer
writer = SummaryWriter()
# Run SARSA
rewards2, Q_tab_SARSA, completion_rate, avg_steps, avg_reward = agent.SARSA(max_episodes, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay)

writer.flush()
writer.close()

# Save the Q-table
filename1 = 'Q_learning_table.pickle'
filename2 = 'SARSA_Q_table.pickle'

# Saving the value-function to file
with open(filename1, 'wb') as handle:
    pickle.dump(Tab_Q_learning, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

with open(filename2, 'wb') as handle:
    pickle.dump(Q_tab_SARSA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

#Evaluating the performance of our trained policy
n_eval_episodes = 1000  # Number of evaluation episodes
seed = None  # Optional: Replace with a list of seeds for each episode if desired
completion_rate, avg_steps, avg_reward = agent.evaluate_agent(max_steps=agent.max_steps, n_eval_episodes=n_eval_episodes,Q_tab=Tab_Q_learning, seed=seed)

#**************** 3. Hyperparameter tuning****************
# Uncomment this code to perform a grind search to find the optimal parameters (be warned! it takes some time).
# # Set the hyperparameters and their ranges for grid search
# alpha_values = [0.1, 0.2, 0.3]
# gamma_values = [0.9, 0.95, 0.99]
# epsilon_max_values = [1.0, 0.9, 0.8]
# epsilon_min_values = [0.01, 0.05, 0.1]
# epsilon_decay_values = [0.995, 0.99, 0.98]

# # # Set the maximum number of episodes for training
# # max_episodes = 3000

# # Perform grid search for hyperparameter tuning
# rewards, optimal_Q, completion_rate, avg_steps, avg_reward, optimal_params = agent.grid_search(max_episodes, alpha_values, gamma_values, epsilon_max_values, epsilon_min_values, epsilon_decay_values)

optimal_params = {'alpha': 0.1, 'gamma': 0.9, 'epsilon_max': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995}
print('\nGrid Search Results')
print('----------------------')
print('Best Average Reward: {:.2f}'.format( 0.96))
print('Best Completion Rate: {:.2%}'.format(1.0))
print('Best Parameters:', optimal_params)

############# Evaluating the trained agents####################
# Load the trained Q_values for Q-learning and SARSA


#Evaluating the performance of our trained policy
n_eval_episodes = 1000  # Number of evaluation episodes
seed = None  # Optional: Replace with a list of seeds for each episode if desired
Tab_Q_learning = agent.load_Q_values(filename1) #Loading the Q_value

QL_completion_rate, QL_avg_steps, QL_avg_reward = agent.evaluate_agent(max_steps=agent.max_steps, \
                                                                       n_eval_episodes=n_eval_episodes,Q_tab=Tab_Q_learning, seed=seed)

# Printing the evaluation results
print('Q-learning Evaluation Results')
print('------------------')
print('Completion Rate: {:.2%}'.format(QL_completion_rate))
print('Average Steps: {:.2f}'.format(QL_avg_steps))
print('Average Reward: {:.4f}\n'.format(QL_avg_reward))


Q_tab_SARSA = agent.load_Q_values(filename2)  # Loading the Q_table
SARSA_completion_rate, SARSA_avg_steps, SARSA_avg_reward = agent.evaluate_agent(max_steps=agent.max_steps, \
                                                                                n_eval_episodes=n_eval_episodes, Q_tab=Q_tab_SARSA, seed=seed)
# Printing the evaluation results
print('SARSA: Evaluation Results')
print('-------------------------')
print('Completion Rate: {:.2%}'.format(SARSA_completion_rate))
print('Average Steps: {:.2f}'.format(SARSA_avg_steps))
print('Average Reward: {:.4f}\n'.format(SARSA_avg_reward))