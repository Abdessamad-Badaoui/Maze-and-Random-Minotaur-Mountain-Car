# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Abdessamad Badaoui 20011228-T118
# Nasr Allah Aghelias 20010616-T318
# Load packages
import numpy as np
import gym
# import torch
import matplotlib.pyplot as plt
import pickle

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 200        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

# Training process
# for i in range(N_episodes):
#     # Reset enviroment data
#     done = False
#     state = scale_state_variables(env.reset()[0])
#     total_episode_reward = 0.

#     while not done:
#         # Take a random action
#         # env.action_space.n tells you the number of actions
#         # available
#         action = np.random.randint(0, k)
            
#         # Get next state and reward.  The done variable
#         # will be True if you reached the goal position,
#         # False otherwise
#         next_state, reward, done, truncated, _ = env.step(action)
#         next_state = scale_state_variables(next_state[0])

#         # Update episode reward
#         total_episode_reward += reward
            
#         # Update state for next iteration
#         state = next_state

#     # Append episode reward
#     episode_reward_list.append(total_episode_reward)

#     # Close environment
#     env.close()



eta = np.array([[0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [2, 0],
                [1, 2],
                [2, 1], 
                [2, 2]
                ])

def Q(weights, phi_state, action):
    return np.dot(weights[action], phi_state)

def epsilon_greedy(epsilon, phi_state, weights):
    p = np.random.random()
    action_space = weights.shape[0]
    if p < epsilon:
        return np.random.choice(range(action_space))
    else:
        return np.argmax([Q(weights, phi_state, i) for i in range(action_space)])
    
def calculate_phi(state):
    phi = np.ones((eta.shape[0], ))
    for j in range(phi.shape[0]):
        phi[j] = np.cos(np.pi * np.dot(eta[j], state))
    return phi

data = dict()
data['N'] = eta

def SARSA_lmbda(gamma, lmbda, epsilon, learning_rate):
    weights = np.ones((k, eta.shape[0]))
    alpha = learning_rate
    for i in range(N_episodes):
        # Reset enviroment data
        done = False
        state = env.reset()[0]
        phi_s = calculate_phi(scale_state_variables(state))
        # phi_s = np.ones((eta.shape[0], ))
        z = np.zeros((k, phi_s.shape[0]))
        m = 0.01
        velocity = np.zeros((k, phi_s.shape[0]))
        total_episode_reward = 0.

        counter = 0
        # flag = True
        while not done:
            if counter == 200:
                break
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available

            action = epsilon_greedy(1/(i+1), phi_s, weights)
                
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, _ = env.step(action)
            
            # next_state = next_state[0]
            phi_next_s = calculate_phi(scale_state_variables(next_state))

            # if done:
            #     print(reward)
            #     print("Done", done)

            next_action = epsilon_greedy(1/(i+1), phi_next_s, weights)
            for j in range(k):
                if j == action:
                    z[j] = gamma * lmbda * z[j] + phi_s
                else:
                    z[j] = gamma * lmbda * z[j]

            np.clip(z, -5, 5)
            
            # if i > 10 and flag:
            #     if abs(running_average(episode_reward_list, 10)[-1] + 135) < 20:
            #         alpha = alpha - 0.3 * alpha
            #         flag = False

            delta_t = reward + gamma * Q(weights, phi_next_s, next_action) - Q(weights, phi_s, action)
            velocity = m * velocity + alpha * delta_t * z
            weights = weights + velocity

            # Update episode reward
            total_episode_reward += reward
                
            # Update state for next iteration
            state = next_state
            phi_s = phi_next_s

            counter += 1

        # print(weights)


        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    data['W'] = weights
    

SARSA_lmbda(1, 0.5, 0.1, 0.001)

overrite = False

if overrite:
    weights_filename = 'weights.pkl'

    with open(weights_filename, 'wb') as file:
        pickle.dump(data, file)


# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

try:
    f = open('weights_1.pkl', 'rb')
    data = pickle.load(f)
    if 'W' not in data or 'N' not in data:
        print('Matrix W (or N) is missing in the dictionary.')
        exit(-1)
    weights = data['W']

except:
    print('File weights.pkl not found!')
    exit(-1)

plot_3d_graph = False

if plot_3d_graph:

    s1_values = np.linspace(-1.2, 0.6, 200)
    s2_values = np.linspace(-0.07, 0.07, 200)

    s1_mesh, s2_mesh = np.meshgrid(s1_values, s2_values)

    v_values_mesh = np.ones(s1_mesh.shape)

    for i in range(v_values_mesh.shape[0]):
        for j in range(v_values_mesh.shape[1]):
            state = scale_state_variables(np.array([s1_mesh[i, j], s2_mesh[i, j]]))
            v_values_mesh[i, j] = np.argmax([Q(weights, calculate_phi(state), l) for l in range(k)])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot_surface(s1_mesh, s2_mesh, v_values_mesh, cmap='viridis')

    # ax.set_xlabel('s1')
    # ax.set_ylabel('s2')
    # ax.set_zlabel('pi(s1, s2)')

    # plt.show()

    plt.imshow(v_values_mesh, extent=[s1_values.min(), s1_values.max(), s2_values.min(), s2_values.max()],
            origin='lower', cmap='viridis', aspect='auto')

    plt.colorbar(label='Actions')

    plt.xlabel('s1')
    plt.ylabel('s2')

    plt.show()

# lambda_list = np.linspace(0.2, 0.9, 8)
# average_total_reward = np.zeros(lambda_list.shape)

# for i, lmbda in enumerate(lambda_list):
#     episode_reward_list = []
#     SARSA_lmbda(1, lmbda, 0.1, 0.001)
#     average_total_reward[i] = running_average(episode_reward_list, 50)[-1]

# plt.plot(lambda_list, average_total_reward, label = 'Average Total Reward')
# plt.xlabel("Eligibility Trace")
# plt.ylabel("Average Total Reward over 50 Episodes")
# plt.show()