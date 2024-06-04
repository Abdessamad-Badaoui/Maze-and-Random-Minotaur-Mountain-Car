# Abdessamad Badaoui 20011228-T118
# Nasr Allah Aghelias 20010616-T318

import maze_h as mz
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 3],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    n_episodes = 50000
    env = mz.Maze(maze, minotaur_allowed_stay=False)
    gamma   = 49/50; 
    epsilon = 0.1
    start  = ((0,0), (6, 5), 0);
    method = 'Q-Learning';
    alpha0 = 2/3
    alpha1 = 0.6
    alpha2 = 0.7
    alpha3 = 0.9
    V_alpha_1, policy = mz.Q_learning(env, n_episodes, env.map[start], epsilon, gamma, alpha0)
    # V_alpha_2, policy = mz.Q_learning(env, n_episodes, env.map[start], epsilon, gamma, alpha2)
    path, _ = env.simulate(start, policy, method)
    plt.plot(range(0, n_episodes+1), V_alpha_1, label = f"alpha = {alpha1}")
    # plt.plot(range(0, n_episodes+1), V_alpha_2, label = f"alpha = {alpha2}")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Value function")
    plt.show()
    mz.animate_solution(maze, path)

    

    # V, policy = mz.SARSA(env, n_episodes, env.map[start], epsilon, gamma, alpha1)
    # method = 'SARSA'
    # path, _ = env.simulate(start, policy, method)

    # V = np.load("V_values_SARSA.npy")

    # plt.plot(range(0, len(V)), V, label = f"alpha = 0.6")
    # plt.legend()
    # plt.xlabel("Episode")
    # plt.ylabel("Value function")
    # plt.show()
    # mz.animate_solution(maze, path)

    # V_epsilon_1 = np.load("V_values_SARSA_epsilon_1e-1.npy")
    # V_epsilon_2 = np.load("V_values_SARSA_epsilon_2e-1.npy")

    # plt.plot(range(0, n_episodes+1), V_epsilon_1, label = f"epsilon = {0.1}")
    # plt.plot(range(0, n_episodes+1), V_epsilon_2, label = f"epsilon = {0.2}")
    # plt.legend()
    # plt.xlabel("Episode")
    # plt.ylabel("Value function")
    # plt.show()

    # Q = np.load('./Q-Learning/Q_values_q-learning.npy')
    # policy = np.argmax(Q, 1)
    