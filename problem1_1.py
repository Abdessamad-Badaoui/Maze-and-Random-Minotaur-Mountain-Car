# Abdessamad Badaoui 20011228-T118
# Nasr Allah Aghelias 20010616-T318

import maze as mz
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    # mz.draw_maze(maze)

    # Create an environment maze
    # Finite horizon
    # Solve the MDP problem with dynamic programming
    """
    n_exp = 100
    horizon_range = range(1, 31)
    probabilities_allowed_stay = []
    probabilities_not_allowed_stay = []
    env_stay = mz.Maze(maze, minotaur_allowed_stay=True)
    env = mz.Maze(maze, minotaur_allowed_stay=False)
    method = 'DynProg';
    start  = ((0,0), (6, 5));
    # horizon = 20
    for horizon in horizon_range:
        probability_allowed = 0
        probability_not_allowed = 0
        V_stay, policy_stay = mz.dynamic_programming(env_stay,horizon);
        V, policy= mz.dynamic_programming(env,horizon);
        for _ in range(n_exp):
            path, out_of_maze = env_stay.simulate(start, policy_stay, method);
            if out_of_maze:
                probability_allowed += 1
            path, out_of_maze = env.simulate(start, policy, method);
            if out_of_maze:
                probability_not_allowed += 1
        probabilities_allowed_stay.append(probability_allowed/n_exp)
        probabilities_not_allowed_stay.append(probability_not_allowed/n_exp)
        print(f"Horizon : {horizon}")

    plt.plot(horizon_range, probabilities_allowed_stay, label = "minotaur can stay")
    plt.plot(horizon_range, probabilities_not_allowed_stay, label = "minotaur can\'t stay")
    plt.legend()
    plt.xlabel("Time horizon")
    plt.ylabel("Probability of exit")
    plt.show()
    """

    
    prob_getting_out = 0
    n_exp = 40000
    env = mz.Maze(maze, minotaur_allowed_stay=False)
    gamma   = 29/30; 
    # Accuracy treshold 
    epsilon = 0.0001;
    V, policy = mz.value_iteration(env, gamma, epsilon)
    method = 'ValIter';
    start  = ((0,0), (6, 5));
    for i in range(n_exp):
        path, out_of_maze = env.simulate(start, policy, method);
        if out_of_maze:
            prob_getting_out += 1
        if i%1000 == 0:
            print(f"Experiment : {i}")
    prob_getting_out /= n_exp

    print(prob_getting_out)
    

    # Show the shortest path 
    # mz.animate_solution(maze, path)