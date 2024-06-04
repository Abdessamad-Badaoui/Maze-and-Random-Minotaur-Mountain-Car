# Abdessamad Badaoui 20011228-T118
# Nasr Allah Aghelias 20010616-T318

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter', 'Q-Learning', 'SARSA'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
CYAN = '#01A9FB';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False, minotaur_allowed_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.minotaur_stay            = minotaur_allowed_stay;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        for t in range(2):
                            if self.maze[i,j] != 1:
                                if t == 0:
                                    if self.maze[i, j] != 3:
                                        states[s] = ((i,j), (k, l), t);
                                        map[((i,j), (k, l), t)] = s;
                                        s += 1;
                                else:
                                    states[s] = ((i,j), (k, l), t);
                                    map[((i,j), (k, l), t)] = s;
                                    s += 1;
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        state_player, state_minotaur, has_key = self.states[state]
        row = state_player[0] + self.actions[action][0];
        col = state_player[1] + self.actions[action][1];
        minotaur_possible_next_moves = [1, 2, 3, 4]
        if self.minotaur_stay:
            minotaur_possible_next_moves.append(0)
        k, l = state_minotaur
        if k == 0:
            minotaur_possible_next_moves.remove(3)
        elif k == self.maze.shape[0] - 1:
            minotaur_possible_next_moves.remove(4)
        if l == 0:
            minotaur_possible_next_moves.remove(1)
        elif l == self.maze.shape[1] - 1:
            minotaur_possible_next_moves.remove(2)
        p = np.random.random()
        if p < .35:
            moves_towards_player = []
            initial_distance_from_player = abs(state_player[0] - state_minotaur[0]) + abs(state_player[1] - state_minotaur[1])
            for movement in minotaur_possible_next_moves:
                new_row_minotaur = state_minotaur[0] + self.actions[movement][0] 
                new_col_minotaur = state_minotaur[1] + self.actions[movement][1]
                new_distance_from_player = abs(state_player[0] - new_row_minotaur) + abs(state_player[1] - new_col_minotaur)
                if new_distance_from_player < initial_distance_from_player:
                    moves_towards_player.append(movement)
            minotaur_next_move = np.random.choice(moves_towards_player)
            row_minotaur = state_minotaur[0] + self.actions[minotaur_next_move][0] 
            col_minotaur = state_minotaur[1] + self.actions[minotaur_next_move][1]
        else:
            minotaur_next_move = np.random.choice(minotaur_possible_next_moves)
            row_minotaur = state_minotaur[0] + self.actions[minotaur_next_move][0] 
            col_minotaur = state_minotaur[1] + self.actions[minotaur_next_move][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                            (col == -1) or (col == self.maze.shape[1]) or \
                            (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return self.map[(state_player, (row_minotaur, col_minotaur), has_key)], minotaur_possible_next_moves;
        else:
            if not has_key and self.maze[row, col] == 3:
                return self.map[((row, col), (row_minotaur, col_minotaur), 1)], minotaur_possible_next_moves;
            else:
                return self.map[((row, col), (row_minotaur, col_minotaur), has_key)], minotaur_possible_next_moves;

    def move(self, state, action):
        return self.__move(state, action)

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s, minotaur_next_possible_moves = self.__move(s,a);
                state_player = self.states[next_s][0]
                old_state_minotaur = self.states[s][1]
                for j in range(len(minotaur_next_possible_moves)):
                    row_minotaur = old_state_minotaur[0] + self.actions[minotaur_next_possible_moves[j]][0] 
                    col_minotaur = old_state_minotaur[1] + self.actions[minotaur_next_possible_moves[j]][1]
                    potential_next_state = self.map[(state_player, (row_minotaur, col_minotaur))]
                    transition_probabilities[potential_next_state, s, a] = 1 / len(minotaur_next_possible_moves);
        return transition_probabilities;

    def reward(self, current_state, action, next_state):
        has_key = self.states[current_state][2]
        if self.states[next_state][1] == self.states[next_state][0]:
            return self.IMPOSSIBLE_REWARD
        else:
            # Reward for hitting a wall
            if self.states[current_state][0] == self.states[next_state][0] and action != self.STAY:
                return self.IMPOSSIBLE_REWARD;
            # Reward for retrieving the key
            elif self.maze[self.states[next_state][0]] == 3 and not has_key:
                return self.GOAL_REWARD;
            # Reward for exiting using the key
            elif self.maze[self.states[next_state][0]] == 2 and has_key:
                return self.GOAL_REWARD;
            else:
                return self.STEP_REWARD;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        def r_step(s, next_s, a):
            if self.states[s][0] == self.states[next_s][0] and a != self.STAY:
                return self.IMPOSSIBLE_REWARD;
            # Reward for reaching the exit
            # self.states[s][0] == self.states[next_s][0] and 
            elif self.maze[self.states[next_s][0]] == 2:
                return self.GOAL_REWARD;
            # Reward for taking a step to an empty cell that is not the exit
            else:
                return self.STEP_REWARD;

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s, minotaur_next_possible_moves = self.__move(s,a);
                    old_state_minotaur = self.states[s][1]
                    p_MP = 0
                    for j in range(len(minotaur_next_possible_moves)):
                        row_minotaur = old_state_minotaur[0] + self.actions[minotaur_next_possible_moves[j]][0] 
                        col_minotaur = old_state_minotaur[1] + self.actions[minotaur_next_possible_moves[j]][1]
                        if self.states[next_s][0] == (row_minotaur, col_minotaur):
                            p_MP += 1
                    p_MP /= len(minotaur_next_possible_moves)
                    rewards[s, a] = p_MP * self.IMPOSSIBLE_REWARD + (1 - p_MP) * r_step(s, next_s, a)

                    # # Player and Minotaur not in the same position
                    # if self.states[next_s][0] != self.states[next_s][1]:
                    #     # Rewrd for hitting a wall
                    #     if self.states[s][0] == self.states[next_s][0] and a != self.STAY:
                    #         rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    #     # Reward for reaching the exit
                    #     # self.states[s][0] == self.states[next_s][0] and 
                    #     elif self.maze[self.states[next_s][0]] == 2:
                    #         rewards[s,a] = self.GOAL_REWARD;
                    #     # Reward for taking a step to an empty cell that is not the exit
                    #     else:
                    #         rewards[s,a] = self.STEP_REWARD;
                    # else:
                    #     rewards[s, a] = self.IMPOSSIBLE_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];
        
        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        out_of_maze = False
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s, _ = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                if self.states[next_s][0] == self.states[next_s][1]:
                    break
                if self.maze[self.states[next_s][0]] == 2:
                    out_of_maze = True
                    break
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while t < 31:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s, _ = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                dead =  np.random.choice([0] * 29 + [1])
                if dead == 1:
                    break
                if self.states[next_s][0] == self.states[next_s][1]:
                    break
                if self.maze[self.states[next_s][0]] == 2:
                    out_of_maze = True
                    break
                # Update time and state for next iteration
                t +=1;
        if method == 'Q-Learning' or method == 'SARSA':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while t < 51:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s, _ = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                dead = np.random.choice([0] * 49 + [1])
                if dead:
                    break
                if self.states[next_s][0] == self.states[next_s][1]:
                    break
                if self.maze[self.states[next_s][0]] == 2:
                    out_of_maze = True
                    break
                # Update time and state for next iteration
                t +=1;

        return path, out_of_maze


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def epsilon_greedy(Q, epsilon, state):
    p = np.random.random()
    if p < epsilon:
        return np.random.choice(range(Q.shape[1]))
    else:
        return np.argmax(Q[state])

def Q_learning(env, number_episodes, initial_state, epsilon, gamma, alpha):
    N = number_episodes
    n_states  = env.n_states
    n_actions = env.n_actions
    Q = np.ones((n_states, n_actions)) * (-57)
    visits_state_action = np.zeros((n_states, n_actions))
    V = []
    for k in range(N+1):
        state = initial_state
        cond = False
        if not k%1000:
            print(f"Episode : {k}")
        while not cond:
            action = epsilon_greedy(Q, epsilon, state)
            visits_state_action[state, action] += 1
            next_state, _ = env.move(state, action)
            reward = env.reward(state, action, next_state)
            Q[state, action] = Q[state, action] + 1 / (visits_state_action[state, action] ** alpha) * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if (env.maze[env.states[next_state][0]] == 2 and env.states[next_state][2]) or env.states[next_state][1] == env.states[next_state][0]:
                cond = True
        V.append(np.max(Q[initial_state]))
    policy = np.argmax(Q, 1)
    np.save('./Q-Learning/Q_values_q-learning.npy', Q)
    np.save('./Q-Learning/V_values_q-learning.npy', V)
    return V, policy

def SARSA(env, number_episodes, initial_state, epsilon, gamma, alpha, delta=0.8):
    N = number_episodes
    n_states  = env.n_states
    n_actions = env.n_actions
    Q = np.ones((n_states, n_actions)) * (-55)
    visits_state_action = np.zeros((n_states, n_actions))
    V = []
    counter = 0
    for k in range(N+1):
        state = initial_state
        cond = False
        epsilon_k = 1 / ((k + 1) ** delta)
        if not k%1000:
            print(f"Episode : {k}")
        while not cond:
            action = epsilon_greedy(Q, epsilon_k, state)
            visits_state_action[state, action] += 1
            next_state, _ = env.move(state, action)
            next_action = epsilon_greedy(Q, epsilon_k, next_state)
            reward = env.reward(state, action, next_state)
            Q[state, action] = Q[state, action] + 1 / (visits_state_action[state, action] ** alpha) * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state = next_state
            if (env.maze[env.states[next_state][0]] == 2 and env.states[next_state][2]) or env.states[next_state][1] == env.states[next_state][0]:
                if env.maze[env.states[next_state][0]] == 2 and env.states[next_state][2]:
                    counter += 1
                cond = True
        V.append(np.max(Q[initial_state]))
    # np.save('Q_values.npy', Q)
    # np.save('V_values_SARSA.npy', V)
    policy = np.argmax(Q, 1)
    return V, policy

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: CYAN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0])].get_text().set_text('Player')

        grid.get_celld()[(path[i][1])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][1])].get_text().set_text('Minotaur')
        if i > 0:
            if maze[path[i][0]] == 2:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is out')
            if path[i-1][0] != path[i][1] and path[i-1][0] != path[i][0]:
                grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
                grid.get_celld()[(path[i-1][0])].get_text().set_text('')
            if path[i-1][1] != path[i][0] and path[i-1][1] != path[i][1]:
                grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
                grid.get_celld()[(path[i-1][1])].get_text().set_text('')
        display.display(fig)
        plt.savefig(f"frame{i}.png")
        display.clear_output(wait=True)
        # time.sleep(1)
