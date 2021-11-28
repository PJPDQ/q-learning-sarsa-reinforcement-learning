import math
import random
import time
import itertools
import numpy as np

from game_env import GameEnv
from game_state import GameState

"""
solution.py

Template file for you to implement your solution to Assignment 3.

You must implement the following method stubs, which will be invoked by the simulator during testing:
    __init__(game_env)
    run_training()
    select_action()
    
To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

COMP3702 2021 Assignment 3 Support Code

Last updated by njc 10/10/21
"""


class RLAgent:

    def __init__(self, game_env):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor, and should be placed in the
        plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator if not completed within
        the limit.
        """
        self.solver_type = 'sarsa'
        self.game_env = game_env
        # HYPERPARAMETERS
        self.gamma = 0.9999  # discount
        self.alpha = 0.01  # learning rate
        self.epsilon = 0.4  # exploit_probability
        self.EXP_BIAS = 1.4
        self.MAX_STEPS = 300
        # ---------------
        self.q_values = {}
        self.actions = list(self.game_env.ACTIONS)
        self.reachable_actions = {}  # state, action
        self.num_of_states = {}  # state: number of times
        self.num_of_actions = {}  # state, action: number of times
        self.reach_state = []
        self.persistent_state = self.game_env.get_init_state()
        self.EXIT_STATE = GameState(self.game_env.exit_row, self.game_env.exit_col,
                                    tuple(1 for g in self.game_env.gem_positions))
        self.visited_states = set()
        self.walk_jump_actions = [self.game_env.WALK_LEFT, self.game_env.WALK_RIGHT, self.game_env.JUMP]
        self.glide_drop_actions = [action for action in self.actions if action not in self.walk_jump_actions]

        self.learning_rates = [0.1, 0.01, 0.001]
        self.total_rewards = []

    def get_reachable_action(self, curr_state):
        actions = []
        # pos = self.game_env.gem_positions[0]
        pos = (curr_state.row, curr_state.col)
        for action in self.game_env.ACTIONS:
            if (action in self.game_env.WALK_AND_JUMP_ACTIONS
                and self.game_env.grid_data[pos[0] + 1][pos[1]] in self.game_env.WALK_JUMP_ALLOWED_TILES) \
                    or (action in self.game_env.GLIDE_AND_DROP_ACTIONS
                        and self.game_env.grid_data[pos[0] + 1][pos[1]] in self.game_env.GLIDE_DROP_ALLOWED_TILES):
                actions.append(action)
        return actions

    def get_reachable_position_and_actions(self):
        # init_pos = self.game_env.gem_positions[0]
        visited = set()
        # container = [init_pos]
        valid_actions = {}
        for i in range(1, self.game_env.n_rows - 1):
            for j in range(1, self.game_env.n_cols - 1):
                if self.game_env.grid_data[i][j] in self.game_env.VALID_TILES and \
                        self.game_env.grid_data[i][j] != self.game_env.SOLID_TILE:
                    pos = (i, j)
                    temp_state = GameState(*pos, tuple(0 for g in self.game_env.gem_positions))
                    for a in self.get_reachable_action(temp_state):
                        is_valid, _, next_state, _ = self.game_env.perform_action(temp_state, a)
                        if is_valid:
                            # new_pos = (next_state.row, next_state.col)
                            if pos in valid_actions.keys():
                                valid_actions[pos].append(a)
                            else:
                                valid_actions[pos] = [a]
                            if pos not in visited:
                                visited.add(pos)
        return visited, valid_actions

    def get_reachable_status(self):
        return list(itertools.product(*[(0, 1)] * len(self.game_env.gem_positions)))

    def choose_action(self, state):
        best_q = -math.inf
        best_a = None
        ## unvisited state-action
        unvisited = self.reachable_actions[state]
        for a in unvisited:
            if (state, a) in self.q_values.keys() and self.q_values[(state, a)] > best_q:
                best_q = self.q_values[(state, a)]
                best_a = a
        if best_a is None or random.random() < self.epsilon:
            return random.choice(unvisited)
        else:
            return best_a

    def ucb1(self, state):
        unvisited = []
        unvisited_exists = False
        best_u = -math.inf
        best_a = None
        for a in self.reachable_actions[state]:
            if (state, a) not in self.q_values:
                unvisited.append(a)
                unvisited_exists = True
            elif not unvisited_exists:
                u = self.q_values[(state, a)] + (self.EXP_BIAS * np.sqrt(np.log(self.num_of_states[state])
                                                                         / self.num_of_actions[(state, a)]))
                if u > best_u:
                    best_u = u
                    best_a = a
        if unvisited_exists:
            return random.choice(unvisited)
        else:
            return best_a

    def run_q_learning_one(self):
        t0 = time.time()
        epis_state = random.choice(self.reach_state)
        terminal = False
        while self.game_env.get_total_reward() > self.game_env.training_reward_tgt and \
                time.time() - t0 < self.game_env.training_time - 1:
            t = 0
            while t < self.MAX_STEPS:
                t += 1
                epis_action = self.ucb1(epis_state)
                # epis_action = self.choose_action(epis_state)
                if not terminal:
                    _, reward, next_state, is_terminal = self.game_env.perform_action(epis_state, epis_action)
                    best_q = -math.inf
                    best_a = None
                    for a1 in self.reachable_actions[next_state]:
                        q = self.q_values.get((next_state, a1))
                        if q is not None and q > best_q:
                            best_q = q
                            best_a = a1
                    if best_a is None:
                        best_q = 0
                    target = reward + (self.gamma * best_q)

                    if (epis_state, epis_action) in self.q_values:
                        old_q = self.q_values[(epis_state, epis_action)]
                    else:
                        old_q = 0
                    self.q_values[(epis_state, epis_action)] = old_q + (self.alpha * (target - old_q))

                    # UCB1 number counts
                    if epis_state in self.num_of_states:
                        self.num_of_states[epis_state] += 1
                    else:
                        self.num_of_states[epis_state] = 1
                    if (epis_state, epis_action) in self.num_of_actions:
                        self.num_of_actions[(epis_state, epis_action)] += 1
                    else:
                        self.num_of_actions[(epis_state, epis_action)] = 1

                    if is_terminal:
                        terminal = True
                        continue
                    self.visited_states.add(epis_state)
                    # print(self.visited_states)
                    epis_state = next_state
                else:
                    terminal = False
                    epis_state = random.choice(self.reach_state)
                    # row = epis_state.row
                    # col = epis_state.col
                    # while (row == self.EXIT_STATE.row and col == col) \
                    #         or (self.game_env.grid_data[row][col] == self.game_env.LAVA_TILE):
                    #     epis_state = random.choice(self.reach_state)
                    #     row = epis_state.row
                    #     col = epis_state.col
                    # break

    def run_sarsa_one(self):
        t0 = time.time()
        epi_state = random.choice(self.reach_state)
        terminal = False
        while self.game_env.get_total_reward() > self.game_env.training_reward_tgt and \
                time.time() - t0 < self.game_env.training_time - 1:
            t = 0
            # epi_action = self.ucb1(epi_state)
            epi_action = self.choose_action(epi_state)
            while t < self.MAX_STEPS:
                t += 1
                if not terminal:
                    _, reward, next_state, is_terminal = self.game_env.perform_action(epi_state, epi_action)
                    # next_action = self.ucb1(next_state)
                    next_action = self.choose_action(next_state)
                    q_next = self.q_values.get((next_state, next_action))
                    if q_next is None:
                        q_next = 0
                    target = reward + (self.gamma * q_next)
                    if (epi_state, epi_action) in self.q_values:
                        old_q = self.q_values[(epi_state, epi_action)]
                    else:
                        old_q = 0
                    self.q_values[(epi_state, epi_action)] = old_q + (self.alpha * (target - old_q))

                    if epi_state in self.num_of_states:
                        self.num_of_states[epi_state] += 1
                    else:
                        self.num_of_states[epi_state] = 1
                    if (epi_state, epi_action) in self.num_of_actions:
                        self.num_of_actions[(epi_state, epi_action)] += 1
                    else:
                        self.num_of_actions[(epi_state, epi_action)] = 1

                    if is_terminal:
                        # self.q_values[(next_state, next_action)] = 0
                        terminal = True
                        continue
                    self.visited_states.add(epi_state)
                    epi_state = next_state
                    epi_action = next_action
                else:
                    terminal = False
                    epi_state = random.choice(self.reach_state)
                    row = epi_state.row
                    col = epi_state.col
                    while (row == self.EXIT_STATE.row and col == col) \
                            or (self.game_env.grid_data[row][col] == self.game_env.LAVA_TILE):
                        epi_state = random.choice(self.reach_state)
                        row = epi_state.row
                        col = epi_state.col
                    break

    def run_training(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform training (e.g. via Q-Learning or SARSA).

        The allowed run time for this method is given by 'game_env.training_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.
        """
        t0 = time.time()
        i = 0
        ## Initialization
        reach_pos, reach_actions = self.get_reachable_position_and_actions()
        reach_statuses = self.get_reachable_status()
        reachable_state = []
        for i, pos in enumerate(reach_pos):
            for status in reach_statuses:
                reachable_state.append(GameState(*pos, status))
        for state in reachable_state:
            self.num_of_states[state] = 0
            self.reachable_actions[state] = reach_actions[(state.row, state.col)]
        self.reach_state = reachable_state
        self.persistent_state = random.choice(self.reach_state)
        # for state in self.reach_state:
        #     for action in self.reachable_actions[state]:
        #         self.q_values[(state, action)] = 0
        #         self.num_of_actions[(state, action)] = 0
        #         self.num_of_states[state] = 0

        if self.solver_type == 'q-learning':
            self.run_q_learning_one()
        elif self.solver_type == 'sarsa':
            self.run_sarsa_one()
        t1 = time.time()
        print("Time to complete", self.solver_type + " " + str(i) + " iterations is about ")
        print(t1 - t0)

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to perform (once for each
        step of the episode).

        You can use this method to select an action based on the Q-value estimates learned during training.

        The allowed run time for this method is 1 second. The method will be terminated by the simulator if it does not
        complete within this limit - you should design your algorithm to ensure this method exits before the time limit
        is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """
        #self.print_values_and_policy()
        best_q = -math.inf
        best_a = None
        for action in self.actions:
            if (state, action) in self.q_values.keys() and self.q_values[(state, action)] > best_q:
                best_q = self.q_values[(state, action)]
                best_a = action
        if best_a is None:
            return random.choice(self.reachable_actions[state])
        else:
            return best_a

    def print_values_and_policy(self):
        values = {state: 0 for state in self.reach_state}
        policy = {state: '_' for state in self.reach_state}
        for state in self.reach_state:
            best_q = -math.inf
            best_a = None
            for a in self.reachable_actions[state]:
                if (state, a) in self.q_values.keys() and self.q_values[(state, a)] > best_q:
                    best_q = self.q_values[(state, a)]
                    best_a = a
            n_state = state
            values[state] = best_q
            policy[state] = best_a
        print('================Values==================')
        for r in range(self.game_env.n_rows):
            line = ''
            for c in range(self.game_env.n_cols):
                for state, val in values.items():
                    if state.row == r and state.col == c:
                        line += self.game_env.grid_data[r][c] + str(round(val, 1)) + self.game_env.grid_data[r][c]
                    else:
                        line += self.game_env.grid_data[r][c]
            print(line)
        print('\n' * 2)
        print('===============Policy====================')
        for r in range(self.game_env.n_rows):
            line = ''
            for c in range(self.game_env.n_cols):
                for state, action in policy.items():
                    if state.row == r and state.col == c:
                        line += self.game_env.grid_data[r][c] + action + self.game_env.grid_data[r][c]
                    else:
                        line += self.game_env.grid_data[r][c]
            print(line)
        print('\n' * 2)
