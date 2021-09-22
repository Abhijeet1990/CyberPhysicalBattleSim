import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import gym
import cyberbattle._env.cyberbattle_env as cyberbattle_env

# defines the option class

class Option():
    def __init__(self, option):
        self.name = option
        # Set I (initiation set), beta (termination set), pi (policy)
        self._setIBetaPi()

    def pickAction(self, state):
        action_number = self.pi[state]
        if action_number == 1:
            action = "left"
        elif action_number == 2:
            action = "up"
        elif action_number == 3:
            action = "right"
        elif action_number == 4:
            action = "down"
        # Return action number, used for intra-option model learning
        return action, action_number

    def visualize(self):
        plt.imshow(self.I)
        plt.colorbar()
        plt.title("Initiation set")
        plt.show()
        plt.imshow(self.beta)
        plt.colorbar()
        plt.title("Termination set")
        plt.show()
        plt.imshow(self.pi)
        plt.colorbar()
        plt.title("Policy")
        plt.show()

    def _setIBetaPi(self):
        if self.name in ["left", "up", "right", "down"]:
            self.I = np.ones((13, 13))  # Available everywhere
            self.beta = np.ones((13, 13))  # Terminates everywhere

            if self.name == "left":
                self.pi = np.ones((13, 13))  # Left  (1 everywhere)
            elif self.name == "up":
                self.pi = np.ones((13, 13)) + 1  # Up    (2 everywhere)
            elif self.name == "right":
                self.pi = np.ones((13, 13)) + 2  # Right (3 everywhere)
            elif self.name == "down":
                self.pi = np.ones((13, 13)) + 3  # Down  (4 everywhere)

        else:
            self.I = np.zeros((13, 13))
            self.beta = np.ones((13, 13))
            self.pi = np.zeros((13, 13))

            if self.name == "topleft->topright":
                self.I[1:6, 1:6] = self.I[6, 2] = 1
                self.beta[1:6, 1:6] = 0
                self.pi[1:6, 1:5] = self.pi[3, 5] = 3  # Right
                self.pi[1:3, 5] = 4  # Down
                self.pi[6, 2] = self.pi[4:6, 5] = 2  # Up

            if self.name == "topleft->botleft":
                self.I[1:6, 1:6] = self.I[3, 6] = 1
                self.beta[1:6, 1:6] = 0
                self.pi[1:5, 1:6] = self.pi[5, 2] = 4  # Down
                self.pi[5, 1] = 3  # Right
                self.pi[3, 6] = self.pi[5, 3:6] = 1  # Left

            if self.name == "botleft->topleft":
                self.I[7:12, 1:6] = self.I[10, 6] = 1
                self.beta[7:12, 1:6] = 0
                self.pi[8:12, 1:6] = self.pi[7, 2] = 2  # Up
                self.pi[7, 1] = 3  # Right
                self.pi[7, 3:6] = self.pi[10, 6] = 1  # Left

            if self.name == "botleft->botright":
                self.I[7:12, 1:6] = self.I[6, 2] = 1
                self.beta[7:12, 1:6] = 0
                self.pi[7:12, 1:6] = self.pi[10, 5] = 3  # Right
                self.pi[7:10, 5] = self.pi[6, 2] = 4  # Down
                self.pi[11, 5] = 2  # Up

            if self.name == "topright->topleft":
                self.I[1:7, 7:12] = self.I[7, 9] = 1
                self.beta[1:7, 7:12] = 0
                self.pi[1:7, 8:12] = self.pi[3, 7] = 1  # Left
                self.pi[7, 9] = self.pi[4:7, 7] = 2  # Up
                self.pi[1:3, 7] = 4  # Down

            if self.name == "topright->botright":
                self.I[3, 6] = self.I[1:7, 7:12] = 1
                self.beta[1:7, 7:12] = 0
                self.pi[1:6, 7:12] = self.pi[6, 9] = 4  # Down
                self.pi[3, 6] = self.pi[6, 7:9] = 3  # Right
                self.pi[6, 10:12] = 1  # Left

            if self.name == "botright->botleft":
                self.I[8:12, 7:12] = self.I[7, 9] = 1
                self.beta[8:12, 7:12] = 0
                self.pi[8:12, 8:12] = self.pi[10, 7] = 1  # Left
                self.pi[7, 9] = self.pi[8:10, 7] = 4  # Down
                self.pi[11, 7] = 2  # Up

            if self.name == "botright->topright":
                self.I[8:12, 7:12] = self.I[10, 6] = 1
                self.beta[8:12, 7:12] = 0
                self.pi[9:12, 7:12] = self.pi[8, 9] = 2  # Up
                self.pi[10, 6] = self.pi[8, 7:9] = 3  # Right
                self.pi[8, 10:12] = 1  # Left

    def __str__(self):
        return self.name


""" Simple agent planning using Q-learning """
class SMDPQLearningAgent():
    def __init__(self, gamma=0.9):

        # Abhijeet: need to change the list of options
        self.options = \
            [Option("left"), Option("up"), Option("right"), Option("down"), # primitive action
             Option("topleft->topright"), Option("topleft->botleft"),
             Option("topright->topleft"), Option("topright->botright"),
             Option("botleft->topleft"), Option("botleft->botright"),
             Option("botright->botleft"), Option("botright->topright")]

        self.gamma = gamma  # Discount factor, 0.9 by default as in paper
        self.current_option = None
        self.starting_state = None  # Starting state of current option
        self.k = 0  # Number of time steps elapsed in current option
        self.cumulative_reward = 0  # Total reward for current option

        # Abhijeet: Initialize option value table, and occurrence counts table
        n_states = 13 * 13
        n_options = len(self.options)
        self.Q = np.zeros((n_states, n_options))
        self.N = np.zeros((n_states, n_options))

    def epsilonGreedyPolicy(self, state, epsilon=0.1):
        # If we are not currently following an option
        if self.current_option is None:
            # Pick a new option and record starting state
            self._pickNewOptionEpsilonGreedily(state, epsilon)

        # Select action according to policy of current option
        action, _ = self.current_option.pickAction(state)
        return action

    # Remark : state argument is unused, we update only for the starting
    # state of the finishing option (which is recorded in the agent)
    def recordTransition(self, state, reward, next_state):
        # Add reward discounted by current discounting factor
        self.cumulative_reward += (self.gamma ** self.k) * reward
        self.k += 1  # Increment k after

        # If current option terminates at next state
        if self.current_option.beta[next_state] == 1:
            # Update Q table
            self._updateQValue(next_state)
            # Reset current option to None
            self._resetCurrentOption()

    def _updateQValue(self, next_state):
        s1 = self._sIdx(self.starting_state)
        o = self._oIdx(self.current_option)
        s2 = self._sIdx(next_state)

        self.N[s1, o] += 1
        alpha = (1. / self.N[s1, o])

        target = self.cumulative_reward + \
                 (self.gamma ** self.k) * np.max(self.Q[s2])
        self.Q[s1, o] += alpha * (target - self.Q[s1, o])

    # Pick new option according to model and state value function greedily
    def _pickNewOptionEpsilonGreedily(self, state, epsilon):
        # Iterate over options, keeping track of all available options
        # and the index of best option seen so far
        available_options = []
        best_option_index = 0
        s = self._sIdx(state)
        for i in range(len(self.options)):
            if self.options[i].I[state] == 1:
                available_options.append(self.options[i])
                if self.Q[s, i] > self.Q[s, best_option_index]:
                    best_option_index = i

        # Pick greedy option with probability (1 - epsilon)
        if random.uniform(0, 1) > epsilon:
            self.current_option = self.options[best_option_index]

        # Pick random action with probability epsilon
        else:
            self.current_option = random.choice(available_options)

        # Set starting state of option
        self.starting_state = state

    def _sIdx(self, state):
        return state[0] * 13 + state[1]

    def _oIdx(self, option):
        return self.options.index(option)

    def _resetCurrentOption(self):
        self.k = 0
        self.cumulative_reward = 0
        self.current_option = None
        self.starting_state = None




env = gym.make('CyberBattleChain-v0',size=4,attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=1.0,reward=2180))
agent = SMDPQLearningAgent()


def run_episode(verbose=False):
    state = env.reset()
    while True:
        action = agent.epsilonGreedyPolicy(state)
        if verbose:
            print("State = {}, Option = {}, Action = {}".format(
                state, agent.current_option, action))
        next_state, reward, done = env.step(action)
        agent.recordTransition(state, reward, next_state)
        state = next_state
        if done:
            break


for i in range(10000):
    run_episode()
