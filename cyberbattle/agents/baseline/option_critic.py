import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Softmax, LogSoftmax, Sigmoid
import random

class Option():
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        # Policy parameters
        self.theta = Variable(torch.Tensor(
            np.random.rand(n_states, n_actions)), requires_grad=True)
        # Termination parameters
        self.upsilon = Variable(torch.Tensor(
            np.random.rand(n_states)), requires_grad=True)

    # Input : index in [0, n_states - 1]
    # Return : log pi(.|state), variable of shape (1, n_actions)
    def pi(self, state_index, T=0.1):
        state_var = self._varFromStateIndex(state_index)
        logprobs = Softmax()(torch.matmul(state_var, self.theta) / T)
        return logprobs

    # Input : index in [0, n_states - 1]
    # Return : beta(state), variable of shape (1)
    def beta(self, state_index):
        state_var = self._varFromStateIndex(state_index)
        return Sigmoid()(torch.matmul(state_var, self.upsilon))

    # Input : index in [0, n_states - 1]
    # Return : one of "left", "up", "right", or "down",
    #          index of action chosen in [0, n_actions]
    #          and one-hot variable of shape (1, n_actions)
    def pickAction(self, state_index):
        probs = self.pi(state_index).data.numpy().reshape(-1)
        action_index = np.random.choice(self.n_actions, size=1, p=probs)[0]
        action, action_one_hot = self._actionFromActionIndex(action_index)
        return action, action_index, action_one_hot

    # Input : index in [0, n_states - 1]
    # Return : one-hot variable of shape (1, n_states)
    def _varFromStateIndex(self, state_index):
        s = np.zeros(self.n_states)
        s[state_index] = 1
        return Variable(torch.Tensor(s)).view(1, -1)

    # Input : index in [0, n_actions - 1]
    # Return : one of "left", "up", "right", or "down"
    #          and one-hot variable of shape (1, n_actions)
    def _actionFromActionIndex(self, action_index):
        if action_index == 0:
            action = "left"
        elif action_index == 1:
            action = "up"
        elif action_index == 2:
            action = "right"
        elif action_index == 3:
            action = "down"
        a = np.zeros(self.n_actions)
        a[action_index] = 1
        return action, Variable(torch.Tensor(a)).view(1, -1)


# Option Critic Agent
class OptionCritic():
    def __init__(self, gamma=0.99, alpha_critic=0.5, alpha_theta=0.25,
                 alpha_upsilon=0.25, n_options=4):
        self.gamma = gamma  # Discount factor
        self.alpha_critic = alpha_critic  # Critic lr
        self.alpha_theta = alpha_theta  # Intra-option policies lr
        self.alpha_upsilon = alpha_upsilon  # Termination functions lr

        n_states = 13 * 13
        n_actions = 4
        self.options = [Option(n_states, n_actions) \
                        for _ in range(n_options)]

        self.current_option = None
        # Keep track of one hot var and index of last action taken
        self.last_action_one_hot = None
        self.last_action_index = None

        # Action values in the context of (state, option) pairs
        self.Q_U = np.zeros((n_states, n_options, n_actions))
        # Option values (computed from Q_U)
        self.Q = np.zeros((n_states, n_options))
        # State values (computed from Q)
        self.V = np.zeros(n_states)

    def epsilonGreedyPolicy(self, state_tuple, epsilon=0.01):
        state_index = self._sIdx(state_tuple)
        # If current option is None, pick a new one epsilon greedily
        if self.current_option is None:
            # Pick greedy option with probability (1 - epsilon)
            if random.uniform(0, 1) > epsilon:
                best_option_idx = np.argmax(self.Q[state_index])
                self.current_option = self.options[best_option_idx]
            # Pick random action with probability epsilon
            else:
                self.current_option = random.choice(self.options)

        # Pick action according to current option
        action, action_index, action_one_hot = \
            self.current_option.pickAction(state_index)
        # Record one hot var and index of last action taken
        self.last_action_one_hot = action_one_hot
        self.last_action_index = action_index
        return action

    def recordTransition(self, state, reward, next_state):
        pi = self.current_option.pi(self._sIdx(state))
        beta = self.current_option.beta(self._sIdx(next_state))

        # 1) Critic improvement
        # Update estimate of Q_U[state, current_option, next_state]
        self._evaluateOption(state, reward, next_state, pi, beta)

        # 2) Actor improvement
        # Take a gradient step for policy and termination parameters
        # of current option
        self._improveOption(state, next_state, pi, beta)

        # If current option ends, set current option to None
        beta = self.current_option.beta(self._sIdx(next_state)).data[0]
        if random.uniform(0, 1) < beta:
            self.current_option = None

    def _evaluateOption(self, state, reward, next_state, pi, beta):
        s1 = self._sIdx(state)
        s2 = self._sIdx(next_state)
        o = self._oIdx(self.current_option)
        a = self.last_action_index

        # Update Q_U
        beta = beta.data[0]
        target = reward + self.gamma * (1 - beta) * self.Q[s2, o] \
                 + self.gamma * beta * np.max(self.Q[s2])
        self.Q_U[s1, o, a] += \
            self.alpha_critic * (target - self.Q_U[s1, o, a])

        # Update Q since Q_U has changed
        self.Q[s1, o] = pi.data.numpy().reshape(-1).dot(self.Q_U[s1, o])

        # Update V since Q has changed
        # This update is only valid if the policy over options is greedy
        self.V[s1] = np.max(self.Q[s1, o])

    def _improveOption(self, state, next_state, pi, beta):
        s1 = self._sIdx(state)
        s2 = self._sIdx(next_state)
        o = self._oIdx(self.current_option)
        a = self.last_action_index

        # 1) Policy update
        # Compute log pi(last_action_taken | state)
        logprobs = torch.log(pi)
        logprob = torch.sum(logprobs * self.last_action_one_hot)
        # Compute gradient of theta w.r.t this quantity
        logprob.backward()
        grad_theta = self.current_option.theta.grad.data
        # Take a gradient step
        self.current_option.theta.data += self.alpha_theta * \
                                          self.Q_U[s1, o, a] * grad_theta
        # Zero gradient
        self.current_option.theta.grad.data.zero_()

        # 2) Termination function update
        # Compute gradient of upsilon w.r.t beta(next_state)
        beta.backward()
        grad_upsilon = self.current_option.upsilon.grad.data
        # Take a gradient step
        self.current_option.upsilon.data += self.alpha_upsilon * \
                                            (self.Q[s2, o] - self.V[s2]) * grad_upsilon
        # Zero gradient
        self.current_option.upsilon.grad.data.zero_()

    def _sIdx(self, state):
        return state[0] * 13 + state[1]

    def _oIdx(self, option):
        return self.options.index(option)
