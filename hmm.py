"""
COMS W4701 Artificial Intelligence - Programming Homework 5

A HMM object with forward, Viterbi, and (optionally) smoothing algorithm implementations

@author: Anton Priyanka P(ap3901)
"""

import numpy as np


class HMM(object):
    """
    states: List of state values (e.g., strings, integers, etc.)
    emissions: List of emission values (e.g., strings, integers, etc.)
    initial: Initial state probability distribution (row vector)
    tprob: Transition matrix. tprob[i,j] = Pr(X_t = j | X_{t-1} = i)
    eprob: Emissions matrix. eprob[i,j] = Pr(E_t = j | X_t = i)
    """

    def __init__(self, model):
        self.states = model["states"]
        self.emissions = model["emissions"]
        self.initial = np.array(model["initial"])
        self.tprob = np.array(model["tprob"])
        self.eprob = np.array(model["eprob"])

    """YOUR CODE STARTS HERE"""

    # Forward algorithm for state estimation
    """
    Input: List of observation indices
    Outputs: 2d array, each row is belief distribution P(X_t | e_{1:t})
    """

    def forward(self, observations):
        f = np.array([np.multiply(self.initial, self.eprob[:, observations[0]])])
        f_sum = sum(f[0])
        f[0] = f[0] / f_sum
        for i in range(1, len(observations)):
            f_prime = np.array([np.dot(f[i - 1], self.tprob)])
            f_norm = np.multiply(f_prime, self.eprob[:, observations[i]])
            f_sum = sum(f_norm[0])
            f_norm[0] = f_norm[0] / f_sum
            f = np.append(f, f_norm, axis=0)
        return f

    # Elapse time for most likely sequence (Viterbi)
    """
    Input: Message distribution m_t = max P(x_{1:t-1}, X_t, e_{1:t})
    Outputs: max P(x_{1:t}, X_{t+1}, e_{1:t}), 
             list of most likely prior state indices
    """

    def propagate_joint(self, m):
        m_inter = []
        prior_state = []
        for i in range(len(self.tprob)):
            m_trans = np.multiply(m, self.tprob[:, i])
            prior_state.append(np.argmax(m_trans))
            m_inter.append(max(m_trans))
        m_prime = np.array([m_inter])
        return m_prime, prior_state

    # Viterbi algorithm for state sequence estimation
    """
    Input: List of observation indices
    Outputs: List of most likely sequence of state indices
    """

    def viterbi(self, observations):
        m = np.array([np.multiply(self.initial, self.eprob[:, observations[0]])])
        m_sum = sum(m[0])
        m[0] = m[0] / m_sum
        prior_states = []
        for i in range(1, len(observations)):
            m_prime, prior_state = self.propagate_joint(m[i - 1])
            m_norm = np.multiply(m_prime, self.eprob[:, observations[i]])
            m_sum = sum(m_norm[0])
            m_norm[0] = m_norm[0] / m_sum
            m = np.append(m, m_norm, axis=0)
            prior_states.append(prior_state)
        index = int(np.argmax(m[len(observations) - 1]))
        likely_seq = [index]
        for state in prior_states[::-1]:
            likely_seq.append(state[index])
            index = state[index]
        return likely_seq[::-1]

    # Backward pass for computing likelihood of future evidence given current state
    """
    Input: List of observations indices
    Output: 2d array, each row is likelihood P(e_{k+1:T} | X_k)
    """

    def backward(self, observations):
        m = np.array([[1, 1, 1]])
        i = 0
        for e in observations[::-1]:
            m_obs = np.multiply(m[i], self.eprob[:, e])
            m_trans = np.array([np.dot(m_obs, self.tprob.T)])
            i = i + 1
            m = np.append(m, m_trans, axis=0)
            if i == len(observations) - 1:
                break
        return m[::-1]

    """YOUR CODE STOPS HERE"""

    def smooth(self, observations):
        forward = self.forward(observations)
        backward = self.backward(observations)
        smoothed = np.multiply(forward, backward)
        return smoothed / np.linalg.norm(smoothed, ord=1, axis=1).reshape(len(observations), 1)
