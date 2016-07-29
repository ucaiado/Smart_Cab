#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement and run an agent to learn in reinforcment learning scenario

@author: Udacity, ucaiado

Created on 07/15/2016
"""

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import logging
import sys
import time
from collections import defaultdict
import pandas as pd

# Log finle enabled. global variable
DEBUG = True

# setup logging messages
s_format = '%(asctime)s;%(message)s'
s_now = time.strftime('%c')
s_now = s_now.replace('/', '').replace(' ', '_').replace(':', '')
s_file = 'log/sim_{}.log'.format(s_now)
logging.basicConfig(filename=s_file, format=s_format)
root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(s_format)
ch.setFormatter(formatter)
root.addHandler(ch)


'''
Begin help functions
'''


def save_q_table(e):
    '''
    Log the final Q-table of the algorithm
    :param e: Environment object. The grid-like world
    '''
    agent = e.primary_agent
    try:
        q_table = agent.q_table
        pd.DataFrame(q_table).T.to_csv('log/qtable.log', sep='\t')
    except:
        print 'No Q-table to be printed'

'''
End help functions
'''


class BasicAgent(Agent):
    '''
    A Basic agent representation that learns to drive in the smartcab world.
    '''
    def __init__(self, env):
        '''
        Initialize a BasicLearningAgent. Save all parameters as attributes
        :param env: Environment object. The grid-like world
        '''
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(BasicAgent, self).__init__(env)
        # override color
        self.color = 'green'
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        '''
        Prepare for a new trip
        :param destination: tuple. the coordinates of the destination
        '''
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None
        self.old_state = None
        self.last_action = None
        self.last_reward = None

    def update(self, t):
        '''
        Update the state of the agent
        :param t: integer. Environment step attribute value
        '''
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs, self.next_waypoint, deadline)

        # TODO: Select action according to your policy
        action = self._take_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self._apply_policy(self.state, action, reward)

        # [debug]
        s_rtn = 'LearningAgent.update(): deadline = {}, inputs = {}, action'
        s_rtn += ' = {}, reward = {}'
        if DEBUG:
            s_rtn += ', next_waypoint = {}'
            root.debug(s_rtn.format(deadline, inputs, action, reward,
                       self.next_waypoint))
        else:
            print s_rtn.format(deadline, inputs, action, reward)

    def _take_action(self, t_state):
        '''
        Return an action according to the agent policy
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        return random.choice(Environment.valid_actions)

    def _apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward
        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        pass


class BasicLearningAgent(BasicAgent):
    '''
    A representation of an agent that learns using a basic implementation of
    Q-learning that is suited for deterministic Markov decision processes
    '''
    def __init__(self, env, f_gamma=0.9):
        '''
        Initialize a BasicLearningAgent. Save all parameters as attributes
        :param env: Environment object. The grid-like world
        :*param f_gamma: float. weight of delayed versus immediate rewards
        '''
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(BasicLearningAgent, self).__init__(env)
        # override color
        self.color = 'white'
        # TODO: Initialize any additional variables here
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.f_gamma = f_gamma
        self.old_state = None
        self.last_action = None
        self.last_reward = None

    def _take_action(self, t_state):
        '''
        Return an action according to the agent policy
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        # set a random action in case of exploring world
        max_val = 0
        best_Action = random.choice(Environment.valid_actions)
        # arg max Q-value choosing a action better than zero
        for action, val in self.q_table[str(t_state)].iteritems():
            if val > max_val:
                max_val = val
                best_Action = action
        return best_Action

    def _apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward
        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        # check if there is some state in cache
        if self.old_state:
            # apply: Q <- r + y max_a' Q(s', a')
            # note that s' is the result of apply a in s. a' is the action that
            # would maximize the Q-value for the state s'
            s_state = str(state)
            max_Q = 0.
            l_aux = self.q_table[s_state].values()
            if len(l_aux) > 0:
                max_Q = max(l_aux)
            gamma_f_max_Q_a_prime = self.f_gamma * max_Q
            f_new = self.last_reward + gamma_f_max_Q_a_prime
            self.q_table[str(self.old_state)][self.last_action] = f_new
        # save current state, action and reward to use in the next run
        # apply s <- s'
        self.old_state = state
        self.last_action = action
        self.last_reward = reward
        # make sure that the current state has at the current reward
        self.q_table[str(self.old_state)][self.last_action] = self.last_reward


class LearningAgent_k(BasicLearningAgent):
    '''
    A representation of an agent that learns to drive adopts a probabilistic
    approach to select actions
    '''
    def __init__(self, env, f_gamma=0.9, f_k=0.3):
        '''
        Initialize a LearningAgent2. Save all parameters as attributes
        :param env: Environment object. The grid-like world
        :*param f_gamma: float. weight of delayed versus immediate rewards
        :*param f_k: float. How strongly should favor high Q-hat values
        '''
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(LearningAgent_k, self).__init__(env)
        # override color
        self.color = 'yellow'
        # TODO: Initialize any additional variables here
        self.f_k = f_k

    def _take_action(self, t_state):
        '''
        Return an action according to the agent policy
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        # set a random action in case of exploring world
        max_val = 0
        cum_prob = 0.
        f_count = 0.
        best_Action = random.choice(Environment.valid_actions)
        # arg max Q-value choosing a action better than zero
        for action, val in self.q_table[str(t_state)].iteritems():
            # just consider action with positive rewards
            if val > 0:
                f_count += 1
                cum_prob += self.f_k ** val
            if val > max_val:
                max_val = val
                best_Action = action
        # choose the best_action just if: eps <= k**thisQhat / sum(k**Qhat)
        # if the agent still did not test all actions: (4. - f_count) * k**0.25
        f_prob = ((self.f_k ** max_val) /
                  ((4. - f_count) * 0.2 + cum_prob))
        if (random.random() <= f_prob):
            root.debug('action: explotation, k = {}'.format(self.f_k))
            return best_Action
        else:
            root.debug('action: exploration, k = {}'.format(self.f_k))
            return random.choice(Environment.valid_actions)


class LearningAgent(LearningAgent_k):
    '''
    A representation of an agent that learns to drive assuming that the world
    is a non-deterministic MDP using Q-learning and adopts a probabilistic
    approach to select actions
    '''
    def __init__(self, env, f_gamma=0.9, f_k=2.):
        '''
        Initialize a LearningAgent. Save all parameters as attributes
        :param env: Environment object. The grid-like world
        :*param f_gamma: float. weight of delayed versus immediate rewards
        :*param f_k: float. How strongly should favor high Q-hat values
        '''
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(LearningAgent, self).__init__(env)
        # override color
        self.color = 'red'
        # TODO: Initialize any additional variables here
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.f_gamma = f_gamma
        self.f_k = f_k
        self.old_state = None
        self.last_action = None
        self.last_reward = None

    def _apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward
        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        # check if there is some state in cache
        if self.old_state:
            # apply: Q <- r + y max_a' Q(s', a')
            # note that s' is the result of apply a in s. a' is the action that
            # would maximize the Q-value for the state s'
            s_state = str(state)
            max_Q = 0.
            l_aux = self.q_table[s_state].values()
            if len(l_aux) > 0:
                max_Q = max(l_aux)
            gamma_f_max_Q_a_prime = self.f_gamma * max_Q
            f_new = self.last_reward + gamma_f_max_Q_a_prime
            self.q_table[str(self.old_state)][self.last_action] = f_new
        # save current state, action and reward to use in the next run
        # apply s <- s'
        self.old_state = state
        self.last_action = action
        self.last_reward = reward
        # make sure that the current state has at the current reward
        self.q_table[str(self.old_state)][self.last_action] = self.last_reward


def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(BasicAgent)  # create agent
    a = e.create_agent(BasicLearningAgent)  # create agent
    # a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow
    # longer trials

    # Now simulate it
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.01, display=False)
    # NOTE: To speed up simulation,reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C
    # on the command-line
    # save the Q table of the primary agent
    save_q_table(e)

    # k tests
    # for f_k in [0.05, 0.1, 0.3, 0.5, 1., 1.5, 2., 3., 5.]:
    #     e = Environment()
    #     a = e.create_agent(LearningAgent_k, f_k=f_k)
    #     e.set_primary_agent(a, enforce_deadline=True)
    #     sim = Simulator(e, update_delay=0.01, display=False)
    #     sim.run(n_trials=100)


if __name__ == '__main__':
    # run the code
    run()
