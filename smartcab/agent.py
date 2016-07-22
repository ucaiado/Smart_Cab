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
import defaultdict

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


class BasicAgent(Agent):
    """
    A Basic agent representation that learns to drive in the smartcab world.
    """

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(BasicAgent, self).__init__(env)
        # override color
        self.color = 'green'
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None

    def update(self, t):
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

    def _take_action(self, d_state):
        '''
        Return an action according to the agent policy
        :param d_state: dictionary. The inputs to be considered by the agent
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


class LearningAgent(BasicAgent):
    """
    An agent that learns to drive in the smartcab world.
    """

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(LearningAgent, self).__init__(env)
        # override color
        self.color = 'red'
        # TODO: Initialize any additional variables here
        self.

    def _take_action(self):
        '''
        Return an action according to the agent policy
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


def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(BasicAgent)  # create agent
    a = e.create_agent(LearningAgent)  # create agent
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


if __name__ == '__main__':
    # run the code
    run()
