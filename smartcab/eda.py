#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Perform simple statistical data analysis in log files produced by simulation

@author: ucaiado

Created on 07/24/2016
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


def simple_counts(s_fname):
    '''
    Count the number of times the agent reached its destination and other basic
    stats
    :param s_fname: string. path to a log file
    '''
    with open(s_fname) as fr:
        i_trial = None
        i_fail = 0
        i_success = 0
        i_reached = 0
        l_last = []
        i_last_step = 0
        i_that_time = 0
        d_count = {'forward': 0, 'left': 0, 'right': 0, 'None': 0}
        for idx, row in enumerate(fr):
            s_aux = row.strip().split(';')[1]
            if 'LearningAgent.update' in s_aux:
                i_last_step = int(s_aux.split(',')[0].split('=')[1].strip())
                s_action = s_aux.split('action = ')[1].split(',')[0]
                d_count[s_action] += 1
            elif 'Environment.reset' in s_aux:
                b_already_finish = False
                if not i_trial:
                    i_trial = 1
                    i_initial_dealline = int(s_aux.split('deadline = ')[1])
                else:
                    i_trial += 1
                    l_last.append({'steps': i_initial_dealline - i_last_step,
                                   'success': i_that_time})
                    i_initial_dealline = int(s_aux.split('deadline = ')[1])
            elif 'Environment.step' in s_aux:
                if not b_already_finish:
                    i_fail += 1
                    i_that_time = 0
                    b_already_finish = True
            elif 'Environment.act' in s_aux:
                if not b_already_finish:
                    if i_last_step > 0:
                        i_success += 1
                    i_reached += 1
                    i_that_time = 1
                    b_already_finish = True

    print 'Number of Trials: {}'.format(i_trial)
    s_aux = 'Times that the agent reached the target location: {}'
    print s_aux.format(i_reached)
    print 'Times the agent reached the hard deadline: {}'.format(i_fail)
    s_aux = 'Times the agent SUCCESSFULLY reached the target location: {}'
    print s_aux.format(i_success)
    print 'Counting of moves made:\n{}'.format(d_count)
    return l_last, (i_trial, i_reached, i_fail, i_success, d_count)


def performance_measures(s_fname, i_trials=100):
    '''
    Return dictionaries counting the number of times the agent eplore and
    exploit, the cumulated average rewards and the number of times that
    compleated the route successfuly. Split results by k
    :param s_fname: string. path to a log file
    :*param i_trials: integer. number of trials at each simulation
    '''
    with open(s_fname) as fr:
        i_trial = 0
        s_k = None
        d_kact = defaultdict(lambda: defaultdict(int))
        d_krew = defaultdict(lambda: defaultdict(int))
        d_ksuccess = defaultdict(int)
        f_reward = 0.
        f_count_step = 0
        k = None
        last_k = None
        last_reward = 0.
        i_fail = 0
        i_success = 0
        i_reached = 0
        l_last = []
        i_last_step = 0
        i_that_time = 0

        for idx, row in enumerate(fr):
            s_aux = row.strip().split(';')[1]
            if 'LearningAgent.update' in s_aux:
                i_last_step = int(s_aux.split(',')[0].split('=')[1].strip())
                s_action = s_aux.split('action = ')[1].split(',')[0]
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1

            elif 'Environment.reset' in s_aux:
                if k:
                    d_krew[float(k)][i_trial] = f_reward/f_count_step

                if i_trial == i_trials:
                    d_ksuccess[float(k)] = i_success * 1. / i_trial
                    i_success = 0.
                i_trial += 1
                b_already_finish = False

            elif 'Environment.step' in s_aux:
                if not b_already_finish:
                    i_fail += 1
                    i_that_time = 0
                    b_already_finish = True

            elif 'Environment.act' in s_aux:
                if not b_already_finish:
                    if i_last_step > 0:
                        i_success += 1
                    i_reached += 1
                    i_that_time = 1
                    b_already_finish = True

            elif 'action:' in s_aux:
                s_type, s_k = s_aux[8:].split(',')
                _, k = s_k.split('k = ')
                d_kact[float(k)][s_type] += 1
                if k != last_k:
                    i_trial = 1
                    f_reward = last_reward
                    f_count_step = 1.
                last_k = k
        d_ksuccess[float(k)] = i_success * 1. / i_trial

    return d_kact, d_krew, d_ksuccess


def performance_measures_gamma(s_fname, i_trials=100):
    '''
    Return dictionaries counting the number of times the agent eplore and
    exploit, the cumulated average rewards and the number of times that
    compleated the route successfuly. Split results by Gamma
    :param s_fname: string. path to a log file
    :*param i_trials: integer. number of trials at each simulation
    '''
    with open(s_fname) as fr:
        i_trial = 0
        s_gamma = None
        d_kact = defaultdict(lambda: defaultdict(int))
        d_krew = defaultdict(lambda: defaultdict(int))
        d_ksuccess = defaultdict(int)
        f_reward = 0.
        f_count_step = 0
        gamma = None
        last_gamma = None
        last_reward = 0.
        i_fail = 0
        i_success = 0
        i_reached = 0
        l_last = []
        i_last_step = 0
        i_that_time = 0

        for idx, row in enumerate(fr):
            s_aux = row.strip().split(';')[1]
            if 'LearningAgent.update' in s_aux:
                i_last_step = int(s_aux.split(',')[0].split('=')[1].strip())
                s_action = s_aux.split('action = ')[1].split(',')[0]
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1

            elif 'Environment.reset' in s_aux:
                if gamma:
                    d_krew[float(gamma)][i_trial] = f_reward/f_count_step

                if i_trial == i_trials:
                    d_ksuccess[float(gamma)] = i_success * 1. / i_trial
                    i_success = 0.
                i_trial += 1
                b_already_finish = False

            elif 'Environment.step' in s_aux:
                if not b_already_finish:
                    i_fail += 1
                    i_that_time = 0
                    b_already_finish = True

            elif 'Environment.act' in s_aux:
                if not b_already_finish:
                    if i_last_step > 0:
                        i_success += 1
                    i_reached += 1
                    i_that_time = 1
                    b_already_finish = True

            elif 'LearningAgent.__init__(): ' in s_aux:
                s_gamma, s_k = s_aux[26:].split(',')
                _, gamma = s_gamma.split('gamme = ')
                d_kact[float(gamma)]['count'] += 1
                if gamma != last_gamma:
                    i_trial = 1
                    f_reward = last_reward
                    f_count_step = 1.
                last_gamma = gamma
        d_ksuccess[float(gamma)] = i_success * 1. / i_trial

    return d_krew, d_ksuccess
