import random
import numpy as np
from scipy.stats import percentileofscore
from behave_analysis.utils.flatten import flatten

def permutation_test(data: np.ndarray, groups: np.ndarray, sessions: np.ndarray, group_1:int=1, group_2:int=2, iterations:int = 1000, two_tailed:bool= True):
    ''' the test statistic is the pooled mean and data are shuffled by session'''
    group_1_data,     group_2_data     = data    [groups==group_1], data    [groups==group_2]
    group_1_sessions, group_2_sessions = sessions[groups==group_1], sessions[groups==group_2]
    group_1_by_session = [[group_1_data[group_1_sessions==session_num]] for session_num in np.unique(group_1_sessions)]
    group_2_by_session = [[group_2_data[group_2_sessions==session_num]] for session_num in np.unique(group_2_sessions)]
    both_groups_by_session = np.array(group_1_by_session + group_2_by_session)
    group_1_label = np.zeros(len(np.unique(group_1_sessions)))
    group_2_label = np.ones (len(np.unique(group_2_sessions)))
    labels = np.concatenate((group_1_label, group_2_label))
    null_distribution = np.ones(iterations) * np.nan
    for i in range(iterations):
        random.shuffle(labels)
        group_1_mean = np.mean(list(flatten(both_groups_by_session[labels == 0])))
        group_2_mean = np.mean(list(flatten(both_groups_by_session[labels == 1])))
        if two_tailed: null_distribution[i] = abs(group_1_mean - group_2_mean)
        else:          null_distribution[i] = group_2_mean - group_1_mean
    if two_tailed:     test_statistic = abs(np.mean(group_1_data) - np.mean(group_2_data))
    else:              test_statistic = np.mean(group_2_data) - np.mean(group_1_data)
    p = 1 - percentileofscore(null_distribution, test_statistic, kind='mean') / 100
    return p

def print_stat_test_results(p, comparison: str, two_tailed: bool, binarized: bool, group_1: int, group_2: int):
    if p > 0.0001: p = np.round(p, 4)
    if p < 0.05: result = 'SIGNIFICANT.'
    else:        result = 'not significant'
    if two_tailed: test_type = 'Two-tailed'
    else:          test_type = 'One-tailed'
    if binarized:  test_data = ', binarized'
    else:          test_data = ''
    print("\n{}{} permutation test on {}\n  group {} vs. group {}\n  p = {} \n  {}".format(test_type, test_data, comparison, group_1, group_2, p, result))
