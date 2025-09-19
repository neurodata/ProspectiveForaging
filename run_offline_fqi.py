import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from src.utils import reward_simulate, simulate_data_raw
from src.utils import path_opt, compute_normalized_future_rewards
from src.rl import FQILearner

base_reward = 10.0
map_name = "short"
decay_rate = 0.6
reward_period = 10 
session_duration = 500000
rewards_in_period = []

total_steps = session_duration
for period_start in range(0, total_steps, reward_period):
    period_end = min(period_start + reward_period, total_steps)
    steps_in_period = np.arange(period_start, period_end)
    rewards_in_period.extend(base_reward * (decay_rate ** (steps_in_period - period_start)))

pattern = [1,1,1,1,1,1,1,2,3,4,5,5,5,5,5,5,5,4,3,2]
optimal_states = np.array(pattern*(session_duration // len(pattern))).reshape(-1,1)

def next_state(current_state, action):
    next_state = current_state + action
    if current_state == 0 and next_state < 0:
        next_state = 0
    if current_state == 6 and next_state > 6:
        next_state = 6
    return next_state

def fit_and_evaluate(t, learner, experiences, fit_iterations=20, gamma=0.5, eval_period=100):
    # fit the learner
    learner.reset()
    if t > 0:
        learner.fit(experiences[:t], num_iterations=fit_iterations, n_trees=1000)

    # get the future states using the learned policy
    current_state, _, _, _ = experiences[t]
    pred_states = [current_state]
    for step in range(t, t+eval_period):
        action = learner.greedy_policy(current_state, step)
        current_state = next_state(current_state, action)
        pred_states.append(current_state)

    # evaluate
    irewards_test = [reward_simulate(pred_states[i], t+i, rewards_in_period) for i in np.arange(1, len(pred_states))]
    preward = compute_normalized_future_rewards(irewards_test, eval_period, gamma)
    optimal_states, ireward_opt = path_opt(pred_states[0], t, eval_period)
    preward_opt_test = compute_normalized_future_rewards(ireward_opt[1:], eval_period, gamma)
    pregret = (np.sum(preward_opt_test).item() - np.sum(preward).item()) / eval_period

    return pregret

def run_replicate(t_list, experiences, include_time=False):
    learner = FQILearner(include_time=include_time)
    pregret_list = []
    for t in tqdm(t_list):
        result = fit_and_evaluate(t, learner, experiences, fit_iterations=20)
        pregret_list.append(result)
    return pregret_list

data = np.load("results/RL/experiences_list.npz")
experiences_list = data["experiences_list"]

# t_list = [128, 512, 1024, 2048, 4096, 10000, 20000]
t_list = [0]


pregret_list_notime = Parallel(n_jobs=5)(delayed(run_replicate)(t_list, experiences, include_time=False) for experiences in experiences_list)

pregret_list_time = Parallel(n_jobs=5)(delayed(run_replicate)(t_list, experiences, include_time=True) for experiences in experiences_list)

np.savez("results/RL/pregret_list_fqi_0.npz", pregret_list_notime=pregret_list_notime, pregret_list_time=pregret_list_time)