import numpy as np

def get_normalized_future_rewards(rewards, gamma):
    Y = np.zeros_like(rewards, dtype=float)
    H = len(rewards)
    
    # Pre-calculate powers of gamma up to the maximum possible horizon
    gammas = gamma ** np.arange(H)

    for t in range(H - 1):
        # The number of steps remaining after time t
        L = H - 1 - t
        
        # Slice the rewards and weights
        future_rewards = rewards[t+1:]
        weights = gammas[:L]
        
        # Normalize and compute the weighted sum
        Y[t] = np.dot(weights, future_rewards) / weights.sum()

    return Y

def evaluate(learner, env, state, time, eval_period=100, gamma=0.5):
    _, pred_rewards = learner.run_current_policy(env, state, time, eval_period)
    _, optimal_rewards = env.run_optimal_policy(state, time, eval_period)

    pred_prewards = get_normalized_future_rewards(pred_rewards[1:], gamma)
    optimal_prewards = get_normalized_future_rewards(optimal_rewards[1:], gamma)
    pregret = np.mean(optimal_prewards - pred_prewards)
    return pregret