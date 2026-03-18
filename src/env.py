import numpy as np

class ForagingEnv:
    def __init__(
        self,
        initial_reward: int = 10,
        decay_rate: float = 0.6,
        window_size: int = 10,
        session_length: int = 500000,
        stochastic: bool = False,
        noise: float = 0.1
    ):
        self.initial_reward = initial_reward
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.period = 2 * window_size
        self.session_length = session_length
        self.stochastic = stochastic

        self.patch_A = 1
        self.patch_B = 5

        self.rewards_in_session = self._compute_session_rewards()

        self.action_space = [-1, 0, 1]  # left, stay, right
        self.state_space = [0, 1, 2, 3, 4, 5, 6]
        self.transition_tensor = self.generate_transition_tensor(noise=noise)
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.state = 1  # Start at state 1
        return self.state
    
    def generate_transition_tensor(self, noise=0.1):
        num_actions = len(self.action_space)
        num_states = len(self.state_space)
        # Initialize tensor with zeros: (Actions, States, Next_States)
        tensor = np.zeros((num_actions, num_states, num_states))
        
        for a_idx, action in enumerate(self.action_space):
            for s in range(num_states):
                # 1. Determine the "intended" outcome
                intended_s_prime = np.clip(s + action, 0, num_states - 1)
                
                # 2. Assign probabilities
                # (1 - noise) chance to land in the intended state
                tensor[a_idx, s, intended_s_prime] += (1.0 - noise)
                
                # noise chance to stay in the current state (the "slip")
                tensor[a_idx, s, s] += noise
                
                # Note: If intended_s_prime == s, the probability correctly sums to 1.0
        return tensor
    
    def get_next_state(self, action, state=None):
        state = self.state if state is None else state
        probs = self.transition_tensor[action, state]

        if self.stochastic:
            # Sample based on the probability distribution
            return np.random.choice(len(probs), p=probs)
        else:
            # Return the most likely state (deterministic)
            return np.argmax(probs)
    
    def step(self, action):
        # Action: -1 (left), 0 (stay), +1 (right)
        reward = self._get_current_reward(self.state, self.current_step)
        next_state = self.get_next_state(action)
        self.state = next_state
        self.current_step += 1
        return next_state, reward
    
    def get_state(self):
        return self.state

    def get_time(self):
        return self.current_step
    
    def run_optimal_policy(self, state, time, horizon):
        curr_time = time
        curr_state = state
        trajectory = [curr_state]
        rewards = [self._get_current_reward(curr_state, curr_time)]
        if horizon: 
            for _ in range(horizon):
                target = self._get_current_target(curr_time)
                if curr_state > target:
                    curr_state -= 1
                elif curr_state < target:
                    curr_state += 1
                curr_time += 1
                trajectory.append(curr_state)
                rewards.append(self._get_current_reward(curr_state, curr_time))
        return trajectory, rewards
    
    def _compute_session_rewards(self):
        rewards_per_window = self.initial_reward * self.decay_rate ** np.arange(self.window_size)
        rewards_per_window = rewards_per_window.tolist()
        num_windows, remainder = divmod(self.session_length, self.window_size)
        return rewards_per_window * num_windows + rewards_per_window[:remainder] 

    def _get_current_reward(self, state, time):
        phase = time % self.period
        if state == self.patch_A and phase < self.window_size:
            return self.rewards_in_session[time]
        elif state == self.patch_B and phase >= self.window_size:
            return self.rewards_in_session[time]
        else:
            return 0.0
        
    def _get_current_target(self, time):
        phase = time % self.period
        q1 = self.period // 4
        q3 = self.period // 4 * 3
        if phase <= q1: 
            return self.patch_A
        elif phase > q3:
            return self.patch_A
        else:
            return self.patch_B