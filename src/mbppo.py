import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

from .ppo import (
    PPOLearner
)

# -------------------------
# How it works
# -------------------------

# 1. Train world model: Sample a batch from the history buffer and update the world model
# to predict next state and reward give current state and action (World Model: MLP with a separate heads for next state and reward)

# 2. Selecting starting points to dream: Sample N random states from the history buffer

# 3. For each starting state, use the current policy and the world model to simulate rollouts that extend H steps into the future

# 4. Run the PPO update on these "dreamed" trajectories (N x H) for K epochs.


# -------------------------
# Time-embedding
# -------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        t = t.unsqueeze(-1)
        freqs = self.freqs.to(t.device)
        sin = torch.sin(freqs * t)
        cos = torch.cos(freqs * t)

        return torch.cat([sin, cos], dim=-1)
    
# -------------------------
# History Buffer
# -------------------------
    
class HistoryBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, time, reward, log_prob, value, next_state):
        self.buffer.append((state, action, time, reward, log_prob, value, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer.clear()
    
# -------------------------
# Dynamics Model
# -------------------------

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, t_dim=50, include_time=False):
        super(DynamicsModel, self).__init__()
        input_dim = state_dim + action_dim + t_dim if include_time else state_dim + action_dim

        # Input: state (one-hot) + action (one-hot or scalar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Heads: one for state prediction, one for reward prediction
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        self.include_time = include_time
        self.time_embed = TimeEmbedding(t_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action, time):
        state = F.one_hot(state, num_classes=self.state_dim).float()
        action = F.one_hot(action, num_classes=self.action_dim).float()
        if self.include_time:
            t_emb = self.time_embed(time)
            x = torch.cat([state, action, t_emb], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Logits for the next state (categorical) and scalar reward
        next_state_logits = self.state_head(x)
        reward = self.reward_head(x)
        return next_state_logits, reward
    
    
class MBPPOLearner(PPOLearner):
    def __init__(
        self,
        imagination_horizon=10,
        imagination_batch=64,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.imagination_horizon = imagination_horizon
        self.imagination_batch = imagination_batch

        self.world_model = DynamicsModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            include_time=self.include_time
        )
        self.wm_optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=self.lr
        )
        
        self.history = HistoryBuffer()
        self.curr_policy_history = HistoryBuffer()

    def get_next_state_and_reward(self, state, action, time):
        with torch.no_grad():
            state_t = torch.LongTensor([state])
            action_t = torch.LongTensor([action])
            time_t = torch.FloatTensor([time])
            next_state_logits, reward = self.world_model(state_t, action_t, time_t)
            next_state = torch.argmax(next_state_logits, dim=-1).item()
            # probs = F.softmax(next_state_logits, dim=-1)
            # dist = torch.distributions.Categorical(probs)
            # next_state = dist.sample()
        return next_state, reward

    def simulate_rollouts(self):
        """
        Collects multiple synthetic trajectories into the buffer, 
        then performs a single PPO update on the whole batch.

        N (num trajectories) x H (horizon)
        """
        if len(self.curr_policy_history.buffer) < self.imagination_batch:
            return

        # 1. Sample all starting points at once
        seeds = self.curr_policy_history.sample(self.imagination_batch)
        
        all_dream_data = [] # To store (s, a, t, r, lp, v) per trajectory

        for seed in seeds:
            trajectory = []
            curr_state = seed[-1] # start from the 'next_state' of the history
            curr_time = seed[2] + 1
            
            for _ in range(self.imagination_horizon):
                # pick action using the current policy
                action, log_prob, value = self.select_action(curr_state, curr_time)
                
                # Predict next state and reward
                next_state, reward = self.get_next_state_and_reward(curr_state, action, curr_time)
                
                # Store temporarily
                trajectory.append((curr_state, action, curr_time, reward, log_prob, value))
                
                curr_state = next_state
                curr_time += 1
            
            # Add this dream to the master list
            all_dream_data.append((trajectory, curr_state, curr_time))

        return all_dream_data
    
    def prepare_mixed_data(self):
        real_segment = list(self.curr_policy_history.buffer)
        horizon = self.imagination_horizon

        all_trajectories = []

        for i in range(0, len(real_segment) - horizon, horizon):
            chunk = real_segment[i : i + horizon]
            final_s = real_segment[i + horizon][0] # state of next step
            final_t = real_segment[i + horizon][2] # time of next step
            all_trajectories.append((chunk, final_s, final_t))

        all_trajectories.extend(self.simulate_rollouts())
        return all_trajectories

    def train_dynamics(self, batch_size=64, epochs=5):
        if len(self.history.buffer) < batch_size:
            return 0.0, 0.0 # Not enough data yet

        self.world_model.train()
        total_state_loss = 0
        total_reward_loss = 0

        for _ in range(epochs):
            # Sample a random batch of (s, a, r, s_next, t)
            batch = self.history.sample(batch_size)
            
            states = torch.LongTensor([b[0] for b in batch])
            actions = torch.LongTensor([b[1] for b in batch])
            rewards = torch.FloatTensor([b[4] for b in batch]).unsqueeze(-1)
            next_states = torch.LongTensor([b[-1] for b in batch])
            times = torch.FloatTensor([b[2] for b in batch])

            # Predict
            state_logits, pred_rewards = self.world_model(states, actions, times)

            # Losses: CrossEntropy for discrete states, MSE for scalar rewards
            state_loss = F.cross_entropy(state_logits, next_states)
            reward_loss = F.mse_loss(pred_rewards, rewards)
            
            loss = state_loss + reward_loss

            self.wm_optimizer.zero_grad()
            loss.backward()
            self.wm_optimizer.step()
            
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()

        return total_state_loss / epochs, total_reward_loss / epochs

    def update(self, all_dream_data):
        """
        Computes GAE for each trajectory independently, 
        then pools them for the K-epoch PPO optimization.

        Each trajectory has 
        (curr_state, action, curr_time, reward, log_prob, value)
        """
        all_states, all_actions, all_times = [], [], []
        all_old_log_probs, all_values, all_returns, all_advantages = [], [], [], []

        for trajectory, final_state, final_time in all_dream_data:
            # Extract data for this specific dream
            states = torch.LongTensor([t[0] for t in trajectory])
            actions = torch.LongTensor([t[1] for t in trajectory])
            times = torch.FloatTensor([t[2] for t in trajectory])
            rewards = [t[3] for t in trajectory]
            old_log_probs = torch.FloatTensor([t[4] for t in trajectory])
            values = torch.FloatTensor([t[5] for t in trajectory])

            # Bootstrap from the end of THIS dream
            with torch.no_grad():
                next_value = self.critic(torch.LongTensor([final_state]), torch.FloatTensor([final_time])).item()
                
                traj_advantages = []
                gae = 0
                for i in reversed(range(len(rewards))):
                    delta = rewards[i] + self.gamma * next_value - values[i]
                    gae = delta + self.gamma * self.lam * gae
                    traj_advantages.insert(0, gae)
                    next_value = values[i]
                
                traj_advantages = torch.FloatTensor(traj_advantages)
                traj_returns = traj_advantages + values

            # Pool the processed data
            all_states.append(states)
            all_actions.append(actions)
            all_times.append(times)
            all_old_log_probs.append(old_log_probs)
            all_values.append(values)
            all_returns.append(traj_returns)
            all_advantages.append(traj_advantages)

        # Concatenate everything into one giant batch for PPO epochs
        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        times = torch.cat(all_times)
        old_log_probs = torch.cat(all_old_log_probs)
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Now run your K_epochs loop using S, A, T, LP, RET, ADV
        for _ in range(self.K_epochs):
            # Get current policy distribution and values
            probs, log_probs_all = self.policy(states, times)
            curr_values = self.critic(states, times).squeeze()
            
            curr_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute the entropy bonus (encourage the policy to explore more by maximizing entropy)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()

            # PPO Ratio calculation
            ratio = torch.exp(curr_log_probs - old_log_probs)

            # Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Loss composition: Policy Loss - Entropy Bonus + Value Loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(curr_values, returns)
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()