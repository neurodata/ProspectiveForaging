import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
# Rollout Buffer (On-Policy)
# -------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.times = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def push(self, state, action, time, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.times.append(time)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.times.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

# -------------------------
# Critic Network (Value Function)
# -------------------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, t_dim=50, include_time=False):
        super(ValueNetwork, self).__init__()
        input_dim = state_dim + t_dim if include_time else state_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.include_time = include_time
        self.time_embed = TimeEmbedding(t_dim)
        self.state_dim = state_dim

    def forward(self, state, time):
        state = F.one_hot(state, num_classes=self.state_dim).float()
        if self.include_time:
            x = torch.cat([state, self.time_embed(time)], dim=-1)
        else:
            x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
# -------------------------
# Actor Network (Policy)
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, t_dim=50, include_time=False):
        super(PolicyNetwork, self).__init__()
        # Input dimension depends on whether we use time embeddings
        input_dim = state_dim + t_dim if include_time else state_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        self.include_time = include_time
        self.time_embed = TimeEmbedding(t_dim)
        self.state_dim = state_dim

    def forward(self, state, time):
        # Convert discrete state to one-hot encoding
        state_one_hot = F.one_hot(state, num_classes=self.state_dim).float()
        
        if self.include_time:
            t_emb = self.time_embed(time)
            # Ensure dimensions match for concatenation (batch_size, dim)
            x = torch.cat([state_one_hot, t_emb], dim=-1)
        else:
            x = state_one_hot
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def get_action_dist(self, state, time):
        """
        Helper for the update phase to get a Categorical distribution
        object directly for entropy and log_prob calculations.
        """
        probs, _ = self.forward(state, time)
        return torch.distributions.Categorical(probs)

# -------------------------
# PPO Learner
# -------------------------
class PPOLearner:
    def __init__(self, include_time=False, state_dim=7, action_dim=3, 
                 gamma=0.99, lam=0.95, eps_clip=0.2, K_epochs=10, lr=3e-4):
        # GAE parameters
        self.gamma = gamma
        self.lam = lam 

        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr = lr

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.include_time = include_time

        self.policy = PolicyNetwork(state_dim, action_dim, include_time=include_time)
        self.critic = ValueNetwork(state_dim, include_time=include_time)
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

        self.buffer = RolloutBuffer()

    def select_action(self, state, time, eval_mode=False):
        with torch.no_grad():
            state_t = torch.LongTensor([state])
            time_t = torch.FloatTensor([time])
            probs, log_probs_all = self.policy(state_t, time_t)
            value = self.critic(state_t, time_t)

            if eval_mode:
                action = torch.argmax(probs, dim=-1).item()
                return action
            else:
                # Sample action
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = log_probs_all[0, action.item()]
                return action.item(), log_prob.item(), value.item()

    def update(self, next_state, next_time):
        # 1. Prepare data from buffer
        states = torch.LongTensor(self.buffer.states)
        actions = torch.LongTensor(self.buffer.actions)
        times = torch.FloatTensor(self.buffer.times)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)
        values = torch.FloatTensor(self.buffer.values)
        rewards = self.buffer.rewards

        # 2. Compute GAE and Returns (The Bootstrapping Part)
        with torch.no_grad():
            next_value = self.critic(torch.LongTensor([next_state]), torch.FloatTensor([next_time])).item()
            
            advantages = []
            gae = 0
            for i in reversed(range(len(rewards))):
                # Delta is the temporal difference error
                delta = rewards[i] + self.gamma * next_value - values[i]
                gae = delta + self.gamma * self.lam * gae
                advantages.insert(0, gae)
                next_value = values[i]
            
            advantages = torch.FloatTensor(advantages)
            returns = advantages + values # rewards-to-go
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Optimization Loop (K Epochs)
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

        self.buffer.clear()