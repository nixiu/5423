import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from copy import deepcopy
import os

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, z


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class SACAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 replay_buffer_capacity=50000,
                 gamma=0.99,
                 tau=0.003,
                 alpha=0.2,
                 lr=3e-4,
                 hidden_dim=256,
                 target_entropy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Handle action_range as scalar or [low, high]
        try:
            self.action_range = float(action_range)
        except (TypeError, ValueError):
            self.action_range = float(action_range[1])

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic1 = deepcopy(self.critic1).to(device)
        self.target_critic2 = deepcopy(self.critic2).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Entropy temperature (in float32 for compatibility)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.action_range
            return action.cpu().detach().numpy()[0]
        else:
            action, _, _ = self.actor.sample(state)
            return (action * self.action_range).cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0, 0, 0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Compute target Q value
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            next_actions = next_actions * self.action_range
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Update critics
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        actions_pi, log_pi, _ = self.actor.sample(states)
        actions_pi = actions_pi * self.action_range
        q1_pi = self.critic1(states, actions_pi)
        q2_pi = self.critic2(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def save(self, save_dir="Models_SAC"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(save_dir, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(save_dir, "critic2.pth"))
        torch.save(self.log_alpha, os.path.join(save_dir, "log_alpha.pth"))

    def load(self, save_dir="Models_SAC"):
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, "actor.pth"), map_location=device))
        self.critic1.load_state_dict(torch.load(os.path.join(save_dir, "critic1.pth"), map_location=device))
        self.critic2.load_state_dict(torch.load(os.path.join(save_dir, "critic2.pth"), map_location=device))
        self.log_alpha = torch.load(os.path.join(save_dir, "log_alpha.pth"), map_location=device)

    def evaluate_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q_min = torch.min(q1, q2)
        return q_min.item()


# Example usage:

# Example usage:
if __name__ == "__main__":
    from OpenAIGym import ArmEnv

    env = ArmEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = env.action_bound  # [low, high]

    agent = SACAgent(state_dim, action_dim, action_bound)
    num_episodes = 200
    batch_size = 64

    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            episode_reward += reward
        print(f"Episode {ep}: Reward = {episode_reward}")