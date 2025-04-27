"""
Our DRL algorithm is called Extensively Explored and Evaluated Actor-Critic (E3AC).
We have briefly tested E3AC's performance on Mofan's 2-DoF robot environment. (https://static.mofanpy.com/static/results/ML-practice/arm5-1.mp4)

Success rate in training:
E3AC(82.93%) VS DDPG(47.47%)

In E3AC, we proposed an Extensive Exploration Strategy (EES) to enhance action exploration ability of DRL algorithms
based on deterministic policy, where we generate diverse action candidates.
Please refer to extensive_exploration_strategy() function.

We also proposed an Extensive Evaluation Architecture (EEA) to comprehensively evaluate the values of action candidates
in a fairer manner, and select the optimal action for the agent to execute.
Please refer to evaluate_and_choose_optimal_action() function.

This code is written by Ying Fengkang, Huang Huishi, and Liu Yulin from National University of Singapore.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

# 超参数
Actor_LR = 0.001  # actor学习率
Critic_LR = 0.001  # critic学习率（E3AC中有5个critic）
Gamma = 0.98  # 折扣因子
Tau = 0.01  # 软更新系数
Memory_Size = 30000
Batch_Size = 64

# Loss权重参数
Eta_1 = 0.4
Eta_2 = 1 - Eta_1  # Eta_1 + Eta_2 = 1
Omiga = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# OU噪声实现，用于探索
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.out(x))
        return action


# Critic网络：第一层对state和action分别进行线性变换，再求和加上偏置后用ReLU激活
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 256, bias=False)
        self.fc_action = nn.Linear(action_dim, 256, bias=False)
        self.bias = nn.Parameter(torch.zeros(256))
        self.fc_hidden = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = F.relu(self.fc_state(state) + self.fc_action(action) + self.bias)
        x = F.relu(self.fc_hidden(x))
        q_value = self.out(x)
        return q_value


# 软更新函数：target = (1 - tau)*target + tau*source
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


# E3AC算法类
class E3AC:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound  # 假设格式为 [min, max]

        # 初始化经验回放池：每个transition格式为 [state, action, reward, next_state]
        self.memory = np.zeros((Memory_Size, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.counter = 0
        self.current_size = 0
        self.is_memory_full = False

        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)

        # 初始化5个Critic及对应的Target Critic
        self.critics = [Critic(state_dim, action_dim).to(device) for _ in range(5)]
        self.target_critics = [copy.deepcopy(critic).to(device) for critic in self.critics]

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Actor_LR)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=Critic_LR) for critic in self.critics]

        # OU噪声
        self.ou_noise = OUNoise(action_dim)

    def output_one_action(self, state):
        # state为shape (state_dim,)的numpy数组
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        return action

    def extensive_exploration_strategy(self, state, number_of_exploration):
        """
        扩展探索策略（EES）：使用高斯噪声和OU噪声生成多样化动作候选
        """
        variance_1 = 0.1
        variance_2 = 0.5

        # 获取原始动作（确定性策略输出）
        raw_action = self.output_one_action(state)
        action_candidates = [raw_action]

        # 用方差为0.1的高斯噪声探索
        for _ in range(number_of_exploration):
            action = np.clip(np.random.normal(raw_action, variance_1),
                             self.action_bound[0], self.action_bound[1])
            action_candidates.append(action)

        # 用方差为0.5的高斯噪声探索
        for _ in range(number_of_exploration):
            action = np.clip(np.random.normal(raw_action, variance_2),
                             self.action_bound[0], self.action_bound[1])
            action_candidates.append(action)

        # 用OU噪声探索
        for _ in range(number_of_exploration):
            action = np.clip(raw_action + self.ou_noise.noise(),
                             self.action_bound[0], self.action_bound[1])
            action_candidates.append(action)

        return action_candidates

    def evaluate_and_choose_optimal_action(self, state, action_candidates):
        """
        扩展评估架构（EEA）：使用5个Critic求平均Q值，选出最优动作
        """
        q_values = []
        self.actor.eval()
        for action in action_candidates:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            q_sum = 0.0
            with torch.no_grad():
                for target_critic in self.target_critics:
                    q_sum += target_critic(state_tensor, action_tensor).item()
            q_avg = q_sum / len(self.target_critics)
            q_values.append(q_avg)
        self.actor.train()
        optimal_index = np.argmax(q_values)
        optimal_action = action_candidates[optimal_index]
        return optimal_action, q_values

    def train(self):
        if self.current_size < Batch_Size:
            return  # 样本不足时不训练

        # 软更新目标网络
        soft_update(self.target_actor, self.actor, Tau)
        for target_critic, critic in zip(self.target_critics, self.critics):
            soft_update(target_critic, critic, Tau)

        # 随机采样一个batch
        indices = np.random.randint(0, self.current_size, size=Batch_Size)
        batch = self.memory[indices]
        # 解析数据：顺序为 [state, action, reward, next_state]
        batch_s = torch.FloatTensor(batch[:, :self.state_dim]).to(device)
        batch_a = torch.FloatTensor(batch[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        batch_r = torch.FloatTensor(batch[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]).to(
            device)
        batch_next_s = torch.FloatTensor(batch[:, -self.state_dim:]).to(device)

        # 用target actor计算下一个状态的动作（不需要梯度）
        with torch.no_grad():
            a_target = self.target_actor(batch_next_s)

        # 计算每个target critic的Q值
        q_target_list = []
        for target_critic in self.target_critics:
            q_target = target_critic(batch_next_s, a_target)
            q_target_list.append(q_target)
        q_target_average = sum(q_target_list) / len(q_target_list)

        # 用在线critic计算当前动作的Q值（这里使用actor输出的动作）
        # 注意：此处的 actor 输出仅用于critic损失计算，后面 actor 损失会重新前向传播计算
        actor_output = self.actor(batch_s)
        q_list = []
        for critic in self.critics:
            q_val = critic(batch_s, actor_output)
            q_list.append(q_val)
        q_average = sum(q_list) / len(q_list)

        # 计算目标Q值
        q_target_avg = batch_r + Gamma * q_target_average
        td_error_avg = F.mse_loss(q_average, q_target_avg)

        # 对每个critic计算损失（包括TD误差和惩罚项）
        critic_losses = []
        for i, critic in enumerate(self.critics):
            q_val = critic(batch_s, actor_output)
            with torch.no_grad():
                q_target_i = batch_r + Gamma * q_target_list[i]
            td_loss = F.mse_loss(q_val, q_target_i)
            penalty = F.mse_loss(q_val, q_average)
            loss_i = Eta_1 * td_error_avg + Eta_2 * td_loss + Omiga * penalty
            critic_losses.append(loss_i)

        # 合并所有critic损失，一次性反向传播更新所有critic
        total_critic_loss = sum(critic_losses)
        for opt in self.critic_optimizers:
            opt.zero_grad()
        total_critic_loss.backward()
        for opt in self.critic_optimizers:
            opt.step()

        # ===== Actor更新阶段 =====
        # 重新前向传播计算actor输出及其对应的critic评估，构造新的计算图
        self.actor_optimizer.zero_grad()
        new_actor_output = self.actor(batch_s)
        new_q_list = []
        for critic in self.critics:
            new_q_val = critic(batch_s, new_actor_output)
            new_q_list.append(new_q_val)
        new_q_average = sum(new_q_list) / len(new_q_list)
        actor_loss = - torch.mean(new_q_average)
        actor_loss.backward()
        self.actor_optimizer.step()

    def store_transition(self, state, action, reward, next_state):
        index = self.counter % Memory_Size
        self.counter += 1
        transition = np.hstack((state, action, [reward], next_state))
        self.memory[index, :] = transition
        if self.counter >= Memory_Size:
            self.is_memory_full = True
        self.current_size = min(self.counter, Memory_Size)

    def save(self, path='E3AC_model.pth'):
        # 保存actor和critic网络参数
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critics': [critic.state_dict() for critic in self.critics],
            'target_critics': [target_critic.state_dict() for target_critic in self.target_critics]
        }, path)

    def load(self, path='E3AC_model.pth'):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        for critic, state in zip(self.critics, checkpoint['critics']):
            critic.load_state_dict(state)
        for target_critic, state in zip(self.target_critics, checkpoint['target_critics']):
            target_critic.load_state_dict(state)


if __name__ == '__main__':
    state_dim = 21
    action_dim = 7
    action_bound = [-1, 1]
    rl = E3AC(state_dim, action_dim, action_bound)

    # 随机生成一个状态用于测试
    s = np.random.rand(state_dim)
    print('state\n', s)

    # 使用扩展探索策略（EES）生成动作候选
    action_candidates = rl.extensive_exploration_strategy(s, 3)
    print('action_candidates\n', action_candidates)

    # 使用扩展评估架构（EEA）选择最优动作
    optimal_action, q_list = rl.evaluate_and_choose_optimal_action(s, action_candidates)
    print('q_value list\n', q_list)
    print('optimal action\n', optimal_action)






