"""
This is the main training loop that combine the OpenAI Gym (with the proposed obstacle-avoidable reward)
and the proposed E3AC algorithm.

This code is written by Ying Fengkang, Huang Huishi, and Liu Yulin from National University of Singapore.
"""
from OpenAIGym import ArmEnv
from Neural_Network_E3AC import E3AC
from Neural_Network_SAC import SACAgent
MAX_EPISODES = 500
MAX_EPISODES_STEPS = 500
batch_size = 64
ON_TRAIN = True  # set "ON_TRAIN=True" to train agent, set is as False to evaluate a trained agent.

# set up the OpenAI Gym environment
env = ArmEnv()

# set up the parameters for E3AC
state_dim = env.state_dim
action_dim = env.action_dim
action_bound = env.action_bound   # clip the action value with the bound

# choose the DRL algorithm that you want to use. E3AC is used here.
# rl = DDPG(a_dim, s_dim, a_bound)
rl = SACAgent(state_dim, action_dim, action_bound)

import time
def train():
    #rl.load()
    for i in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0.
        collision_flag = False
        start_time = time.time()
        if i%100 == 0:
            rl.save()
        for j in range(MAX_EPISODES_STEPS):
            action = rl.select_action(state, evaluate=False)
            next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(action)

            if minimum_distance <= 0.005:
                collision_flag = True

            # 存储 transition
            rl.store_transition(state, action, reward, next_state, done)
            # **立即更新网络**
            rl.update(batch_size)

            episode_reward += reward
            state = next_state

            if done or j == MAX_EPISODES_STEPS - 1:
                print(f"Ep: {i+1} | {'done' if done else '---'} | Collision: {int(collision_flag)} "
                      f"| ep_r: {episode_reward:.1f} | step: {j+1} "
                      f"| pose_error: {pose_error:.4f} | orient_error: {orient_error:.4f}")
                break

        print("Time for one episode:", time.time() - start_time)
    rl.save()


def eval():
    """
    function eval() is used to evaluate the performance of a well-trained policy.
    It follows the same loop as train(). The only difference is we do not need to store transitions and train networks.
    """
    # load a well-trained network model.
    rl.load()
    print('Load well-trained network parameters.')
    test_episodes = 10
    for i in range(test_episodes):
        state = env.reset()
        episode_reward = 0.
        collision_flag = False

        for j in range(MAX_EPISODES_STEPS):
            action = rl.select_action(state, evaluate=False)
            #action_candidates = rl.extensive_exploration_strategy(state, 3)
            #optimal_action, q_list_for_action_candidates = rl.evaluate_and_choose_optimal_action(state, action_candidates)
            # print('action_candidates', action_candidates)
            # print('optimal_action', optimal_action)
            # print('q_list_for_action_candidates', q_list_for_action_candidates)

            next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(action)
            if minimum_distance <= 0.005:
                collision_flag = True
            episode_reward += reward

            state = next_state

            if done or j == MAX_EPISODES_STEPS - 1:
                print(
                    'Ep: %i | %s | Collision: %i | ep_r: %.1f | step: %i | pose_error: %.4f | orient_error: %.4f'
                    % (i + 1, '---' if not done else 'done', collision_flag, episode_reward, j + 1, pose_error,
                       orient_error))
                break


if ON_TRAIN:
    train()
else:
    eval()



