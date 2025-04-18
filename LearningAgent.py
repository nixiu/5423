"""
This is the main training loop that combine the OpenAI Gym (with the proposed obstacle-avoidable reward)
and the proposed E3AC algorithm.

This code is written by Ying Fengkang, Huang Huishi, and Liu Yulin from National University of Singapore.
"""
from OpenAIGym import ArmEnv
from Neural_Network_E3AC import E3AC

MAX_EPISODES = 1500
MAX_EPISODES_STEPS = 300
ON_TRAIN = True  # set "ON_TRAIN=True" to train agent, set is as False to evaluate a trained agent.

# set up the OpenAI Gym environment
env = ArmEnv()

# set up the parameters for E3AC
state_dim = env.state_dim
action_dim = env.action_dim
action_bound = env.action_bound   # clip the action value with the bound

# choose the DRL algorithm that you want to use. E3AC is used here.
# rl = DDPG(a_dim, s_dim, a_bound)
rl = E3AC(state_dim, action_dim, action_bound)

import time
def train():
    """
    function train() establishes a training loop for agent to interact with the environment and optimize the policy.
    """
    for i in range(MAX_EPISODES):
        # For each episode, initialize the environment with random target position.
        state = env.reset()
        # episode accumulative reward
        episode_reward = 0.
        # If collision occurs, "collision_flag" will be True.
        collision_flag = False
        start_time = time.time()
        for j in range(MAX_EPISODES_STEPS):

            # Explore diverse action candidates by extensive exploration strategy (EES).
            action_candidates = rl.extensive_exploration_strategy(state, 3)

            '''
            Evaluate each action by extensive evaluation architecture (EEA) 
            and output the optimal action with maximum Q_EEA.
            q_list_for_action_candidates: records Q_EEA for each action candidates. 
            '''
            optimal_action, q_list_for_action_candidates = rl.evaluate_and_choose_optimal_action(state, action_candidates)
            # print('action_candidates', action_candidates)
            # print('optimal_action', optimal_action)
            # print('q_list_for_action_candidates', q_list_for_action_candidates)

            # The agent interact with the environment by the optimal action.
            next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(optimal_action)

            '''
            Judge if collision occurs.
            Note we allow the agent to go on exploration until it reaches the target or uses up the training steps,
            even if collision occurs. This could help the agent fully explore the rest of the workspace.
            Also, it is fine if we start a new training episode immediately when collision occurs. But it could be
            less efficient to explore the environment and learn the policy.
            After all, collision does not bring real damage if we train in simulation.
            '''
            #print('minimum_distance', minimum_distance)
            if minimum_distance <= 0:
                collision_flag = True

            # store the transition to the memory pool.
            rl.store_transition(state, optimal_action, reward, next_state)

            episode_reward += reward

            # Randomly sample batch-size data from memory pool, calculate losses and optimize network parameters.
            rl.train()

            state = next_state

            # If the agent reaches the target or uses up the training steps, terminate the episode and print some data.
            if done or j == MAX_EPISODES_STEPS - 1:
                print(
                    'Ep: %i | %s | Collision: %i | ep_r: %.1f | step: %i | pose_error: %.4f | orient_error: %.4f'
                    % (i + 1, '---' if not done else 'done', collision_flag, episode_reward, j + 1, pose_error,
                       orient_error))
                break
        end_time = time.time()
        print('Time for one episode:', end_time - start_time)
    # save the trained network parameters
    rl.save()


def eval():
    """
    function eval() is used to evaluate the performance of a well-trained policy.
    It follows the same loop as train(). The only difference is we do not need to store transitions and train networks.
    """
    # load a well-trained network model.
    rl.load()
    print('Load well-trained network parameters.')

    for i in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0.
        collision_flag = False

        for j in range(MAX_EPISODES_STEPS):

            action_candidates = rl.extensive_exploration_strategy(state, 3)
            optimal_action, q_list_for_action_candidates = rl.evaluate_and_choose_optimal_action(state, action_candidates)
            # print('action_candidates', action_candidates)
            # print('optimal_action', optimal_action)
            # print('q_list_for_action_candidates', q_list_for_action_candidates)

            next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(optimal_action)

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



