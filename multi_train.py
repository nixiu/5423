import multiprocessing
import time
from OpenAIGym import ArmEnv
from Neural_Network_E3AC import E3AC

MAX_EPISODES = 500  # 例如，每个实例训练150个episode
MAX_EPISODES_STEPS = 300


def train_instance(sim_port):
    print(f"开始训练，实例端口：{sim_port}")
    env = ArmEnv(sim_port=sim_port)
    # 初始化RL算法（假设 E3AC 训练代码在 Neural_Network_E3AC 中）
    rl = E3AC(env.state_dim, env.action_dim, env.action_bound)

    # 训练循环（简化示例，可根据实际代码扩展）
    for i in range(MAX_EPISODES):
        #print(f"[Port {sim_port}] 开始第 {i + 1} 个 episode")
        state = env.reset()
        episode_reward = 0.
        collision_flag = False
        for j in range(MAX_EPISODES_STEPS):
            #print(f"[Port {sim_port}] 第 {i + 1} 个 episode, 第 {j + 1} 步")
            # 生成动作：这里可以调用你的探索策略
            action = env.sample_action()
            # 与环境互动
            next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(action)

            # 可加入存储与训练步骤
            rl.store_transition(state, action, reward, next_state)
            rl.train()  # 一次训练更新
            state = next_state
            episode_reward += reward

            if minimum_distance <= 0:
                collision_flag = True

            if done or j == MAX_EPISODES_STEPS - 1:
                print(
                    f"[Port {sim_port}] Ep: {i + 1} | Collision: {collision_flag} | ep_r: {episode_reward:.1f} | step: {j + 1}")
                break
    # 保存该实例训练得到的网络参数（如需要，可以单独保存，或者在后续统一更新）
    rl.save()
    print(f"实例 {sim_port} 训练结束.")


if __name__ == '__main__':
    # 要并行训练的端口号，与启动实例时的端口号一致
    ports = [23000, 23001, 23002]

    # 使用 multiprocessing 启动多个训练进程
    processes = []
    for port in ports:
        p = multiprocessing.Process(target=train_instance, args=(port,))
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("所有训练实例均已结束。")