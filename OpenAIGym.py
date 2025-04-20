"""
This code is authored by Ying Fengkang, Huang Huishi, and Liu Yulin from National University of Singapore.
"""
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
SUBSTEPS = 4
class FuzzyReward(object):
    """
    Fuzzy system is mainly used to calculate rewards without manually establishing perfect explicit reward functions.
    For example, to calculate a pose reward, it is difficult to determine the relationship between the position reward
    and the orientation reward. Is it linear or nonlinear? What is the exact coefficient or weight of the two parts?
    Here, we use fuzzy system to represent the relationship/weight by several if-then rules.
    In the current edition, we only apply fuzzy system to pose reward.
    In the future, we hope to apply it to collision avoidance reward, too.
    """
    # Note: ensure 1 is reachable by setting a value slightly larger than 1
    Delta_x_range = np.arange(0, 1.001, 0.01)
    Delta_y_range = np.arange(0, 1.001, 0.01)
    fuzzy_reward1_range = np.arange(-1, 0.001, 0.01)

    # Create fuzzy control variables (Inputs & Outputs)
    Delta_x = ctrl.Antecedent(Delta_x_range, 'Delta_x')
    Delta_y = ctrl.Antecedent(Delta_y_range, 'Delta_y')
    fuzzy_reward1 = ctrl.Consequent(fuzzy_reward1_range, 'fuzzy_reward1')

    # Define fuzzy set and its membership function
    Delta_x['VG'] = fuzz.trimf(Delta_x_range, [0, 0, 1 / 4])
    Delta_x['G'] = fuzz.trimf(Delta_x_range, [0, 1 / 4, 1 / 2])
    Delta_x['N'] = fuzz.trimf(Delta_x_range, [1 / 4, 1 / 2, 3 / 4])
    Delta_x['B'] = fuzz.trimf(Delta_x_range, [1 / 2, 3 / 4, 1])
    Delta_x['VB'] = fuzz.trimf(Delta_x_range, [3 / 4, 1, 1])

    Delta_y['VG'] = fuzz.trimf(Delta_y_range, [0, 0, 1 / 4])
    Delta_y['G'] = fuzz.trimf(Delta_y_range, [0, 1 / 4, 1 / 2])
    Delta_y['N'] = fuzz.trimf(Delta_y_range, [1 / 4, 1 / 2, 3 / 4])
    Delta_y['B'] = fuzz.trimf(Delta_y_range, [1 / 2, 3 / 4, 1])
    Delta_y['VB'] = fuzz.trimf(Delta_y_range, [3 / 4, 1, 1])

    fuzzy_reward1['VB'] = fuzz.trimf(fuzzy_reward1_range, [-1, -1, -3 / 4])
    fuzzy_reward1['B'] = fuzz.trimf(fuzzy_reward1_range, [-1, -3 / 4, -1 / 2])
    fuzzy_reward1['N'] = fuzz.trimf(fuzzy_reward1_range, [-3 / 4, -1 / 2, -1 / 4])
    fuzzy_reward1['G'] = fuzz.trimf(fuzzy_reward1_range, [-1 / 2, -1 / 4, 0])
    fuzzy_reward1['VG'] = fuzz.trimf(fuzzy_reward1_range, [-1 / 4, 0, 0])

    # Defuzzification —— centroid method
    fuzzy_reward1.defuzzify_method = 'centroid'

    # Rule with a VB (very bad) output
    rule1 = ctrl.Rule(antecedent=((Delta_x['VB'] & Delta_y['VB']) |
                                  (Delta_x['VB'] & Delta_y['B']) |
                                  (Delta_x['B'] & Delta_y['VB'])),
                      consequent=fuzzy_reward1['VB'], label='rule VB')

    # Rule with a B (bad) output
    rule2 = ctrl.Rule(antecedent=((Delta_x['VB'] & Delta_y['N']) |
                                  (Delta_x['VB'] & Delta_y['G']) |
                                  (Delta_x['B'] & Delta_y['B']) |
                                  (Delta_x['B'] & Delta_y['N']) |
                                  (Delta_x['N'] & Delta_y['VB']) |
                                  (Delta_x['N'] & Delta_y['B']) |
                                  (Delta_x['G'] & Delta_y['VB'])),
                      consequent=fuzzy_reward1['B'], label='rule B')

    # Rule with a N (normal) output
    rule3 = ctrl.Rule(antecedent=((Delta_x['VB'] & Delta_y['VG']) |
                                  (Delta_x['B'] & Delta_y['G']) |
                                  (Delta_x['N'] & Delta_y['N']) |
                                  (Delta_x['G'] & Delta_y['B']) |
                                  (Delta_x['VG'] & Delta_y['VB'])),
                      consequent=fuzzy_reward1['N'], label='rule N')

    # Rule with a G (good) output
    rule4 = ctrl.Rule(antecedent=((Delta_x['B'] & Delta_y['VG']) |
                                  (Delta_x['N'] & Delta_y['G']) |
                                  (Delta_x['N'] & Delta_y['VG']) |
                                  (Delta_x['G'] & Delta_y['N']) |
                                  (Delta_x['G'] & Delta_y['G']) |
                                  (Delta_x['VG'] & Delta_y['B']) |
                                  (Delta_x['VG'] & Delta_y['N'])),
                      consequent=fuzzy_reward1['G'], label='rule G')

    # Rule with a VG (very good) output
    rule5 = ctrl.Rule(antecedent=((Delta_x['G'] & Delta_y['VG']) |
                                  (Delta_x['VG'] & Delta_y['G']) |
                                  (Delta_x['VG'] & Delta_y['VG'])),
                      consequent=fuzzy_reward1['VG'], label='rule VG')

    # Initialization
    fuzzy_system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5])
    reward_calculator = ctrl.ControlSystemSimulation(fuzzy_system)

    def fuzzy_reward(self, dx, dy):
        # calculate the output/reward
        self.reward_calculator.input['Delta_x'] = dx  # position, normalized to 0~1
        self.reward_calculator.input['Delta_y'] = dy  # Orientation, normalized to 0~1
        self.reward_calculator.compute()
        output_reward = self.reward_calculator.output['fuzzy_reward1']
        return output_reward


class ArmEnv(object):
    action_bound = [-1.5, 1.5]
    state_dim = 21
    action_dim = 7
    distance_old = 0
    distance_new = 0
    orient_old = 0
    orient_new = 0
    def __init__(self, sim_port=23000):
        self.sim_port = sim_port
        self.connect_to_Coppeliasim()
        self.retrieve_Object_Handles()
        # 目标位姿和机械臂初始状态
        self.goal_pose = {'x': 1.00, 'y': 0.10, 'z': 0.70, 'pe': 0.02,
                          'alpha': 0, 'beta': 0, 'gamma': 0, 'oe': 6 * np.pi / 180}
        self.arm_info = [0, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0]
        self.fuzzy_reward_system = FuzzyReward()
        self.on_goal = 0
        self.dist_norm = 3
        self.dist_norm2 = 0.05

    def connect_to_Coppeliasim(self):

        print(f'Program started on port {self.sim_port}')
        self.client = RemoteAPIClient(port=self.sim_port)
        self.sim = self.client.require('sim')

        # ----------- 加速基础设置 -----------
        #self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)  # ① 关实时锁
        #self.sim.startSimulation()  # ③ 开始仿真
        #self.client.setStepping(True)  # ② 开同步步进

        # -----------------------------------

        # GUI 专属：只有窗口存在时才无限加速
        try:
            self.sim.setInt32Param(self.sim.intparam_speedmodifier, 0)  # 0 = 无限
        except Exception as e:
            # 356 = headless 实例，没有 GUI，安全忽略
            if '356' not in str(e):
                raise

        print(f'Connected to remote API server on port {self.sim_port}')

    def retrieve_Object_Handles(self):
        self.arm_joint = {}
        self.arm_joint[0] = self.sim.getObject('/LBR_iiwa_7_R800_joint1')
        self.arm_joint[1] = self.sim.getObject('/LBR_iiwa_7_R800_joint2')
        self.arm_joint[2] = self.sim.getObject('/LBR_iiwa_7_R800_joint3')
        self.arm_joint[3] = self.sim.getObject('/LBR_iiwa_7_R800_joint4')
        self.arm_joint[4] = self.sim.getObject('/LBR_iiwa_7_R800_joint5')
        self.arm_joint[5] = self.sim.getObject('/LBR_iiwa_7_R800_joint6')
        self.arm_joint[6] = self.sim.getObject('/LBR_iiwa_7_R800_joint7')
        self.goal = self.sim.getObject('/goal')
        self.tip = self.sim.getObject('/tip')
        self.target = self.sim.getObject('/target')
        self.cylinder = self.sim.getObject('/conferenceChair')
        # 用机械臂集合来代表整条机械臂
        base_link = self.sim.getObject('/LBR_iiwa_7_R800')  # 模型根节点
        coll = self.sim.createCollection(0)
        self.sim.addItemToCollection(
            coll,  # ① collectionHandle
            self.sim.handle_tree,  # ② what
            base_link,  # ③ objectHandle
            2  # ④ options
        )
        self.arm_collection = coll
    def sample_action(self):
        """
        随机采样一个动作，用于测试
        """
        variance = 0.08   # 高斯噪声方差
        action = 2 * np.random.rand(7) - 1  # 在 [-1, 1] 范围内随机生成动作
        action_with_noise = np.clip(np.random.normal(action, variance), *self.action_bound)
        return action_with_noise

    def step(self, action):
        """
        执行一次动作，并通过新版接口获取状态、计算奖励。
        """
        done = False
        action = action * np.pi / 180  # 转换为弧度
        self.arm_info = self.arm_info + action

        # 限制各个关节角度在允许范围内
        self.arm_info[0] = np.clip(self.arm_info[0], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[1] = np.clip(self.arm_info[1], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[2] = np.clip(self.arm_info[2], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[3] = np.clip(self.arm_info[3], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[4] = np.clip(self.arm_info[4], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[5] = np.clip(self.arm_info[5], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[6] = np.clip(self.arm_info[6], -175 * np.pi / 180, 175 * np.pi / 180)

        # 设置各关节的新角度
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], self.arm_info[i])
        # for _ in range(SUBSTEPS):
        #     self.sim.step()  # 让引擎真正前进 SUBSTEPS×dt
        # 获取 TCP（工具末端）的位置信息与朝向
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        #print(finger_xyz)
        #print(finger_orient)
        # 使用机械臂集合与 Cylinder 进行距离检测（偏移量传入独立数值）
        res = self.sim.checkDistance(self.arm_collection, self.cylinder)
        distance_Cylinder = res[1]
        #print("distance_Cylinder", distance_Cylinder)
        if isinstance(distance_Cylinder, list):
            distance_Cylinder = distance_Cylinder[-1]

        # 计算归一化的位置信息及姿态误差
        dist1 = [(self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
                 (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
                 (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm]

        dist2 = [finger_orient[0] / np.pi, finger_orient[1] / np.pi, finger_orient[2] / np.pi]

        distance_tipgoal = np.sqrt((self.goal_pose['x'] - finger_xyz[0]) ** 2 +
                                   (self.goal_pose['y'] - finger_xyz[1]) ** 2 +
                                   (self.goal_pose['z'] - finger_xyz[2]) ** 2)

        delta_pos = np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2)
        delta_orient = (np.abs(dist2[0]) + np.abs(dist2[1]) + np.abs(dist2[2])) / 3

        # 计算基于 fuzzy 系统的奖励
        r = - delta_pos - delta_orient

        """r2"""
        self.distance_new = np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2)
        if self.distance_new < self.distance_old:
            r += 0.05
        else:
            r -= 0.05
        self.distance_old = self.distance_new.copy()

        self.orient_new = np.abs(dist2[0]) + np.abs(dist2[1]) + np.abs(dist2[2])
        if self.orient_new < self.orient_old:
            r += 0.03
        elif self.orient_new > self.orient_old:
            r -= 0.03
        self.orient_old = self.orient_new.copy()

        # 额外奖励（如避障和目标达成奖励）计算，与原代码一致
        c1 = 0.05
        c2 = 1.6
        m1 = distance_Cylinder  # 使用检测到的距离值
        if m1 >= 1 / c1 ** 2:
            m1 = 0.39
        m2 = distance_tipgoal
        a = 1 - 4 * (1 / c2) * m1
        b = 1 / c2 + c2 - 4 * m1
        r_att = a * (m2 ** 2) - b * m2 + 1

        c = 1 - 2 * (1 / c1) * m2
        d = 1 / c1 - c1 + 2 * m2

        if m1 <= 0.05:
            r_rep = c * (m1 ** 2) + d * m1 - 1
        else:
            r_rep = 0

        r += (r_att + r_rep) / 2

        # 目标检测：如果 TCP 在目标区域内连续达到一定步数，则任务结束
        if (self.goal_pose['x'] - self.goal_pose['pe'] < finger_xyz[0] < self.goal_pose['x'] + self.goal_pose['pe'] and
                self.goal_pose['y'] - self.goal_pose['pe'] < finger_xyz[1] < self.goal_pose['y'] + self.goal_pose['pe'] and
                self.goal_pose['z'] - self.goal_pose['pe'] < finger_xyz[2] < self.goal_pose['z'] + self.goal_pose['pe'] and
                -self.goal_pose['oe'] < finger_orient[0] < self.goal_pose['oe'] and
                -self.goal_pose['oe'] < finger_orient[1] < self.goal_pose['oe'] and
                -self.goal_pose['oe'] < finger_orient[2] < self.goal_pose['oe']):
            self.on_goal += 1
            r += 1
            if self.on_goal >= 50:
                done = True
        else:
            self.on_goal = 0

        s = np.concatenate((self.arm_info,
                            np.array(finger_xyz) / self.dist_norm,
                            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
                            dist1,
                            dist2,
                            [distance_Cylinder],
                            [1. if self.on_goal else 0.]))

        return s, r, done, delta_pos, delta_orient, distance_Cylinder

    def test_with_trained_trajectory(self, joint_angle):
        """
        使用经过训练的轨迹测试模型
        """
        self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z'] = 0.97, 0.1, 0.71
        # 将 numpy 数组转换为列表，避免序列化错误
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]).tolist()
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], joint_angle[i])

        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        res = self.sim.checkDistance(self.arm_collection, self.cylinder)
        distance_Cylinder = res[1]
        #print("distance_Cylinder", distance_Cylinder)
        if isinstance(distance_Cylinder, list):
            distance_Cylinder = distance_Cylinder[-1]

        dist1 = [(self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
                 (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
                 (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm]
        dist2 = [finger_orient[0] / np.pi, finger_orient[1] / np.pi, finger_orient[2] / np.pi]

        delta_pos = np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2)
        delta_orient = (np.abs(dist2[0]) + np.abs(dist2[1]) + np.abs(dist2[2])) / 3

        return delta_pos, delta_orient, distance_Cylinder

    def reset(self):
        """
        重置环境，并通过新版接口初始化各对象位置。
        """
        self.goal_pose['x'] = 1.0 - np.random.rand() * 0.05
        self.goal_pose['y'] = 0.1 + (2 * np.random.rand() - 1) * 0.025
        self.goal_pose['z'] = 0.7 + (2 * np.random.rand() - 1) * 0.025
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]).tolist()
        self.on_goal = 0
        self.arm_info = [90 * np.pi / 180, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0]
        cylinder_pos = [0.925, 0.124, 0.90]

        for k in range(7):
            self.sim.setJointPosition(self.arm_joint[k], self.arm_info[k])
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        self.sim.setObjectPosition(self.cylinder, -1, cylinder_pos)

        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        # 使用机械臂集合与 Cylinder 进行距离检测
        res = self.sim.checkDistance(self.arm_collection, self.cylinder)
        distance_Cylinder = res[1]
        print("distance_Cylinder", distance_Cylinder)
        if isinstance(distance_Cylinder, list):
            distance_Cylinder = distance_Cylinder[-1]

        dist1 = [(self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
                 (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
                 (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm]
        dist2 = [finger_orient[0] / np.pi, finger_orient[1] / np.pi, finger_orient[2] / np.pi]

        s = np.concatenate((self.arm_info,
                            np.array(finger_xyz) / self.dist_norm,
                            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
                            dist1,
                            dist2,
                            [distance_Cylinder],
                            [1. if self.on_goal else 0.]))
        return s


# if __name__ == '__main__':
#     client = RemoteAPIClient()
#     # 获取新版的 self.sim 接口（内部已建立 ZMQ 连接）
#     self.sim = client.require('self.sim')
#     env = ArmEnv()
#     MAX_EPISODES = 5
#     MAX_EPISODES_STEPS = 100
#
#     # 根据需要选择测试方式
#     is_test_with_trained_model = True
#
#     if is_test_with_trained_model:
#         import pandas as pd
#
#         data = pd.read_excel('COR_DDPG_scene3_data.xls', header=None)
#         collision_flag = False
#         for k in range(0, len(data.iloc[:, 0]), 10):
#             joint_angle = data.iloc[k, :]
#             pose_error, orient_error, minimum_distance = env.test_with_trained_trajectory(joint_angle)
#             if minimum_distance <= 0:
#                 collision_flag = True
#             if k % 50 == 0:
#                 print('Collision: %i | pose_error: %.4f | orient_error: %.4f'
#                       % (collision_flag, pose_error, orient_error))
#     else:
#         for i in range(MAX_EPISODES):
#             state = env.reset()
#             episode_reward = 0.
#             collision_flag = False
#             for j in range(MAX_EPISODES_STEPS):
#                 action = env.sample_action()
#                 next_state, reward, done, pose_error, orient_error, minimum_distance = env.step(action)
#                 if minimum_distance <= 0:
#                     collision_flag = True
#                 state = next_state
#                 episode_reward += reward
#                 if done or j == MAX_EPISODES_STEPS - 1:
#                     print('Ep: %i | %s | Collision: %i | ep_r: %.1f | step: %i | pose_error: %.4f | orient_error: %.4f'
#                           % (i + 1, '---' if not done else 'done', collision_flag, episode_reward, j + 1, pose_error,
#                              orient_error))