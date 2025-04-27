"""
This code is authored by Ying Fengkang, Huang Huishi, and Liu Yulin from National University of Singapore.
"""
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

SUBSTEPS = 4
MAX_GAP_RANGE = 0.30

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

class ArmEnv(object):
    action_bound = [-1.5, 1.5]
    state_dim = 90  # 更新状态维度为 90（前4个link各7维，后2个link各3×7维，共70维 + 20其它信息）
    action_dim = 7
    distance_old = 0
    distance_new = 0
    orient_old = 0
    orient_new = 0

    def __init__(self, sim_port=23000):
        self.offset = -0.02
        self.sim_port = sim_port
        self.connect_to_Coppeliasim()
        self.retrieve_Object_Handles()
        # 目标位姿和机械臂初始状态
        self.goal_pose = {
            'x': 1.00, 'y': 0.10, 'z': 0.70, 'pe': 0.02,
            'alpha': 0, 'beta': 0, 'gamma': 0, 'oe': 60 * np.pi / 180
        }
        # 预设的 4 个 goal 位置列表
        # ---------- goal sets ----------
        # 区域 A（先到）
        self.goals_a = [
            {'x': 1.95, 'y': 0.75, 'z': 0.75},
            {'x': 1.4, 'y': 0.75, 'z': 0.75},
        ]
        # 区域 B（随后到）
        self.goals_b = [
            {'x': 0.98, 'y': 0.175, 'z': 0.68},
            {'x': 2.20, 'y': 0.175, 'z': 0.68},
        ]

        # 当前阶段：0 = A；1 = B
        self.goal_stage = 0

        # 选初始目标（A 区随机）
        idx = np.random.randint(len(self.goals_a))
        self.goal_pose.update(self.goals_a[idx])
        self.episode_count = 0
        # 初始随机选一个预设目标
        self.arm_info = [0, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0]
        self.fuzzy_reward_system = FuzzyReward()
        self.on_goal = 0
        self.dist_norm = 3
        self.dist_norm2 = 0.05
        self.safe_distance = 0.03    # 安全距离阈值（默认0.05米）
        self.sensor_range = 0.30    # 传感器探测最大范围（默认0.30米）
        self.goal_gain = 5.0         # 引力场强度，可调，越大接近目标奖励越多

    def connect_to_Coppeliasim(self):
        print(f'Program started on port {self.sim_port}')
        self.client = RemoteAPIClient(port=self.sim_port)
        self.sim = self.client.require('sim')

        # ----------- 加速基础设置 -----------
        #self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)  # ① 关实时锁

        self.client.setStepping(True)  # ② 开同步步进
        #self.sim.startSimulation()  # ③ 开始仿真
        # -----------------------------------

        # GUI 专属：只有窗口存在时才无限加速
        try:
            self.sim.setInt32Param(self.sim.intparam_speedmodifier, 0)  # 0 = 无限加速
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
        # self.cylinder = self.sim.getObject('/Cylinder')  # removed
        self.link_names = [
            '/LBR_iiwa_7_R800_link3', '/LBR_iiwa_7_R800_link4', '/LBR_iiwa_7_R800_link5',
            '/LBR_iiwa_7_R800_link6', '/LBR_iiwa_7_R800_link7', '/gripper_base_visible'
        ]
        self.links = [self.sim.getObject(n) for n in self.link_names]
        # 初始化上一帧各连杆位置（用于计算运动方向）
        self.prev_positions = {link: self.sim.getObjectPosition(link, -1) for link in self.links}
        # 创建各连杆表面的传感器
        self.create_surface_sensors()

    # =========================================================
    # 体锥传感器批量生成 & 读取
    # =========================================================
    def create_surface_sensors(self, range_far=0.30, fast=True, explicit=True):
        """
        适用于 CoppeliaSim 4.9 的 5 参数版 sim.createProximitySensor。
        在 link3‑link7 和 gripper_base_visible 每个 bbox 面中心插入 1 个 pyramid volume sensor（共36颗传感器）：
          • offset = 0   • range = range_far
          • x/y near = bbox 面长/宽
          • x/y far  = x/y near × 1.1  (略放大，6面拼起来近似360°)
        """
        PYR_TYPE = self.sim.proximitysensor_pyramid  # constant: volume_pyramid
        SUB_TYPE = 16  # deprecated, 必须 16
        OPT = (1 if explicit else 0) | (32 if fast else 0)  # bit0 + bit5

        # intParams: [faceCnt, faceCntFar, subDiv, subDivFar, rnd1, rnd2, 0, 0]
        INT_P = [4, 4, 0, 0, 0, 0, 0, 0]

        self.link_sensors = []
        TYPE = self.sim.object_proximitysensor_type
        for link in self.links:
            existing = self.sim.getObjectsInTree(link, TYPE)
            if existing:
                # 若已经有传感器，跳过此link
                print(f'[INFO] link "{self.sim.getObjectAlias(link)}" already has {len(existing)} sensors; skip.')
                self.link_sensors.append(existing)
                continue
            xmin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_x)
            xmax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_x)
            ymin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_y)
            ymax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_y)
            zmin = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_min_z)
            zmax = self.sim.getObjectFloatParam(link, self.sim.objfloatparam_objbbox_max_z)
            xm, ym, zm = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
            xLen, yLen, zLen = xmax - xmin, ymax - ymin, zmax - zmin

            # 六个面: (轴, 符号, 本地欧拉角deg)
            faces = [
                ('x', -1, (0, -90, 0)),
                ('x',  1, (0,  90, 0)),
                ('y', -1, (90, 0, 0)),
                ('y',  1, (-90, 0, 0)),
                ('z',  1, (0, 0, 0)),
                ('z', -1, (180, 0, 0))
            ]
            sensors_this_link = []
            for axis, sgn, rotDeg in faces:
                # ——浮点参数数组 (15)—————————————————————
                if axis == 'x':
                    xNear, yNear = zLen, yLen
                elif axis == 'y':
                    xNear, yNear = xLen, zLen
                else:  # 'z'
                    xNear, yNear = xLen, yLen
                xFar, yFar = xNear * 3, yNear * 3  # 稍放大探测范围
                FLT_P = [self.offset, range_far,  # offset, range
                         xNear, yNear, xFar, yFar,  # 面尺寸参数
                         0, 0, 0, 0, 0, 0, 0, 0, 0]  # 其余备用参数填0

                # ——创建传感器—————————————————————————
                h = self.sim.createProximitySensor(PYR_TYPE, SUB_TYPE, OPT, INT_P, FLT_P)
                self.sim.setObjectParent(h, link, True)
                # 将传感器贴在面中心 (offset=0时传感器顶点在obj原点)
                pos = [xm, ym, zm]
                if axis == 'x':
                    pos[0] = xmax if sgn > 0 else xmin
                elif axis == 'y':
                    pos[1] = ymax if sgn > 0 else ymin
                else:  # 'z'
                    pos[2] = zmax if sgn > 0 else zmin
                self.sim.setObjectPosition(h, link, pos)
                # 设置传感器朝向
                r = [math.radians(r_) for r_ in rotDeg]
                self.sim.setObjectOrientation(h, link, r)
                # 设置别名方便调试
                alias = f'{self.sim.getObjectAlias(link)}_{axis}{sgn}_sensor'
                self.sim.setObjectAlias(h, alias)
                sensors_this_link.append(h)
            self.link_sensors.append(sensors_this_link)
        print(f'[4.9] created {sum(len(s) for s in self.link_sensors)} pyramid sensors')

    def get_link_directions(self):
        """
        获取每个 link 在当前时刻的运动方向（单位向量列表）。
        返回一个列表，与 self.links 顺序对应，每个元素为该连杆速度方向的单位向量 [vx, vy, vz]。
        若该连杆线速度为零则返回 [0.0, 0.0, 0.0]。
        """
        directions = []
        for link in self.links:
            new_pos = self.sim.getObjectPosition(link, -1)
            old_pos = self.prev_positions[link]
            delta = [new_pos[i] - old_pos[i] for i in range(3)]
            norm = math.sqrt(sum(d * d for d in delta))
            if norm > 1e-6:
                dir_vec = [d / norm for d in delta]
            else:
                dir_vec = [0.0, 0.0, 0.0]
            # 更新 prev_positions 为当前帧位置
            self.prev_positions[link] = new_pos
            directions.append(dir_vec)
        return directions

    def step(self, action):
        """
        执行一次动作，并通过新版接口获取状态、计算奖励。
        """
        done = False
        action = action * np.pi / 180  # 将输入角度转换为弧度增量
        self.arm_info = self.arm_info + action

        # 限制各个关节角度在允许范围内
        self.arm_info[0] = np.clip(self.arm_info[0], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[1] = np.clip(self.arm_info[1], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[2] = np.clip(self.arm_info[2], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[3] = np.clip(self.arm_info[3], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[4] = np.clip(self.arm_info[4], -170 * np.pi / 180, 170 * np.pi / 180)
        self.arm_info[5] = np.clip(self.arm_info[5], -120 * np.pi / 180, 120 * np.pi / 180)
        self.arm_info[6] = np.clip(self.arm_info[6], -175 * np.pi / 180, 175 * np.pi / 180)

        # 设置各关节的新角度并推进仿真
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], self.arm_info[i])
        for _ in range(SUBSTEPS):
            self.sim.step()  # 推进仿真 SUBSTEPS×dt 时间

        # 获取 TCP（末端执行器）的位姿
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)

        # 计算归一化的位姿和朝向误差
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        distance_tipgoal = math.sqrt((self.goal_pose['x'] - finger_xyz[0])**2 +
                                     (self.goal_pose['y'] - finger_xyz[1])**2 +
                                     (self.goal_pose['z'] - finger_xyz[2])**2)
        delta_pos = math.sqrt(dist1[0]**2 + dist1[1]**2 + dist1[2]**2)
        delta_orient = (abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2]))/3
        link_dirs = self.get_link_directions()  # 获取各连杆当前运动方向
        global_min_dist = self.sensor_range  # 全局最小距离初始化
        # 基础奖励：逼近目标（距离和朝向误差的负值）
        #r = - delta_pos - delta_orient
        # 引力场奖励：距离目标越近，奖励越高
        #print(distance_tipgoal)
        #goal_attraction = self.goal_gain * (0.5 - distance_tipgoal)**3/0.125

        # 计算 tip->goal 单位向量
        goal_vec = [
            self.goal_pose['x'] - finger_xyz[0],
            self.goal_pose['y'] - finger_xyz[1],
            self.goal_pose['z'] - finger_xyz[2]
        ]
        g_norm = math.sqrt(sum(v * v for v in goal_vec))
        goal_dir = [v / g_norm for v in goal_vec]
        gripper_dir = link_dirs[-2]
        dp_goal = (
                goal_dir[0] * gripper_dir[0] +
                goal_dir[1] * gripper_dir[1] +
                goal_dir[2] * gripper_dir[2]
        )

        # print("dogoal:",dp_goal)
        r = -np.log2(distance_tipgoal)
        pos_r= r
        orient_r= 0
        #r += -np.log2(delta_orient)
        #r -= 5*delta_orient
        if distance_tipgoal < 0.1:
            #if dp_goal>0 and distance_tipgoal < 0.1:
            r += 1 / (distance_tipgoal ** 0.5) * dp_goal/10
            r += -np.log2(delta_orient)
             # 朝向误差惩罚
            #print(f'delta_orient: {delta_orient}, np.log2(delta_orient): {-np.log2(delta_orient)}')
            # 奖励 shaping：若朝向误差比上一帧减小则加分，增大则减分
            self.orient_new = abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2])
            #r -= delta_orient*3
            if self.orient_new < self.orient_old:
                r += 0.5
                #orient_r=2
            elif self.orient_new > self.orient_old:
                r -= 0.5
                #orient_r=-2
            self.orient_old = self.orient_new

        # 奖励 shaping：若距离比上一帧减小则加分，增大则减分
        # self.distance_new = delta_pos
        # if self.distance_new < self.distance_old:
        #     r += 0.08
        # else:
        #     r -= 0.08
        # self.distance_old = self.distance_new
        # # 奖励 shaping：若朝向误差比上一帧减小则加分，增大则减分
        # self.orient_new = abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2])
        # if self.orient_new < self.orient_old:
        #     r += 0.03
        # elif self.orient_new > self.orient_old:
        #     r -= 0.03
        # self.orient_old = self.orient_new

        # 基于传感器的避障奖励计算（取代原 distance_Cylinder 奖励）
        link_dirs = self.get_link_directions()  # 获取各连杆当前运动方向
        # 构建每个 link 的最近障碍信息（7 维），并在此循环内集成逐传感器避障奖励
        nearest_info = []
        num_links = len(self.link_sensors)
        for link_idx, sensors in enumerate(self.link_sensors):
            # 对最后两个 link，我需要最近3个障碍；其余 link 只要最近1个
            k = 1 if link_idx < num_links - 2 else 3
            # 存储 top-k 最近障碍：每项为 (dist, px, py, pz, sensor_handle)
            topk = []
            # 遍历该 link 所有传感器
            for s in sensors:
                out = self.sim.handleProximitySensor(s)
                detected, dist = out[0], out[1]
                # 逐传感器避障奖励逻辑同原始
                if detected and dist < self.safe_distance:
                    if dist < 0.015:
                        r -= 20.0
                        if dist<global_min_dist:
                            global_min_dist = dist
                    else:
                        m2 = self.sim.getObjectMatrix(s, -1)
                        px, py, pz = out[2]
                        vx = m2[0]*px + m2[1]*py + m2[2]*pz
                        vy = m2[4]*px + m2[5]*py + m2[6]*pz
                        vz = m2[8]*px + m2[9]*py + m2[10]*pz
                        norm = math.sqrt(vx*vx + vy*vy + vz*vz)
                        if norm > 1e-6:
                            world_dir = [vx/norm, vy/norm, vz/norm]
                        else:
                            world_dir = [0.0, 0.0, 0.0]
                        dp = (world_dir[0]*link_dirs[link_idx][0] +
                              world_dir[1]*link_dirs[link_idx][1] +
                              world_dir[2]*link_dirs[link_idx][2])
                        if dp > 0:
                            r += -1.0 / dist * dp / 20.0
                        elif dp < 0:
                            r += 1.0 / dist * abs(dp) / 20.0
                # 记录到 topk
                if detected:
                    px, py, pz = out[2]
                else:
                    dist, px, py, pz = self.sensor_range, 0.0, 0.0, 0.0
                # 插入或替换 topk 中的最大 dist
                if len(topk) < k:
                    topk.append((dist, px, py, pz, s))
                else:
                    # 找到当前 topk 中距离最大的索引
                    max_i = max(range(k), key=lambda i: topk[i][0])
                    if dist < topk[max_i][0]:
                        topk[max_i] = (dist, px, py, pz, s)
            # 如果 topk 数量不足 k，填零
            while len(topk) < k:
                topk.append((self.sensor_range, 0.0, 0.0, 0.0, None))
            # 对 topk 中每个障碍点，按距离升序排序再依次追加 7 维信息
            topk.sort(key=lambda x: x[0])
            for dist, px, py, pz, s in topk:
                # 计算世界坐标下接触点
                if s is not None:
                    m = self.sim.getObjectMatrix(s, -1)
                    world_point = [
                        m[3] + m[0]*px + m[1]*py + m[2]*pz,
                        m[7] + m[4]*px + m[5]*py + m[6]*pz,
                        m[11] + m[8]*px + m[9]*py + m[10]*pz
                    ]
                    link_pos = self.sim.getObjectPosition(self.links[link_idx], -1)
                    vector = [world_point[i] - link_pos[i] for i in range(3)]
                else:
                    vector = [0.0, 0.0, 0.0]
                    dist = self.sensor_range
                # 获取该 link 运动方向
                link_dir = link_dirs[link_idx]
                # 追加 7 维信息
                nearest_info.extend([
                    vector[0], vector[1], vector[2],
                    link_dir[0], link_dir[1], link_dir[2],
                    dist / self.dist_norm
                ])
        sensor_data = nearest_info

        # 目标检测：若 TCP 持续在目标区域内一定步数，则视为完成任务
        # ------------- 目标检测：先到 A，再到 B -------------
        reached = (
                self.goal_pose['x'] - self.goal_pose['pe'] < finger_xyz[0] < self.goal_pose['x'] + self.goal_pose[
            'pe'] and
                self.goal_pose['y'] - self.goal_pose['pe'] < finger_xyz[1] < self.goal_pose['y'] + self.goal_pose[
                    'pe'] and
                self.goal_pose['z'] - self.goal_pose['pe'] < finger_xyz[2] < self.goal_pose['z'] + self.goal_pose[
                    'pe'] and
                -self.goal_pose['oe'] < finger_orient[0] < self.goal_pose['oe'] and
                -self.goal_pose['oe'] < finger_orient[1] < self.goal_pose['oe'] and
                -self.goal_pose['oe'] < finger_orient[2] < self.goal_pose['oe']
        )

        if reached:
            self.on_goal += 1
            r += 10  # 每步在目标区奖励
            if self.on_goal >= 5:  # 停留够久算真正到达
                if self.goal_stage == 0:  # ------- 完成 A -------
                    self.goal_stage = 1
                    self.on_goal = 0
                    # 切到随机 B 目标
                    idx = np.random.randint(len(self.goals_b))
                    self.goal_pose.update(self.goals_b[idx])
                    goal_pos = [self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]
                    self.sim.setObjectPosition(self.goal, -1, goal_pos)
                    r += 30  # 额外奖励
                else:  # ------- 完成 B -------
                    done = True
        else:
            self.on_goal = 0
        # -----------------------------------------------------

        # 构建状态向量（机械臂状态 + 目标/末端位姿差 + on_goal 标志 + 传感器数据）
        s = np.concatenate((
            self.arm_info,
            np.array(finger_xyz) / self.dist_norm,
            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
            dist1, dist2,
            [1. if self.on_goal else 0.],
            np.array(sensor_data)
        ))
        #print(f'pos_r = {pos_r:.4f}, orient_r = {orient_r:.4f}, other_r = {other_r:.4f}, r = {r:.4f}, goal = {self.on_goal}, min_dist = {min_dist:.4f}')
        #print(s)  # （调试）打印状态向量
        # 返回状态、奖励、终止标志以及误差信息和本步全局最小距离
        return s, r, done, delta_pos, delta_orient, global_min_dist

    def test_with_trained_trajectory(self, joint_angle):
        """
        使用经过训练的轨迹测试模型。
        """
        # 设置特定的目标位置
        self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z'] = 0.97, 0.1, 0.71
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]).tolist()
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        # 将机械臂各关节设置为提供的关节角（轨迹点）
        for i in range(7):
            self.sim.setJointPosition(self.arm_joint[i], joint_angle[i])
        # 推进仿真以更新物理和传感器状态
        for _ in range(SUBSTEPS):
            self.sim.step()
        # 获取末端执行器当前位姿和相对目标的误差
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        delta_pos = math.sqrt(dist1[0]**2 + dist1[1]**2 + dist1[2]**2)
        delta_orient = (abs(dist2[0]) + abs(dist2[1]) + abs(dist2[2])) / 3
        # 计算与障碍物的最小距离（用于评估碰撞情况）
        min_dist = self.sensor_range
        for sensors in self.link_sensors:
            for s in sensors:
                out = self.sim.readProximitySensor(s)
                if out[0] and out[1] < min_dist:
                    min_dist = out[1]
        return delta_pos, delta_orient, min_dist

    def reset(self):
        """
        重置环境，并通过新版接口初始化各对象位置。
        """
        # Episode 计数，并每 10 个 episode 随机切换一个预设的 goal
        self.episode_count += 1
        self.goal_stage = 0
        idx = np.random.randint(len(self.goals_a))
        self.goal_pose.update(self.goals_a[idx])
        # --- add local random perturbation around the chosen A‑region goal ---
        # x, y ∈ [−0.1, 0.1],  z ∈ [−0.02, 0.02]
        self.goal_pose['x'] += np.random.uniform(-0.1,  0.1)
        self.goal_pose['y'] += np.random.uniform(-0.1,  0.1)
        self.goal_pose['z'] += np.random.uniform(-0.02, 0.02)
        goal_pos = [self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        self.on_goal = 0
        # 重置机械臂初始关节角度
        self.arm_info = [180 * np.pi / 180, 0, 0, -90 * np.pi / 180, 0, 90 * np.pi / 180, 0]
        # 设置机械臂各关节和目标物位置
        for k in range(7):
            self.sim.setJointPosition(self.arm_joint[k], self.arm_info[k])
        self.sim.setObjectPosition(self.goal, -1, goal_pos)
        # 推进仿真一步以更新传感器
        for _ in range(SUBSTEPS):
            self.sim.step()
        # 重置各 link 的 prev_positions（因为直接设置位置，相当于无运动）
        self.prev_positions = {link: self.sim.getObjectPosition(link, -1) for link in self.links}
        # 获取末端执行器位置和朝向
        finger_xyz = self.sim.getObjectPosition(self.tip, -1)
        finger_orient = self.sim.getObjectOrientation(self.tip, self.goal)
        # 计算归一化的位置和朝向误差
        dist1 = [
            (self.goal_pose['x'] - finger_xyz[0]) / self.dist_norm,
            (self.goal_pose['y'] - finger_xyz[1]) / self.dist_norm,
            (self.goal_pose['z'] - finger_xyz[2]) / self.dist_norm
        ]
        dist2 = [
            finger_orient[0] / np.pi,
            finger_orient[1] / np.pi,
            finger_orient[2] / np.pi
        ]
        # 获取每个 link 的最近障碍信息（7 维 × k），与 step() 保持一致
        link_dirs = self.get_link_directions()
        nearest_info = []
        num_links = len(self.link_sensors)
        for link_idx, sensors in enumerate(self.link_sensors):
            # 对最后两个 link，我需要最近3个障碍；其余 link 只要最近1个
            k = 1 if link_idx < num_links - 2 else 3
            # 存储 top-k 最近障碍：每项为 (dist, px, py, pz, sensor_handle)
            topk = []
            for s in sensors:
                out = self.sim.readProximitySensor(s)
                detected, dist = out[0], out[1]
                if detected:
                    px, py, pz = out[2]
                else:
                    dist, px, py, pz = self.sensor_range, 0.0, 0.0, 0.0
                if len(topk) < k:
                    topk.append((dist, px, py, pz, s))
                else:
                    max_i = max(range(k), key=lambda i: topk[i][0])
                    if dist < topk[max_i][0]:
                        topk[max_i] = (dist, px, py, pz, s)
            while len(topk) < k:
                topk.append((self.sensor_range, 0.0, 0.0, 0.0, None))
            topk.sort(key=lambda x: x[0])
            for dist, px, py, pz, s in topk:
                if s is not None:
                    m = self.sim.getObjectMatrix(s, -1)
                    world_point = [
                        m[3] + m[0]*px + m[1]*py + m[2]*pz,
                        m[7] + m[4]*px + m[5]*py + m[6]*pz,
                        m[11] + m[8]*px + m[9]*py + m[10]*pz
                    ]
                    link_pos = self.sim.getObjectPosition(self.links[link_idx], -1)
                    vector = [world_point[i] - link_pos[i] for i in range(3)]
                else:
                    vector = [0.0, 0.0, 0.0]
                    dist = self.sensor_range
                link_dir = link_dirs[link_idx]
                nearest_info.extend([
                    vector[0], vector[1], vector[2],
                    link_dir[0], link_dir[1], link_dir[2],
                    dist / self.dist_norm
                ])
        sensor_data = nearest_info
        # 构建初始状态（不包含 distance_Cylinder）
        s = np.concatenate((
            self.arm_info,
            np.array(finger_xyz) / self.dist_norm,
            np.array([self.goal_pose['x'], self.goal_pose['y'], self.goal_pose['z']]) / self.dist_norm,
            dist1, dist2,
            [1. if self.on_goal else 0.],
            np.array(sensor_data)
        ))
        return s