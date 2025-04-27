#!/usr/bin/env python3
import time, math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

if __name__ == "__main__":
    # 1. 连接 CoppeliaSim
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    sim.setStepping(True)

    # 2. 获取句柄
    sensor = sim.getObject('/proximitySensor')
    cuboid = sim.getObject('/Cuboid')

    # 3. 主循环：先让场景稳定几步
    for _ in range(10):
        sim.step()

    print("开始检测 & 移动测试……按 Ctrl+C 停止")
    try:
        while True:
            # 4. 读取传感器
            out = sim.handleProximitySensor(sensor)
            detected = out[0]
            if not detected:
                print("no hit")
            else:
                # out[2] 是 [px,py,pz]
                px, py, pz = out[2]

                # 获取世界变换矩阵
                M = sim.getObjectMatrix(sensor, -1)  # 长度 12：3×4

                # 本地点投影到世界
                vx = M[0]*px + M[1]*py + M[2]*pz
                vy = M[4]*px + M[5]*py + M[6]*pz
                vz = M[8]*px + M[9]*py + M[10]*pz
                norm = math.sqrt(vx*vx + vy*vy + vz*vz)
                world_dir = [vx/norm, vy/norm, vz/norm] if norm>1e-6 else [0.0,0.0,0.0]

                # 5. 移动 sensor 小步测试运动方向
                old_pos = sim.getObjectPosition(sensor, -1)
                new_pos = [old_pos[0] + 0.002, old_pos[1], old_pos[2]]
                sim.setObjectPosition(sensor, -1, new_pos)
                sim.step()
                # 计算运动方向
                pos2 = sim.getObjectPosition(sensor, -1)
                delta = [pos2[i] - old_pos[i] for i in range(3)]
                dnorm = math.sqrt(sum(d*d for d in delta))
                move_dir = [d/dnorm for d in delta] if dnorm>1e-6 else [0.0,0.0,0.0]

                # 6. 点积
                dot = sum(world_dir[i]*move_dir[i] for i in range(3))

                # 输出
                print(f"world_dir = {world_dir}, move_dir = {move_dir}, dot = {dot:.4f}")

            # 等待下一步
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("测试结束")
        sim.finishSimulation()
        client.close()