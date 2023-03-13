"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""



"""Self Play
"""

import os
import numpy as np
import magent2 as magent
import random
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_order(orientations, n):
    vel_x = [np.cos(orientations[j]) for j in range(n)]
    vel_y = [np.sin(orientations[j]) for j in range(n)]
    # [print('vel: (', vel_x[j], ', ', vel_y[j], ')') for j in range(nums[i])]
    sum_order = [np.sum(vel_x), np.sum(vel_y)]
    sum_order = np.linalg.norm(sum_order)
    order = sum_order / n
    # print('order: ', order)
    return order

def calculate_avg_num_neighbors(env, handle):
    n = env.get_num(handle)
    neighbor_agents = env.get_observation(handle)[0][:, :, :, 1]
    sum = 0
    for i in range(n):
        neighbors = neighbor_agents[i]
        neighbors_n = np.sum(neighbors) - 1    # 把自己扣除掉
        sum += neighbors_n
    avg = sum / n
    return avg


def calculate_agents_info(env, handle, acts):
    # print(acts)
    neighbor_agents = env.get_observation2(handle)[0][:, :, :, 0]  # 使用get_observation2 ?!
    orientation_neighbors = env.get_observation2(handle)[0][:, :, :, 1]

    move_angle = 2 * np.pi / 3
    move_n = 361
    step = move_angle / move_n  # 以弧度为单位

    agents_num = orientation_neighbors.shape[0]
    # print('agents_num: ', agents_num)
    # action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    vision_width = orientation_neighbors.shape[1]
    vision_height = orientation_neighbors.shape[2]
    dims = 5 + 2 * vision_width * vision_height
    agents_info = np.zeros((agents_num,dims), dtype=np.float32)
    # neighbors_mean_orientation = np.zeros((agents_num,), dtype=np.float32)
    # diffs_orientation = np.zeros((agents_num,), dtype=np.float32)
    for i in range(agents_num):
        # 将action变换为转角（弧度）
        act_angle = -0.5 * move_angle + acts[i] * step + 0.5 * step  # act对应的旋转角

        # 提取自身的orientation
        neighbors_orientation = orientation_neighbors[i]
        self_orientation = neighbors_orientation[int(neighbors_orientation.shape[0] / 2), int(neighbors_orientation.shape[1] / 2)]

        # 计算邻居的平均orientation
        neighbors_pos = neighbor_agents[i]
        neighbors_orientation[int(neighbors_orientation.shape[0] / 2), int(neighbors_orientation.shape[1] / 2)] = 0  # 把自己（中心元素）去掉
        neighbors_pos[int(neighbors_pos.shape[0] / 2), int(neighbors_pos.shape[1] / 2)] = 0
        neighbors_n = np.sum(neighbors_pos)
        if neighbors_n > 0:
            sum = np.sum(neighbors_orientation)
            mean_neighbors_orientation = sum / neighbors_n
        else:
            mean_neighbors_orientation = self_orientation

        # 计算自身orientation与邻居平均orientation的差
        diff_orientation = mean_neighbors_orientation - self_orientation
        diff_orientation = diff_orientation + 2 * np.pi if diff_orientation < -np.pi else diff_orientation
        diff_orientation = diff_orientation - 2 * np.pi if diff_orientation > np.pi else diff_orientation

        # 处理邻居pos和orientation
        pos = neighbors_pos.flatten()
        orientation = neighbors_orientation.flatten()

        agents_info[i][0] = acts[i]
        agents_info[i][1] = act_angle
        agents_info[i][2] = self_orientation
        agents_info[i][3] = mean_neighbors_orientation
        agents_info[i][4] = diff_orientation
        agents_info[i][5: 5+vision_width*vision_height] = pos
        agents_info[i][5+vision_width*vision_height:5+2*vision_width*vision_height] = orientation

        # print('agents_info1: ', neighbors_n, acts[i], act_angle, self_orientation, mean_neighbors_orientation, diff_orientation)
        # print('pos: ', neighbors_pos)
        # print('orientation: ', neighbors_orientation)
        # print('agents_info: ', agents_info[i])

    return agents_info


if __name__ == '__main__':
    map_size = 60
    env = magent.GridWorld('fishschool_alignment', map_size=map_size)  # 根据python/magent/builtin/config/batlle.py的配置文件创建env对象，里面包含2组predators和2组preys
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render_vicsek'))
    # env.set_render_dir('___debug___')
    handles = env.get_handles()
    env.reset()
    # print('YYYYYYYYYYYYY')

    # pos = [[2, 0.5*map_size, 0],
    #        [2+2, 0.5*map_size+2, 0],
    #        [map_size - 3, 0.5 * map_size, 0],
    #        [map_size - 2, 0.5 * map_size + 2, 0]
    #        ]
    # env.add_agents(handles[0], method="custom", pos=pos)  # fishes
    # pos = [[map_size - 3, 0.5 * map_size + 3, 0]]
    # env.add_agents(handles[1], method="custom", pos=pos)  # fishes
    env.add_agents(handles[0], method="random", n=140)  # fishes
    # env.add_agents(handles[1], method="random", n=1)  # predators
    # print('SSSSSSSSSSSSS')
    n_group = len(handles)
    acts = [None for _ in range(n_group)]
    prey_action_n = env.get_action_space(handles[0])[0]
    print(prey_action_n)

    nums = [env.get_num(handle) for handle in handles]
    neighbor_agents = env.get_observation2(handles[0])[0][:, :, :, 0]
    vision_width = neighbor_agents.shape[1]
    vision_height = neighbor_agents.shape[2]
    # action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    dims = 5 + 2 * vision_width * vision_height
    agents_infos = [np.zeros((500, nums[i], dims), dtype=np.float32) for i in range(n_group)]

    options = [-42, 42]

    order_params = []
    avg_neighbors_num = []
    k = 0
    round = 0
    while(k<500):
        # acts[1] = env.get_predator_actions2(handles[1])

        num = env.get_num(handles[0])
        num_alive = len(env.get_alive(handles[0]))
        #print(num, num_alive)
        # preys_acts = [random.randint(0,prey_action_n-1) for _ in range(num_alive) ]
        # preys_acts = [180 for _ in range(num_alive)]
        preys_acts = env.vicsek_policy(handles[0])
        # preys_acts = [options[random.randint(0,1)]+random.randint(-3, 3) for _ in range(num_alive)]
        acts[0] = np.array(preys_acts, dtype=np.int32)

        # agents_infos[0][k] = calculate_agents_info(env, handles[0], acts[0])

        # print(acts[0])

        # fish_view = env.get_observation2(handles[0])
        # print('fish_view:')

        # fish_view = env.get_observation(handles[0])[0]
        # predator_view = env.get_observation(handles[1])[0]
        # print('fish_view:')
        # print(fish_view[0, :, :, 1])
        # print(fish_view[0, :, :, 2])
        # print(fish_view[0, :, :, 3])
        # print('predator_view:')
        # print(predator_view[0, :, :, 1])
        # print(predator_view[0, :, :, 2+1])




        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # preys_acts = [1]
        # acts[0] = np.array(preys_acts, dtype=np.int32)
        # env.set_action(handles[0], acts[0])

        env.step()
        env.get_reward(handles[0])
        env.clear_dead()
        env.render()
        # env.render_real_time()

        # time.sleep(2)

        # # 计算alignment指标order params和邻居平均数
        # n = env.get_num(handles[0])
        # orientations = env.get_orientation(handles[0])
        # order = calculate_order(orientations, n)
        # avg_num = calculate_avg_num_neighbors(env, handles[0])
        # order_params.append(order)
        # avg_neighbors_num.append(avg_num)

        k += 1
        print(round, ', ', k)
        if k == 500:
            round += 1
            k = 0
            env.reset()
            env.add_agents(handles[0], method="random", n=140)  # fishes

    # print('Order_params: ')
    # print(order_params)
    # print('Avg_neighbors_num: ')
    # print(avg_neighbors_num)

    results_dir = os.path.join(BASE_DIR, 'examples/battle_model', 'build/render_vicsek')
    n = env.get_num(handles[0])
    file_name = '/order_{}.txt'.format(n)
    np.savetxt(results_dir + file_name, np.array(order_params), fmt="%.2f")
    file_name = '/neighbors_num_{}.txt'.format(n)
    np.savetxt(results_dir + file_name, np.array(order_params), fmt="%.2f")

    file_name = '/agent_infos_{}'.format(n)
    np.save(results_dir + file_name, agents_infos[0])  # .tofile(file_name, sep=',', format='%s')
