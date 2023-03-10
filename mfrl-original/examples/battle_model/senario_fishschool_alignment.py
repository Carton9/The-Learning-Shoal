import random
import math
import numpy as np
import sys


def generate_circle(map_size, x0, y0, r):
    pos = []
    for x in range(map_size):
        for y in range(map_size):
            dx = x - x0
            dy = y - y0
            dist = math.sqrt(dx * dx + dy * dy)
            if math.fabs(dist - r) < 1.5:
                pos.append([x, y, 0])

    return pos


def generate_agents(map_size, x0, y0, r, R, n):
    pos = []
    cnt = 0
    while cnt < n:
        x = random.randint(0, map_size - 1)
        y = random.randint(0, map_size - 1)
        dx = x - x0
        dy = y - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if r + 1 < dist and dist < R - 1:
            pos.append([x, y, 0])
            cnt += 1

    return pos


def generate_map(env, map_size, handles):
    env.add_agents(handles[0], method="random", n=50)  # fishes

    # wall_pos0 = generate_circle(map_size, int(0.5 * map_size), int(0.5 * map_size), 0.25 * map_size)
    # wall_pos1 = generate_circle(map_size, int(0.5 * map_size), int(0.5 * map_size), 0.48 * map_size)
    # print('wall_pos0:', len(wall_pos0))
    # print('wall_pos1:', len(wall_pos1))
    # env.add_agents(-1, method="custom", pos=wall_pos0)
    # env.add_agents(-1, method="custom", pos=wall_pos1)
    #
    # prey_num = 80
    # predator_num = 5
    # group_pos0 = generate_agents(map_size, int(0.5 * map_size), int(0.5 * map_size), 0.25 * map_size, 0.48 * map_size,
    #                              prey_num)
    # group_pos1 = generate_agents(map_size, int(0.5 * map_size), int(0.5 * map_size), 0.25 * map_size, 0.48 * map_size,
    #                              predator_num)
    #
    # print('group_pos0:', len(group_pos0))
    # print('group_pos1:', len(group_pos1))
    # env.add_agents(handles[0], method="custom", pos=group_pos0)
    # env.add_agents(handles[1], method="custom", pos=group_pos1)

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

def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    # generate_map(env, map_size, handles)        # 重新布阵
    env.add_agents(handles[0], method="random", n=env.agents_num)  # fishes
    # print('agents_num: ', env.agents_num)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    agent_acc_rewards = [np.zeros((nums[i],), dtype=float) for i in range(n_group)]
    order_params = [[] for _ in range(n_group)]
    avg_neighbors_num = [[] for _ in range(n_group)]

    neighbor_agents = env.get_observation2(handles[0])[0][:, :, :, 0]
    vision_width = neighbor_agents.shape[1]
    vision_height = neighbor_agents.shape[2]
    # action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    dims = 5 + 2 * vision_width * vision_height
    agents_infos = [np.zeros((max_steps, nums[i], dims), dtype=np.float32) for i in range(n_group)]

    # former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])),
    #                    np.zeros((1, env.get_action_space(handles[1])[0]))]
    # former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0]))]
    former_act_prob = [np.zeros((nums[i], env.get_action_space(handles[i])[0]), dtype=np.float32) for i in range(n_group)]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation2(handles[i]))         # 获取“活着”智能体的state（viewspace+featurespace）
            ids[i] = env.get_agent_id(handles[i])                    # 获取“活着”智能体的id

        for i in range(n_group):
            # # 将former_act_prob[i]平铺复制，确保每个agent对应一个former_act_prob
            # former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            former_act_prob[i] = env.get_neighbors_mean_act(handles[i])
            # 同一组的所有“活着”智能体的agents使用同一策略模型确定action，是编号，不是one-hot向量
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

            # analysis data
            agents_infos[i][step_ct] = calculate_agents_info(env, handles[0], acts[i])

        for i in range(n_group):
            env.set_action(handles[i], acts[i])          # 给所有活着的agents设置action

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])    # 得到这一步的立即奖励，这一步挂掉的以后就没有了，包含即将死亡的
            alives[i] = env.get_alive(handles[i])      # 这一步被kill的agent还算活着，感觉跟env.get_num函数一样呢？！

        #print('rewards: ', rewards[0])
        # 收集第一组智能体的数据并训练
        # #for i in range(n_group):
        # for i in range(n_group):
        #     buffer = {
        #         'state': state[i], 'acts': acts[i], 'rewards': rewards[i],
        #         'alives': alives[i], 'ids': ids[i]
        #     }
        #
        #     buffer['prob'] = former_act_prob[i]
        #     former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)    # 该组全体活着的智能体的平均值。先扩展成onehot形式
        #     if train:
        #         models[i].flush_buffer(**buffer)

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }
        buffer['prob'] = former_act_prob[0]         # shape=(agents_n, act_n)
        # former_act_prob[0] = np.mean(list(map(lambda x: np.eye(n_action[0])[x], acts[0])), axis=0,
        #                              keepdims=True)  # 该组全体活着的智能体的平均值。先扩展成onehot形式



        if train:
            models[0].flush_buffer(**buffer)

        # stat info
        nums = [env.get_num(handle) for handle in handles]            # 获取尚存的智能体数量，包含即将死亡的
        orientations = [env.get_orientation(handle) for handle in handles]

        # print('nums: ', nums[0])
        # for i in range(nums[0]):
        #     print(ids[0][i], ': ', rewards[0][i])

        for i in range(n_group):
            for j in range(len(rewards[i])):
                agent_id = ids[i][j] - env.get_agent_id(handles[i])[0]
                #print(i, j, agent_id)
                agent_acc_rewards[i][agent_id] += rewards[i][j]

            sum_reward = sum(rewards[i])
            if nums[i] > 0:
                rewards[i] = sum_reward / nums[i]
            else:
                rewards[i] = 0
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

            order = calculate_order(orientations[i], nums[i])
            order_params[i].append(order)
            avg_num = calculate_avg_num_neighbors(env, handles[i])
            avg_neighbors_num[i].append(avg_num)


        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        #models[1].train()



    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    # print('agent_acc_rewards: ', agent_acc_rewards)
    # print('mean_rewards: ', mean_rewards)
    # print('total_rewards: ', total_rewards)
    # print('===: ', sum(agent_acc_rewards[0]), sum(agent_acc_rewards[1]))

    return max_nums, nums, mean_rewards, total_rewards, agent_acc_rewards, order_params, avg_neighbors_num, agents_infos

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


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    # generate_map(env, map_size, handles)
    env.add_agents(handles[0], method="random", n=env.agents_num)  # fishes

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    agent_acc_rewards = [np.zeros((nums[i],), dtype=float) for i in range(n_group)]
    order_params = [[] for _ in range(n_group)]
    avg_neighbors_num = [[] for _ in range(n_group)]

    neighbor_agents = env.get_observation2(handles[0])[0][:, :, :, 0]
    vision_width = neighbor_agents.shape[1]
    vision_height = neighbor_agents.shape[2]
    # action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    dims = 5 + 2 * vision_width * vision_height
    agents_infos = [np.zeros((max_steps, nums[i], dims), dtype=np.float32) for i in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation2(handles[i]))         # 获取“活着”智能体的state（viewspace+featurespace）
            ids[i] = env.get_agent_id(handles[i])                    # 获取“活着”智能体的id

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            # 同一组的所有“活着”智能体的agents使用同一策略模型确定action，是编号，不是one-hot向量
            # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH", state[i], former_act_prob[i])
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)
            # 增加随机性
            for j in range(nums[i]):
                acts[i][j] += random.randint(-3, 3)
                if acts[i][j] < 0:
                    acts[i][j] = 0
                if acts[i][j] > n_action[0]-1:
                    acts[i][j] = n_action[0]-1

            # analysis data
            agents_infos[i][step_ct] = calculate_agents_info(env, handles[0], acts[i])

            # print(agents_infos[i][step_ct].shape)
            # print('agents_infos: ', step_ct, agents_infos[i][step_ct])


        for i in range(n_group):
            env.set_action(handles[i], acts[i])          # 给所有活着的agents设置action

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.get_num(handle) for handle in handles]
        orientations = [env.get_orientation(handle) for handle in handles]
        # print('orientations: ', orientations)

        for i in range(n_group):
            # sum_reward = sum(rewards[i])
            # rewards[i] = sum_reward / nums[i]
            # mean_rewards[i].append(rewards[i])
            # total_rewards[i].append(sum_reward)

            for j in range(len(rewards[i])):
                agent_id = ids[i][j] - env.get_agent_id(handles[i])[0]
                #print(i, j, agent_id)
                agent_acc_rewards[i][agent_id] += rewards[i][j]

            sum_reward = sum(rewards[i])
            if nums[i] > 0:
                rewards[i] = sum_reward / nums[i]
            else:
                rewards[i] = 0
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

            order = calculate_order(orientations[i], nums[i])
            order_params[i].append(order)
            avg_num = calculate_avg_num_neighbors(env, handles[i])
            avg_neighbors_num[i].append(avg_num)

        # print('order: ', order)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    # print('order_params: ', order_params[0])
    # print('avg_neighbors_num: ', avg_neighbors_num[0])

    # print(agents_infos[0].shape)
    # print('agents_infos: ', agents_infos[0])

    return max_nums, nums, mean_rewards, total_rewards, agent_acc_rewards, order_params, avg_neighbors_num, agents_infos
