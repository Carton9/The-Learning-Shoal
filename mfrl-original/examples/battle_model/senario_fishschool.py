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
    env.add_agents(handles[0], method="random", n=80)  # fishes
    env.add_agents(handles[1], method="random", n=5)  # predators

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



def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)        # 重新布阵

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
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    agent_acc_rewards = [np.zeros((nums[i],), dtype=float) for i in range(n_group)]

    # former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])),
    #                    np.zeros((1, env.get_action_space(handles[1])[0]))]
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))         # 获取“活着”智能体的state（viewspace+featurespace）
            ids[i] = env.get_agent_id(handles[i])                    # 获取“活着”智能体的id

        for i in range(n_group):
            # former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            # acts[i] = models[i].act(state=state[i], prob=former_act_prob[i],
            #                         eps=eps)  # 同一组的所有“活着”智能体的agents使用同一策略模型确定action，是编号，不是one-hot向量

            if i == 0:
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)       # 同一组的所有“活着”智能体的agents使用同一策略模型确定action，是编号，不是one-hot向量
            elif i == 1:
                acts[i] = env.get_predator_actions2(handles[i])
            else:
                print('EEEEEEERROR!')
        #print('======state: ', state[1][0].shape)
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
        buffer['prob'] = former_act_prob[0]
        former_act_prob[0] = np.mean(list(map(lambda x: np.eye(n_action[0])[x], acts[0])), axis=0,
                                     keepdims=True)  # 该组全体活着的智能体的平均值。先扩展成onehot形式

        if train:
            models[0].flush_buffer(**buffer)

        # stat info
        nums = [env.get_num(handle) for handle in handles]            # 获取尚存的智能体数量，包含即将死亡的

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

    return max_nums, nums, mean_rewards, total_rewards, agent_acc_rewards


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

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

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

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

    return max_nums, nums, mean_rewards, total_rewards
