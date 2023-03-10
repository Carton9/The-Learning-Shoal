# --coding:utf-8--
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def draw_mean_heatmap_per_step(file_name, model_name):
    # data.shape = (max_steps, agents_num, dims)，
    # 其中dims包含：action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    data = np.load(file_name)
    move_angle = 2 * np.pi / 3
    move_n = 361
    move_step = 9
    step = move_angle / move_n
    state_n = 36#int(2 * np.pi / step)
    step = 2 * np.pi / state_n
    xticklabels = ['{}'.format(int(-180+i*10)) for i in range(36+1)]
    yticklabels = ['{}'.format(int(-60+i*3)) for i in range(40+1)]

    agents_n = data.shape[1]
    # results = np.zeros((int(move_n / move_step) + 1, state_n), dtype=np.float32)
    for timestep in range(400):
        print('timestep {}'.format(timestep))
        step_record = data[timestep]
        results = np.zeros((int(move_n / move_step) + 1, state_n), dtype=np.float32)
        for i in range(agents_n):
            diff = step_record[i][4]  # diff_orientation
            state = int((diff + np.pi) / step)
            act = int(step_record[i][0])  # action(int)
            act = int(act / move_step)
            results[act][state] += 1

        plt.figure(figsize=(15, 8))
        heatmap = sns.heatmap(results, annot=True, cmap="OrRd", xticklabels=xticklabels, yticklabels = yticklabels)
        plt.xlabel("Difference between agent's orientation and mean orientation")
        plt.ylabel('Turning angle')
        plt.savefig('./figs/by_step/{}_step_{}'.format(model_name, timestep))
        plt.close()
        # plt.show()

    # plt.figure(figsize=(15, 8))
    # heatmap = sns.heatmap(results, annot=True, cmap="OrRd", xticklabels=xticklabels, yticklabels=yticklabels)
    # plt.xlabel("Difference between agent's orientation and mean orientation")
    # plt.ylabel('Turning angle')
    # plt.show()

def draw_heatmap_per_agent(file_name, model_name):
    # data.shape = (max_steps, agents_num, dims)，
    # 其中dims包含：action(int), act_angle(rad), self_orientation, mean_neighbors_orientation, diff_orientation, neighbors_pos, neighbors_orientation
    data = np.load(file_name)
    move_angle = 2 * np.pi / 3
    move_n = 361
    move_step = 9
    step = move_angle / move_n
    state_n = 36  # int(2 * np.pi / step)
    step = 2 * np.pi / state_n
    xticklabels = ['{}'.format(int(-180 + i * 10)) for i in range(36 + 1)]
    yticklabels = ['{}'.format(int(-60 + i * 3)) for i in range(40 + 1)]

    agents_n = data.shape[1]
    for i in range(agents_n):
        print('agent {}'.format(i))
        results = np.zeros((int(move_n / move_step) + 1, state_n), dtype=np.float32)
        for timestep in range(400):
            step_record = data[timestep]
            diff = step_record[i][4]  # diff_orientation
            state = int((diff + np.pi) / step)
            act = int(step_record[i][0])  # action(int)
            act = int(act / move_step)
            results[act][state] += 1

        plt.figure(figsize=(15, 8))
        heatmap = sns.heatmap(results, annot=True, cmap="OrRd", xticklabels=xticklabels, yticklabels=yticklabels)
        plt.xlabel("Difference between agent's orientation and mean orientation")
        plt.ylabel('Turning angle')
        plt.savefig('./figs/by_agent/{}_agent_{}'.format(model_name, i))
        plt.close()
        # plt.show()


# draw_mean_heatmap_per_step('/home/hill/mtmfrl/mfrl-original/examples/battle_model/build/render_vicsek/agent_infos_140.npy', 'model_vicsek')
# draw_mean_heatmap_per_step('/home/hill/mtmfrl/mfrl-original/data/tmp/mfq(fishschool_alignment)_VR5_0.025/agent_infos_90_0.npy', 'model_90')
draw_mean_heatmap_per_step('/home/hill/mtmfrl/mfrl-original/data/results/fishschool_alignment(140)_1/agent_infos_140_0.npy', 'model_140')
# draw_mean_heatmap_per_step('/home/hill/mtmfrl/mfrl-original/data/tmp/mfq(fishschool_alignment)_VR5_0.025/agent_infos_90_67.npy')
# draw_mean_heatmap_per_step('/home/hill/mtmfrl/mfrl-original/data/tmp/mfq(fishschool_alignment)_VR4_local_mean_act/agent_infos_90_69.npy')

# draw_heatmap_per_agent('/home/hill/mtmfrl/mfrl-original/examples/battle_model/build/render_vicsek/agent_infos_140.npy', 'model_vicsek')
# draw_heatmap_per_agent('/home/hill/mtmfrl/mfrl-original/data/tmp/mfq(fishschool_alignment)_VR5_0.025/agent_infos_90_0.npy', 'model_90')
# draw_heatmap_per_agent('/home/hill/mtmfrl/mfrl-original/data/results/fishschool_alignment(140)_1/agent_infos_140_0.npy', 'model_140')

# def calculate_potential_energy(pos, i, r):
#     focal_pos = pos[i]
#     pos = np.delete(pos, i, axis=0)
#     m = 2
#     l = pow(2, 2)
#     t = [[0, 0], [0, 1], [1, 0], [1, 1]]
#     n = pos.shape[0]
#     x_ks = np.zeros((n, l, m), dtype=np.float32)
#     for k in range(n):
#         for s in range(l):
#             for j in range(m):
#                 if t[s][j] == 0:
#                     x_ks[k][s][j] = pos[k][j]
#                 else:
#                     if pos[k][j] < focal_pos[j]:
#                         x_ks[k][s][j] = pos[k][j] + r
#                     else:
#                         x_ks[k][s][j] = pos[k][j] - r
#
#     # print('focal_pos: ', focal_pos)
#     # print('x_ks: ')
#     # print(x_ks)
#     potential_energy = 0
#     for k in range(n):
#         sub_sum = 0
#         for s in range(l):
#             dist = np.linalg.norm(focal_pos - x_ks[k][s])
#             sub_sum += 1.0 / dist
#         potential_energy += sub_sum
#
#     # print('potential_energy: ', potential_energy)
#
#     return potential_energy
#
#
# def calculate_uniformity(neighbors):
#     # r = 0.5 * (neighbors.shape[0] - 1)
#     r = neighbors.shape[0]
#     # 把agents占据的栅格转化为相对坐标
#     idx = np.nonzero(neighbors)
#     n = idx[0].shape[0]
#     pos = np.zeros((n, 2), dtype=np.float32)
#     for i in range(n):
#         pos[i][0] = idx[1][i] - r
#         pos[i][1] = r - idx[0][i]
#
#     # print('neighbors')
#     # print(neighbors)
#     # print('pos', pos)
#     # print('r', r)
#     uniformity = 0.0
#     for i in range(n):
#         uniformity += calculate_potential_energy(pos, i, r)
#
#     # print('uniformity: ', uniformity)
#
#     return uniformity
#

# import math
# neighbors = [[1, 0, 1, 0, 1],
#              [0, 1, 0, 1, 0],
#              [1, 0, 1, 0, 1],
#              [0, 1, 0, 1, 0],
#              [1, 0, 1, 0, 1]]
# neighbors = np.array(neighbors)
# uni = calculate_uniformity(neighbors)
# print('uni: ', uni)
# idx = np.nonzero(neighbors)
# # plt.scatter(x=idx[1], y=4-idx[0])
# # plt.xlim([-0.5, 4.5])
# # plt.ylim([-0.5, 4.5])
# content = 'uniformity: {0:.2f}'.format(uni)
# # plt.text(x=1.1, y=4, s=content, fontsize=16, color = "r", fontweight='bold')
# # plt.savefig('./data/results/figs/uniformity_5.jpg')
# x = [4, 5, 9, 13]
# y = [14, 25, 91, 195]
# y = [math.sqrt(y[i]) for i in range(4)]
# plt.plot(x, y)
# plt.show()
