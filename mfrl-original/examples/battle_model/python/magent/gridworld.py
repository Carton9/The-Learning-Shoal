"""gridworld interface"""
from __future__ import absolute_import

import ctypes
import os
import importlib

import numpy as np
import random

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment

def sigmoid(x):
    return 1/(1+np.exp(-x))


class GridWorld(Environment):
    # constant
    OBS_INDEX_VIEW = 0
    OBS_INDEX_HP   = 1

    def __init__(self, config, **kwargs):
        """
        Parameters
        ----------
        config: str or Config Object
            if config is a string, then it is a name of builtin config,
                builtin config are stored in python/magent/builtin/config
                kwargs are the arguments to the config
            if config is a Config Object, then parameters are stored in that object
        """
        Environment.__init__(self)

        # if kwargs is not None:
        #     for key, value in kwargs.items():
        #         if key == 'agents_num':
        #             self.agents_num = value
        #             print('agents_num: ', self.agents_num)

        self.agents_num = 0

        # if is str, load built in configuration
        if isinstance(config, str):
            # built-in config are stored in python/magent/builtin/config
            try:
                demo_game = importlib.import_module('magent.builtin.config.' + config)
                config = getattr(demo_game, 'get_config')(**kwargs)
            except AttributeError:
                raise BaseException('unknown built-in game "' + config + '"')

        # create new game
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"GridWorld")
        self.game = game
        #print('============game:', self.game)

        # set global configuration
        config_value_type = {
            'map_width': int, 'map_height': int,
            'food_mode': bool, 'turn_mode': bool, 'minimap_mode': bool,
            'revive_mode': bool, 'goal_mode': bool, 'pbc_mode': bool,
            'embedding_size': int,
            'render_dir': str,
            # 'agents_num': int,
        }

        self.cfg = config

        self.view_space_spec = config.view_space_spec

        for key in config.config_dict:
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_int(config.config_dict[key])))
            elif value_type is bool:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_bool(config.config_dict[key])))
            elif value_type is float:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_float(config.config_dict[key])))
            elif value_type is str:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.c_char_p(config.config_dict[key]))

        # register agent types
        for name in config.agent_type_dict:
            type_args = config.agent_type_dict[name]

            # special pre-process for view range and attack range
            for key in [x for x in type_args.keys()]:
                if key == "view_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["view_radius"] = val.radius
                    type_args["view_angle"]  = val.angle
                elif key == "attack_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["attack_radius"] = val.radius
                    type_args["attack_angle"]  = val.angle

            length = len(type_args)
            keys = (ctypes.c_char_p * length)(*[key.encode("ascii") for key in type_args.keys()])
            values = (ctypes.c_float * length)(*type_args.values())
            #print('==============name:', name)
            #print('==============keys:', keys)
            #print('==============values:', values)

            _LIB.gridworld_register_agent_type(self.game, name.encode("ascii"), length, keys, values)

        # serialize event expression, send to C++ engine
        self._serialize_event_exp(config)

        # init group handles
        self.group_handles = []
        for item in config.groups:
            handle = ctypes.c_int32()
            _LIB.gridworld_new_group(self.game, item.encode("ascii"), ctypes.byref(handle))
            self.group_handles.append(handle)

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view space, feature space, action space
        self.view_space = {}
        self.feature_space = {}
        self.action_space = {}
        buf = np.empty((3,), dtype=np.int32)
        for handle in self.group_handles:
            _LIB.env_get_info(self.game, handle, b"view_space",
                              buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.view_space[handle.value] = (buf[0], buf[1], buf[2])
            _LIB.env_get_info(self.game, handle, b"feature_space",
                                  buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.feature_space[handle.value] = (buf[0],)
            _LIB.env_get_info(self.game, handle, b"action_space",
                                  buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.action_space[handle.value] = (buf[0],)

        self.collision_counters = {}

    def reset(self):
        """reset environment"""
        _LIB.env_reset(self.game)

        self.collision_counters_init = False
        self.previous_positions = [self.get_pos(handle) for handle in self.group_handles]


    def add_walls(self, method, **kwargs):
        """add wall to environment

        Parameters
        ----------
        method: str
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_walls(method="random", n=1000)

        # add 3 walls to (1,2), (4,5) and (9, 8) in map
        >>> env.add_walls(method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        # handle = -1 for walls
        kwargs["dir"] = 0
        self.add_agents(-1, method, **kwargs)

    # ====== AGENT ======
    def new_group(self, name):
        """register a new group into environment"""
        handle = ctypes.c_int32()
        _LIB.gridworld_new_group(self.game, ctypes.c_char_p(name.encode("ascii")), ctypes.byref(handle))
        print('new_group!!!')
        return handle

    def add_agents(self, handle, method, **kwargs):
        """add agents to environment

        Parameters
        ----------
        handle: group handle
        method: str
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_agents(handle, method="random", n=1000)

        # add 3 agents to (1,2), (4,5) and (9, 8) in map
        >>> env.add_agents(handle, method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        if method == "random":
            _LIB.gridworld_add_agents(self.game, handle, int(kwargs["n"]), b"random", 0, 0, 0)
        elif method == "custom":
            n = len(kwargs["pos"])
            pos = np.array(kwargs["pos"], dtype=np.int32)
            if len(pos) <= 0:
                return
            if pos.shape[1] == 3:  # if has dir
                xs, ys, dirs = pos[:, 0], pos[:, 1], pos[:, 2]
            else:                  # if do not has dir, use zero padding
                xs, ys, dirs = pos[:, 0], pos[:, 1], np.zeros((n,), dtype=np.int32)
            # copy again, to make these arrays continuous in memory
            xs, ys, dirs = np.array(xs), np.array(ys), np.array(dirs)
            _LIB.gridworld_add_agents(self.game, handle, n, b"custom", as_int32_c_array(xs),
                                      as_int32_c_array(ys), as_int32_c_array(dirs))
        elif method == "fill":
            x, y = kwargs["pos"][0], kwargs["pos"][1]
            width, height = kwargs["size"][0], kwargs["size"][1]
            dir = kwargs.get("dir", np.zeros_like(x))
            bind = np.array([x, y, width, height, dir], dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0,  b"fill", as_int32_c_array(bind),
                                      0, 0, 0)
        elif method == "maze":
            # TODO: implement maze add
            x_start, y_start, x_end, y_end = kwargs["pos"][0], kwargs["pos"][1], kwargs["pos"][2], kwargs["pos"][3]
            thick = kwargs["pos"][4]
            bind = np.array([x_start, y_start, x_end, y_end, thick], dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0, b"maze", as_int32_c_array(bind),
                                      0, 0, 0)
        else:
            print("Unknown type of position")
            exit(-1)

        num_agents = self.get_num(handle)
        self.collision_counters[handle.value] = np.zeros((num_agents,), dtype=np.int32)

    # ====== RUN ======
    def _get_obs_buf(self, group, key, shape, dtype):
        """get buffer to receive observation from c++ engine"""
        obs_buf = self.obs_bufs[key]
        if group in obs_buf:
            ret = obs_buf[group]
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
        else:
            ret = obs_buf[group] = np.empty(shape=shape, dtype=dtype)

        return ret

    def _init_obs_buf(self):
        """init observation buffer"""
        self.obs_bufs = []
        self.obs_bufs.append({})
        self.obs_bufs.append({})

    def get_observation(self, handle):
        """ get observation of a whole group

        Parameters
        ----------
        handle : group handle

        Returns
        -------
        obs : tuple (views, features)
            views is a numpy array, whose shape is n * view_width * view_height * n_channel
            features is a numpy array, whose shape is n * feature_size
            for agent i, (views[i], features[i]) is its observation at this step
        """
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, self.OBS_INDEX_VIEW, (n,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_INDEX_HP, (n,) + feature_space, np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)

        return view_buf, feature_buf

    def get_observation2(self, handle):
        """ get observation of a whole group

        Parameters
        ----------
        handle : group handle

        Returns
        -------
        obs : tuple (views, features)
            views is a numpy array, whose shape is n * view_width * view_height * n_channel
            features is a numpy array, whose shape is n * feature_size
            for agent i, (views[i], features[i]) is its observation at this step
        """
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, self.OBS_INDEX_VIEW, (n,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_INDEX_HP, (n,) + feature_space, np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)

        channel_list = []
        if 'map' in self.view_space_spec:
            channel_list.append(0)
        if 'food_mode' in self.view_space_spec:
            print('暂时未支持！！')

        scale = 5       # pos, hp, orientation, previous_act, id
        for i in range(len(self.group_handles)):
            if 'pos' in self.view_space_spec:
                channel_list.append(i*scale + 1)
            if 'hp' in self.view_space_spec:
                channel_list.append(i*scale + 2)
            if 'orientation' in self.view_space_spec:
                channel_list.append(i*scale + 3)
            if 'prev_act' in self.view_space_spec:
                channel_list.append(i*scale + 4)

        # print('view_buf[5, :, :, 1]', view_buf[5, :, :, 1])
        # print('view_buf[5, :, :, 4]', view_buf[5, :, :, 4])
        # print('view_buf[5, :, :, 5]', view_buf[5, :, :, 5])

        view_buf = view_buf[:, :, :, channel_list]


        return view_buf, feature_buf

    def get_neighbors_mean_act(self, handle):
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, self.OBS_INDEX_VIEW, (n,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_INDEX_HP, (n,) + feature_space, np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)
        # view_buf的channel： obstacle, pos, hp, orientation, previous_act, id
        agents_in_view = view_buf[:, :, :, 1]
        agents_prev_act_in_view = view_buf[:, :, :, 4]
        n_action = self.get_action_space(handle)[0]
        agents_neighbors_mean_act = np.zeros((n, n_action), dtype=np.float32)
        for i in range(n):
            idx = np.nonzero(agents_in_view[i])
            acts = agents_prev_act_in_view[i, idx[0], idx[1]] - 1
            acts = np.trunc(acts).astype(int).tolist()

            # print(agents_prev_act_in_view[i])
            # print('idx: ', idx)
            # print('acts: ', acts)
            # v = np.eye(n_action)[acts]
            # print(v)
            mean_act = np.mean(list(map(lambda x: np.eye(n_action)[x], acts)), axis=0,
                                     keepdims=True)  # 该组全体活着的智能体的平均值。先扩展成onehot形式
            agents_neighbors_mean_act[i] = mean_act
            # print(agents_neighbors_mean_act[i])

        # print('agents_neighbors_mean_act: ')
        # print(agents_neighbors_mean_act)

        return agents_neighbors_mean_act


    def set_action(self, handle, actions):
        """ set actions for whole group

        Parameters
        ----------
        handle: group handle
        actions: numpy array
            the dtype of actions must be int32
        """
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.int32
        _LIB.env_set_action(self.game, handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    def step(self):
        """simulation one step after set actions

        Returns
        -------
        done: bool
            whether the game is done
        """
        self.previous_positions = self.get_pos(self.group_handles[0])
        self.previous_orientations = self.get_orientation(self.group_handles[0])
        done = ctypes.c_int32()
        _LIB.env_step(self.game, ctypes.byref(done))
        return bool(done)

    def set_agent_rand_pos(self, handle, idx):
        res = _LIB.env_set_agent_rand_pos(self.game, handle, idx)
        return int(res)

    def get_rotating_reward(self, handle):
        eps = np.finfo(np.float32).eps
        positions = self.get_pos(handle)
        previous_center_of_mass = np.mean(self.previous_positions[0], axis=0)
        # print('previous_center_of_mass', previous_center_of_mass)
        previous_position_vector_center_of_mass = np.array(
            self.previous_positions[:, :2] - previous_center_of_mass) + eps
        position_vector_center_of_mass = np.array(positions[:, :2] - previous_center_of_mass) + eps
        y = np.cross(previous_position_vector_center_of_mass, position_vector_center_of_mass)
        x = np.sum(previous_position_vector_center_of_mass * position_vector_center_of_mass, axis=1)
        angular_displacement = np.arctan2(y, x)
        # print('angular_displacement', angular_displacement)

        self.activate_symetry = True
        self.alpha_angular = 2
        self.beta_angular = 40
        a_temp = np.arange(0, 0.2, 0.001)
        self.scalling_angular_reward = 1 / np.max(
            a_temp ** self.alpha_angular * (1 - a_temp) ** (self.beta_angular - 1))
        if self.activate_symetry:
            n_side1 = np.sum(angular_displacement >= 0)
            n_side2 = np.sum(angular_displacement <= 0)
            f1 = np.clip(angular_displacement[angular_displacement >= 0], 0, 1)
            f2 = np.clip(-angular_displacement[angular_displacement <= 0], 0, 1)
            b_f1 = np.sum(
                self.scalling_angular_reward * (f1 ** self.alpha_angular * (1 - f1) ** (self.beta_angular - 1)))
            b_f2 = np.sum(
                self.scalling_angular_reward * (f2 ** self.alpha_angular * (1 - f2) ** (self.beta_angular - 1)))
            return np.abs(b_f1 - b_f2) / (n_side1 + n_side2)
        else:
            angular_d_cliped = np.clip(np.mean(-angular_displacement), 0, 1)
            return self.scalling_angular_reward * (
                        angular_d_cliped ** self.alpha_angular * (1 - angular_d_cliped) ** (self.beta_angular - 1))

    def get_attraction_reward(self, handle):
        positions = self.get_pos(handle)
        center_of_mass = np.mean(positions, axis=0)
        dist_center_xy = np.linalg.norm(positions[:, :2] - center_of_mass[:2], axis=1)

        r_xy = np.mean(50 * sigmoid(10 * (dist_center_xy - 25)))

        cost_being_in_center = np.mean(np.heaviside(dist_center_xy - 10, 0.5) - 1)

        # print('r_xy: ', r_xy)
        # print('cost_being_center: ', cost_being_in_center)
        return -np.minimum((r_xy - 2 * cost_being_in_center), 50)

    def get_collision_reward(self, handle):
        n = self.get_num(handle)
        neighbor_agents = self.get_observation(handle)[0][:, :, :, 1]
        too_close = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            neighbors = neighbor_agents[i]
            rows = neighbors.shape[0]
            cols = neighbors.shape[1]
            core_neighbors = neighbor_agents[i][int(0.5 * (rows - 1)) - 1:int(0.5 * (rows - 1)) + 2,
                             int(0.5 * (cols - 1)) - 1:int(0.5 * (cols - 1)) + 2]
            core_neighbors_n = np.sum(core_neighbors) - 1
            too_close[i] = 1 if core_neighbors_n > 0 else 0

        penalty_for_too_close = -2
        col_reward = np.mean(too_close) * penalty_for_too_close
        return col_reward

    def get_group_reward(self, handle):
        rotatation_reward = self.get_rotating_reward(handle)
        attraction_reward = self.get_attraction_reward(handle)
        collision_reward = self.get_collision_reward(handle)
        reward_weights = []
        reward_weights.append(1.5)
        reward_weights.append(1.0)
        reward_weights.append(1.0)
        # print('rot, att, col')
        # print(rotatation_reward, attraction_reward, collision_reward)
        group_reward = np.mean(
            reward_weights[0] * rotatation_reward + reward_weights[1] * attraction_reward + reward_weights[
                2] * collision_reward) / (np.sum(10 * reward_weights))

        # print('group_reward: ', group_reward)


        return group_reward

    def calculate_potential_energy(self, pos, i, r):
        focal_pos = pos[i]
        pos = np.delete(pos, i, axis=0)
        m = 2
        l = pow(2, 2)
        t = [[0, 0], [0, 1], [1, 0], [1, 1]]
        n = pos.shape[0]
        x_ks = np.zeros((n, l, m), dtype=np.float32)
        for k in range(n):
            for s in range(l):
                for j in range(m):
                    if t[s][j] == 0:
                        x_ks[k][s][j] = pos[k][j]
                    else:
                        if pos[k][j] < focal_pos[j]:
                            x_ks[k][s][j] = pos[k][j] + r
                        else:
                            x_ks[k][s][j] = pos[k][j] - r

        # print('focal_pos: ', focal_pos)
        # print('x_ks: ')
        # print(x_ks)
        potential_energy = 0
        for k in range(n):
            sub_sum = 0
            for s in range(l):
                dist = np.linalg.norm(focal_pos - x_ks[k][s])
                sub_sum += 1.0 / dist
            potential_energy += sub_sum

        # print('potential_energy: ', potential_energy)

        return potential_energy


    def calculate_uniformity(self, neighbors):
        # r = 0.5 * (neighbors.shape[0] - 1)
        r = neighbors.shape[0]
        # 把agents占据的栅格转化为相对坐标
        idx = np.nonzero(neighbors)
        n = idx[0].shape[0]
        pos = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            pos[i][0] = idx[1][i] - r
            pos[i][1] = r - idx[0][i]

        # print('neighbors')
        # print(neighbors)
        # print('pos', pos)
        # print('r', r)
        uniformity = 0.0
        for i in range(n):
            uniformity += self.calculate_potential_energy(pos, i, r)

        uniformity = math.sqrt(uniformity)
        # print('uniformity: ', uniformity)


        return uniformity


    def get_reward(self, handle):
        """ get reward for a whole group

        Returns
        -------
        rewards: numpy array (float32)
            reward for all the agents in the group
        """
        # print('self.collision_counters', self.collision_counters[handle.value])
        n = self.get_num(handle)
        rewards = np.zeros((n,), dtype=np.float32)
        # print(rewards)
        _LIB.env_get_reward(self.game, handle,
                            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # Yuanshan Lin
        # additional reward
        neighbor_agents = self.get_observation(handle)[0][:, :, :, 1]
        neighbor_orientations = self.get_observation(handle)[0][:, :, :, 3]
        # agents_feature = self.get_observation(handle)[1]
        agents_last_event_op = self.get_last_event_op(handle)
        # agents_id = self.get_agent_id(handle)
        # print(agents_last_event_op)

        # group_reward = self.get_group_reward(handle)

        for i in range(n):
            # print('agent_id: ', agents_id[i])
            last_op = agents_last_event_op[i]
            collision = (last_op == 6 or last_op == 7)
            if last_op == 6:          # 6=碰撞到其他agent
                self.collision_counters[handle.value][i] += 1
                rewards[i] += self.collision_counters[handle.value][i] * (-0.1)
                if self.collision_counters[handle.value][i] > 10:
                    self.set_agent_rand_pos(handle, i)
                    self.collision_counters[handle.value][i] = 0
            elif last_op == 7:        # 7=碰撞到墙
                self.collision_counters[handle.value][i] += 1
                rewards[i] += self.collision_counters[handle.value][i] * (-0.2)
                if self.collision_counters[handle.value][i] > 10:
                    self.set_agent_rand_pos(handle, i)
                    self.collision_counters[handle.value][i] = 0
            else:
                self.collision_counters[handle.value][i] = 0

            neighbors = neighbor_agents[i]
            uniformity = self.calculate_uniformity(neighbors)
            rewards[i] += -0.05 * uniformity
            rows = neighbors.shape[0]
            cols = neighbors.shape[1]
            core_neighbors = neighbor_agents[i][int(0.5*(rows-1))-1:int(0.5*(rows-1))+2, int(0.5*(cols-1))-1:int(0.5*(cols-1))+2]
            # print(core_neighbors)
            # print(neighbors)
            neighbors[int(neighbors.shape[0] / 2), int(neighbors.shape[1] / 2)] = 0  # 把自己（中心元素）去掉
            neighbors_theta = neighbor_orientations[i]
            agent_theta = neighbors_theta[int(neighbors.shape[0] / 2), int(neighbors.shape[1] / 2)]
            neighbors_theta[int(neighbors.shape[0] / 2), int(neighbors.shape[1] / 2)] = 0  # 把自己（中心元素）去掉
            neighbors_n = np.sum(neighbors)
            core_neighbors_n = np.sum(core_neighbors)-1
            if neighbors_n > 0:
                sum = np.sum(neighbors_theta)
                avg_neighbors_theta = sum / neighbors_n
                diff = avg_neighbors_theta - agent_theta

                rewards[i] += 0.1 * neighbors_n #- 0.3 * core_neighbors_n    # 可以用diff构造reward
            else:
                rewards[i] += -1

            # print('private reward: ', rewards[i])
            # print('group_reward: ', group_reward)
            # rewards[i] += group_reward
            # rewards[i] = group_reward


        # print('rewards: ', rewards)
        return rewards

    def clear_dead(self):
        """ clear dead agents in the engine
        must be called after step()
        """
        _LIB.gridworld_clear_dead(self.game)

    # ====== INFO ======
    def get_handles(self):
        """ get all group handles in the environment """
        return self.group_handles

    def get_num(self, handle):
        """ get the number of agents in a group"""
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b'num', ctypes.byref(num))
        return num.value

    def get_action_space(self, handle):
        """get action space

        Returns
        -------
        action_space : tuple
        """
        return self.action_space[handle.value]

    def get_view_space(self, handle):
        """get view space

        Returns
        -------
        view_space : tuple
        """
        return self.view_space[handle.value]

    def get_view_space2(self, handle):
        """get view space

        Returns
        -------
        view_space : tuple
        """
        view_space = self.view_space[handle.value]

        n = len(self.view_space_spec)
        if n == 0:           # 不指定使用哪些通道的数据
            return view_space
        else:                # 指定了使用特定通道的数据
            base = 0
            if 'map' in self.view_space_spec:
                base += 1
            if 'food_mode' in self.view_space_spec:
                base += 1
            n = n - base
            n_groups = len(self.group_handles)
            view_space = (view_space[0], view_space[1], base + n * n_groups)
            print('view_space: ', view_space)
            return view_space

    def get_feature_space(self, handle):
        """ get feature space

        Returns
        -------
        feature_space : tuple
        """
        return self.feature_space[handle.value]

    def get_agent_id(self, handle):
        """ get agent id

        Returns
        -------
        ids : numpy array (int32)
            id of all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"id",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_orientation(self, handle):
        """ get agent orientation

        Returns
        -------
        ids : numpy array (float)
            id of all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.float32)
        _LIB.env_get_info(self.game, handle, b"orientation",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def get_real_pos(self, handle):
        """ get real position of agents in a group

        Returns
        -------
        pos: numpy array (float)
            the shape of pos is (n, 2)
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.float32)
        _LIB.env_get_info(self.game, handle, b"real_pos",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def get_last_event_op(self, handle):
        """ get last event operation of agents in a group

        Returns
        -------
        pos: numpy array (int)
            the shape of last event op is (n, 1)
        """
        n = self.get_num(handle)
        buf = np.empty((n, ), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"last_event_op",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_alive(self, handle):
        """ get alive status of agents in a group

        Returns
        -------
        alives: numpy array (bool)
            whether the agents are alive
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.bool)
        _LIB.env_get_info(self.game, handle, b"alive",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))
        return buf

    def get_pos(self, handle):
        """ get position of agents in a group

        Returns
        -------
        pos: numpy array (int)
            the shape of pos is (n, 2)
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"pos",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_mean_info(self, handle):
        """ deprecated """
        buf = np.empty(2 + self.action_space[handle.value][0], dtype=np.float32)
        _LIB.env_get_info(self.game, handle, b"mean_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def get_view2attack(self, handle):
        """ get a matrix with the same size of view_range,
            if element >= 0, then it means it is a attackable point, and the corresponding
                                    action number is the value of that element
        Returns
        -------
        attack_back: int
        buf: numpy array
            map attack action into view
        """
        size = self.get_view_space(handle)[0:2]
        buf = np.empty(size, dtype=np.int32)
        attack_base = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b"view2attack",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        _LIB.env_get_info(self.game, handle, b"attack_base",
                          ctypes.byref(attack_base))
        return attack_base.value, buf

    def get_global_minimap(self, height, width):
        """ compress global map into a minimap of given size
        Parameters
        ----------
        height: int
            the height of minimap
        width:  int
            the width of minimap

        Returns
        -------
        minimap : numpy array
            the shape (n_group + 1, height, width)
        """
        buf = np.empty((height, width, len(self.group_handles)), dtype=np.float32)
        buf[0, 0, 0] = height
        buf[0, 0, 1] = width
        _LIB.env_get_info(self.game, -1, b"global_minimap",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def set_seed(self, seed):
        """ set random seed of the engine"""
        _LIB.env_config_game(self.game, b"seed", ctypes.byref(ctypes.c_int(seed)))

    # ====== RENDER ======
    def set_render_dir(self, name):
        """ set directory to save render file"""
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name.encode("ascii"))

    def render(self):
        """ render a step """
        _LIB.env_render(self.game)

    def _get_groups_info(self):
        """ private method, for interactive application"""
        n = len(self.group_handles)
        buf = np.empty((n, 5), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"groups_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def _get_walls_info(self):
        """ private method, for interactive application"""
        n = 100 * 100
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"walls_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        n = buf[0, 0]  # the first line is the number of walls
        return buf[1:1+n]

    def _get_render_info(self, x_range, y_range):
        """ private method, for interactive application"""
        n = 0
        for handle in self.group_handles:
            n += self.get_num(handle)

        buf = np.empty((n+1, 4), dtype=np.int32)
        buf[0] = x_range[0], y_range[0], x_range[1], y_range[1]
        _LIB.env_get_info(self.game, -1, b"render_window_info",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))

        # the first line is for the number of agents in the window range
        info_line = buf[0]
        agent_ct, attack_event_ct = info_line[0], info_line[1]
        buf = buf[1:1 + info_line[0]]

        agent_info = {}
        for item in buf:
            agent_info[item[0]] = [item[1], item[2], item[3]]

        buf = np.empty((attack_event_ct, 3), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"attack_event",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))
        attack_event = buf

        return agent_info, attack_event

    def __del__(self):
        _LIB.env_delete_game(self.game)

    # ====== SPECIAL RULE ======
    def set_goal(self, handle, method, *args, **kwargs):
        """ deprecated """
        if method == "random":
            _LIB.gridworld_set_goal(self.game, handle, b"random", 0, 0)
        else:
            raise NotImplementedError

    # ====== PRIVATE ======
    def _serialize_event_exp(self, config):
        """serialize event expression and sent them to game engine"""
        game = self.game

        # collect agent symbol
        symbol2int = {}
        config.symbol_ct = 0

        def collect_agent_symbol(node, config):
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_agent_symbol(item, config)
                elif isinstance(item, AgentSymbol):
                    if item not in symbol2int:
                        symbol2int[item] = config.symbol_ct
                        config.symbol_ct += 1

        for rule in config.reward_rules:
            on = rule[0]
            receiver = rule[1]
            for symbol in receiver:
                if symbol not in symbol2int:
                    symbol2int[symbol] = config.symbol_ct
                    config.symbol_ct += 1
            collect_agent_symbol(on, config)

        # collect event node
        event2int = {}
        config.node_ct = 0

        def collect_event_node(node, config):
            if node not in event2int:
                event2int[node] = config.node_ct
                config.node_ct += 1
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_event_node(item, config)

        for rule in config.reward_rules:
            collect_event_node(rule[0], config)

        # send to C++ engine
        for sym in symbol2int:
            no = symbol2int[sym]
            _LIB.gridworld_define_agent_symbol(game, no, sym.group, sym.index)

        for event in event2int:
            no = event2int[event]
            inputs = np.zeros_like(event.inputs, dtype=np.int32)
            for i, item in enumerate(event.inputs):
                if isinstance(item, EventNode):
                    inputs[i] = event2int[item]
                elif isinstance(item, AgentSymbol):
                    inputs[i] = symbol2int[item]
                else:
                    inputs[i] = item
            n_inputs = len(inputs)
            _LIB.gridworld_define_event_node(game, no, event.op, as_int32_c_array(inputs), n_inputs)

        for rule in config.reward_rules:
            # rule = [on, receiver, value, terminal]
            on = event2int[rule[0]]

            receiver = np.zeros_like(rule[1], dtype=np.int32)
            for i, item in enumerate(rule[1]):
                receiver[i] = symbol2int[item]
            if len(rule[2]) == 1 and rule[2][0] == 'auto':
                value = np.zeros(receiver, dtype=np.float32)
            else:
                value = np.array(rule[2], dtype=np.float32)
            n_receiver = len(receiver)
            _LIB.gridworld_add_reward_rule(game, on, as_int32_c_array(receiver),
                                           as_float_c_array(value), n_receiver, rule[3])


    def vicsek_policy(self, group_handle):

        # import cv2
        # scale = 100
        # width = scale * 9
        # height = scale * 9
        # img = np.empty((width, height, 3), dtype=np.uint8)
        # img[...] = 255

        move_angle = 2*np.pi/3
        move_n = 361

        neighbor_agents = self.get_observation(group_handle)[0][:, :, :, 1]        # 使用get_observation2 ?!
        neighbor_orientations = self.get_observation(group_handle)[0][:, :, :, 3]

        agents_n = len(neighbor_agents)
        # print('agents_n: ', agents_n)
        agents_act = np.zeros((agents_n), dtype=np.int32)
        for i in range(agents_n):
            neighbors = neighbor_agents[i]
            neighbors[int(neighbors.shape[0]/2), int(neighbors.shape[1]/2)] = 0      # 把自己（中心元素）去掉
            neighbors_theta = neighbor_orientations[i]
            agent_theta = neighbors_theta[int(neighbors.shape[0]/2), int(neighbors.shape[1]/2)]
            neighbors_theta[int(neighbors.shape[0]/2), int(neighbors.shape[1]/2)] = 0      # 把自己（中心元素）去掉
            # print(neighbors_theta)
            neighbors_n = int(np.sum(neighbors))
            neighbors_idx = np.nonzero(neighbors)
            # print(neighbors_n, '  ', len(neighbors_idx[0]))

            # img[...] = 255
            if neighbors_n > 0:
                # sum = np.sum(neighbors_theta)
                # avg_neighbors_theta = sum / neighbors_n

                v_sum = [0, 0]
                for j in range(neighbors_n):
                    v_sum[0] += math.cos(neighbors_theta[neighbors_idx[0][j], neighbors_idx[1][j]])
                    v_sum[1] += math.sin(neighbors_theta[neighbors_idx[0][j], neighbors_idx[1][j]])


                    # pos = np.array([neighbors_idx[0][j], neighbors_idx[1][j]])
                    # cv2.circle(img, tuple((pos * scale).astype(int)), int(0.5 * scale), (0, 0, 255),
                    #            lineType=cv2.LINE_AA)
                    # displace = [math.cos(neighbors_theta[neighbors_idx[0][j], neighbors_idx[1][j]]), math.sin(neighbors_theta[neighbors_idx[0][j], neighbors_idx[1][j]])]
                    # cv2.line(img,
                    #          tuple((pos * scale).astype(int)),
                    #          tuple(((pos + displace) * scale).astype(int)), (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    # cv2.putText(img, "{:.2f}".format(neighbors_theta[neighbors_idx[0][j], neighbors_idx[1][j]]), tuple(((pos + displace) * scale).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                v_avg = v_sum / np.linalg.norm(v_sum)
                avg_neighbors_theta = math.atan2(v_avg[1], v_avg[0])

                # print('avg_or:', avg_neighbors_theta)
                # 构造该agent的action，以avg_theta为目标朝向
                diff = avg_neighbors_theta - agent_theta
                noise = np.random.uniform(-1*np.pi/180, 1*np.pi/180)
                diff += noise
                diff = diff
                diff = diff + 2*np.pi if diff < -np.pi else diff
                diff = diff - 2*np.pi if diff > np.pi else diff
                if diff < -0.5 * move_angle:  # 超过左边最大范围
                    agents_act[i] = 0
                elif diff > 0.5 * move_angle:  # 超过右边最大范围
                    agents_act[i] = move_n - 1
                else:  # 正好在运动范围之内
                    step = move_angle / move_n
                    agents_act[i] = int((diff + 0.5 * move_angle) / step)

                # pos = np.array([4, 4])
                # cv2.circle(img, tuple((pos * scale).astype(int)), int(0.5 * scale), (0, 255, 0), lineType=cv2.LINE_AA)
                # dir = agent_theta + (-0.5 * move_angle + agents_act[i] * move_angle / move_n)
                # dir = dir + 2 * np.pi if dir < -np.pi else dir
                # dir = dir - 2 * np.pi if dir > np.pi else dir
                # offset = [math.cos(dir), math.sin(dir)]
                # cv2.line(img,
                #          tuple((pos * scale).astype(int)),
                #          tuple(((pos + offset) * scale).astype(int)), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                # cv2.putText(img, "{:.2f}".format(dir), tuple(((pos + offset) * scale).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                # offset0 = [math.cos(agent_theta), math.sin(agent_theta)]
                # cv2.line(img,
                #          tuple((pos * scale).astype(int)),
                #          tuple(((pos + offset0) * scale).astype(int)), (255, 0, 0), 1, lineType=cv2.LINE_AA)
                # cv2.putText(img, "{:.2f}".format(agent_theta), tuple(((pos + offset0) * scale).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

                # print('agents_act:', agents_act[i])
            else:
                agents_act[i] = random.randint(0, move_n - 1)  # 只随机选择“移动”的动作

            # opacity = 0.4
            # rate = 60
            # # bg = np.ones((width, height, 3), dtype=np.uint8) * 255
            # # cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
            # cv2.imshow('{}.jpg'.format(i), img)
            # cv2.imwrite('./figs/vicsek/{}.jpg'.format(i), img)
            # cv2.waitKey(rate)
            # print('i = {}'.format(i))
            # import time
            # time.sleep(2)

        return agents_act

    def render_real_time(self, rate=10, mode='human'):
        import cv2
        scale = 15
        # print(self.cfg.config_dict['map_width'])
        width = scale * self.cfg.config_dict['map_width']
        height = scale * self.cfg.config_dict['map_height']
        img = np.empty((width, height, 3), dtype=np.uint8)
        img[...] = 255

        # agents
        for handle in self.group_handles:
            agents_pos = self.get_real_pos(handle)
            agents_dir = self.get_orientation(handle)

            agents_num = len(agents_pos)
            # print(agents_pos)
            for i in range(agents_num):
                pos = agents_pos[i]
                cv2.circle(img, tuple((pos * scale).astype(int)), int(0.5*scale), (0,0,255), lineType=cv2.LINE_AA)
                # cv2.rectangle(img, tuple(((pos - [0.5, 0.5]) * scale).astype(int)), tuple(((pos + [0.5, 0.5]) * scale).astype(int)), (0, 0, 255))
                offset = [math.cos(agents_dir[i]), math.sin(agents_dir[i])]
                cv2.line(img,
                         tuple((pos * scale).astype(int)),
                         tuple(((pos + offset) * scale).astype(int)), (0,0,255), 1, lineType=cv2.LINE_AA)

        opacity = 0.4
        bg = np.ones((width, height, 3), dtype=np.uint8) * 255
        cv2.addWeighted(bg, opacity, img, 1 - opacity, 0, img)
        cv2.imshow('Fish School', img)
        cv2.waitKey(rate)
        return np.asarray(img)[..., ::-1]



    def get_predator_actions2(self, predator_group_handle):    # 攻击行为好像不好使！！！！！
        view_radius = 9
        speed = 2
        move_n = 121
        move_angle = 2*np.pi
        attack_radius = 2
        agent_size = 2
        turn_mode = 0
        parity = agent_size % 2
        turn_base = move_n
        attack_base = turn_base + 2 if turn_mode else turn_base

        # 获取所有preys的实数值位置
        preys_real_pos = self.get_real_pos(self.group_handles[0])
        predators_real_pos = self.get_real_pos(predator_group_handle)
        predators_real_dir = self.get_orientation(predator_group_handle)
        predator_obs_views = self.get_observation(predator_group_handle)[0]
        view_range = MyCircleRange(view_radius, 0, parity)
        attack_range = MyCircleRange(attack_radius, agent_size / 2.0, parity)
        # print(preys_real_pos, predators_real_pos)
        preys_num = len(preys_real_pos)
        pred_num = len(predators_real_pos)
        predator_acts = np.zeros((pred_num), dtype=np.int32)

        for i in range(pred_num):
            # print('i: ', i)
            # 1. 确定当前predator视野范围内的preys分布, spatial view通道的排序规则为：主体所在组，接着按创建的顺序排。所以preys在第1组
            preys_in_view = predator_obs_views[i, :, :, 3 * 1 + 1]

            preys_in_view = preys_in_view * view_range.is_in_range  # 确定出视野圈内的preys分布
            # print('predator ', predator)
            # [print(predator_obs_views[predator, :, :, i]) for i in range(9)]
            # 2. 确定当前predator攻击范围内的preys分布
            offset = int(0.5 * (preys_in_view.shape[0] - attack_range.get_width()))
            # 确定攻击矩形区内的preys分布，暂时未考虑offset不为整数的情况！！！！！
            preys_in_attack = preys_in_view[offset: preys_in_view.shape[0] - offset,
                              offset: preys_in_view.shape[0] - offset]
            preys_in_attack = preys_in_attack * attack_range.is_in_range  # 攻击圈内的prey分布

            # 3. 如果在攻击范围内有prey，则随机选择一个prey进行攻击，构建攻击动作
            idx = np.nonzero(preys_in_attack)
            n = len(idx[0])
            # print('n: ', n)
            if n > 0:  # 在攻击矩形区有prey
                j = random.randint(0, len(idx[0]) - 1)
                prey_pos = [int(idx[1][j] - attack_radius), int(idx[0][j] - attack_radius)]
                act = int(attack_range.delta2num(prey_pos))

                # # 调试用
                # print(preys_in_attack)
                # print('prey pos: ', i, prey_pos)
                # print('Attack act: ', act)


                # print('n: ', n, '  act: ', act)

                if act != -1:
                    predator_acts[i] = attack_base + act
                    continue


            preys_id_in_view = []
            # 找出在视野范围内的preys
            for j in range(preys_num):
                dx = preys_real_pos[j][0] - predators_real_pos[i][0]
                dy = preys_real_pos[j][1] - predators_real_pos[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= view_radius:
                    preys_id_in_view.append(j)

            # 随机挑选一个prey进行追逐
            n = len(preys_id_in_view)
            if n >= 1:
                target_prey = random.randint(0, n - 1)
                # 构造action
                dx = preys_real_pos[target_prey][0] - predators_real_pos[i][0]
                dy = preys_real_pos[target_prey][1] - predators_real_pos[i][1]
                angle = math.atan2(dy, dx)
                diff = angle - predators_real_dir[i]
                diff = diff + 2*np.pi if diff < -np.pi else diff
                diff = diff - 2*np.pi if diff > np.pi else diff
                # print(i, preys_real_pos[target_prey], predators_real_dir[i], angle * 180.0 / math.pi, diff)
                if diff < -0.5 * move_angle:  # 超过左边最大范围
                    predator_acts[i] = 0
                    # print(predator_acts[i])
                    continue
                elif diff > 0.5 * move_angle:  # 超过右边最大范围
                    predator_acts[i] = move_n - 1
                    # print(predator_acts[i])
                    continue
                else:  # 正好在运动范围之内
                    step = move_angle / move_n
                    predator_acts[i] = int((diff + 0.5 * move_angle) / step)
                    # print(predator_acts[i])
                    continue

            # 否则随机选择一个移动的动作
            rnd = random.randint(0, move_n - 1)  # 只随机选择“移动”的动作
            predator_acts[i] = rnd


        return predator_acts


    def get_predator_actions(self, predator_group_handle):
        view_radius = 8
        speed = 2
        move_n = 11
        move_angle = 110*np.pi/180
        attack_radius = 2
        agent_size = 2
        turn_mode = 0
        parity = agent_size % 2
        view_range = MyCircleRange(view_radius, 0, parity)
        # move_range = MyCircleRange(speed, 0, 1)
        attack_range = MyCircleRange(attack_radius, agent_size / 2.0, parity)
        move_base = 0
        # turn_base = move_base + move_range.get_count()
        turn_base = move_base + move_n
        attack_base = turn_base + 2 if turn_mode else turn_base

        # predator_ids = self.get_agent_id(predator_group_handle)
        predator_obs_views = self.get_observation(predator_group_handle)[0]
        predator_orientations = self.get_orientation(predator_group_handle)
        # print(predator_orientations)
        #print(self.get_pos(predator_group_handle))

        # print('view_width:', view_range.get_width())
        # print('move_num:', move_range.get_width())
        # for i in range(len(attack_range.dxy)):
        #     print(i, attack_range.dxy[i])
        # print('attack_num:', attack_range.get_width())
        # print('action_num:', self.get_action_space(predator_group_handle)[0])
        #print('Group ', predator_group_handle)

        predator_acts = np.zeros((len(predator_obs_views)), dtype=np.int32)
        for predator, obs_view in enumerate(predator_obs_views):
            # 1. 确定当前predator视野范围内的preys分布, spatial view通道的排序规则为：主体所在组，接着按创建的顺序排。所以preys在第1组
            preys_in_view = predator_obs_views[predator, :, :, 2 * 1 + 1]
            preys_in_view = preys_in_view * view_range.is_in_range    # 确定出视野圈内的preys分布
            # print('predator ', predator)
            # [print(predator_obs_views[predator, :, :, i]) for i in range(9)]
            # 2. 确定当前predator攻击范围内的preys分布
            offset = int(0.5 * (preys_in_view.shape[0] - attack_range.get_width()))
            # 确定攻击矩形区内的preys分布，暂时未考虑offset不为整数的情况！！！！！
            preys_in_attack = preys_in_view[offset : preys_in_view.shape[0]-offset, offset : preys_in_view.shape[0]-offset]
            preys_in_attack = preys_in_attack * attack_range.is_in_range  # 攻击圈内的prey分布

            # 3. 如果在攻击范围内有prey，则随机选择一个prey进行攻击，构建攻击动作
            idx = np.nonzero(preys_in_attack)
            n = len(idx[0])
            if n > 0:        # 在攻击矩形区有prey
                i = random.randint(0, len(idx[0])-1)
                prey_pos = [int(idx[1][i] - attack_radius), int(idx[0][i] - attack_radius)]
                act = int(attack_range.delta2num(prey_pos))

                # # 调试用
                # print(preys_in_attack)
                # print('prey pos: ', i, prey_pos)
                # print('Attack act: ', act)

                if act != -1:
                    predator_acts[predator] = attack_base + act
                    continue

            # 4. 如果在视野范围之内但在攻击范围之外，则随机选择一个prey作为追逐对象，构建移动动作
            idx = np.nonzero(preys_in_view)
            print(preys_in_view)
            print(idx)
            n = len(idx[0])
            if n > 0:
                # # 将视野圈内的prays分布投影到move_range圈中，以便构造出合适的移动动作
                # scale = 1.0 * move_range.get_width() / predator_obs_views.shape[1]
                # idx_scale0 = idx[0] * scale #+ 0.5
                # idx_scale1 = idx[1] * scale #+ 0.5
                # # print(idx)
                # # print(idx_scale0, idx_scale1)
                # # print(idx_scale0.astype(int), idx_scale1.astype(int))
                # preys_in_view_proj = np.zeros((move_range.get_width(), move_range.get_width()))
                # preys_in_view_proj[idx_scale0.astype(int), idx_scale1.astype(int)] = 1
                # preys_in_view_proj = preys_in_view_proj * move_range.is_in_range   #  move_range圈中的
                # # print(idx)
                # # print(idx_scale0.astype(int), idx_scale1.astype(int))
                # # print(preys_in_view_proj)
                #
                # # 将选中的prey的位置映射到move_range中
                # idx = np.nonzero(preys_in_view_proj)
                # m = len(idx[0])
                # if m > 0:
                #     i = random.randint(0, m - 1)
                #     prey_pos = [int(idx[1][i] - speed), int(idx[0][i] - speed)]
                #     act = move_range.delta2num(prey_pos)
                #     # print('prey pos: ', i, prey_pos)
                #     # print('Move act: ', act)
                #     if act != -1:
                #         predator_acts[predator] = move_base + act
                #         continue

                predator_orientation = predator_orientations[predator]
                i = random.randint(0, n - 1)
                # 获取prey的位置
                prey_pos = [int(idx[1][i] - view_radius), int(idx[0][i] - view_radius)]
                predator_pos = [0, 0]
                angle = math.atan2(prey_pos[1], prey_pos[0])
                diff = predator_orientation - angle * np.pi
                diff = diff + 2*np.pi if diff < -np.pi else diff
                diff = diff - 2*np.pi if diff > np.pi else diff
                print(i, prey_pos, predator_orientation, angle*np.pi, diff)
                if diff < -0.5 * move_angle:  # 超过左边最大范围
                    predator_acts[predator] = move_base + 0
                    print(predator_acts[predator])
                    continue
                elif diff > 0.5 * move_angle: # 超过右边最大范围
                    predator_acts[predator] = move_base + move_n - 1
                    print(predator_acts[predator])
                    continue
                else:   # 正好在运动范围之内
                    step = move_angle / move_n
                    predator_acts[predator] = int((diff + 0.5*move_angle) / step)
                    print(predator_acts[predator])
                    continue



            # 5. 如果视野范围之内没有prey，则随机选择动作
            # i = random.randint(0, self.get_action_space(predator_group_handle)[0] - 1)         # 随机选择包含攻击在内的动作
            i = random.randint(0, turn_base - 1)    # 只随机选择“运动”的动作
            predator_acts[predator] = i

        return predator_acts


'''
the following classes are for reward description
'''
class EventNode:
    """an AST node of the event expression"""
    OP_AND = 0
    OP_OR  = 1
    OP_NOT = 2

    OP_KILL = 3
    OP_AT   = 4
    OP_IN   = 5
    OP_COLLIDE = 6
    OP_ATTACK  = 7
    OP_DIE  = 8
    OP_IN_A_LINE = 9
    OP_ALIGN = 10

    # can extend more operation below

    def __init__(self):
        # for non-leaf node
        self.op = None
        # for leaf node
        self.predicate = None

        self.inputs = []

    def __call__(self, subject, predicate, *args):
        node = EventNode()
        node.predicate = predicate
        if predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == 'at':
            node.op = EventNode.OP_AT
            coor = args[0]
            node.inputs = [subject, coor[0], coor[1]]
        elif predicate == 'in':
            node.op = EventNode.OP_IN
            coor = args[0]
            x1, y1 = min(coor[0][0], coor[1][0]), min(coor[0][1], coor[1][1])
            x2, y2 = max(coor[0][0], coor[1][0]), max(coor[0][1], coor[1][1])
            node.inputs = [subject, x1, y1, x2, y2]
        elif predicate == 'attack':
            node.op = EventNode.OP_ATTACK
            node.inputs = [subject, args[0]]
        elif predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == 'collide':
            node.op = EventNode.OP_COLLIDE
            node.inputs = [subject, args[0]]
        elif predicate == 'die':
            node.op = EventNode.OP_DIE
            node.inputs = [subject]
        elif predicate == 'in_a_line':
            node.op = EventNode.OP_IN_A_LINE
            node.inputs = [subject]
        elif predicate == 'align':
            node.op = EventNode.OP_ALIGN
            node.inputs = [subject]
        else:
            raise Exception("invalid predicate of event " + predicate)
        return node

    def __and__(self, other):
        node = EventNode()
        node.op = EventNode.OP_AND
        node.inputs = [self, other]
        return node

    def __or__(self, other):
        node = EventNode()
        node.op = EventNode.OP_OR
        node.inputs = [self, other]
        return node

    def __invert__(self):
        node = EventNode()
        node.op = EventNode.OP_NOT
        node.inputs = [self]
        return node

Event = EventNode()

class AgentSymbol:
    """symbol to represent some agents"""
    def __init__(self, group, index):
        """ define a agent symbol, it can be the object or subject of EventNode

        group: group handle
            it is the return value of cfg.add_group()
        index: int or str
            int: a deterministic integer id
            str: can be 'all' or 'any', represents all or any agents in a group
        """
        self.group = group if group is not None else -1
        if index == 'any':
            self.index = -1
        elif index == 'all':
            self.index = -2
        else:
            assert isinstance(self.index, int), "index must be a deterministic int"
            self.index = index

    def __str__(self):
        return 'agent(%d,%d)' % (self.group, self.index)


class Config:
    """configuration class of gridworld game"""
    def __init__(self):
        self.config_dict = {}
        self.agent_type_dict = {}
        self.groups = []
        self.reward_rules = []
        self.view_space_spec = []

    def set(self, args):
        """ set parameters of global configuration

        Parameters
        ----------
        args : dict
            key value pair of the configuration

        map_width: int
        map_height: int
        food_mode: bool
        turn_mode: bool
        minimap_mode: bool
        goal_mode: bool
        embedding_size: int
        render_dir: str
        seed: int
        """
        for key in args:
            self.config_dict[key] = args[key]

    def register_agent_type(self, name, attr):
        """ register an agent type

        Parameters
        ----------
        name : str
            name of the type (should be unique)
        attr: dict
            key value pair of the agent type
            see notes below to know the available attributes

        Notes
        -----
        height: int, height of agent body
        width:  int, width of agent body
        speed:  float, maximum speed, i.e. the radius of move circle of the agent
        hp:     float, maximum health point of the agent
        view_range: gw.CircleRange or gw.SectorRange

        damage: float, attack damage
        step_recover: float, step recover of health point (can be negative)
        kill_supply: float, the hp gain when kill this type of agents

        step_reward: float, reward get in every step
        kill_reward: float, reward gain when kill this type of agent
        dead_penalty: float, reward get when dead
        attack_penalty: float, reward get when perform an attack (this is used to make agents do not attack blank grid)

        """
        if name in self.agent_type_dict:
            raise Exception("type name %s already exists" % name)
        self.agent_type_dict[name] = attr
        #print('=====================', self.agent_type_dict)
        return name

    def add_group(self, agent_type):
        """ add a group to the configuration

        Returns
        -------
        group_handle : int
            a handle for the new added group
        """
        no = len(self.groups)
        self.groups.append(agent_type)
        return no

    def add_reward_rule(self, on, receiver, value, terminal=False):
        """ add a reward rule

        Some note:
        1. if the receiver is not a deterministic agent,
           it must be one of the agents involved in the triggering event

        Parameters
        ----------
        on: Expr
            a bool expression of the trigger event
        receiver:  (list of) AgentSymbol
            receiver of this reward rule
        value: (list of) float
            value to assign
        terminal: bool
            whether this event will terminate the game
        """
        if not (isinstance(receiver, tuple) or isinstance(receiver, list)):
            assert not (isinstance(value, tuple) or isinstance(value, tuple))
            receiver = [receiver]
            value = [value]
        if len(receiver) != len(value):
            raise Exception("the length of receiver and value should be equal")
        self.reward_rules.append([on, receiver, value, terminal])

    def set_view_space_spec(self, spec):
        self.view_space_spec = spec


import math
class MyRange:
    def __init__(self):
        self.count = 0
        self.p1 = []
        self.p2 = []
        self.width = self.height = -1
        self.dxy = []
        self.is_in_range = None

    def is_in(self, row, col):
        return self.is_in_range[row][col]

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_count(self):
        return self.count

    def get_dxy(self):
        return self.dxy

    def get_range_rela_offset(self):
        return self.p1, self.p2

    def add_rela_offset(self, pos_offset):
        self.p1 = self.p1 + pos_offset
        self.p2 = self.p2 + pos_offset

    def num2delta(self, n):
        return self.dxy

    def delta2num(self, dxy):
        for i, item in enumerate(self.dxy):
            if item == dxy:
                return i

        return -1

    def print_self(self):
        print(self.is_in_range)
        cnt = 0
        for row in range(self.height):
            for col in range(self.width):
                if self.is_in_range[row][col]:
                    print(int(self.dxy[cnt]))
                    cnt = cnt + 1
                else:
                    print(0,0)

class MyCircleRange(MyRange):
    def __init__(self, radius, inner_radius, parity):
        """ define a circle range for attack or view

        Parameters
        ----------
        radius : float
        """
        super(MyCircleRange, self).__init__()
        self.radius = radius
        self.inner_radius = inner_radius
        self.parity = parity        # agent尺寸为奇数，parity=1

        eps = 1e-8

        self.width = (2 * int(radius + eps) + parity)        # 外接矩形宽，若agent尺寸为奇数，宽度+1使得agent能居中
        center = int(radius)

        if (self.width % 2 != parity):
            self.width=self.width+1
        self.height = self.width

        self.is_in_range = np.zeros((self.height, self.width), dtype=bool)
        self.dxy = []

        delta = 0.5 if parity == 0 else 0
        for i in range(self.width):
            for j in range(self.width):
                dis_x = math.fabs(j - center + delta)
                dis_y = math.fabs(i - center + delta)
                dis = math.sqrt(dis_x * dis_x + dis_y * dis_y)
                if inner_radius - eps < dis and dis < radius + eps:
                    self.is_in_range[i][j] = True
                    self.dxy.append([j - center, i - center])
                    self.count = self.count + 1
                else:
                    self.is_in_range[i][j] = False

        self.p1 = [-center, -center]
        self.p2 = [self.width- center - 1, self.width- center - 1]


    def __str__(self):
        return 'circle(%g)' % self.radius


class CircleRange:
    def __init__(self, radius):
        """ define a circle range for attack or view

        Parameters
        ----------
        radius : float
        """
        self.radius = radius
        self.angle  = 360

    def __str__(self):
        return 'circle(%g)' % self.radius


class SectorRange:
    def __init__(self, radius, angle):
        """ define a sector range for attack or view

        Parameters
        ----------
        radius : float
        angle :  float
            angle should be less than 180
        """
        self.radius = radius
        self.angle  = angle
        if self.angle >= 180:
            raise Exception("the angle of a sector should be smaller than 180 degree")

    def __str__(self):
        return 'sector(%g, %g)' % (self.radius, self.angle)
