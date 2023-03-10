""" tigers eat deer to get health point and reward"""

import magent
import numpy as np


def get_config(map_size):
    gw = magent.gridworld

    cfg = gw.Config()
    '''
        map_width: int       
        map_height: int
        food_mode: bool          食物模式
        turn_mode: bool          转弯模式
        minimap_mode: bool       把小地图加入observation中
        goal_mode: bool          已经过时了
        embedding_size: int
        render_dir: str
        seed: int                使用你给的随机种子
    '''

    cfg.set({"map_width": map_size, "map_height": map_size})      # 全局地图大小 = map_height * map_width
    cfg.set({"embedding_size": 10})
    #cfg.set({"minimap_mode": True})        # True: channel = 7; False: channel = 5, 将全局地图上agent的分布缩小成边长为(2*view_range+1)的正方形大小
    #cfg.set({"turn_mode": True})

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
            hp:     float, maximum health point of the agent, 健康值，小于0则死掉
            view_range: gw.CircleRange or gw.SectorRange

            damage: float, attack damage，攻击力，使被攻击的agent的健康值减少该数值
            step_recover: float, step recover of health point (can be negative), 走一步，hp的增加量
            kill_supply: float, the hp gain when kill this type of agents,自己被杀死后，对方hp增加该值
            food_supply: float, agent死后变成食物，下次有agent攻击此处，他即可获取到该食物，如果他吃不完，该食物会剩余

            step_reward: float, reward get in every step，每走一步获得的reward，但在代码里看只用来初始化奖励
            kill_reward: float, reward gain when kill this type of agent, 被杀死后，对方得到的奖励
            dead_penalty: float, reward get when dead, 死掉后，自己得到的reward，通常为负值
            attack_penalty: float, reward get when perform an attack (this is used to make agents do not attack blank grid)，攻击空白单元，该agent得到的奖励，通常为负值
            
            gridworld这个环境好像没有下面这些属性
            hear_radius:     float
            speak_radius:    float
            speak_ability:   int
            trace:           float
            eat_ability:     float
            attack_in_group: bool

            view_x_offset, view_y_offset: int
            att_x_offset,  att_y_offset:  int
            turn_x_offset, turn_y_offset: int
    """
    fish = cfg.register_agent_type(
        "fish",
        {'width': 1, 'length': 1, 'hp': 5,
         'speed': 3,
         'move_angle': 2*np.pi,        # 弧度为单位
         'move_n': 361,      # 要求奇数
         'view_range': gw.CircleRange(8), #gw.SectorRange(5, 120),     # view_space=(视野范围占据矩形高, 视野范围占据矩形宽, channel)，其中channel= minimap_mode ? 7, 5； gw.CircleRange(5, 120)——view_space=(2*5+1, 2*5+1, channel)， gw.SectorRange(5, 120)——view_space=(5, 9, channel)
         'attack_range': gw.CircleRange(0),
         'damage': 0, 'step_recover': 0.1,     # damage=杀伤力，使得对方健康值hp减少对应的值； step_recover=每走一步而不受攻击hp会恢复
         'food_supply': 0, 'kill_supply': 0,   # kill_supply=自己被kill，对方hp增加
         'step_reward': 0.5, 'dead_penalty': -20,
         'kill_reward': 5,
         })

    predator = cfg.register_agent_type(
        "predator",
        {'width': 2, 'length': 2, 'hp': 10,
         'speed': 2,
         'move_angle': 2*np.pi,
         'move_n': 121,      # 要求奇数
         'view_range': gw.CircleRange(9), 'attack_range': gw.CircleRange(2),
         'damage': 4, 'step_recover': 0.1,
         'food_supply': 0, 'kill_supply': 0,
         'step_reward': 0, 'attack_penalty': 0.1,
         })

    fish_group  = cfg.add_group(fish)
    predator_group = cfg.add_group(predator)

    b = gw.AgentSymbol(fish_group, index='any')
    a = gw.AgentSymbol(predator_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[2, -2])

    return cfg
