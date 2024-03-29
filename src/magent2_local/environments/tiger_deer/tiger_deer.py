# noqa
"""
## Tiger Deer

```{figure} tiger_deer.gif
:width: 140px
:name: tiger_deer
```

| Import             | `from magent2.environments import tiger_deer_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= [deer_[0-100], tiger_[0-19]]`      |
| Agents             | 121                                           |
| Action Shape       | (5),(9)                                       |
| Action Values      | Discrete(5),(9)                               |
| Observation Shape  | (3,3,5), (9,9,5)                              |
| Observation Values | [0,2]                                         |
| State Shape        | (45, 45, 5)                                   |
| State Values       | (0, 2)                                        |


In tiger-deer, there are a number of tigers who are only rewarded for teaming up to take down the deer (two tigers must attack a deer in the same step to receive reward). If they do not eat the deer, they will slowly lose 0.1 HP each turn until they die. If they do eat the deer they regain 8
health (they have 10 health to start). At the same time, the deer are trying to avoid getting attacked. Deer start with 5 HP, lose 1 HP when attacked, and regain 0.1 HP each turn. Deer should run from tigers and tigers should form small teams to take down deer.

### Arguments

``` python
tiger_deer_v3.env(map_size=45, minimap_mode=False, tiger_step_recover=-0.1, deer_attacked=-0.1, max_cycles=500, extra_features=False)
```

`map_size`: Sets dimensions of the (square) map. Increasing the size increases the number of agents.  Minimum size is 10.

`minimap_mode`: Turns on global minimap observations. These observations include your and your opponents piece densities binned over the 2d grid of the observation space. Also includes your `agent_position`, the absolute position on the map (rescaled from 0 to 1).

`tiger_step_recover`: Amount of health a tiger gains/loses per turn (tigers have health 10 and get health 8 from killing a deer)

`deer_attacked`: Reward a deer gets for being attacked

`max_cycles`:  number of frames (a step for each agent) until game terminates

`extra_features`: Adds additional features to observation (see table). Default False

#### Action Space

Key: `move_N` means N separate actions, one to move to each of the N nearest squares on the grid.

Tiger action space: `[do_nothing, move_4, attack_4]`

Deer action space: `[do_nothing, move_4]`

#### Reward

Tiger's reward scheme is:

* 1 reward for attacking a deer alongside another tiger

Deer's reward scheme is:

* -1 reward for dying
* -0.1 for being attacked

#### Observation space

The observation space is a 3x3 map with 5 channels for deer and 9x9 map with 5 channels for tigers, which are (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
other_team_presence| 1
other_team_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 5 Deer/9 Tiger
last_reward(extra_features=True)| 1

### Version History

* v0: Initial MAgent2 release (0.3.0)

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

import magent2
from magent2.environments.magent_env import magent_parallel_env, make_env

default_map_size = 45
max_cycles_default = 300
minimap_mode_default = False
default_env_args = dict(tiger_step_recover=-0.1, deer_attacked=-0.1)


def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    seed=None,
    **env_args
):
    env_env_args = dict(**default_env_args)
    env_env_args.update(env_args)
    return _parallel_env(
        map_size,
        minimap_mode,
        env_env_args,
        max_cycles,
        extra_features,
        render_mode,
        seed,
    )


def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    seed=None,
    **env_args
):
    return parallel_to_aec_wrapper(
        parallel_env(
            map_size, max_cycles, minimap_mode, extra_features, seed=seed, **env_args
        )
    )


env = make_env(raw_env)

def get_config(map_size, minimap_mode, seed, tiger_step_recover, deer_attacked):
    gw = magent2.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})
    cfg.set({"minimap_mode": minimap_mode})
    if seed is not None:
        cfg.set({"seed": seed})

    options = {
        "width": 1,
        "length": 1,
        "hp": 1,
        "speed": 1,
        "view_range": gw.CircleRange(3),
        "attack_range": gw.CircleRange(0),
        "step_recover": 0,
        "kill_supply": 10,
        "dead_penalty": -1.0,
    }

    deer = cfg.register_agent_type("deer", options)

    options = {
        "width": 2,
        "length": 2,
        "hp": 10,
        "speed": 1.25,
        "view_range": gw.CircleRange(5),
        "attack_range": gw.CircleRange(2),
        "damage": 100,
        "step_recover": tiger_step_recover,
    }
    tiger = cfg.register_agent_type("tiger", options)

    deer_group = cfg.add_group(deer)
    tiger_group = cfg.add_group(tiger)

    a = gw.AgentSymbol(tiger_group, index="any")
    b = gw.AgentSymbol(tiger_group, index="any")
    c = gw.AgentSymbol(deer_group, index="any")
    d = gw.AgentSymbol(deer_group, index="any")

    # tigers get reward when they attack a deer simultaneously
    # e1 = gw.Event(a, "attack", c)
    # e2 = gw.Event(b, "attack", c)
    # e2 = gw.Event()
    tiger_attack_rew = 1
    in_a_line_rew = 0.005
    collide_rew = -0.01
    align_rew = 0.005
    # reward is halved because the reward is double counted
    # cfg.add_reward_rule(
    #     e1 & e2, receiver=[a, b], value=[tiger_attack_rew / 2, tiger_attack_rew / 2]
    # )
    cfg.add_reward_rule(gw.Event(a, "kill", c), receiver=[a], value=[tiger_attack_rew])
    cfg.add_reward_rule(gw.Event(a, "kill", b), receiver=[a], value=[-tiger_attack_rew])
    # cfg.add_reward_rule(gw.Event(c, "in_a_line"), receiver=[c], value=[in_a_line_rew])
    # cfg.add_reward_rule(gw.Event(c, "collide", d), receiver=[c], value=[collide_rew])
    # cfg.add_reward_rule(gw.Event(c, "in", (())), receiver=[c], value=[collide_rew])
    # cfg.add_reward_rule(gw.Event(c, "align"), receiver=[c], value=[align_rew])

    return cfg


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "tiger_deer_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
        seed=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
            seed,
        )
        assert map_size >= 10, "size of map must be at least 10"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, seed, **reward_args), map_size=map_size
        )

        handles = env.get_handles()
        reward_vals = np.array([1, -1] + list(reward_args.values()))
        reward_range = [
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]

        names = ["deer", "tiger"]
        super().__init__(
            env,
            handles,
            names,
            map_size,
            max_cycles,
            reward_range,
            minimap_mode,
            extra_features,
            render_mode,
        )

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()

        # env.add_walls(method="random", n=map_size * map_size * 0.04)
        env.add_walls(method="random", n=0)
        # Deer
        # env.add_agents(handles[0], method="random", n=map_size * map_size * 0.05)
        env.add_agents(handles[0], method="random", n=100)
        #Tiger
        # env.add_agents(handles[1], method="random", n=map_size * map_size * 0.007)
        env.add_agents(handles[1], method="random", n=25)
