from magent2.environments import battle_v4,adversarial_pursuit_v4
from pettingzoo.utils import random_demo

# env = battle_v4.env(render_mode='human',max_cycles=5)
env = adversarial_pursuit_v4.env(render_mode='human',max_cycles=200)
random_demo(env, render=True, episodes=2)