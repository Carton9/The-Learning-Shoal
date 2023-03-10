"""Battle
"""

import argparse
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import magent


from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools2_alignment as tools2
from examples.battle_model.senario_fishschool_alignment import battle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, default='mfq',
                        help='choose an algorithm from the preset', required=True)
    parser.add_argument('--n_round', type=int, default=1, help='set the trainning round')
    parser.add_argument('--n_steps', type=int, default=500, help='set the steps in each round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=80, help='set the size of map')  # then the amount of agents is 64

    # parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # tf 配置
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    results_dirs = [os.path.join(BASE_DIR, 'data/results/fishschool_alignment(140)_2')]

    # Initialize the environment
    env = magent.GridWorld('fishschool_alignment', map_size=args.map_size)
    env.set_render_dir(os.path.join(results_dirs[0], 'render_align'))
    handles = env.get_handles()

    # 加载已训练好的模型
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.n_steps)]
    sess.run(tf.global_variables_initializer())
    main_model_dirs = [os.path.join(BASE_DIR, 'data/models/fishschool_alignment(140)/{}-0'.format(args.algo))]
    steps = [49]
    models[0].load(main_model_dirs[0], step=steps[0])

    agents_num = [20, 50, 80, 110, 140, 170, 200]

    for i in range(len(agents_num)):
        env.agents_num = agents_num[i]
        runner = tools2.Runner(sess, env, handles, args.map_size, args.n_steps, models, battle, render_every=5)

        for k in range(0, args.n_round):
            order_params, avg_neighbor_num = runner.run(0.0, k, results_dir=results_dirs[0], n_agents=agents_num[i])
