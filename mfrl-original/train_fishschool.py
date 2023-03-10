"""Self Play
"""

import argparse
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools2
from examples.battle_model.senario_fishschool import play


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=2000, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=60, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')

    args = parser.parse_args()

    # Initialize the environment
    # !!!!!!!!!!------1. 通过config下的fishschool.py配置文件，创建env
    env = magent.GridWorld('fishschool', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/fishschool/{}'.format(args.algo))

    if args.algo in ['mfq', 'mfac']:
        use_mf = True
    else:
        use_mf = False

    sess = tf.Session(config=tf_config)
    # !!!!!!!!!!!!!!----2. 创建对抗的策略模型——models[0], models[1]。 在这里确定选用何种算法'ac', 'mfac', 'mfq', 'il'
    # models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps),
    #           spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent', args.max_steps)]
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    start_from = 0
    # model_file = os.path.join('/home/hill/mtmfrl/mfrl-original/data/models/fishschool/', 'mfq-0')
    # models[0].load(model_file, step=499)

    # !!!!!!!!!!!!!-----3. 定义一个执行器，原来用的是tools, 在此使用的是algo目录下的tools2，将原来的self-play（models[1]直接使用[models[0]来更新）部分去掉了，models[0]和models[1]单独训练
    runner = tools2.Runner(sess, env, handles, args.map_size, args.max_steps, models, play,        # play 是一个非常重要的函数，数据采集和训练都play函数里
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        # !!!!!!!!!!-----4. 执行训练
        runner.run(eps, k)          # 在run里执行play函数，进行数据采集并训练
