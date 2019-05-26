import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

from ann_utils.sess import TfSess
from ann_utils.manager import tf_global_initializer

from agents.curiosity_agent import CuriosityAgent

sess = TfSess( gpu = True )

env = gym.make('MountainCar-v0')
actions_size = env.unwrapped.action_space.n
state_size = env.unwrapped.observation_space.shape[0]

chp_folder = './model_params/'

# create a agent
dqn = CuriosityAgent( actions_size, state_size, sess, chp_folder, "model", soft_update = 0.01, lr = 0.01, epsilon = 0.7 )

# build agent graph
dqn.build_agent_brain()

# initialize variables
tf_global_initializer( sess )

ep_steps = []
loss = [ 0 ]
reward = [ 0 ]
for epi in range(2000):

    s = env.reset()
    steps = 0
    while True:

        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        dqn.store_transition(s, a, r, s_)
        ls = dqn.learn()

        loss.append( ls )
        reward.append( r )

        if done:
            print( 'Epi: ', epi, "| steps: ", steps, 'Loss: ', np.mean( loss ), 'Reward: ', np.mean( reward ) )   
            ep_steps.append(steps)
            break

        s = s_
        steps += 1

plt.plot(ep_steps)
plt.ylabel("steps")
plt.xlabel("episode")
plt.show()
