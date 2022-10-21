from __future__ import print_function
from __future__ import absolute_import

######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

# python modules
import argparse
import importlib
import math
import random
import numpy as np
import os.path
import sys
import collections
import tensorflow as tf
from tensorflow import keras

# project files
import dynamic_obstacle
import bounds
import robot # double integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from utils import ReplayBuffer
from td3 import TD3
from ssa import SafeSetAlgorithm
from cautious_rl import ProbabiilisticShield
from cbf import ControlBarrierFunction
import ISSA_AdamBA as IA
'''
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
'''


def main(display_name, exploration, qp, enable_ssa_buffer, ISSA, pure_AdamBA, obs_num, AdamBA_resol):
    AdamBA_resol = int(AdamBA_resol)
    # testing env
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params, int(obs_num))

    # rl policy
    robot_state_size = 4 #(x,y,v_x,v_y)
    robot_action_size = 2
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = robot_state_size + nearest_obstacle_state_size

    model_update_freq = 1000
    env = simu_env.Env(display, int(obs_num), **(env_params))

    policy_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = robot_action_size, max_size=int(1e6))
    policy = TD3(state_dim, robot_action_size, env.max_acc, env.max_acc, exploration = exploration)
    #policy.load("./model/ssa1")
    ssa_replay_buffer = ReplayBuffer(state_dim = state_dim, action_dim = robot_action_size, max_size=int(1e6))
    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed, is_qp = qp)
    cbf_controller = ControlBarrierFunction(max_speed = env.robot_state.max_speed)
    shield_controller = ProbabiilisticShield(max_speed = env.robot_state.max_speed)
    # parameters
    max_steps = int(1e6)
    start_timesteps = 2e3
    episode_reward = 0
    episode_num = 0
    last_episode_reward = 0
    teacher_forcing_rate = 0
    total_rewards = []
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    state, done = env.reset(), False
    collision_num = 0
    failure_num = 0
    success_num = 0

    # Random Network Distillation
    rnd_fixed = RND()
    rnd_train = RND()
    rnd_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    rnd_loss = keras.losses.MeanSquaredError()

    is_meet_requirement = False
    reward_records = []


    robot_xs = []
    robot_ys = []
    obs_xs = []
    obs_ys = []
    safe_obs_xs = []
    safe_obs_ys = []
    xs_qp = []
    ys_qp = []
    obs_xs_qp = []
    obs_ys_qp = []

    out_s = []
    yes_s = []
    valid_s = []
#Newly added parameters:
    s_next_new = []
    phi = 0
#Collect number of times that collide    
    collision_num = 0



    for t in range(max_steps):
      # disturb the policy parameters at beginning of each episodes when using PSN
      if (exploration == 'psn' and env.cur_step == 0):
        policy.parameter_explore()
        print(f"parameter_explore in {t}")
      
      # train the random network prediction when using rnd
      if (exploration == 'rnd' and t > 1024):
        with tf.GradientTape() as tape:
          state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  policy_replay_buffer.sample(256)
          q_fixed = rnd_fixed.call(state_batch)
          q_train = rnd_train.call(state_batch)
          loss = rnd_loss(q_fixed, q_train)
          gradients = tape.gradient(loss, rnd_train.trainable_weights)
          rnd_optimizer.apply_gradients(zip(gradients, rnd_train.trainable_weights))
      
      action = policy.select_action(state)
      original_action = action
      env.display_start()
      # ssa parameters
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 12)
      action, is_safe, is_unavoidable, danger_obs, phis_unsafe_temp_old = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      all_obs_states = env.find_all_obstacle_loc(state[0],state[1])
# u could be original_action
# u could be action(original action being filtered by the "get safe control")
# #Use the AdamBA to change the control action geneated by either vanilla SSA or adapated SSA:
      if ISSA:
        #need to know phi in the ssa(future phi and current phi)
        if phis_unsafe_temp_old>0:
          action, valid_adamba, NP_vec_tmp, out, yes, valid= IA.AdamBA(s = state[:4], u = original_action, env = env, threshold=0, phi_old = phis_unsafe_temp_old,
                                          obstacles=unsafe_obstacles, pure_AdamBA = pure_AdamBA, AdamBA_resol = AdamBA_resol)
# u= action, this action set as the default action when AdamBA doesn't have solution, or solution itself is safe, this action could be set as nominal action (original_action), 
# or get_safe_control action(action). use the original action means the program purely use AdamBA to find safe control, not using any SSA. If use (get_safe_control action), 
# user will find most of the action in AdamBA are "Yes", means itself is safe, since it already been filtered by SSA, but AdamBA still be effective on the scenario where the 
# safe_control provided for next step, the new phi is still larger than zero, and AdamBA could provide an alternative solution to find a better control. 
      #Record data     
          out_s.append(out)
          yes_s.append(yes)
          valid_s.append(valid)
      np.save('src/trajectory_result/out_s.npy', np.array(out_s))
      np.save('src/trajectory_result/yes_s.npy', np.array(yes_s))
      np.save('src/trajectory_result/valid_s.npy', np.array(valid_s))
      if (len(danger_obs) > 0):
        for obs in danger_obs:
          obs_xs.append(obs[0])
          obs_ys.append(obs[1])
      for obs in env.field.obstacles:
        safe_obs_xs.append(obs.c_x)
        safe_obs_ys.append(obs.c_y)
      robot_xs.append(state[0])
      robot_ys.append(state[1])
      s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
      original_reward = reward
      episode_reward += original_reward
      # add the novelty to reward when using rnd
      if (exploration == 'rnd'):
        rnd_state = tf.convert_to_tensor(state.reshape(1, -1))
        q_fixed = rnd_fixed.call(rnd_state)
        q_train = rnd_train.call(rnd_state)                    
        loss = np.sum(np.square(q_fixed - q_train))      
        reward += loss      
      env.display_end()
      # Store data in replay buffer
      if (enable_ssa_buffer):
        if (is_safe):
          ssa_replay_buffer.add(state, action, s_new, reward, done)          
        else:
          policy_replay_buffer.add(state, action, s_new, reward, done)
      else:
        policy_replay_buffer.add(state, original_action, s_new, reward, done)
      old_state = state
      state = s_new

      # train policy
      if (policy_replay_buffer.size > 1024):
        state_batch, action_batch, next_state_batch, reward_batch, not_done_batch =  [np.array(x) for x in policy_replay_buffer.sample(256)]
        if enable_ssa_buffer and ssa_replay_buffer.size > 128:
            model_batch_size = int(0.4*256) # batch size is 256, ratio is 0.4
            idx = np.random.choice(256, model_batch_size, replace=False)
            state_batch[idx], action_batch[idx], next_state_batch[idx], reward_batch[idx], not_done_batch[idx] =  ssa_replay_buffer.sample(model_batch_size)
        policy.train_on_batch(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch)

      if (done and original_reward == -500):          
        print("collision")
        collision_num += 1      
        safe_controller.plot_control_subspace(old_state[:4], unsafe_obstacles, fx, gx, original_action)
#original setting is when collide with obstacle break from main function
      elif (done and original_reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      
      if (done):   
        np.save('src/trajectory_result/xs.npy', np.array(robot_xs))
        np.save('src/trajectory_result/ys.npy', np.array(robot_ys))
        np.save('src/trajectory_result/obs_xs.npy', np.array(obs_xs))
        np.save('src/trajectory_result/obs_ys.npy', np.array(obs_ys))
        np.save('src/trajectory_result/safe_obs_xs.npy', np.array(safe_obs_xs))
        np.save('src/trajectory_result/safe_obs_ys.npy', np.array(safe_obs_ys))
        np.save('src/trajectory_result/xs_qp.npy', np.array(xs_qp))
        np.save('src/trajectory_result/ys_qp.npy', np.array(ys_qp))
        np.save('src/trajectory_result/obs_xs_qp.npy', np.array(obs_xs_qp))
        np.save('src/trajectory_result/obs_ys_qp.npy', np.array(obs_ys_qp))
        total_steps += env.cur_step
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, reward {episode_reward}, is_qp {qp}, exploration {exploration}, last state {state[:2]}")
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
        if (episode_num >= 10):
          print(collision_num)
          print(success_num)
        if (episode_num >= 10):
          policy.save("./model/ssa1")
          break

      # check reward threshold
      '''
      if (len(total_rewards) >= 20 and np.mean(total_rewards[-20:]) >= 1900 and not is_meet_requirement):
        print(f"\n\n\nWe meet the reward threshold episode_num {episode_num}, total_steps {total_steps}\n\n\n")
        is_meet_requirement = True
        break
      '''

      # evalution part at every 1000 steps
      '''
      if (t % 1000 == 0):
        env.save_env()
        eval_reward = eval(policy, env, safe_controller, fx, gx)
        print(f"t {t}, eval_reward {eval_reward}")
        reward_records.append(eval_reward)
        env.read_env()
        if (len(reward_records) == 100):
          break
      '''
    return reward_records

class RND(keras.Model):
    '''
        RND
    '''
    def __init__(self):
        super().__init__()
        self.l1 = keras.layers.Dense(128, activation="relu")
        self.l2 = keras.layers.Dense(128)

    def call(self, state):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        q1 = self.l1(state)
        q1 = self.l2(q1)
        return q1

def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()

def run_kwargs( params, obs_num):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    obs_num = int(obs_num)
    ret = { 'field': dynamic_obstacle.ObstacleField(obs_num),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }
    return ret

def eval(policy, env, safe_controller, fx, gx):
  episode_num = 0
  episode_reward = 0
  state, done = env.reset(), False
  episode_rewards = []
  arrives = []
  while (True):
    action = policy.select_action(state)  
    unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(env.min_dist * 6)
    action, _, _,_ = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
    s_new, reward, done, info = env.step(action)
    episode_reward += reward
    state = s_new
    if (done):
      state, done = env.reset(), False
      return episode_reward

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.add_argument( '--explore',
                   choices=('psn','rnd','none'),
                   default='none' )
    prsr.add_argument( '--qp',dest='is_qp', action='store_true')
    prsr.add_argument( '--no-qp',dest='is_qp', action='store_false')
    prsr.add_argument( '--ssa-buffer',dest='enable_ssa_buffer', action='store_true')
    prsr.add_argument( '--no-ssa-buffer',dest='enable_ssa_buffer', action='store_false')
    prsr.add_argument( '--ISSA',dest='ISSA', action='store_true')
    prsr.add_argument( '--no-ISSA',dest='ISSA', action='store_false')
    prsr.add_argument( '--pureAdamBA',dest='pure_AdamBA', action='store_true')
    prsr.add_argument( '--no-pureAdamBA',dest='pure_AdamBA', action='store_false')
    prsr.add_argument( '--obs_num',
                choices=('20','50','100'),
                default='none' )
    prsr.add_argument( '--AdamBA_resol',
                   choices=('50','100','300'),
                   default='none' )
    return prsr

if __name__ == '__main__':
    args = parser().parse_args()
    all_reward_records = []
    for i in range(100):
      all_reward_records.append([])
    for i in range(1):
      reward_records = main(display_name = args.display, #'turtle',#
          exploration = args.explore, #'rnd',
          qp = args.is_qp, #False,#
          enable_ssa_buffer = args.enable_ssa_buffer, #True,
          ISSA = args.ISSA, #True,
          pure_AdamBA= args.pure_AdamBA, #False
          obs_num = args.obs_num,
          AdamBA_resol = args.AdamBA_resol
          )
      for j, n in enumerate(reward_records):
        all_reward_records[j].append(n)
      print(all_reward_records)
    #np.save('plot_result/ssa_rl.npy', np.array(all_reward_records))

#Hard code input arguments, for debugging purpose
# if __name__ == '__main__':
#     args = parser().parse_args()
#     all_reward_records = []
#     for i in range(100):
#       all_reward_records.append([])
#     for i in range(1):
#       reward_records = main(display_name = 'turtle',#
#           exploration = 'rnd',
#           qp = False,#
#           enable_ssa_buffer = True,
#           ISSA = True,
#           pure_AdamBA= False,
#           obs_num = 50,
#           AdamBA_resol = 100
#           )
#       for j, n in enumerate(reward_records):
#         all_reward_records[j].append(n)
#       print(all_reward_records)

