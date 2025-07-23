import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_dof(target_q, q, policy_q):

    target_q_array = np.array(target_q)
    q_array = np.array(q)
    # print(target_q_array.shape)
    # print(q_array.shape)
    policy_q_array = np.array(policy_q)

    plt.figure(figsize=(12, 6))
    for i in range(target_q_array.shape[1]):
        plt.subplot(3, 4, i + 1)
        plt.plot(target_q_array[:, i], label=f'Target Q {i}')
        plt.plot(q_array[:, i], label=f'Q {i}', linestyle='--')
        plt.plot(policy_q_array[:, i],label=f'Policy Q {i}', linestyle=':')
        plt.xlabel('Time Step')
        plt.ylabel('Joint Position')
        plt.title(f'Target Q and Q {i} over Time')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

def plot_phase(left_phase, right_phase, sin_phase, cos_phase):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(left_phase, label='Left Phase')
    plt.plot(right_phase, label='Right Phase', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Phase')
    plt.title('Left and Right Phase over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sin_phase, label='Left Sin Phase')
    plt.plot(cos_phase, label='Right Sin Phase', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Phase')
    plt.title('Sin and Cos Phase over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def play(args):
    env_cfg: H1_2_WholeBodyCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5.0
    
    

    env_cfg.env.test = True

    env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]
    # env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0] # scale
    env_cfg.commands.ranges.lin_vel_x = [-0.7, 0.7]  
    
    # env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]
    
    # env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.heading = [3.14, 3.14]
    
    env_cfg.asset.init_arm_weight = 1
    env_cfg.asset.arm_curriculum = False
    env_cfg.domain_rand.action_delay = False
    
    env_cfg.asset.motion_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/motions/all_wave.pkl')
    env_cfg.asset.mean_episode_length_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/curriculum/mean_episode_length.csv')
    
    # prepare environment
    # env: H1_2_WholeBody
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    lower_obs = env.get_lower_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    lower_policy = ppo_runner.lower_actor_critic.act_inference

    
    lootat = env.root_states[0,:3].detach().cpu().numpy()
    camara_position = lootat + [5,-2,2]
    env.set_camera(camara_position, lootat)
    
    left_phase = []
    right_phase = []   
    left_sin_phase = []
    right_sin_phase = []
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, checkpoint=args.checkpoint)
        print('Exported policy as jit script to: ', path)


    # env_cfg.control.arm_action_scale = 1.0


    for i in range(int(10*env.max_episode_length)):
        actions = policy(obs.detach())
        # actions[:,0] = 0
        lower_actions = lower_policy(lower_obs.detach())
        obs, _, rews, dones, infos, lower_obs = env.step(actions.detach(), lower_actions.detach())
        #delta = actions * 1 * env_cfg.control.arm_action_scale -  obs[:, 6:15]
        # print(delta)
        
        left_phase.append(float(env.phase_left[0].detach().cpu()))
        right_phase.append(float(env.phase_right[0].detach().cpu()))
        left_sin_phase.append(float(env.left_sin_phase[0].detach().cpu()))
        right_sin_phase.append(float(env.right_sin_phase[0].detach().cpu()))
        if PLOT_DOF and i > 300:
            plot_dof(env.target_q_list, env.q_list, env.policy_q_list)
            break
        
        if PLOT_PHASE and i > 200:
            plot_phase(left_phase, right_phase, left_sin_phase, right_sin_phase)
            break


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    PLOT_DOF = True
    PLOT_PHASE = False
    args = get_args()
    play(args)
