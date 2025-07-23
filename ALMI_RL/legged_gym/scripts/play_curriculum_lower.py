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



def plot_dof(target_q, q):

    target_q_array = np.array(target_q)
    q_array = np.array(q)

    plt.figure(figsize=(12, 6))
    for i in range(target_q_array.shape[1]):
        plt.subplot(2, 4, i + 1)
        plt.plot(target_q_array[:, i], label=f'Target Q {i}')
        plt.plot(q_array[:, i], label=f'Q {i}', linestyle='--')
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
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5

    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [-0.7, 0.7]#[0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]#[0.0, 0.0]  
    
    env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]#[0.0, 0.0]

    # env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.heading = [3.14, 3.14]
    
    env_cfg.terrain.mesh_type = "plane"
    
    env_cfg.asset.init_arm_weight = 0.8
    env_cfg.asset.arm_weight_increase = 0
    env_cfg.asset.motion_weight_increase = 0
    env_cfg.asset.arm_curriculum = True
    env_cfg.domain_rand.action_delay = True
    
    # env_cfg.asset.motion_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/motions/all_wave.pkl')
    # env_cfg.asset.mean_episode_length_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/curriculum/mean_episode_length.csv')
    env_cfg.asset.motion_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/motions/0-KIT_572_wave_both13_poses.pkl')
    env_cfg.asset.mean_episode_length_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/curriculum/mean_episode_length_xxx.csv')
    
    # prepare environment
    # env: H1_2_WholeBody
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()
    upper_obs = env.get_upper_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    upper_policy = ppo_runner.upper_actor_critic.act_inference
    
    lootat = env.root_states[0,:3].detach().cpu().numpy()
    camara_position = lootat + [5,-2,2]
    env.set_camera(camara_position, lootat)
    
    left_phase = []
    right_phase = []   
    left_sin_phase = []
    right_sin_phase = []
    
    for i in range(1000): #(int(env.max_episode_length)):
        actions = policy(obs.detach())
        upper_actions = upper_policy(upper_obs.detach())
        
        obs, _, rews, dones, infos, upper_obs = env.step(actions.detach(), upper_actions.detach())
        left_phase.append(float(env.phase_left[0].detach().cpu()))
        right_phase.append(float(env.phase_right[0].detach().cpu()))
        left_sin_phase.append(float(env.left_sin_phase[0].detach().cpu()))
        right_sin_phase.append(float(env.right_sin_phase[0].detach().cpu()))

    print("done")
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    PLOT_DOF = False
    PLOT_PHASE = False
    args = get_args()
    play(args)
