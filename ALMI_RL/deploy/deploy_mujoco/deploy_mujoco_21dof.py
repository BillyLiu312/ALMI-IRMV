import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import joblib
from termcolor import colored
# import clip
# from legged_gym.trans_model.vqvae import HumanVQVAE

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    
    config_file = 'h1_2_21dof.yaml'
    with open(f"./deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        lower_policy_path = config["lower_policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        upper_policy_path = config["upper_policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        motion_path = config["motion_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        input_seq_len = config['input_seq_len']
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_lower_actions = config["num_lower_actions"]
        num_lower_obs = config["num_lower_obs"]

        num_upper_actions = config["num_upper_actions"]
        num_upper_obs = config["num_upper_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # load clip model
    # clip_model, clip_preprocess = clip.load("/home/bcj/new_unitree_rl_gym/legged_gym/trans_model/ViT-B-32.pt", device=torch.device('cpu'), jit=False)
    # clip_model, clip_preprocess = clip.load("./pretrained/ViT-B-32.pt", device=torch.device('cpu'), jit=False)
    # clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # clip_model.eval()
    # for p in clip_model.parameters():
    #     p.requires_grad = False
    # print("successfully load clip model")

    # encode text
    # clip_text = "Robot go forward fast and wave left."
    # text = clip.tokenize(num_lower_actions + num_upper_actions, truncate=True).to("cpu")
    # feat_clip_text = clip_model.encode_text(text).float()

    # define context variables
    action = np.zeros(num_lower_actions + num_upper_actions, dtype=np.float32)
    lower_actions = np.zeros(num_lower_actions, dtype=np.float32)
    upper_actions = np.zeros(num_upper_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    lower_obs = np.zeros(num_lower_obs, dtype=np.float32)
    upper_obs = np.zeros(num_upper_obs, dtype=np.float32)
    # obs_history_len = input_seq_len
    # trajectory_history = torch.zeros(size=(1, obs_history_len, 71))
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load motion ref
    print(colored(f"Load motion from {motion_path}......", "green"), end="\n")
    motion_data = joblib.load(motion_path) # shape: (num_motion, num_frames, num_dof)
    random_motion_idx = np.random.randint(0, len(motion_data))
    print(colored(f"Current selected motion: {random_motion_idx}", "blue"), end="\n")
    frames = 0

    # load policy
    lower_policy = torch.jit.load(lower_policy_path)
    upper_policy = torch.jit.load(upper_policy_path)
    # num_frame = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # num_frame += 1
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]  # 21
                dqj = d.qvel[6:] # 21
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                if frames >= len(motion_data[random_motion_idx]):
                    random_motion_idx = np.random.randint(0, len(motion_data))
                    frames = 0
                    print(colored(f"Current selected motion: {random_motion_idx}", "blue"), end="\n")
                ref_dof = motion_data[random_motion_idx][frames][0, -9:]
                frames += 1

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                offset = 0.5
                phase_left = phase
                phase_right = (phase + offset) % 1
                left_sin_phase = np.sin(2 * np.pi * phase_left)
                right_sin_phase = np.sin(2 * np.pi * phase_right)

                lower_obs[:3] = omega
                lower_obs[3:6] = gravity_orientation
                lower_obs[6:9] = cmd * cmd_scale
                lower_obs[9:30] = qj
                lower_obs[30:51] = dqj
                lower_obs[51:63] = lower_actions
                lower_obs[63:65] = np.array([left_sin_phase, right_sin_phase])

                upper_obs[:3] = omega
                upper_obs[3:6] = gravity_orientation
                upper_obs[6:15] = ref_dof
                upper_obs[15:36] = qj
                upper_obs[36:57] = dqj
                upper_obs[57:66] = upper_actions
                upper_obs[66:68] = np.array([left_sin_phase, right_sin_phase])

                lower_obs_tensor = torch.from_numpy(lower_obs).unsqueeze(0).float()  # shape: (1, num_lower_obs)
                upper_obs_tensor = torch.from_numpy(upper_obs).unsqueeze(0).float()  # shape: (1, num_upper_obs)

                with torch.no_grad():
                    lower_action_tensor = lower_policy(lower_obs_tensor)  # 根据实际模型输出调整
                    upper_action_tensor = upper_policy(upper_obs_tensor)  # 根据实际模型输出调整

                lower_actions = lower_action_tensor.squeeze().numpy()
                upper_actions = upper_action_tensor.squeeze().numpy()

                action = np.concatenate([lower_actions, upper_actions], axis=0)

                # obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # if num_frame <= obs_history_len:
                #     trajectory_history[:, num_frame-1] = obs_tensor
                # else:
                #     trajectory_history = torch.concat((trajectory_history[:, 1:], obs_tensor.unsqueeze(1)), dim=1)
                
                # action = policy(trajectory_history, feat_clip_text).detach().numpy().squeeze() # actor(obs_tensor).detach().numpy().squeeze()
                # if num_frame < obs_history_len:
                #     # action = action[num_frame, -23:-2]
                #     action = action[num_frame, :]
                # else:
                #     # action = action[-1, -23:-2]
                #     action = action[-1, :]
                # action[-9:] = 0
                                
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
