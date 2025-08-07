from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class H1_2_WholeBodyCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]
        default_joint_angles = { # = target angles [rad] when action = 0.0 
            'left_hip_yaw_joint': 0,
            'left_hip_pitch_joint': -0.4,
            'left_hip_roll_joint': 0,
            'left_knee_joint': 0.8,
            'left_ankle_pitch_joint': -0.4,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': 0,
            'right_knee_joint': 0.8,
            'right_ankle_pitch_joint': -0.4,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0,

            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0.2,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': -0.2,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
        }

    class env(LeggedRobotCfg.env):
        
        # 3 + 3 + 3 + 21 + 21 + 12 + 2 = 47 + 18 = 65

        # 3 + 3 + 9 + 21 + 21 + 9 + 2 = 68
        
        # with sin/cos phase
        num_observations = 65
        num_privileged_obs = 68

        num_actions = 12


        num_envs = 4096 
        
      

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,
            
            'torso_joint': 300, 
            
            'shoulder_pitch_joint': 120,
            'shoulder_roll_joint': 120,
            'shoulder_yaw_joint': 120,
            'elbow_pitch_joint': 80 
        }  # [N*m/rad]
        
        damping = {
            'hip_yaw_joint': 2.5,  #2.5
            'hip_roll_joint': 2.5, #2.5
            'hip_pitch_joint': 2.5,#2.5
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,
            
            'torso_joint': 3,
            
            'shoulder_pitch_joint': 2,
            'shoulder_roll_joint': 2,
            'shoulder_yaw_joint': 2,
            'elbow_pitch_joint': 1 
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 8
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        # dt =  0.0025
        dt = 0.002

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 5.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1
        
        # 12.23 add rand link mass and base CoM
        randomize_link_mass = True
        randomize_link_body_names = [
            'pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 
            'left_ankle_pitch_link','left_ankle_roll_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 
            'right_ankle_pitch_link','right_ankle_roll_link', 'torso_link']
        link_mass_range = [0.9, 1.1]
        
        randomize_base_com = True

        class base_com_range:
            x = [-0.1, 0.1]
            y = [-0.1, 0.1]
            z = [-0.2, 0.2]
        
        action_delay = True
        action_delay_range = [0, 2]
        

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.7, 0.7] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_21dof.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        knee_name = "knee"
        torso_name = 'torso_link'
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        armature = 1e-3
        
        collapse_fixed_joints = False
        
        motion_path = '{LEGGED_GYM_ROOT_DIR}/resources/motions/all_wave.pkl'
        # motion_path = '{LEGGED_GYM_ROOT_DIR}/resources/motions/all_motion.pkl'
        
        init_arm_weight = 0.0
        arm_curriculum = True


        init_motion_weight = 0.0

        motion_leading = False 

        mean_episode_length_path = "{LEGGED_GYM_ROOT_DIR}/resources/curriculum/mean_episode_length.csv"
        # mean_episode_length_path = "{LEGGED_GYM_ROOT_DIR}/resources/curriculum/mean_episode_length_all_motion_new.csv" 

        motion_weight_increase = 0.005 
        motion_weight_decrease = 0.01
        arm_weight_increase = 0.005 
        arm_weight_decrease = 0.01
        motion_range = 0.0025

    # add 2.28 
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        
        only_rough = True
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.95
        
        min_dist = 0.3 
        max_dist = 0.6
        
        max_contact_force = 700


        feet_swing_height_threshold = 0.08 # 0.20

        phase_sigma = 0.05

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.5
            orientation = -1.0
            torques = -1e-5
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3

            feet_air_time = 0 # 10.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0 
            contact = 0.18

            feet_distance = 1
            knee_distance = 0.2 


            stand_still = -2 
            

            ankle_torque = -5e-5
            ankle_action_rate = -0.02
            stance_base_vel = -1
            feet_contact_forces = -0.01








class H1_2_WholeBodyCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "OnPolicyRunnerLower" 
    class policy:
        init_noise_std = 0.8

        actor_hidden_dims = [64, 32]
        critic_hidden_dims = [64, 32]
       
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        
    class runner( LeggedRobotCfgPPO.runner ):

        policy_class_name = "RNNActorMLPCritic"
        algorithm_class_name = "PPO"

        upper_policy_path = f"{LEGGED_GYM_ROOT_DIR}/logs/h1_2_upper/Jul22_06-23-37_upper_body_iteration2/model_10000.pt" # change to your upper body policy path!


        max_iterations = 10000#0
        save_interval = 100
        run_name = ''
        experiment_name = 'h1_2_lower'


