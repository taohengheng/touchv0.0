import argparse
import time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



from multi_task.manager_based.bunker_quickly_grasp.robot_env_cfg import RobotEnvCfg_PLAY
import torch
from isaaclab.envs import ManagerBasedRLEnv



def main():

    env_cfg = RobotEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # set seed of the environment
    env.seed(666)
    
    count = 1
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()          
                print("-"*80)
                print("[INFO]: Resetting environment...")
            #  joint_efforts赋值的顺序取决于env.action_manager._terms的顺序(它是一个字典,依次遍历键值进行赋值)
            #  按照env.action_manager._terms.keys()顺序： dict_keys(['bunker_action_left', 'bunker_action_right', 'cr5', 'leaphand'])

            joint_efforts = torch.zeros_like(env.action_manager.action)
            # joint_efforts[:, 0] = 20
            # joint_efforts[:, 1] = 20
            # joint_efforts[:, 4] = -1.57
            # joint_efforts[:, 6] = -1.57
            # joint_efforts[:, 7] = 3.14
            # joint_efforts[:, 8] = 0.785
            # joint_efforts[:, 12] = 1.57
            # print(env.scene["ee_frame"].data.target_quat_w.squeeze(1))
            obs, rew, terminated, truncated, info = env.step(joint_efforts)    
            #print('aaa', obs['policy']['object_pos_w_6'])
            # print(env.scene["link1_frame"].data.target_pos_w)
            #print(env.scene["ee_frame"].data.target_pos_w.squeeze(1))
            count += 1 
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
 



























