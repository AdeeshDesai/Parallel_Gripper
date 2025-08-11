import sys
# sys.path.append('/home/adeesh/Projects/diffusion_policy')

from TVB.envs.env_wrapper import IsaacEnvWrapper
# from TVB.utils.shear_tactile_viz_utils import visualize_tactile_shear_image, visualize_penetration_depth
from diffusion_policy.common.replay_buffer import ReplayBuffer
from TVB.utils.teleop_utils.spacemouse import SpaceMouse
from TVB.utils.input_utils import input2action
import numpy as np
import cv2
import hydra
import threading
from omegaconf import DictConfig
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import matplotlib.pyplot as plt
plt.ion()
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ------------- Reward Plotting ---------------

fig, ax = plt.subplots()
reward_history = []
reward_boundary_max = 0.1
reward_boundary_min = -0.6
check_lower = -0.
check_upper = -0.
print_time_step = 1000
print_time_step = round(print_time_step)
x_data = list(range(print_time_step))
line, = ax.plot(x_data, [0]*print_time_step, 'bo-', markersize=5)
ax.set_xlim(0, print_time_step - 1)
ax.set_ylim(reward_boundary_min, reward_boundary_max)
ax.set_xlabel("Time step")
ax.set_ylabel("Reward")

def update_plot(reward):
    reward_value = reward[0].cpu().numpy().item()
    reward_value = np.clip(reward_value, reward_boundary_min, reward_boundary_max)
    reward_value = round(reward_value, 6)
    if len(reward_history) >= print_time_step:
        reward_history.pop(0)
    reward_history.append(reward_value)
    ax.clear()
    ax.set_xlim(0, print_time_step - 1)
    ax.set_ylim(reward_boundary_min, reward_boundary_max)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reward")
    for i, r in enumerate(reward_history):
        color = 'r' if check_lower <= r <= check_upper else 'b'
        ax.plot(i, r, f'{color}o', markersize=5)
    plt.draw()
    plt.pause(0.1)

# ------------ Gripper Control / Keyboard -----------

from pynput import keyboard
import time

gripper_action = 0.05
gripper_min = 0.0
gripper_max = 0.05
gripper_step = 0.001
left_pressed = False
right_pressed = False

def on_press(key):
    global left_pressed, right_pressed
    try:
        if key == keyboard.Key.left:
            left_pressed = True
        elif key == keyboard.Key.right:
            right_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global left_pressed, right_pressed
    if key == keyboard.Key.left:
        left_pressed = False
    elif key == keyboard.Key.right:
        right_pressed = False

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

def gripper_control_loop():
    global gripper_action
    while True:
        if left_pressed:
            gripper_action = max(gripper_min, gripper_action - gripper_step)
            print(f"🔒 gripper close: {gripper_action:.3f}")
        if right_pressed:
            gripper_action = min(gripper_max, gripper_action + gripper_step)
            print(f"🔨 gripper open: {gripper_action:.3f}")
        time.sleep(0.1)

# ------------- ROS Image Publishers -------------

rospy.init_node('image_publisher', anonymous=True)
image_pub_front = rospy.Publisher('/camera/front_image', Image, queue_size=10)
bridge = CvBridge()

# ------------ Action Getter ------------

def get_action(space_mouse):
    d_action, _ = input2action(space_mouse)
    # d_pos = d_action[0:3]
    d_pos = np.array([0,0,-0.3])
    d_rpy = d_action[3:6]
    action = np.concatenate([d_pos, d_rpy, [gripper_action]])
    return action

# ----- Listen for Keyboard Input (threaded) ------

done = False
retry = False

def listen_for_input():
    global done
    global retry
    while True:
        user_input = input() # Wait for User's input
        if user_input.lower() == 'd':
            done = True
            print("Done the current demo")
        elif user_input.lower() == 'r':
            print("Retry the current demo")
            retry = True

def display_demo_controls():
    def print_command(char, info):
        char += " " * (30 - len(char))
        print("{}\t{}".format(char, info))
    print_command("Press", "Command")
    print_command("d", "finish one demo/epoch per task")
    print_command("r", "retry the current demo/epoch")
    print_command("p", "pause the current demo")
    print_command("ESC", "quit")
    print("")

@hydra.main(version_base="1.1", config_path="../config", config_name="isaacgym_config_gui")
def main(cfg: DictConfig):
    global gripper_action
    gripper_thread = threading.Thread(target=gripper_control_loop, daemon=True)
    gripper_thread.start()

    env = IsaacEnvWrapper(cfg)
    # ----- Set path for saving data -----
    output = '../../data/cube1'
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    space_mouse = SpaceMouse(vendor_id=9583, product_id=50741, pos_sensitivity=1.0, rot_sensitivity=1.0)
    space_mouse.start_control()

    display_demo_controls()

    # Keyboard thread for 'd' and 'r'
    input_thread = threading.Thread(target=listen_for_input)
    input_thread.daemon = True
    input_thread.start()

    global retry
    global done

    for _ in range(256):
        episode = list()
        seed = replay_buffer.n_episodes
        print(f'⛳ Starting seed {seed}')
        env.seed(seed)
        obs = env.reset()
        retry = False
        done = False
        gripper_action = 0.05
        leap_actions = np.zeros((16,), dtype=np.float32)    
        while not done:
            if retry:
                break
            front_rgb_image = obs['front'][0].cpu().numpy()
            ee_pos = obs['ee_pos'][0].cpu().numpy()
            ee_quat = obs['ee_quat'][0].cpu().numpy()
            state = np.concatenate([ee_pos, ee_quat])
            actions = get_action(space_mouse=space_mouse)
            # print(actions)
            # actions = np.array([0,0,0,0,0,0], dtype=np.float32)
            # leap_actions+=[0.0025]*16
            # actions = np.concatenate([actions, leap_actions], axis=0)
            data = {
                'front': front_rgb_image,
                'state': state,
                'action': np.float32(actions),
            }
            # print(data["state"])
            # print(actions)
            episode.append(data)
            obs, reward, reset, info = env.step(actions)
            # print(reward)
            # Plot reward (optional)
            update_plot(reward)

            # Convert and publish image as uint8
            front_rgb_image = (front_rgb_image * 255).astype(np.uint8)
            ros_front = bridge.cv2_to_imgmsg(front_rgb_image, encoding="bgr8")
            image_pub_front.publish(ros_front)
            cv2.imshow("Front", cv2.cvtColor(front_rgb_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if not retry:
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')

if __name__ == '__main__':
    start_keyboard_listener()
    main()
