import numpy as np
import os
import cv2
from PIL import Image
from datetime import datetime
from script.eval_3dpolicy import Env
import sys
sys.path.append('./') 

# Create image save directory
def create_image_dir(task_name, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = f"./observation_images/{task_name}_{seed}_{timestamp}"
    os.makedirs(image_dir, exist_ok=True)
    return image_dir

# Save image function
def save_image(image_array, save_dir, step, camera_name="front_camera"):
    """Save numpy format image to file"""
    # Ensure correct image format (RGB format numpy array)
    if image_array.dtype != np.uint8:
        # If float type and range is [0,1], convert to [0,255]
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Save using OpenCV (need to convert RGB to BGR)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(save_dir, f"{camera_name}_step_{step:04d}.png")
    cv2.imwrite(image_path, image_bgr)
    return image_path

# 1. Create environment manager instance
env_manager = Env()

# 2. Create specific task environment and validate seeds
task_name = "block_hammer_beat"
seed = 42
task_num = 5

seed_list, id_list = env_manager.Create_env(
    task_name=task_name,
    head_camera_type="D435", 
    seed=seed,
    task_num=task_num
)

print(f"Found {len(seed_list)} valid task seeds: {seed_list}")

# 3. Run tasks for each valid seed
for i, (seed, task_id) in enumerate(zip(seed_list, id_list)):
    print(f"\nExecuting task {i+1}/{len(seed_list)}, seed: {seed}")
    
    # Create image save directory for current task
    image_dir = create_image_dir(task_name, seed)
    print(f"Images will be saved to: {image_dir}")
    
    # Initialize task environment
    env_manager.Init_task_env(seed, task_id)
    
    # Run task loop
    max_steps = 1000
    for step in range(max_steps):
        # Get observation
        observation = env_manager.get_observation()
        
        # Extract and save images from different cameras
        try:
            # Front camera image
            if 'front_camera' in observation['observation'] and 'rgb' in observation['observation']['front_camera']:
                front_image = observation['observation']['front_camera']['rgb']
                front_path = save_image(front_image, image_dir, step, "front_camera")
                print(f"Saved front camera image: {front_path}")
            
            # Head camera image (if exists)
            if 'head_camera' in observation['observation'] and 'rgb' in observation['observation']['head_camera']:
                head_image = observation['observation']['head_camera']['rgb']
                head_path = save_image(head_image, image_dir, step, "head_camera")
                print(f"Saved head camera image: {head_path}")
            
            # Wrist camera image (if exists)
            if 'wrist_camera' in observation['observation'] and 'rgb' in observation['observation']['wrist_camera']:
                wrist_image = observation['observation']['wrist_camera']['rgb']
                wrist_path = save_image(wrist_image, image_dir, step, "wrist_camera")
                print(f"Saved wrist camera image: {wrist_path}")
            
        except KeyError as e:
            print(f"Warning: Unable to access image data - {e}")
            # Print observation structure for debugging
            print(f"Observation data structure: {list(observation.keys())}")
            if 'observation' in observation:
                print(f"Camera data structure: {list(observation['observation'].keys())}")
        
        # Calculate action based on observation (using random action as substitute)
        action = np.random.uniform(-1, 1, 16)  # 16-dimensional action space
        
        # Fix: Ensure action is in numpy array format
        action = np.array(action)
        
        # Execute action and get status
        status = env_manager.Take_action(action)
        print(f"Step {step}: status = {status}")
        
        # Exit loop if task is completed or failed
        if status != "run":
            break
    
    # Ensure environment is closed
    if status == "run":
        env_manager.Close_env()

print("\nAll tasks completed, images have been saved.")