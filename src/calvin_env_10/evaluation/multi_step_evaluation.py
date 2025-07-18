from collections import defaultdict
from omegaconf import OmegaConf
import hydra
import time 
from tqdm import tqdm
import numpy as np

from calvin_env_10.evaluation.sequences import get_sequences
from calvin_env_10.evaluation.utils import get_log_dir, count_success, get_env_state_for_initial_condition
from calvin_env_10.envs.play_table_env import get_env

import cv2
import os
from pathlib import Path

EP_LEN = 360
NUM_SEQUENCES = 1000
BASE_DIR = Path(__file__).parent.parent

def evaluate_policy(model, env, step_size=16, eval_log_dir=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """

    task_cfg = OmegaConf.load(os.path.join(BASE_DIR, "conf/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(os.path.join(BASE_DIR, "conf/annotations/new_playtable_validation.yaml"))

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []


    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        result, output = evaluate_sequence(env, model, step_size, task_oracle, initial_state, eval_sequence, val_annotations)
        results.append(result)
        create_trajectory(output[0], output[1], f"{eval_log_dir}/trajectory_{i}.mp4")
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )


    return results

def create_trajectory(trajectories, tasks, video_path):
    """
    Create a video from trajectory data and save it to log_dir.
    
    Args:
        trajectories: List of lists, where each inner list contains numpy arrays (RGB images)
        tasks: List of strings describing the actions performed
        log_dir: Directory to save the video
    """

    
    
    # Get video dimensions from first image
    first_img = trajectories[0][0] if trajectories[0] else None
    print(first_img.shape)
    height, width = first_img.shape[:2]
    
    # Create video writer with dimensions that include the white text area
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # 10 frames per second
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height + 50))
    
    try:
        for task_idx, (task_trajectory, task_description) in enumerate(zip(trajectories, tasks)):
            print(f"Processing task {task_idx + 1}/{len(tasks)}: {task_description}")
            
            # Add task description text for all frames of each task
            for frame_idx, frame in enumerate(task_trajectory):
                frame = np.concatenate([frame, np.ones((50, width, 3))*255], axis=0)
                
                # Ensure frame is in the correct format for OpenCV
                if frame.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if frame.shape[2] == 3:
                    # If it's RGB, convert to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Add task description text for all frames of each task
                # Create a copy to avoid modifying original data
                frame_with_text = frame_bgr.copy()
                
                # Add text in black color in the white lower section
                cv2.putText(frame_with_text, f"Task {task_idx + 1}: {task_description}", 
                          (20, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                
               
                video_writer.write(frame_with_text)
        
        print(f"Video saved to: {video_path}")
        
    finally:
        video_writer.release()
    
    return str(video_path)

def evaluate_sequence(env, model, step_size, task_checker, initial_state, eval_sequence, val_annotations):
    """
    Evaluates a sequence of language instructions.
    """
    print(initial_state)
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    tasks = []
    trajectories = []
    for subtask in eval_sequence:
        tasks.append(subtask)
        success, obss = rollout(env, model, step_size, task_checker, subtask, val_annotations)
        trajectories.append(obss)
        if success:
            success_counter += 1
        else:
            return success_counter, (trajectories,tasks)
    
    return success_counter, (trajectories,tasks)

def rollout(env, model, step_size, task_oracle, subtask, val_annotations):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """

    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    step = 0
    obss = []
    while step < EP_LEN:
        action = model.step(obs, lang_annotation)
        for s in range(step_size):
            obs, _, _, current_info = env.step(action[s])
            step += 1
            obss.append(obs["rgb_obs"]["rgb_static"])

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, obss
        
    return False, obss

class TestPolicy():

    def __init__(self, step_size):
        self.step_size = step_size

    def reset(self):
        pass

    def step(self, obs, lang_annotation):
        actions = np.zeros((self.step_size,7))
        actions[:,-1] = np.where(actions[:,-1]>=0,1,-1)
        return actions

if __name__ == "__main__":
    env = get_env("task_D_D", show_gui=False)
    step_size = 16
    policy = TestPolicy(step_size)

    evaluate_policy(policy,env,step_size,eval_log_dir="eval_log")