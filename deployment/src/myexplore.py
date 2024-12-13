import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml
from PIL import Image as PILImage
import numpy as np
import argparse
import time

# UTILS
from my_utils import to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main(args: argparse.Namespace):
    global context_size

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Instead of ROS, load images manually for context
    # For example, loading a series of images from a folder or using single input image
    # (For simplicity, we assume these are loaded images)
    # Manually providing context images as input:
    
    # Load your input images
    #image_paths = args.image_paths  # Example: ['image1.jpg', 'image2.jpg', ...]
    image_paths = ['/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test1.jpg',\
                   '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test2.jpg',\
                   '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test3.jpg',\
                   '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test4.jpg',\
                    ]
    context_queue = []
    for image_path in image_paths:
        img = PILImage.open(image_path).convert("RGB")
        context_queue.append(img)

    # Start exploring
    waypoint_msg = []
    if len(context_queue) > model_params["context_size"]:
        obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
        obs_images = obs_images.to(device)
        fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
        mask = torch.ones(1).long().to(device) # ignore the goal

        # infer action
        with torch.no_grad():
            # encoder vision features
            obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            
            # (B, obs_horizon * obs_dim)
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
            
            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (args.num_samples, model_params["len_traj_pred"], 2), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            start_time = time.time()
            for k in noise_scheduler.timesteps[:]:
                # predict noise
                noise_pred = model(
                    'noise_pred_net',
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            print("time elapsed:", time.time() - start_time)

        naction = to_numpy(get_action(naction))

        # Here, you'd process the action as required
        print("Sampled action:", naction)

        # For now, just print the waypoint as chosen from the action
        naction = naction[0]  # Change based on heuristics or requirements
        chosen_waypoint = naction[args.waypoint]

        if model_params["normalize"]:
            chosen_waypoint *= (MAX_V / RATE)
        waypoint_msg = chosen_waypoint
        print(f"Chosen waypoint: {waypoint_msg}")

    else:
        print("Not enough context images to proceed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION without ROS")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # Example: using middle waypoint
        type=int,
        help="index of the waypoint used for navigation (default: 2)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--image-paths",
        "-i",
        nargs="+",
        type=str,
        help="List of image paths to provide as input for the model",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
