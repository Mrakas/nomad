import os
import argparse
import yaml
import torch
import numpy as np
from typing import List, Optional
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import sys
sys.path.append('/home/marcus/workplace/nav_models/visualnav-transformer/deployment')
sys.path.append('/home/marcus/workplace/nav_models/visualnav-transformer/deployment/src')
# Assuming these are custom utility modules from the original script
from my_utils import to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action

class VisualNavigationAPI:
    def __init__(self, model_name: str = "nomad"):
        # Load configuration paths
        ROBOT_CONFIG_PATH = "../config/robot.yaml"
        MODEL_CONFIG_PATH = "../config/models.yaml"

        # Load robot configuration
        with open(ROBOT_CONFIG_PATH, "r") as f:
            robot_config = yaml.safe_load(f)
        
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.RATE = robot_config["frame_rate"]

        # Load model configuration
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        # Select model configuration
        model_config_path = model_paths[model_name]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        ckpth_path = model_paths[model_name]["ckpt_path"]
        if not os.path.exists(ckpth_path):
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

        self.model = load_model(ckpth_path, self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize noise scheduler
        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.context_queue = []

    def navigate(self,
                 single_img = None,# priority
                 num_samples: int = 8, 
                 waypoint_index: int = 2) -> np.ndarray:
        """
        Main navigation method to generate navigation actions.

        Args:
            context_images (List[str]): List of image paths for context
            num_samples (int, optional): Number of action samples. Defaults to 8.
            waypoint_index (int, optional): Waypoint to use for navigation. Defaults to 2.

        Returns:
            np.ndarray: Chosen navigation waypoint
        """
        # Check context size
        
        # Load and transform images
        
        if len(context_queue) < 4:
            context_queue = [single_img,single_img,single_img,single_img] # 4imgs !!
        else:
            context_queue = context_queue[1:] + [single_img]

        obs_images = transform_images(context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = obs_images.to(self.device)

        # Prepare goal and mask
        fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
        mask = torch.ones(1).long().to(self.device)

        # Inference
        with torch.no_grad():
            # Encode vision features
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            
            # Repeat condition for multiple samples
            obs_cond = obs_cond.repeat(num_samples, 1) if len(obs_cond.shape) == 2 else obs_cond.repeat(num_samples, 1, 1)
            
            # Initialize noisy action
            noisy_action = torch.randn(
                (num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
            naction = noisy_action

            # Run noise scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.model(
                    'noise_pred_net',
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # Process actions
        naction = to_numpy(get_action(naction))
        naction = naction[0]  # Select first sample
        chosen_waypoint = naction[waypoint_index]

        # Normalize if required
        if self.model_params.get("normalize", False):
            chosen_waypoint *= (self.MAX_V / self.RATE)

        return chosen_waypoint

def main():
    parser = argparse.ArgumentParser(description="Visual Navigation API Client")
    parser.add_argument("--images", "-i", nargs="+", required=True, help="Paths to context images")
    parser.add_argument("--model", "-m", default="nomad", help="Model name")
    parser.add_argument("--waypoint", "-w", type=int, default=2, help="Waypoint index")
    parser.add_argument("--samples", "-n", type=int, default=8, help="Number of samples")
    
    args = parser.parse_args()
    
    try:
        nav_api = VisualNavigationAPI(model_name=args.model)
        result = nav_api.navigate(
            num_samples=args.samples, 
            waypoint_index=args.waypoint
        )
        print(f"Chosen Waypoint: {result}")
    except Exception as e:
        print(f"Navigation failed: {e}")

if __name__ == "__main__":
    main()