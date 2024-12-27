from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
import pandas as pd
import time
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config
from metadrive.utils import generate_gif
import numpy as np
from moviepy import *
import pickle
import os, random

import torch
from models import PilotNet

sensor_size = (200, 66)


N = 8
map_str = "".join(random.sample(["C", "S"], k=N, counts=[N*10, N] ))
print("Map string: ", map_str)

cfg = dict(image_observation=True, 
            # OBSERVATION
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=1,

            # PROCEDURAL GENERATION MAP
            map=map_str,
            block_dist_config=PGBlockDistConfig,
            random_lane_width=True,
            random_lane_num=True,
            store_map=True,

            # TRAFFIC
            traffic_mode=TrafficMode.Trigger,
            traffic_density=0.025,

            # RANDOMIZATION
            num_scenarios=1000,
            start_seed=10
)

if __name__ == "__main__":
    # load model
    model = PilotNet()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    env = MetaDriveEnv(config=cfg)
    env.reset()

    obs, rew, term, trunc, info = env.step([0, 0])
    print("Starting simulation")

    images = []

    iter = 0
    stop_count = 10
    while True:
        # get image
        image = obs["image"] 

        # render
        env.render(window=True)

        # infer
        # note that model was trained with (3, 66, 200) images
        image = torch.Tensor(image).permute(3, 2, 0, 1) # (200, 66, 3) -> (3, 66, 200) (correct)
        out = model(image)
        out = out.squeeze(0).detach().numpy()
        images.append(image.to("cpu").detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255.)
        

        # actuate
        new_out = (out[0], 0.5) # (steering, throttle)
        obs, reward, term, trunc, info = env.step(out)
        done = term or trunc

        iter += 1
        print(f"Step {iter}", end="\r")
        if done or iter > 20000:
            print(f"Done in {iter} iterations. {stop_count} more to go.")
            env.reset()
            images.append(np.zeros((66, 200, 3)))

            stop_count -= 1
            if stop_count == 0:
                break
    env.close()

    # create video
    clip = ImageSequenceClip(images, fps=20)
    clip.write_videofile("eval_video.mp4")

    