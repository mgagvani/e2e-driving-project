from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
import pandas as pd
import time
from metadrive.policy.expert_policy import ExpertPolicy
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
import pickle
import os, random
import tqdm
sensor_size = (200, 66)

# generate map string. 
'''
C: Circular
S: Straight
'''
N = 8
map_str = "".join(random.sample(["C", "S"], k=N, counts=[N*10, N] ))
print("Map string: ", map_str)

cfg=dict(image_observation=True, 
        # OBSERVATION
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=3,
        agent_policy=ExpertPolicy, # drive with IDM policy

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
        start_seed=random.randint(0, 1000), # random seed (it's saved in the config.pkl)
)

if __name__ == "__main__":
    env=MetaDriveEnv(cfg)

    # data collection in data/ 
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    curr_session = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # make folder
    curr_session = os.path.join(data_path, curr_session)
    if not os.path.exists(curr_session):
        os.makedirs(curr_session)

    print("Current session: ", curr_session)
   
    # create data CSV
    data_csv = os.path.join(curr_session, "data.csv")
    data = pd.DataFrame(columns=["img_path", "throttle", "steer"])

    try:
        env.reset()
        for i in tqdm.tqdm(range(250)):
            # simulation
            obs, rew, terminated, truncated, info = env.step([0, 1])
            # print(info)
            # rendering, the last one is the current frame
            ret=obs["image"][..., -1]*255 # [0., 1.] to [0, 255]
            ret=ret.astype(np.uint8)

            # save image
            cv2.imwrite(img_path:=os.path.join(curr_session, f"img_{i}.png"), ret)

            # update data (steer, throttle)
            data.loc[i] = [img_path, *info["action"]]

            if terminated: # keep on going to get to 1000 samples
                env.reset()
    finally:
        # save data
        data.to_csv(data_csv, index=False)
        print("Data saved to ", data_csv)

        # save config
        with open(os.path.join(curr_session, "config.pkl"), "wb") as f:
            pickle.dump(cfg, f)

        # close environment
        env.close()