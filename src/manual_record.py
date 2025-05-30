from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
import pandas as pd
import time
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.lange_change_policy import LaneChangePolicy
from policy import *
from metadrive.policy.manual_control_policy import *
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.manager.traffic_manager import TrafficMode
import numpy as np
import pickle
import os, random
import tqdm
sensor_size = (200, 66)

def generate_cfg():
    # generate map string. 
    '''
    C: Circular
    S: Straight
    '''
    N = 10
    map_str = "".join(random.sample(["C", "S"], k=N, counts=[N*4, N] ))
    print("Map string: ", map_str)

    cfg=dict(image_observation=True, 
            # OBSERVATION
            vehicle_config=dict(image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, *sensor_size)},
            stack_size=3,
            agent_policy=None,
            manual_control=True,
            controller="steering_wheel",
            use_render=True,

            # PROCEDURAL GENERATION MAP
            map=map_str,
            block_dist_config=PGBlockDistConfig,
            random_lane_width=True,
            random_lane_num=True,
            store_map=True,

            # TRAFFIC
            traffic_mode=TrafficMode.Trigger,
            traffic_density=0.01, # increase for more dataset diversity

            # RANDOMIZATION
            num_scenarios=1000,
            start_seed=random.randint(0, 1000), # random seed (it's saved in the config.pkl)
    )

    return cfg

if __name__ == "__main__":
    env=MetaDriveEnv(config=(cfg:=generate_cfg()))

    # data collection in data/ (scratch)
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

    env.reset()

    try:
        for i in tqdm.tqdm(range(20000)):
            # simulation
            # calculate throttle from IDM policy
            # output of IDM act is [steer, throttle]
            # throttle = abs((_action:=throttle_policy.act())[1]) # throttle is always positive
            # throttle = min(3, throttle) # cap throttle at 3

            obs, rew, terminated, truncated, info = env.step(
                [0, 1] # 0 means don't change lanes
            )

            # render
            env.render(window=True)

            # print(_action); input()

            # print(info); input()
            # print(info)
            # rendering, the last one is the current frame
            ret=obs["image"][..., -1] * 255 # the data *should* be 0-1
            ret=ret.astype(np.uint8)

            # random hflip (also update steer)
            # if random.random() > 0.5:
            #     ret = cv2.flip(ret, 1)
            #     info["action"][1] = -info["action"][1]

            # save image
            cv2.imwrite(img_path:=os.path.join(curr_session, f"img_{i}.png"), ret)

            # update data (throttle, steer)
            data.loc[i] = [img_path, *info["action"]]
            # print(data.loc[i]); input()

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