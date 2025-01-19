from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy import *
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.idm_policy import IDMPolicy  # Or use ExpertPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.obs.image_obs import *
from metadrive.engine.engine_utils import *

import time

import torch
from models import PilotNet

class HybridPolicy(EnvInputPolicy):
    def __init__(self, control_object, random_seed):
        super(HybridPolicy, self).__init__(control_object, random_seed)
        self.lane_change_policy = LaneChangePolicy(control_object, random_seed)
        self.idm_policy = IDMPolicy(control_object, random_seed)
        # self.expert_policy = ExpertPolicy(control_object, random_seed)

    def act(self, agent_id):
        # Get steering from LaneChangePolicy
        steering = self.lane_change_policy.act(agent_id)[0]
        # Get throttle from IDMPolicy
        throttle = self.idm_policy.act(agent_id)[1]
        # If using ExpertPolicy:
        # throttle = self.expert_policy.act(agent_id)[1]
        # Combine the actions
        action = [steering, throttle]
        self.action_info["action"] = action
        return action
    
class ModelPolicy(EnvInputPolicy):
    '''
    Note self.control_object is an instance of DefaultVehicle
    '''
    def __init__(self, control_object, random_seed):
        super(ModelPolicy, self).__init__(control_object, random_seed)
        self.model = PilotNet()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
        # fix config
        # self.image_obs = ImageStateObservation(control_object.config)
        self.e = get_engine()
        self.timer = time.time()

    def throttle_control(self, model_output):
        if time.time() - self.timer > 2.0:
            return 0.3
        else:
            return model_output[1]
            
    def act(self, agent_id):
        # Get the observation from the environment
        # print(type(self.control_object))
        # print(self.control_object.__dir__())
        # obs = self.image_obs.img_obs.get_image()
        obs = self.e.get_sensor("rgb_camera").perceive(True, None, None, None)
        # print(obs.shape)
        # obs is (66, 200, 3). need to convert to (3, 66, 200)
        image = torch.Tensor(obs).permute(2, 0, 1).unsqueeze(0)
        # Get the action from the model
        action = self.model(image).squeeze(0).detach().numpy()
        # throttle
        action[1] = self.throttle_control(action)
        action = [action[0], action[1]]
        print(action)
        self.action_info["action"] = action
        return action

class HybridManualPolicy(EnvInputPolicy):
    def __init__(self, control_object, random_seed):
        super(HybridManualPolicy, self).__init__(control_object, random_seed)
        # self.idm_policy = IDMPolicy(control_object, random_seed)
        self.expert_policy = ExpertPolicy(control_object, random_seed)
        self.manual_control = ManualControlPolicy(control_object, random_seed)

    def act(self, agent_id):
        # get steering from manual control
        steering = self.manual_control.act(agent_id)[0]
        # get throttle from expert policy
        throttle = self.expert_policy.act(agent_id)[1]
        action = [steering, throttle]
        self.action_info["action"] = action
        return action