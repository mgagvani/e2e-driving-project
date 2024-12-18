from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy import *
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.policy.idm_policy import IDMPolicy  # Or use ExpertPolicy
# ...existing code...

class HybridPolicy(EnvInputPolicy):
    def __init__(self, control_object, random_seed):
        super(HybridPolicy, self).__init__(control_object, random_seed)
        self.lane_change_policy = LaneChangePolicy(control_object, random_seed)
        self.idm_policy = IDMPolicy(control_object, random_seed)
        # If you prefer to use ExpertPolicy:
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

