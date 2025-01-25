import carla
import os, math, random, sys, time
import carla.libcarla
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from agents.navigation import controller

from models import MultiCamWaypointNet, WaypointNet
import torch

WAYPOINT_DIST = 2.0
TOWN_NAME = 'Town01'
TIMESTEP = 1 / 30.0 
N_GOALS = 100
GOAL_THRESH = 4.0
N_EXTRAS = 100

curr_frame = np.zeros((168, 3*224, 3), dtype=np.uint8)
curr_collision = False

class Prediction:
    def __init__(self, pred, vehicle_transform):
        pred = pred.squeeze()
        self.throttle = pred[1]
        self.steer = pred[0]

        self.waypoints = []
        for i in range(2, len(pred), 2):
            self.waypoints.append((pred[i], pred[i+1]))

        self.vehicle_transform = vehicle_transform

    def generate_control_waypoint(self, map):
        local_x, local_y = self.waypoints[0]
        yaw_rad = math.radians(self.vehicle_transform.rotation.yaw)
        cos_t = math.cos(yaw_rad)
        sin_t = math.sin(yaw_rad)
        global_x = self.vehicle_transform.location.x + cos_t*local_x - sin_t*local_y
        global_y = self.vehicle_transform.location.y + sin_t*local_x + cos_t*local_y

        loc = carla.Location(x=float(global_x), 
                             y=float(global_y), 
                             z=float(self.vehicle_transform.location.z))
        waypoint = map.get_waypoint(loc, project_to_road=False)
        if waypoint is None:
            waypoint = map.get_waypoint(loc, project_to_road=True)
            # print("waypoint is none")
            return waypoint, False
        # print(f"Control waypoint: {waypoint.transform.location}")
        return waypoint, True

@torch.no_grad()
def model_predict(model):
    # get image
    global curr_frame
    img = torch.from_numpy(curr_frame).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
    img /= 255.0
    # crop just the left camera
    # img = img[:, :, :, :224] # (B, C, H, W)
    # crop just the center camera
    # img = img[:, :, :, 224:448]
    img = img[:, :, :, 448:]
    # get prediction
    pred = model.forward(img)
    return pred

def camera_callback(image, cam_index):
    # Convert to a NumPy array for visualization
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    new_arr = np.zeros((600, 800, 3), dtype=np.uint8)
    new_arr[:, :, :] = arr[:, :, :3]

    # resize to 224x168
    new_arr = cv2.resize(new_arr, (224, 168), interpolation=cv2.INTER_CUBIC)
    
    # update section of global variable
    global curr_frame
    if cam_index == 0:
        curr_frame[:, :224, :] = new_arr
    elif cam_index == 1:
        curr_frame[:, 224:448, :] = new_arr
    elif cam_index == 2:
        curr_frame[:, 448:, :] = new_arr

def collision_callback(event):
    print(event)
    global curr_collision
    curr_collision = time.time()


def get_transform_matrix(x, y, yaw_deg):
    # Build a 4x4 homogeneous transform for (x,y,θ)
    theta = math.radians(yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array([
        [ cos_t, -sin_t, 0, x ],
        [ sin_t,  cos_t, 0, y ],
        [   0,      0,   1, 0 ],
        [   0,      0,   0, 1 ]
    ], dtype=np.float32)

def clip(val, lo, hi):
    return max(lo, min(hi, val))

def set_random_weather(world: carla.World):
    all_weather = [getattr(carla.WeatherParameters, x) for x in dir(carla.WeatherParameters) if '__' not in x]
    weather = random.choice(all_weather)
    try:
        # make sure it's possible to see
        weather.fog_density = clip(weather.fog_density, 0.0, 5.0)
        weather.fog_distance = clip(weather.fog_distance, 0.0, 10.0)
        weather.sun_altitude_angle = clip(weather.sun_altitude_angle, 10.0, 170.0)
        # print(f"Setting weather to {weather}")
        world.set_weather(weather)
    except Exception as e:
        print(f"Error setting weather: {e}")

def classify_turn(angvel_z):
    THRESH = 2.0
    if angvel_z > THRESH:
        return "left"
    elif angvel_z < -THRESH:
        return "right"
    else:
        return "straight"
    
def Vec3d_norm(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        do_viz = False
    else:
        do_viz = args[0] == "viz"

    # instantiate model
    model = WaypointNet()
    model.load_state_dict(torch.load('model_r.pth')) # and model_r.pth
    model.to('cuda')
    model.eval()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world = client.load_world(TOWN_NAME)
    blueprint_lib = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(8000)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TIMESTEP
    world.apply_settings(settings)

    # Spawn ego vehicle
    vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle_controller = controller.VehiclePIDController(
        ego_vehicle, 
        args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': TIMESTEP},
        args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': TIMESTEP}
    )
    # ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    # traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0) # dont bother with this

    # set random weather
    set_random_weather(world)
    world.tick()

    # Define camera transforms (left, center, right)
    camera_transforms = [
        carla.Transform(carla.Location(x=1.5, y=-0.5, z=1.7), carla.Rotation(yaw=-45)),
        carla.Transform(carla.Location(x=1.5, y=0.0, z=1.7), carla.Rotation(yaw=0)),
        carla.Transform(carla.Location(x=1.5, y=0.5,  z=1.7), carla.Rotation(yaw=45))
    ]
    cameras = []
    for idx, transform in enumerate(camera_transforms):
        cam_bp = blueprint_lib.find('sensor.camera.rgb')
        cam = world.spawn_actor(cam_bp, transform, attach_to=ego_vehicle)
        cam.listen(lambda img, index=idx: camera_callback(img, index))
        cameras.append(cam)

    # collision sensor
    collision_bp = blueprint_lib.find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
    collision.listen(lambda event: collision_callback(event))

    # spawn extra cars to make it harder
    print(f"There are {len(world.get_map().get_spawn_points())} spawn points")
    spawn_choices = random.choices(world.get_map().get_spawn_points(), k=N_EXTRAS)
    for _ in range(N_EXTRAS):
        vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*'))
        spawn_point = spawn_choices.pop()
        _car = world.try_spawn_actor(vehicle_bp, spawn_point)
        if _car is not None:
            _car.set_autopilot(True, traffic_manager.get_port())

    if do_viz:
        plt.ion()
        fig_cam, axs_cam = plt.subplots(1, 2)
        ax_way = axs_cam[1]
        ax_cam = axs_cam[0]
        ax_cam.set_aspect('equal', adjustable=None)
        ax_cam = ax_cam.imshow(np.ones((168, 3*224, 3), dtype=np.uint8))

    last_reset = 0
    all_crashes = dict()

    try:
        for frame in range(50_000):
            # step sim - do first to avoid confusion
            world.tick()

            # globals 
            global curr_frame
            global curr_collision

            # get model output
            pred = model_predict(model).cpu().detach().numpy()
            _p_class = Prediction(pred, ego_vehicle.get_transform())
            waypoint, on_road = _p_class.generate_control_waypoint(world.get_map())
            pred = pred[0]
            next_waypoints = [
                [pred[2], pred[3]],
                [pred[4], pred[5]],
                [pred[6], pred[7]]
            ]
            # print("- - - - - - - - - - - - - - - - - -")
            # for i, _wp in enumerate(next_waypoints):
            #     print(f"Waypoint {i}: {_wp}")
            control = vehicle_controller.run_step(30.0, waypoint) 
            # print(f"Gen. Control: {control.steer}, {control.throttle}")
            # print(f"Pred Control: {pred[0]}, {pred[1]}")
            ego_vehicle.apply_control(control) 

            # check we aren't stuck
            ego_vel = ego_vehicle.get_velocity()
            collision = time.time() - curr_collision < 0.5
            off_road = not on_road
            if Vec3d_norm(ego_vel) < 2.0 and last_reset + 100 < frame:
                print(f"Vehicle is stuck, resetting at frame {frame}")
                ego_vehicle.set_transform(random.choice(world.get_map().get_spawn_points()))
                set_random_weather(world)         
                last_reset = frame   
                all_crashes[frame] = "stuck"
            elif collision:
                print(f"Collision detected at frame {frame}")
                ego_vehicle.set_transform(random.choice(world.get_map().get_spawn_points()))
                set_random_weather(world)
                last_reset = frame
                all_crashes[frame] = "collision"
            elif off_road:
                print(f"Off road detected at frame {frame}")
                ego_vehicle.set_transform(random.choice(world.get_map().get_spawn_points()))
                set_random_weather(world)
                last_reset = frame
                all_crashes[frame] = "off_road"

            # Visualization
            if do_viz:
                ax_way.clear()
                ax_way.set_aspect('equal', adjustable=None)
                min_x, min_y, max_x, max_y = -7, -1, 7, 7
                ax_way.set_xbound(min_x, max_x)
                ax_way.set_ybound(min_y, max_y)
                ax_way.set_xlim(min_x, max_x)
                ax_way.set_ylim(min_y, max_y)
                ax_way.plot(0, 0, 'bo', label='Vehicle') # vehicles always at origin in waypoints frame!
                wx = [wp[0] for wp in next_waypoints]
                wy = [wp[1] for wp in next_waypoints]
                color = 'g-o' if on_road else 'r-o'
                ax_way.plot(wy, wx, color, label='Predicted Waypoints') # reverse x/y to show 0 deg as up
                ax_cam.set_data(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))    
                ax_way.legend()
                fig_cam.canvas.draw()
                plt.tight_layout()
                plt.autoscale(False)
                plt.draw()
                plt.pause(0.001)

                
    except KeyboardInterrupt:
        print("Control C received, exiting")
    finally:
        # Clean up actors
        for cam in cameras:
            cam.destroy()
        ego_vehicle.destroy()
        print(f"Crashes: {all_crashes}")

if __name__ == '__main__':
    main()
