import carla
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import math
import random
import sys
import pathlib 

CARLA_PATH = "F:\CARLA_0.9.15\WindowsNoEditor\PythonAPI"
sys.path.append(CARLA_PATH)
sys.path.append(os.path.join(CARLA_PATH, "carla"))
sys.path.append(os.path.join(CARLA_PATH, "carla", "agents"))
from navigation import global_route_planner, behavior_agent


WAYPOINT_DIST = 2.0
TOWN_NAME = 'Town04'
TIMESTEP = 1 / 30.0 
N_EXTRAS = 0

cam_images = [None, None, None]

def camera_callback(image, cam_index):
    # Convert to a NumPy array for visualization
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    new_arr = np.zeros((600, 800, 3), dtype=np.uint8)
    new_arr[:, :, :] = arr[:, :, :3]
    new_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2RGB)

    # Save image to disk
    file_path = f"data/images/cam{cam_index}_{image.frame:06d}.jpg" 
    cv2.imwrite(file_path, new_arr)

    # Update global variable for visualization
    cam_images[cam_index] = new_arr

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

def set_random_weather(world: carla.World):
    all_weather = [getattr(carla.WeatherParameters, x) for x in dir(carla.WeatherParameters) if '__' not in x]
    weather = random.choice(all_weather)
    try:
        world.set_weather(weather)
    except Exception as e:
        print(f"Error setting weather: {e}")


def main():
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

    # Spawn ego vehicle and others
    print(f"There are {len(world.get_map().get_spawn_points())} spawn points")
    spawn_points = random.choices(world.get_map().get_spawn_points(), k=N_EXTRAS)
    for _ in range(N_EXTRAS):
        vehicle_bp = random.choice(blueprint_lib.filter('vehicle.*.*'))
        spawn_point = spawn_points.pop()
        print(f"Spawning {vehicle_bp.id} at {spawn_point}")
        world.spawn_actor(vehicle_bp, spawn_point).set_autopilot(True)

    vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(type(world.get_map()))
    route_planner = global_route_planner.GlobalRoutePlanner(world.get_map(), sampling_resolution=WAYPOINT_DIST/2)
    agent = behavior_agent.BehaviorAgent(ego_vehicle, behavior="normal", map_inst=world.get_map(), grp_inst=route_planner)

    # set random goal
    all_spawn_points =  world.get_map().get_spawn_points()
    current_goal = random.choice(all_spawn_points).location
    agent.set_destination(current_goal)
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

    os.makedirs("data/images", exist_ok=True)

    # Set up CSV logging
    os.makedirs("data", exist_ok=True)
    csv_file = open("data/data_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "img_path_l","img_path","img_path_r","throttle","steer",
        "way1_x","way1_y","way2_x","way2_y","way3_x","way3_y","pos_x","pos_y"
    ])

    plt.ion()
    fig_cam, axs_cam = plt.subplots(1, 4)
    plt.tight_layout()
    img_plots = []
    for ax in axs_cam:
        ax.axis('off')
        img_plots.append(ax.imshow(np.zeros((600, 800, 3), dtype=np.uint8)))
    ax = axs_cam[3] # 0,1,2 cam, 3 waypoint
    ax.axis('equal')    
    ax.set_aspect('equal', adjustable=None)

    try:
        for frame in tqdm(range((100_000))):
            # Check distance to current goal
            pos_x = ego_vehicle.get_location().x
            pos_y = ego_vehicle.get_location().y
            dist_to_goal = math.hypot(pos_x - current_goal.x, pos_y - current_goal.y)
            if dist_to_goal < 5.0:
                # reset stuff
                print(f"Resetting goal, distance to goal: {dist_to_goal}")
                set_random_weather(world)
                current_goal = random.choice(all_spawn_points).location
                agent.set_destination(current_goal)

            # Retrieve control inputs and waypoints
            control = agent.run_step()
            control.manual_gear_shift = False
            ego_vehicle.apply_control(control)
            plan = list(agent.get_local_planner()._waypoints_queue)
            waypoint_list = world.get_map().get_waypoint(ego_vehicle.get_location())
            next_waypoints = []
            w = waypoint_list
            for _ in range(3):
                w = w.next(WAYPOINT_DIST)[0] if w.next(WAYPOINT_DIST) else w
                wx, wy = w.transform.location.x, w.transform.location.y
                car_tf = get_transform_matrix(pos_x, pos_y, ego_vehicle.get_transform().rotation.yaw)
                inv_car_tf = np.linalg.inv(car_tf)
                p_world = np.array([wx, wy, 0, 1], dtype=np.float32)
                p_local = inv_car_tf @ p_world # local = car^-1 * world b/c world is in car's frame
                next_waypoints.append((p_local[0], p_local[1]))

            # Data to put in csv
            pos_x = ego_vehicle.get_location().x
            pos_y = ego_vehicle.get_location().y
            ego_vel = ego_vehicle.get_velocity()

            # Visualization
            ax.clear()
            min_x, min_y, max_x, max_y = -50, -50, 50, 50
            ax.set_xbound(min_x, max_x)
            ax.set_ybound(min_y, max_y)
            ax.plot(0, 0, 'bo', label='Vehicle') # vehicles always at origin in waypoints frame!
            wx = [wp[0] for wp in next_waypoints]
            wy = [wp[1] for wp in next_waypoints]
            ax.plot(wy, wx, 'ro', label='Waypoints') # reverse x/y to show 0 deg as up
            ax.legend()
            plt.draw()
            plt.pause(0.001)

            # Paths for left, center, right images
            img_path_l = f"data/images/cam0_{frame:06d}.jpg"
            img_path_c = f"data/images/cam1_{frame:06d}.jpg"
            img_path_r = f"data/images/cam2_{frame:06d}.jpg"

            # Write CSV row
            csv_writer.writerow([
                img_path_l, img_path_c, img_path_r,
                control.throttle, control.steer,
                next_waypoints[0][0], next_waypoints[0][1],
                next_waypoints[1][0], next_waypoints[1][1],
                next_waypoints[2][0], next_waypoints[2][1],
                pos_x, pos_y
            ])

            # Update camera plots
            for i in range(3):
                if cam_images[i] is not None:
                    img_plots[i].set_data(cam_images[i])
            fig_cam.canvas.draw()

            # Advance simulation
            world.tick()
                
    except KeyboardInterrupt:
        print("Control C received, exiting")
    finally:
        csv_file.close()
        # Clean up actors
        for cam in cameras:
            cam.destroy()
        ego_vehicle.destroy()
        for actor in world.get_actors():
            actor.destroy()

if __name__ == '__main__':
    main()
