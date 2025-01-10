import carla
import os, math, random, sys, csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

WAYPOINT_DIST = 2.0
TOWN_NAME = 'Town01'
TIMESTEP = 1 / 30.0 
N_GOALS = 100
GOAL_THRESH = 4.0

cam_images = [None, None, None]
curr_frame = 0

def camera_callback(image, cam_index):
    # Convert to a NumPy array for visualization
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    new_arr = np.zeros((600, 800, 3), dtype=np.uint8)
    new_arr[:, :, :] = arr[:, :, :3]
    new_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2RGB)

    # Save image to disk
    global curr_frame
    file_path = f"data/images/cam{cam_index}_{curr_frame:06d}.jpg" 
    cv2.imwrite(file_path, cv2.cvtColor(new_arr, cv2.COLOR_RGB2BGR))

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
        print(f"Setting weather to {weather}")
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
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0) # dont bother with this

    # set random goal
    all_spawn_points =  world.get_map().get_spawn_points()
    future_goals = [x.location for x in random.choices(all_spawn_points, k=N_GOALS)]
    traffic_manager.set_path(ego_vehicle, future_goals.copy())
    current_goal = future_goals.pop(0)
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
        "way1_x","way1_y","way2_x","way2_y","way3_x","way3_y","pos_x","pos_y", "turntype"
    ])

    if do_viz:
        plt.ion()
        fig_cam, axs_cam = plt.subplots(1, 4)
        plt.tight_layout()
        img_plots = []
        for ax in axs_cam:
            ax.axis('off')
            img_plots.append(ax.imshow(np.zeros((600, 800, 3), dtype=np.uint8)))
        ax = axs_cam[3] # 0,1,2 cam, 3 waypoints

    try:
        for frame in tqdm(range((100_000))):
            # global image index
            global curr_frame
            curr_frame = frame

            # step sim - do it before writing CSV to sync image path with callback
            world.tick()

            # Check distance to current goal
            pos_x = ego_vehicle.get_location().x
            pos_y = ego_vehicle.get_location().y
            dist_to_goal = math.hypot(pos_x - current_goal.x, pos_y - current_goal.y)
            if dist_to_goal < GOAL_THRESH:
                # reset position to start
                ego_vehicle.set_transform(random.choice(all_spawn_points))
                print(f"Reached goal, setting new goal. Distance to goal: {dist_to_goal}")
                set_random_weather(world)
                current_goal = future_goals.pop(0)

            # check we aren't stuck
            ego_vel = ego_vehicle.get_velocity()
            if Vec3d_norm(ego_vel) < 2.0 and frame > 50: # also wait to let car get up to speed
                print("Vehicle is stuck, resetting")
                ego_vehicle.set_transform(random.choice(all_spawn_points))
                set_random_weather(world)
                # unlike reached goal case, we dont want to set a new goal here

            # Retrieve control inputs and waypoints
            control = ego_vehicle.get_control()
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
            ego_angvel = ego_vehicle.get_angular_velocity()
            # left turn - angvel.z positive, greater than 1
            turn = classify_turn(ego_angvel.z)

            # Visualization
            if do_viz:
                ax.clear()
                plt.tight_layout()
                ax.axis('equal')    
                ax.set_aspect('equal', adjustable=None)
                min_x, min_y, max_x, max_y = -50, -50, 50, 50
                ax.set_xbound(min_x, max_x)
                ax.set_ybound(min_y, max_y)
                ax.plot(0, 0, 'bo', label='Vehicle') # vehicles always at origin in waypoints frame!
                wx = [wp[0] for wp in next_waypoints]
                wy = [wp[1] for wp in next_waypoints]
                ax.plot(wy, wx, 'ro', label='Waypoints') # reverse x/y to show 0 deg as up
                ax.legend()
                for i in range(3):
                    if cam_images[i] is not None:
                        img_plots[i].set_data(cam_images[i])
                fig_cam.canvas.draw()
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
                pos_x, pos_y, turn
            ])
                
    except KeyboardInterrupt:
        print("Control C received, exiting")
    finally:
        csv_file.close()
        # Clean up actors
        for cam in cameras:
            cam.destroy()
        ego_vehicle.destroy()

if __name__ == '__main__':
    main()
