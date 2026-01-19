import glob
import json
import os
import queue
import sys
import time
import math
import numpy as np
from scipy.interpolate import CubicSpline
from mmengine import fileio
import io
from scipy.interpolate import splprep, splev

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')

import carla
from agents.navigation.controller import VehiclePIDController

RENDER = True
RESULT_DIR = None
PATH_LOG_DIR = None
ENVS_DIR = '/home/qian/dataset_V7/'
SIM_COLLISION = 0
RAW_COLLISION = 0
MAX_SPEED = 18
MIN_SPEED = 5
MAX_LAT_ACCEL = 2.5
class MiniWaypoint:
    def __init__(self, location, rotation=carla.Rotation()):
        self.transform = carla.Transform(location, rotation)
        
import matplotlib.pyplot as plt

class TrajectoryEvaluator:
    def __init__(self):
        self.history_cte = []
        self.history_heading_err = []
        self.history_steer = []

        self.total_steps = 0
        
        self.last_closest_idx = 0 

    def compute_step_metrics(self, vehicle, dense_path, dense_yaws):
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        v_yaw_rad = math.radians(v_trans.rotation.yaw)

        search_start = self.last_closest_idx
        search_end = min(self.last_closest_idx + 100, len(dense_path))
        
        min_dist = float('inf')
        closest_idx = self.last_closest_idx

        # 局部搜索
        for i in range(search_start, search_end):
            p = dense_path[i]
            d = math.sqrt((v_loc.x - p.x)**2 + (v_loc.y - p.y)**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        self.last_closest_idx = closest_idx
        
        self.history_cte.append(min_dist)

        path_yaw_rad = math.radians(dense_yaws[closest_idx])
        diff_yaw = abs(v_yaw_rad - path_yaw_rad)
        diff_yaw = diff_yaw % (2 * math.pi)
        if diff_yaw > math.pi:
            diff_yaw = (2 * math.pi) - diff_yaw
        if diff_yaw > (math.pi / 2):
            diff_yaw = (math.pi) - diff_yaw
        self.history_heading_err.append(math.degrees(diff_yaw))

        control = vehicle.get_control()
        self.history_steer.append(control.steer)

        self.total_steps += 1

    def get_final_scores(self, feasibility_threshold=0.5):
        if self.total_steps == 0:
            print("⚠️ Warning: No data has been recorded!")
            return None

        cte_arr = np.array(self.history_cte)
        heading_arr = np.array(self.history_heading_err)
        steer_arr = np.array(self.history_steer)

        rmse_cte = np.sqrt(np.mean(cte_arr ** 2))
        
        max_cte = np.max(cte_arr)

        avg_heading = np.mean(heading_arr)

        feasible_count = np.sum(cte_arr < feasibility_threshold)
        feasibility_ratio = (feasible_count / self.total_steps) * 100.0

        if len(steer_arr) > 1:
            steer_diff = np.diff(steer_arr)
            smoothness_score = np.sum(np.abs(steer_diff)) / (self.total_steps - 1)
        else:
            smoothness_score = 0.0

        return {
            "RMSE_CTE (m)": round(rmse_cte, 4),
            "Max_CTE (m)": round(max_cte, 4),
            "Avg_Heading_Err (deg)": round(avg_heading, 4),
            "Feasibility_Ratio (%)": round(feasibility_ratio, 2),
            "Control_Smoothness": round(smoothness_score, 5)
        }

    def plot_results(self):
        if self.total_steps == 0:
            return

        plt.figure(figsize=(12, 8))

        # Subfigure 1: Cross Track Error
        plt.subplot(3, 1, 1)
        plt.plot(self.history_cte, label='Cross Track Error (CTE)', color='red', linewidth=1.5)
        # Draw a threshold red line to show feasibility
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Feasibility Threshold (0.3m)')
        plt.title('Tracking Accuracy: Cross Track Error')
        plt.ylabel('Error (m)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        # Subfigure 2: Heading Error
        plt.subplot(3, 1, 2)
        plt.plot(self.history_heading_err, label='Heading Error', color='green', linewidth=1.5)
        plt.title('Tracking Stability: Heading Error')
        plt.ylabel('Error (deg)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        # Figure 3: Steering Wheel Angle (Showing Smoothness)
        plt.subplot(3, 1, 3)
        plt.plot(self.history_steer, label='Steering Input', color='blue', linewidth=1.5)
        plt.title('Control Smoothness: Steering Input')
        plt.ylabel('Steer (-1 to 1)')
        plt.xlabel('Simulation Steps')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def save_result(self, save_path):
        """
        Save intermediate results for later plotting
        """
        if self.total_steps == 0:
            print("⚠️ Warning: No data has been recorded!")
            return

        np.savez_compressed(save_path,
                            history_cte=self.history_cte,
                            history_heading_err=self.history_heading_err,
                            history_steer=self.history_steer)
        print(f"✅ Mid result saved to {save_path}")

def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)
    return npz_data

def xy2xy_heading(xy):
    xy = np.array(xy)
    N = xy.shape[0]
    
    if N < 2:
        raise ValueError("Points must contain at least 2 points to calculate heading.")

    headings = np.zeros(N)
    
    dx_mid = xy[2:, 0] - xy[:-2, 0]
    dy_mid = xy[2:, 1] - xy[:-2, 1]
    
    headings[1:-1] = np.arctan2(dy_mid, dx_mid)
    
    dx_start = xy[1, 0] - xy[0, 0]
    dy_start = xy[1, 1] - xy[0, 1]
    headings[0] = np.arctan2(dy_start, dx_start)
    
    dx_end = xy[-1, 0] - xy[-2, 0]
    dy_end = xy[-1, 1] - xy[-2, 1]
    headings[-1] = np.arctan2(dy_end, dx_end)

    xy_heading = np.hstack((xy, headings[:, np.newaxis]))
    
    return xy_heading

import numpy as np
import math

def has_reversing_segment(waypoints, vehicle_yaw=0, strictness=0.5, start_strictness=0.0):
    if len(waypoints) < 2:
        return False
        
    diff_raw = np.diff(waypoints, axis=0)
    dist_raw = np.linalg.norm(diff_raw, axis=1)
    
    valid_mask = dist_raw > 0.05 
    
    if np.sum(valid_mask) < 1:
        return False
        
    vectors = diff_raw[valid_mask]
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / (norms + 1e-6)
    
    first_move_vector = normalized_vectors[0]
    
    car_heading_vector = np.array([math.cos(vehicle_yaw), math.sin(vehicle_yaw)])
    
    initial_dot = np.dot(first_move_vector, car_heading_vector)
    
    if initial_dot < -start_strictness:
        return True
    
    if len(normalized_vectors) < 2:
        return False

    v_current = normalized_vectors[:-1]
    v_next = normalized_vectors[1:]
    
    dot_products = np.sum(v_current * v_next, axis=1)
    
    if np.any(dot_products < -strictness):
        return True
        
    return False

def smooth_path(waypoints, smoothing_factor=0.5):
    """
    waypoints: np.array shape (N, 2) [[x1, y1], [x2, y2], ...]
    """
    diff = np.diff(waypoints, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    valid_indices = np.where(dist > 0.01)[0] + 1
    valid_indices = np.insert(valid_indices, 0, 0) # 加上起点
    clean_waypoints = waypoints[valid_indices]

    if len(clean_waypoints) < 4:
        return clean_waypoints 

    try:
        tck, u = splprep(clean_waypoints.T, s=smoothing_factor) 
        
        u_new = np.linspace(u.min(), u.max(), len(clean_waypoints) * 5)
        x_new, y_new = splev(u_new, tck)
        
        return np.vstack((x_new, y_new)).T
    except Exception as e:
        return clean_waypoints

def get_curvature(p1, p2, p3):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y

    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    len_a = np.hypot(x1 - x2, y1 - y2)
    len_b = np.hypot(x2 - x3, y2 - y3)
    len_c = np.hypot(x3 - x1, y3 - y1)
    
    if area < 1e-4:
        return 0.0
        
    curvature = 4 * area / (len_a * len_b * len_c)
    return curvature

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def generate_path(index, method='hard', file_list=None):
    if method == 'hard':
        path_data_file = f'path_data/hard/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
        path_numpy = np.vstack(([0, 0], path_numpy))
    elif method == 'Soft':
        path_data_file = f'path_data/Soft/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
        path_numpy = np.vstack(([0, 0], path_numpy))
    elif method == 'RRT':
        path_data_dir = f'path_data/RRT'
        file_name = file_list[index]
        path_data_file = f'path_data/RRT/batch_{file_name}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
        path_numpy = path_numpy[::-1]
        path_numpy = smooth_path(path_numpy, smoothing_factor=0.5)
    elif method == 'Astar':
        path_data_dir = f'/home/qian/dataset_V7_labels_5'
        file_name = file_list[index]
        path_data_file = f'{path_data_dir}/{file_name}'
        path_numpy = np.load(path_data_file, allow_pickle=True)['path']
        path_numpy = np.vstack(([0, 0, 0], path_numpy))
    elif method == 'IL_Soft':
        path_data_file = f'path_data/IL_Soft/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
        path_numpy = np.vstack(([0, 0], path_numpy))
    elif method == 'NMPC':
        path_data_dir = 'path_data/NMPC2'
        path_data_file = file_list[index]
        path_data_file = f'{path_data_dir}/{path_data_file}'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    return path_numpy

def customize_physics(vehicle):
    physics_control = vehicle.get_physics_control()
    
    new_wheels = []
    for i, wheel in enumerate(physics_control.wheels):
        if i == 0 or i == 1:
            wheel.max_steer_angle = 40
        new_wheels.append(wheel)
    
    physics_control.wheels = new_wheels
    physics_control.mass = 1500
    vehicle.apply_physics_control(physics_control)

def get_dynamic_look_ahead(speed_kmh, current_idx, dense_path, dense_yaws):
    base_dist = np.clip(speed_kmh * 0.2, 1.0, 5.0)
    
    
    look_ahead_idx = min(current_idx + 10, len(dense_path)-1)
    vec1 = dense_path[current_idx]
    vec2 = dense_path[look_ahead_idx]
    current_yaw = dense_yaws[current_idx]
    future_yaw = dense_yaws[look_ahead_idx]
    diff = abs(current_yaw - future_yaw)
    if diff > 180: diff = 360 - diff
    
    if diff > 10.0:
        return max(0.5, base_dist * 0.5)

    return base_dist*0.1

def smooth_path_cubic_spline(sparse_points, resolution=0.1):
    x, y = [], []
    for p in sparse_points:
        if isinstance(p, carla.Location):
            x.append(p.x); y.append(p.y)
        else:
            x.append(p[0]); y.append(p[1])
            
    if len(x) < 2: return sparse_points, [], []

    s = [0]
    for i in range(1, len(x)):
        dist = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        s.append(s[-1] + max(dist, 0.001))
    
    try:
        cs_x = CubicSpline(s, x)
        cs_y = CubicSpline(s, y)
    except Exception:
        return [], [], []

    total_length = s[-1]
    s_dense = np.arange(0, total_length, resolution)
    
    x_dense = cs_x(s_dense)
    y_dense = cs_y(s_dense)
    
    dx = cs_x(s_dense, 1)
    dy = cs_y(s_dense, 1)
    
    ddx = cs_x(s_dense, 2)
    ddy = cs_y(s_dense, 2)
    
    k_dense = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    yaw_dense = np.arctan2(dy, dx)
    
    dense_path_objects = []
    dense_yaw_degrees = []
    
    for i in range(len(x_dense)):
        z_val = sparse_points[0].z if isinstance(sparse_points[0], carla.Location) else 0
        loc = carla.Location(x=float(x_dense[i]), y=float(y_dense[i]), z=float(z_val))
        dense_path_objects.append(loc)
        dense_yaw_degrees.append(math.degrees(yaw_dense[i]))
        
    return dense_path_objects, dense_yaw_degrees, k_dense

def calculate_feedforward(curvature, wheelbase=2.87, max_steer_deg=40.0):
    steer_rad = math.atan(wheelbase * curvature)
    
    max_steer_rad = math.radians(max_steer_deg)
    steer_norm = steer_rad / max_steer_rad
    
    return np.clip(steer_norm, -1.0, 1.0)

def get_rect_points(xy_heading, width=1.9, length=4.7):
    if not isinstance(xy_heading, np.ndarray):
        xy_heading = np.array(xy_heading)
        
    N = xy_heading.shape[0]
    heading = xy_heading[:, 2]
    
    cos_h = np.cos(heading)[:, np.newaxis] 
    sin_h = np.sin(heading)[:, np.newaxis] 

    half_l = length / 2.0
    half_w = width / 2.0
    
    dx = np.array([half_l, half_l, -half_l, -half_l])
    dy = np.array([half_w, -half_w, -half_w, half_w])

    
    rot_x = dx * cos_h - dy * sin_h  # shape: (N, 4)
    rot_y = dx * sin_h + dy * cos_h  # shape: (N, 4)

    rotated_offsets = np.stack((rot_x, rot_y), axis=-1)

    centers = xy_heading[:, :2][:, np.newaxis, :]
    
    rectangles = centers + rotated_offsets
    
    return rectangles

def get_rect_points2(xy_heading, width=1.942, LF=3.76, LB=0.929):
    if not isinstance(xy_heading, np.ndarray):
        xy_heading = np.array(xy_heading)
        
    N = xy_heading.shape[0]
    heading = xy_heading[:, 2]
    
    cos_h = np.cos(heading)[:, np.newaxis] 
    sin_h = np.sin(heading)[:, np.newaxis] 

    half_w = width / 2.0
    
    dx = np.array([LF, LF, -LB, -LB]) 
    
    dy = np.array([half_w, -half_w, -half_w, half_w])

    rot_x = dx * cos_h - dy * sin_h  # shape: (N, 4)
    rot_y = dx * sin_h + dy * cos_h  # shape: (N, 4)

    rotated_offsets = np.stack((rot_x, rot_y), axis=-1)

    centers = xy_heading[:, :2][:, np.newaxis, :]
    
    rectangles = centers + rotated_offsets
    
    return rectangles

def check_containment(poly1, poly2):
    if all(point_in_polygon(poly1[:,i], poly2) for i in range(4)):
        return True
    
    if all(point_in_polygon(poly2[:,i], poly1) for i in range(4)):
        return True
    
    return False

def point_in_polygon(point, polygon):
    x, y = point
    n = 4
    inside = False
    
    px, py = polygon[0, 0], polygon[1, 0]
    for i in range(n + 1):
        qx, qy = polygon[0, i % n], polygon[1, i % n]
        if y > min(py, qy):
            if y <= max(py, qy):
                if x <= max(px, qx):
                    if py != qy:
                        xinters = (y - py) * (qx - px) / (qy - py) + px
                    if px == qx or x <= xinters:
                        inside = not inside
        px, py = qx, qy
    
    return inside

def check_edges_intersection(poly1, poly2):
    for i in range(4):
        p1 = poly1[:, i]
        p2 = poly1[:, (i+1)%4]
        
        for j in range(4):
            p3 = poly2[:, j]
            p4 = poly2[:, (j+1)%4]
            
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def segments_intersect(a1, a2, b1, b2):
    a1 = np.asarray(a1).flatten()[:2]
    a2 = np.asarray(a2).flatten()[:2]
    b1 = np.asarray(b1).flatten()[:2]
    b2 = np.asarray(b2).flatten()[:2]
    
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    case1 = ccw(a1, b1, b2) != ccw(a2, b1, b2)
    case2 = ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    if case1 and case2:
        return True
    
    if (np.array_equal(a1, b1) or np.array_equal(a1, b2) or 
        np.array_equal(a2, b1) or np.array_equal(a2, b2)):
        return True
    
    if is_point_on_segment(a1, b1, b2) or is_point_on_segment(a2, b1, b2):
        return True
    if is_point_on_segment(b1, a1, a2) or is_point_on_segment(b2, a1, a2):
        return True
    
    return False

def is_point_on_segment(p, a, b):
    p = np.asarray(p).flatten()[:2]
    a = np.asarray(a).flatten()[:2]
    b = np.asarray(b).flatten()[:2]
    
    cross = (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])
    if not np.isclose(cross, 0, atol=1e-8):
        return False
    
    min_x = min(a[0], b[0])
    max_x = max(a[0], b[0])
    min_y = min(a[1], b[1])
    max_y = max(a[1], b[1])
    
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)

def check_polygon_intersection(poly1, poly2):
    if poly1.shape == (4,2):
        poly1 = poly1.T
    if poly2.shape == (4,2):
        poly2 = poly2.T
    
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("The input must be a 2x4 numpy array")
    
    if check_edges_intersection(poly1, poly2):
        return True
    
    if check_containment(poly1, poly2):
        return True
    
    return False

def collision_check(path, obstacles):
    path_rects = get_rect_points(path)

    for rect in path_rects:
        for obs in obstacles:
            if check_polygon_intersection(rect, obs):
                return True
    return False

def compute_curvature(prev_point, curr_point, next_point):
    dx1 = curr_point.x - prev_point.x
    dy1 = curr_point.y - prev_point.y
    dx2 = next_point.x - curr_point.x
    dy2 = next_point.y - curr_point.y
    d1 = math.hypot(dx1, dy1)
    d2 = math.hypot(dx2, dy2)
    if d1 * d2 == 0:
        return 0
    return (dx1 * dy2 - dy1 * dx2) / (d1 * d2)


def main(index, methods='hard', file_list=None):
    actor_list = []
    evaluator = TrajectoryEvaluator()
    result_save_file = f'{RESULT_DIR}/{index}.json'
    # if os.path.exists(result_save_file):
    #     return

    actual_path = []
    if methods == 'hard' or methods == 'Soft':
        obstacle_file = os.path.join(ENVS_DIR, f'{index}.npz')
        obstacles = opendata(obstacle_file)['obstacles_vertices']
    elif methods == 'IL_Soft' or methods == 'NMPC':
        filename = file_list[index]
        # filename = 'batch_544.npy'
        file_base_name = os.path.basename(filename)
        file_base_name = file_base_name.replace('batch_', '')
        file_base_name = file_base_name.replace('npy', 'npz')
        print(file_base_name)
        obstacle_file = os.path.join(ENVS_DIR, f'{file_base_name}')
        obstacles = opendata(obstacle_file)['obstacles_vertices']
    elif methods == 'RRT':
        # filename = file_list[index]
        filename = 'batch_544.npy'
        file_base_name = os.path.basename(filename)
        # 把batch_index.npy 变成 index.npz
        file_base_name = file_base_name.replace('batch_', '')
        file_base_name = file_base_name.replace('npy', 'npz')
        obstacle_file = os.path.join(ENVS_DIR, f'{file_base_name}')
        obstacles = opendata(obstacle_file)['obstacles_vertices']
    elif methods == 'Astar':
        filename = file_list[index]
        file_base_name = os.path.basename(filename)
        obstacle_file = os.path.join(ENVS_DIR, f'{file_base_name}')
        obstacles = opendata(obstacle_file)['obstacles_vertices']
    # if os.path.exists(result_save_file):
    #     return
    c_raw = 0
    c_sim = 0
    
    path_interval = 0.1
    path_points_numpy = generate_path(index, methods, file_list=file_list)
    # collision = collision_check(path_points_numpy, obstacles)
    if methods == 'Astar':
        if has_reversing_segment(path_points_numpy[:, :2]):
            return
        # path_points_numpy = smooth_path(path_points_numpy, smoothing_factor=0.5)
    path_points = []
    dense_yaws = []
    for point in path_points_numpy:
        path_points.append(carla.Location(x=float(point[0]), y=float(point[1]), z=0))
    path_points, dense_yaws, dense_curvatures = smooth_path_cubic_spline(path_points, resolution=path_interval)
    begin_point = path_points[6]
    begin_yaw = math.degrees(math.atan2(begin_point.y, begin_point.x))
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = not RENDER
    settings.fixed_delta_seconds = 0.02 # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2.0), carla.Rotation(yaw=begin_yaw))
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    
    if not vehicle:
        print("❌ Generation failed, the location may be occupied")
        return
    
    actor_list.append(vehicle)

    vehicle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
    for _ in range(20):
        world.tick()

    customize_physics(vehicle)

    pid_controller = VehiclePIDController(vehicle,
                                    args_lateral={'K_P': 3.5, 'K_D': 0.9, 'K_I': 0.1},
                                    args_longitudinal={'K_P': 0.8, 'K_D': 0.2, 'K_I': 0.0})

    
    last_closest_idx = 0
    if RENDER:
        for i in range(len(path_points)-1):
            world.debug.draw_line(path_points[i]+carla.Location(z=0.1), path_points[i+1]+carla.Location(z=0.1), thickness=0.1, color=carla.Color(0,0,255), life_time=0)
        for obs in obstacles:
            for i in range(4):
                start_point = carla.Location(x=float(obs[i][0]), y=float(obs[i][1]), z=0.5)
                end_point = carla.Location(x=float(obs[(i+1)%4][0]), y=float(obs[(i+1)%4][1]), z=0.5)
                world.debug.draw_line(start_point, end_point, thickness=0.1, color=carla.Color(0,0,0), life_time=0)
    while True:
        world.tick()
        current_loc = vehicle.get_location()
        v_trans = vehicle.get_transform()
        v_yaw = math.radians(v_trans.rotation.yaw)
        actual_path.append([current_loc.x, current_loc.y, v_yaw])

        search_range = 50
        start_idx = last_closest_idx
        end_idx = min(last_closest_idx + search_range, len(path_points))

        min_dist = float('inf')
        current_closest_idx = last_closest_idx

        for i in range(start_idx, end_idx):
            dist = vehicle.get_location().distance(path_points[i])
            if dist < min_dist:
                min_dist = dist
                current_closest_idx = i

        last_closest_idx = current_closest_idx
        speed = get_speed(vehicle)
        look_ahead_dist = get_dynamic_look_ahead(speed, current_closest_idx, path_points, dense_yaws)
        look_ahead_step = int(look_ahead_dist / path_interval)

        target_idx = current_closest_idx + look_ahead_step
        if target_idx >= len(path_points):
            target_idx = len(path_points) - 1

        target_loc = path_points[target_idx]
        target_yaw = dense_yaws[target_idx]
        if current_closest_idx >= len(path_points) - 6:
            break
        target_loc.z = current_loc.z
        
        step = 8
        p1 = path_points[target_idx]
        p2 = path_points[min(target_idx + step, len(path_points)-1)]
        p3 = path_points[min(target_idx + 2*step, len(path_points)-1)]
        curvature = get_curvature(p1, p2, p3)
        v_limit_ms = np.sqrt(MAX_LAT_ACCEL / (abs(curvature) + 1e-6))
        v_limit_kmh = v_limit_ms * 3.6
        target_speed = np.clip(v_limit_kmh, MIN_SPEED, MAX_SPEED)
        if RENDER:
            world.debug.draw_point(current_loc, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),carla.Rotation(pitch=-90)))
        curvature_idx = min(current_closest_idx, len(dense_curvatures) - 1)
        k_current = dense_curvatures[curvature_idx]
        
        fake_waypoint = MiniWaypoint(target_loc, carla.Rotation(yaw=target_yaw))
        control = pid_controller.run_step(target_speed, fake_waypoint)
        pid_steer = control.steer

        ff_steer = calculate_feedforward(k_current, wheelbase=2.87, max_steer_deg=45.0)
        
        direction_fix = 1.0
        ff_steer = ff_steer * direction_fix * 1
        final_steer = pid_steer + ff_steer * 0.1
        
        control.steer = np.clip(final_steer, -1.0, 1.0)
        vehicle.apply_control(control)
        evaluator.compute_step_metrics(vehicle, path_points, dense_yaws)
    if RENDER:
        vehicle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        spectator = world.get_spectator()
        location = carla.Location(x=17.5, y=0, z=25)
        spectator.set_transform(carla.Transform(location, carla.Rotation(pitch=-90, yaw=-90)))
        
        for i in range(len(actual_path)-1):
            start = carla.Location(x=actual_path[i][0], y=actual_path[i][1], z=0.1)
            end = carla.Location(x=actual_path[i+1][0], y=actual_path[i+1][1], z=0.1)
            world.debug.draw_line(start, end, thickness=0.1, color=carla.Color(255,0,0), life_time=10)
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        spawn_point = carla.Transform(
            carla.Location(17.5, 0, z=20),
            carla.Rotation(pitch=-90, yaw=-90, roll=0)
        )
        camera = world.spawn_actor(camera_bp, spawn_point)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        try:
            for i in range(10): 
                world.tick()
                if not image_queue.empty():
                    image = image_queue.get()

            print("Capturing high-res map...")
            world.tick()
            image = image_queue.get()
            
            filename = f'{methods}_01.png'
            image.save_to_disk(filename)
            print(f"Saved {filename} successfully!")

        finally:
            if camera:
                camera.stop()
                camera.destroy()
                print("Camera destroyed.")
    scores = evaluator.get_final_scores()
    print(f"------ SUMMARY_{index} ------")
    print(scores)
    evaluator.plot_results()
    evaluator.save_result(f'data_record/{methods}_{index}.npz')
    print("🧹 clearing...")
    world.apply_settings(original_settings)
    
    # paths = {
    #     'raw': path_points_numpy,
    #     'sim': np.array(actual_path)
    # }
    # save_paths = f'paths/{methods}_{index}.npz'
    # np.savez_compressed(save_paths, **paths)

    for actor in actor_list:
        actor.destroy()
    print("👋 finished!")
    
    with open(result_save_file, 'w') as f:
        json.dump(scores, f)
    actual_path = actual_path[::20]
    collision = collision_check(actual_path, obstacles)
    if collision:
        c_sim = 1
        print("Simulated path collided with obstacles!")
    path_save_file = os.path.join(PATH_LOG_DIR, f"simpath_{index}.json")
    path_data = {
        'path': actual_path,
        'raw_collision': c_raw,
        'sim_collision': c_sim
    }
    with open(path_save_file, 'w') as f:
        json.dump(path_data, f)
    return

def analyze_collision():
    json_files = glob.glob(os.path.join(PATH_LOG_DIR, '*.json'))

    total_cases = len(json_files)
    if total_cases == 0:
        print("No result files found!")
        return
    raw_c = 0
    sim_c = 0
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_file}")
            raw_c += data['raw_collision']
            sim_c += data['sim_collision']
    print(f"------ SUMMARY ------")
    print(f"Original Path Collision Rate: {raw_c / total_cases * 100:.2f}%")
    print(f"Simulated Path Collision Rate: {sim_c / total_cases * 100:.2f}%")

def analyze_results():
    json_files = glob.glob(os.path.join(RESULT_DIR, '*.json'))

    total_cases = len(json_files)
    if total_cases == 0:
        print("No result files found!")
        return
    results = {
        "RMSE_CTE (m)": [],
        "Max_CTE (m)": [],
        "Avg_Heading_Err (deg)": [],
        "Feasibility_Ratio (%)": [],
        "Control_Smoothness": []
    }
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_file}")
            for key in results.keys():
                if key in data:
                    results[key].append(data[key])

    for key in results.keys():
        if results[key]:
            results[key] = np.mean(results[key])
    print(f"------ SUMMARY ------")
    print(results)

if __name__ == '__main__':
    RENDER = False
    RESULT_DIR = 'scores_logs_1.9_ILSoft2'
    PATH_LOG_DIR = '19/simpath_ILSoft2'
    SIM_COLLISION = 0
    RAW_COLLISION = 0
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PATH_LOG_DIR, exist_ok=True)
    # replace with path data dir
    file_list = os.listdir('path_data/IL_Soft')
    indexs = range(0, 1000)
    for i in indexs:
        main(i, 'IL_Soft', file_list=file_list)
    analyze_results()
    analyze_collision()