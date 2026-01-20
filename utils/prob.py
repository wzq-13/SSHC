import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import globalvar
import math
from torch.nn.functional import grid_sample
def LSE_max(a, dim, rho=10.0):
    return (1.0 / rho) * torch.logsumexp(rho * a, dim=dim)

def LSE_min(a, dim, rho=10.0):
    return -(1.0 / rho) * torch.logsumexp(-rho * a, dim=dim)

def vertices_to_edges(polygons):
    # vertices shape: (batch_size, K, 4, 2)
    # Get vertex pairs (for quadrilateral)
    v1 = polygons
    v2 = torch.roll(polygons, shifts=-1, dims=2)  # Roll along the vertex dimension
    
    # Compute edge coefficients
    a = v2[:, :, :, 1] - v1[:, :, :, 1]  # y2 - y1
    b = v1[:, :, :, 0] - v2[:, :, :, 0]  # x1 - x2
    c = v2[:, :, :, 0] * v1[:, :, :, 1] - v1[:, :, :, 0] * v2[:, :, :, 1]  # x2*y1 - x1*y2
    
    # Stack coefficients along last dimension
    edges = torch.stack([a, b, c], dim=-1)  # (batch_size, K, 4, 3)
    return edges

def h(x, y, polygons, rho):
    # x, y: (batch_size, N) - N coordinates per batch
    # polygons: tensor of shape (batch_size, K, 4, 2) where K is number of quadrilaterals per batch
    
    batch_size, N = x.shape
    K = polygons.shape[1]
    
    # Convert vertices to edges
    edges = vertices_to_edges(polygons)  # (batch_size, K, 4, 3)
    
    # Expand dimensions for broadcasting
    # x and y will be (batch_size, N, 1, 1) to match (batch_size, K, 4) edges
    x_exp = x.view(batch_size, N, 1, 1)
    y_exp = y.view(batch_size, N, 1, 1)
    
    # Get edge coefficients
    a = edges[:, :, :, 0]  # (batch_size, K, 4)
    b = edges[:, :, :, 1]  # (batch_size, K, 4)
    c = edges[:, :, :, 2]  # (batch_size, K, 4)
    
    # Expand edge coefficients to match coordinates
    a_exp = a.unsqueeze(1)  # (batch_size, 1, K, 4)
    b_exp = b.unsqueeze(1)  # (batch_size, 1, K, 4)
    c_exp = c.unsqueeze(1)  # (batch_size, 1, K, 4)
    
    # Compute distances for all edges and all points
    numerator = a_exp * x_exp + b_exp * y_exp + c_exp  # (batch_size, N, K, 4)
    denominator = torch.sqrt(a_exp**2 + b_exp**2)  # (batch_size, N, K, 4)
    distances = numerator / denominator  # (batch_size, N, K, 4)
    
    # Determine the minimum distance from each point to the edges of each quadrilateral.
    quadrilateral_distances = LSE_min(distances, dim=-1)  # (batch_size, N, K)
    
    # For each point, find maximum distance across quadrilaterals
    point_results = LSE_max(quadrilateral_distances, dim=-1)  # (batch_size, N)
    point_results = point_results + globalvar.vehicle_geometrics_.Safety_margin  # Safety margin

    return point_results # (batch_size, N)

def xy2xy_heading(xy):
    '''
    输入: xy: (B, N, 2)
    输出: xy_heading: (B, N+1, 3)
    '''
    if not isinstance(xy, torch.Tensor):
        xy = torch.from_numpy(xy)
    
    device = xy.device
    B, N, _ = xy.shape
    
    d = 0.8
    
    theta0 = torch.zeros((), device=device) 
    
    last_point_x = -d * torch.cos(theta0)
    last_point_y = -d * torch.sin(theta0)
    
    last_point = torch.stack([last_point_x, last_point_y], dim=-1)
    last_point = last_point.view(1, 1, 2).expand(B, -1, -1)
    
    init_point = torch.zeros((B, 1, 2), device=device)
    
    traj_points = torch.cat([last_point, init_point, xy], dim=1)

    h_start = torch.zeros((B, 2), device=device)

    if N > 0:
        delta = traj_points[:, 3:, :] - traj_points[:, 1:-2, :]
        h_mid = torch.atan2(delta[:, :, 1], delta[:, :, 0])
    else:
        h_mid = torch.empty((B, 0), device=device)

    delta_end = traj_points[:, -1, :] - traj_points[:, -2, :]
    h_end = torch.atan2(delta_end[:, 1], delta_end[:, 0]).unsqueeze(1)
    
    headings = torch.cat([h_start, h_mid, h_end], dim=1)
    xy_heading = torch.cat([traj_points, headings.unsqueeze(2)], dim=2)
    
    return xy_heading

def get_safe_circle_centers(xy_heading):
    B, N, _ = xy_heading.shape
    delta_l = globalvar.vehicle_geometrics_.vehicle_length / 3.0
    
    x = xy_heading[:, :, 0]
    y = xy_heading[:, :, 1]
    heading = xy_heading[:, :, 2]
    
    offset_x = delta_l * torch.cos(heading)
    offset_y = delta_l * torch.sin(heading)

    center_mid = xy_heading[:, :, :2] 
    center_front = torch.stack([x + offset_x, y + offset_y], dim=-1)
    center_rear = torch.stack([x - offset_x, y - offset_y], dim=-1)

    circle_centers = torch.stack([center_mid, center_front, center_rear], dim=2)

    return circle_centers.view(B, N*3, 2)

def compute_turning_radius(xy):
    B, N, _ = xy.shape
    
    p0, p1, p2 = xy[:, :-2], xy[:, 1:-1], xy[:, 2:]
    
    v1 = p1 - p0  # (B, N-2, 2)
    v2 = p2 - p1  # (B, N-2, 2)
    v3 = p2 - p0  # (B, N-2, 2)
    
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # (B, N-2)
    
    norm_v1 = torch.norm(v1, dim=2)  # (B, N-2)
    norm_v2 = torch.norm(v2, dim=2)  # (B, N-2)  
    norm_v3 = torch.norm(v3, dim=2)  # (B, N-2)

    area = 0.5 * torch.abs(cross)  # 三角形面积
    denominator = norm_v1 * norm_v2 * norm_v3

    curvatures = 4 * area / (denominator + 1e-6)

    turning_radius = 1.0 / (curvatures + 1e-6)
    
    min_disp = 0.1
    large_radius = 1e6
    small_disp_mask = (norm_v1 < min_disp) | (norm_v2 < min_disp) | (norm_v3 < min_disp)  # (B, N-2)
    large_radius_tensor = torch.tensor(large_radius, dtype=turning_radius.dtype, device=turning_radius.device)
    turning_radius = torch.where(small_disp_mask, large_radius_tensor, turning_radius)

    return turning_radius


def compute_kappa_diff(xy):
    B, N, _ = xy.shape
    dx = torch.gradient(xy[:, :, 0], dim=1)[0] # (B, N)
    dy = torch.gradient(xy[:, :, 1], dim=1)[0]
    
    ddx = torch.gradient(dx, dim=1)[0] # (B, N)
    ddy = torch.gradient(dy, dim=1)[0]
    kappa = torch.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5

    step_len = torch.sqrt(dx**2 + dy**2)
    weight = torch.tanh(step_len * 5.0)
    kappa = kappa * weight

    return kappa #(B, N)


def soft_constraints(xy_heading, obs_constraints_weight, obstacles_vertices, mode='single_value'):
    #d
    distances = torch.norm(xy_heading[:, 1:, :2] - xy_heading[:, :-1, :2], dim=2)  # (B, N-1)
    max_distance = 1.0  # 最大增量
    distance_violations = F.relu(distances - max_distance)  # (B, N-1)
    distance_violations = (distance_violations ** 2 * 20.0)
    distance_violations = distance_violations.sum(dim=1, keepdim=True)  # (B, 1)
    
    # kappa
    turning_radius = compute_turning_radius(xy_heading[:, :, :2])  # (B, N-2)
    min_turning_radius = globalvar.vehicle_kinematics_.min_turning_radius
    turning_violations = F.relu(min_turning_radius - turning_radius)  # (B, N-2)
    turning_violations = turning_violations.mean(dim=1, keepdim=True)#**2  # (B, 1)
    
    # obs
    xy_heading = xy_heading[:, 1:, :]  # (batch_size, N, 3)
    safe_centers = get_safe_circle_centers(xy_heading)  # (batch_size, N*3, 2)
    point_x = safe_centers[:, :, 0]  # (batch_size, N*3)
    point_y = safe_centers[:, :, 1]  # (batch_size, N*3)
    safety_distances = h(point_x, point_y, obstacles_vertices, rho=10.0)  # (batch_size, N*3)
    safety_distances = F.relu(safety_distances)
    safety_distances = safety_distances.view(safety_distances.shape[0], -1, 3) # (B, N, 3)
    safety_distances = LSE_max(safety_distances, dim=2)  # (B, N)
    safety_distances = torch.mean(safety_distances, dim=1, keepdim=True) # (B, 1)
    
    soft_loss = distance_violations.mean() + turning_violations.mean() + safety_distances.mean() * obs_constraints_weight
    return soft_loss
    # return torch.cat([distance_violations, turning_violations, safety_distances], dim=1)  # (B, 3) if mode=='single_value' else (B, N-1 + N-2)

def _create_objective_function():
    def objective_function(data, y):
        xy = y.view(y.shape[0], -1, 7)  # (batch_size, N, 5)
        s = xy[:, :, 2:5]  # (batch_size, N, 3)
        s_2 = s ** 2
        s_h = xy[:, :, 5]  # (batch_size, N)
        s_h_2 = s_h ** 2
        s_d = xy[:, :, 6]  # (batch_size, N)
        s_d_2 = s_d ** 2

        xy_heading = xy2xy_heading(xy[:, :, :2])  # (batch_size, N+2, 3)

        distances = torch.norm(xy_heading[:, 2:, :2] - xy_heading[:, 1:-1, :2], dim=2)  # (B, N)
        max_distance = 1.0
        distance_violations = distances - max_distance  # (B, N)
        residuals_distance = distance_violations + s_d_2  # (B, N)

        kappas = compute_kappa_diff(xy_heading[:, :, :2])[:, 1:-1]  # (B, N)
        kappa_max = 1.0 / globalvar.vehicle_kinematics_.min_turning_radius
        safety_kappas = kappas - kappa_max  # (B, N-1)
        residuals_kappa = safety_kappas + s_h_2  # (B, N-1)
        
        xy_heading = xy_heading[:, 2:, :]  # (batch_size, N, 3)
        safe_centers = get_safe_circle_centers(xy_heading)  # (batch_size, N*3, 2)
        point_x = safe_centers[:, :, 0]  # (batch_size, N*3)
        point_y = safe_centers[:, :, 1]  # (batch_size, N*3)
        obstacles_vertices = data['obstacles_vertices']  # (batch_size, K, 4, 2)
        safety_distances = h(point_x, point_y, obstacles_vertices, rho=10.0)  # (batch_size, N*3)
        safety_distances = safety_distances.view(y.shape[0], -1, 3)  # (batch_size, N, 3)

        residuals = safety_distances + s_2 # (batch_size, N, 3)
        residuals = LSE_max(residuals, dim=2, rho=20)  # (batch_size, N)

        residuals = torch.cat([residuals, residuals_kappa, residuals_distance], dim=1)  # (batch_size, N + N + N)
        residuals = residuals.view(y.shape[0], -1, 3)  # (batch_size, num_constraints, 3)
        residuals = LSE_max(residuals, dim=2, rho=20)  # (batch_size, num_constraints)
        
        residuals = residuals.view(y.shape[0], -1, 4)  # (batch_size, num_groups, X)
        residuals = LSE_max(residuals, dim=2, rho=20)  # (batch_size, num_groups)
        return residuals
    
    return objective_function

def get_map_distance(distance_map, grid_x, grid_y, config):
    B, W, H = distance_map.shape
    B, N = grid_x.shape
    grid_x_no_grad = grid_x.detach() # (B, N)
    grid_y_no_grad = grid_y.detach() # (B, N)
    
    x_round = torch.round(grid_x_no_grad).long()
    y_round = torch.round(grid_y_no_grad).long()
    
    x_offsets = torch.tensor([-1, -1, -1, 0, 0, 1, 1, 1, 0], device=grid_x.device).view(1, 1, 9)  # (1, 1, 9)
    y_offsets = torch.tensor([-1, 0, 1, -1, 1, -1, 0, 1, 0], device=grid_y.device).view(1, 1, 9)  # (1, 1, 9)
    x_neighbors = x_round.unsqueeze(2) + x_offsets  # (B, N, 9)
    y_neighbors = y_round.unsqueeze(2) + y_offsets  # (B, N, 9)
    x_neighbors = torch.clamp(x_neighbors, 0, W-1)
    y_neighbors = torch.clamp(y_neighbors, 0, H-1)
    neighbor_potential = distance_map[torch.arange(B).unsqueeze(1).unsqueeze(2), x_neighbors, y_neighbors]  # (B, N, 9)
    min_indices = torch.argmin(neighbor_potential, dim=-1)  # (B, N)
    max_indices = torch.argmax(neighbor_potential, dim=-1)  # (B, N)
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
    point_indices = torch.arange(N).unsqueeze(0).expand(B, N)
    min_dist_x = x_neighbors[batch_indices, point_indices, min_indices]  # (B, N)
    min_dist_y = y_neighbors[batch_indices, point_indices, min_indices]  # (B, N)
    min_potential = neighbor_potential[batch_indices, point_indices, min_indices]  # (B, N)
    
    distances_with_min_point_8 = torch.sqrt((grid_x - min_dist_x.float())**2 + (grid_y - min_dist_y.float())**2)  # (B, N)
    
    reformulated_potential = neighbor_potential - min_potential.unsqueeze(2)  # (B, N, 9)
    reformulated_potential = reformulated_potential ** 2
    distance_with_neighbors = torch.sqrt((grid_x.unsqueeze(2) - x_neighbors.float())**2 + (grid_y.unsqueeze(2) - y_neighbors.float())**2)  # (B, N, 9)
    repulsive_force = reformulated_potential * torch.exp(-10.0 * distance_with_neighbors)  # (B, N, 9)
    repulsive_force = repulsive_force.sum(dim=2)  # (B, N)

    x_down = torch.floor(grid_x_no_grad).long()
    y_down = torch.floor(grid_y_no_grad).long()
    x_up = x_down + 1
    y_up = y_down + 1
    x_up = torch.clamp(x_up, 0, W-1)
    x_down = torch.clamp(x_down, 0, W-1)
    y_up = torch.clamp(y_up, 0, H-1)
    y_down = torch.clamp(y_down, 0, H-1)
    dist_11 = distance_map[torch.arange(B).unsqueeze(1), x_up, y_up]  # (B, N)
    dist_10 = distance_map[torch.arange(B).unsqueeze(1), x_up, y_down]  # (B, N)
    dist_01 = distance_map[torch.arange(B).unsqueeze(1), x_down, y_up]  # (B, N)
    dist_00 = distance_map[torch.arange(B).unsqueeze(1), x_down, y_down]  # (B, N)
    
    all_dists = torch.stack([dist_00, dist_01, dist_10, dist_11], dim=-1)
    
    min_indices = torch.argmin(all_dists, dim=-1)
    
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
    point_indices = torch.arange(N).unsqueeze(0).expand(B, N)
    
    min_dist = all_dists[batch_indices, point_indices, min_indices]  # (B, N)

    dist_11 = dist_11 - min_dist
    dist_10 = dist_10 - min_dist
    dist_01 = dist_01 - min_dist
    dist_00 = dist_00 - min_dist
    
    mu = 2.0
    dist_11 = dist_11 ** mu
    dist_10 = dist_10 ** mu
    dist_01 = dist_01 ** mu
    dist_00 = dist_00 ** mu
    
    max_delta = 100.0
    dist_11 = torch.clamp(dist_11, max=max_delta)
    dist_10 = torch.clamp(dist_10, max=max_delta)
    dist_01 = torch.clamp(dist_01, max=max_delta)
    dist_00 = torch.clamp(dist_00, max=max_delta)

    dist_11 = dist_11 + min_dist
    dist_10 = dist_10 + min_dist
    dist_01 = dist_01 + min_dist
    dist_00 = dist_00 + min_dist
    
    wa = grid_x - x_down.float()  # (B, N)
    wb = grid_y - y_down.float()  # (B, N)
    distances = (1 - wa) * (1 - wb) * dist_00 + wa * (1 - wb) * dist_10 + \
                (1 - wa) * wb * dist_01 + wa * wb * dist_11  # (B, N)
    return distances + distances_with_min_point_8 * config['guide_weight']

def world_to_grid(x, y, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    i = (x - xmin) / resolution
    j = (y - ymin) / resolution
    return i, j

def obj_fn(data, y, config):
    # Use the distance map
    distance_map = data['distance_map']  # (batch_size, H, W)
    target = data['target']  # (batch_size, 2)
    xy = y.view(y.shape[0], -1, 7)[:, :, :2]  # (batch_size, N, 2)
    B, N, _ = xy.shape
    end_point = xy[:, -1, :2]  # (batch_size, 2)
    world_x_end = xy[:, -1, 0].unsqueeze(1)  # (B, 1)
    world_y_end = xy[:, -1, 1].unsqueeze(1)  # (B, 1)
    world_x = xy[:, :, 0]  # (B, N)
    world_y = xy[:, :, 1]  # (B, N)
    i_end, j_end = world_to_grid(world_x_end, world_y_end)  # (B, 1)
    i, j = world_to_grid(world_x, world_y)  # (B, N)

    distances = get_map_distance(distance_map, i, j, config)  # (B, N)
    map_loss = distances.mean()
    # return map_loss

    terminal_point = target  # (batch_size, 2)
    obj = torch.norm(end_point - terminal_point, dim=1)  # (B,)
    obj_loss = -1/(torch.exp(1.5*obj-5) + 1.0) + 1.0  # (B, )
    
    xy_heading = xy2xy_heading(xy)  # (batch_size, N+1, 3)
    soft_loss = soft_constraints(xy_heading, config['obs_constraints_weight']*1e3, data['obstacles_vertices'], mode='single_value')  # (B, 3)

    return  obj_loss.mean() + soft_loss + map_loss  # stage 1
    return  obj_loss.mean() + map_loss  # stage 2