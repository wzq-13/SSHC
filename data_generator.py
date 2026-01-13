import numpy as np
import math
import random
from random import randint, random as rand
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import globalvar
from collections import deque
from itertools import chain
import multiprocessing
import os
MAX_LEN = 4.0
MIN_LEN = 1.0
def triArea(a, b, c):
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def grid_to_world(i, j, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    x = xmin + i * resolution
    y = ymin + j * resolution
    return (x, y)

def world_to_grid(x, y, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    i = round((x - xmin) / resolution)
    j = round((y - ymin) / resolution)
    return (i, j)

def obstacle_blowup_quadrilateral(obstacle, blowup_distance):
    poly = np.array(obstacle)
    signed_area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], 1) - 
                                poly[:, 1] * np.roll(poly[:, 0], 1))
    if signed_area < 0:
        poly = poly[::-1]

    edges = np.roll(poly, -1, axis=0) - poly
    edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    edge_lengths[edge_lengths < 1e-6] = 1e-6 
    unit_edges = edges / edge_lengths
    normals = np.stack([unit_edges[:, 1], -unit_edges[:, 0]], axis=1)

    n_curr = normals
    n_prev = np.roll(normals, 1, axis=0)
    denom = n_prev[:, 0] * n_curr[:, 1] - n_prev[:, 1] * n_curr[:, 0]
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    d = blowup_distance
    delta_x = d * (n_prev[:, 1] - n_curr[:, 1]) / denom
    delta_y = d * (n_curr[:, 0] - n_prev[:, 0]) / denom
    delta = np.stack([delta_x, delta_y], axis=1)
    
    new_poly = poly + delta # (4,2)

    return new_poly

def inpolygon(x, y, xv, yv):
    n = len(xv)
    inside = False
    p1x, p1y = xv[0], yv[0]
    for i in range(1, n+1):
        p2x, p2y = xv[i % n], yv[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
def check_segment_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    A, B = p1, p2
    C, D = p3, p4
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def is_simple_polygon(poly):
    n = poly.shape[1]
    for i in range(n):
        for j in range(i+1, n):
            if check_segment_intersection(
                poly[:,i], poly[:,(i+1)%n],
                poly[:,j], poly[:,(j+1)%n]
            ):
                return False
    return True

def polygon_edges(vertices):
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    n = vertices.shape[0]
    edges = []
    for i in range(n):
        v = vertices[i]
        edge_set = []
        for j in range(4):
            x1, y1 = v[j]
            x2, y2 = v[(j + 1) % 4]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            edge_set.append((a, b, c))
        edges.append(edge_set)
    return edges

def check_polygon_intersection(poly1, poly2):
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("input must be 2x4 numpy arrays")
    
    if check_edges_intersection(poly1, poly2):
        return True

    # Check containment
    if check_containment(poly1, poly2):
        return True
    
    return False

def h(x, y, polygons_edges, rho=10.0):
    '''
    -x: point's x coordinate
    -y: point's y coordinate
    -polygons_edges: list of edges for obstacle polygons, each edge is a triplet (a, b, c) with shape (m,4,3)
    '''
    all_distances = []
    for edge_set in polygons_edges:
        distances = []
        for a, b, c in edge_set:
            d = (a * x + b * y + c) / np.sqrt(a**2 + b**2)
            distances.append(d)
        all_distances.append(np.min(np.array(distances)))
    h1 = np.max(np.array(all_distances))
    return h1  + 1.25

def check_edges_intersection(poly1, poly2):
    """Check if the edges of two polygons intersect."""
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
    """Check if two line segments intersect."""
    # Ensure all points are 2D coordinates
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

class VehiclePolygon:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.polygon = self._create_polygon()
        
    def _create_polygon(self):
        length = globalvar.vehicle_geometrics_.vehicle_length
        width = globalvar.vehicle_geometrics_.vehicle_width
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        corners = np.array([
            [self.x + length/2*cos_theta - width/2*sin_theta, 
             self.y + length/2*sin_theta + width/2*cos_theta],
            [self.x + length/2*cos_theta + width/2*sin_theta,
             self.y + length/2*sin_theta - width/2*cos_theta],
            [self.x - length/2*cos_theta + width/2*sin_theta,
             self.y - length/2*sin_theta - width/2*cos_theta],
            [self.x - length/2*cos_theta - width/2*sin_theta,
             self.y - length/2*sin_theta + width/2*cos_theta]
        ])
        return corners

def CreateVehiclePolygon(x, y, theta):
    length = globalvar.vehicle_geometrics_.vehicle_length
    width = globalvar.vehicle_geometrics_.vehicle_width
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    corners = np.array([
        [x + length/2*cos_theta - width/2*sin_theta, 
         y + length/2*sin_theta + width/2*cos_theta],
        [x + length/2*cos_theta + width/2*sin_theta,
         y + length/2*sin_theta - width/2*cos_theta],
        [x - length/2*cos_theta + width/2*sin_theta,
         y - length/2*sin_theta - width/2*cos_theta],
        [x - length/2*cos_theta - width/2*sin_theta,
         y - length/2*sin_theta + width/2*cos_theta]
    ])
    return VehiclePolygon(corners[:,0], corners[:,1], theta)

def visualize_environment(planning_scale, vehicle_TPBV, obstacles):
    plt.figure(figsize=(10, 10))
    
    plt.plot([planning_scale['xmin'], planning_scale['xmax']], 
             [planning_scale['ymin'], planning_scale['ymin']], 'k-')
    plt.plot([planning_scale['xmin'], planning_scale['xmax']], 
             [planning_scale['ymax'], planning_scale['ymax']], 'k-')
    plt.plot([planning_scale['xmin'], planning_scale['xmin']], 
             [planning_scale['ymin'], planning_scale['ymax']], 'k-')
    plt.plot([planning_scale['xmax'], planning_scale['xmax']], 
             [planning_scale['ymin'], planning_scale['ymax']], 'k-')
    
    V_initial = VehiclePolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0'])
    V_terminal = VehiclePolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf'])
    from matplotlib.patches import Polygon
    initial_poly = Polygon(V_initial.polygon, closed=True, fill=True, color='green', alpha=0.5, label='Initial Position')
    terminal_poly = Polygon(V_terminal.polygon, closed=True, fill=True, color='blue', alpha=0.5, label='Terminal Position')
    
    ax = plt.gca()
    ax.add_patch(initial_poly)
    ax.add_patch(terminal_poly)
    
    for i, obs in enumerate(obstacles):
        poly = Polygon(np.column_stack((obs['x'], obs['y'])), closed=True, 
                      fill=True, color='red', alpha=0.3)
        ax.add_patch(poly)
        
        centroid_x = sum(obs['x']) / 4
        centroid_y = sum(obs['y']) / 4
        plt.text(centroid_x, centroid_y, str(i+1), ha='center', va='center', color='black')
    
    for obs in obstacles:
        margin = obs['margin']
        x_margin = [
            obs['x'][0] - margin, obs['x'][1] + margin,
            obs['x'][2] + margin, obs['x'][3] - margin
        ]
        y_margin = [
            obs['y'][0] + margin, obs['y'][1] + margin,
            obs['y'][2] - margin, obs['y'][3] - margin
        ]
        plt.plot(x_margin + [x_margin[0]], y_margin + [y_margin[0]], 'r--', linewidth=0.5)
    
    plt.title('Generated Obstacles Environment')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.xlim(planning_scale['xmin'] - 5, planning_scale['xmax'] + 5)
    plt.ylim(planning_scale['ymin'] - 5, planning_scale['ymax'] + 5)
    
    plt.show()

def GenerateStaticObstacles_unstructured(planning_scale, vehicle_TPBV, vehicle_geometrics, Nobs):
    lx = globalvar.planning_scale_.obs_x_min
    ux = globalvar.planning_scale_.obs_x_max
    ly = globalvar.planning_scale_.obs_y_min
    uy = globalvar.planning_scale_.obs_y_max

    V_initial = CreateVehiclePolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0'])
    V_terminal = CreateVehiclePolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf'])

    s_k = (vehicle_TPBV['ytf'] - vehicle_TPBV['y0']) / (vehicle_TPBV['xtf'] - vehicle_TPBV['x0'] + 1e-6)
    s_b = vehicle_TPBV['y0'] - s_k * vehicle_TPBV['x0']
    
    obstacles = []
    obj = None
    margin = 2.5
    count = 0
    max_attempts = 100 * Nobs
    
    while count < Nobs and max_attempts > 0:
        max_attempts -= 1
        
        # 生成随机四边形
        while True:
            x = (ux - lx) * rand() + lx
            if count == 1:
                y = s_k * x + s_b + 1 + rand() 
            elif count == 2:
                y = -1 * rand() + uy
            elif count == 3:
                y = 1 * rand() + ly
            else:
                y = (uy - ly) * rand() + ly
            theta = 2 * math.pi * rand() - math.pi
            
            xru = x + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)
            yru = y + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            xrd = xru + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            yrd = yru - (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)
            xld = x + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            yld = y - (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)

            if (xru < lx or xru > ux or xrd < lx or xrd > ux or 
                xld < lx or xld > ux or yru < ly or yru > uy or 
                yrd < ly or yrd > uy or yld < ly or yld > uy):
                continue
                
            temp_obj = np.array([[x, xru, xrd, xld], [y, yru, yrd, yld]])
            temp_obj_margin  = obstacle_blowup_quadrilateral(temp_obj.T, margin).T
            
            if is_simple_polygon(temp_obj):
                break
        
        xv = temp_obj_margin[0, :].tolist() + [temp_obj_margin[0, 0]]
        yv = temp_obj_margin[1, :].tolist() + [temp_obj_margin[1, 0]]
        
        if (inpolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], xv, yv) or 
            inpolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], xv, yv)):
            continue
        
        vehicle_poly = np.array([V_initial.x, V_initial.y])#.T
        if check_polygon_intersection(temp_obj_margin, vehicle_poly):
            continue
            
        vehicle_poly = np.array([V_terminal.x, V_terminal.y])#.T
        if check_polygon_intersection(temp_obj_margin, vehicle_poly):
            continue
        
        collision = False
        if obj is not None:
            n = obj.shape[1] // 4
            for i in range(n):
                existing_obs = obj[:, 4*i:4*i+4]
                if check_polygon_intersection(temp_obj_margin, existing_obs):
                    collision = True
                    break
        if collision:
            continue
        
        obstacle = {
            'x': temp_obj[0, :].tolist(),
            'y': temp_obj[1, :].tolist(),
        }
        obstacles.append(obstacle)
        
        if obj is None:
            obj = temp_obj.copy()
        else:
            obj = np.hstack((obj, temp_obj))
        
        count += 1
    
    if count < Nobs:
        raise RuntimeError("can not generate enough obstacles")
    
    return obstacles

def generate_navigation_graph(obstacles_vertices, obstacles_numpy, car_width, resolution, target_node, init_node):
    obstacles = [Polygon(vertices) for vertices in obstacles_vertices]
    obstacle_edges = polygon_edges(obstacles_numpy)
    min_x, max_x = globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax
    min_y, max_y = globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax
    xs = np.arange(min_x, max_x + resolution, resolution)
    ys = np.arange(min_y, max_y + resolution, resolution)
    W = len(xs)
    H = len(ys)
    valid_points = {}
    invalid_points = {}
    point_to_index = {}
    index_to_point = {}
    point_index = 0
    
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            is_inside_obstacle = False
            if abs(y) >= 10 or h(x, y, obstacle_edges) > 0:
                is_inside_obstacle = True
            
            if not is_inside_obstacle:
                valid_points[(i, j)] = (x, y)
                point_to_index[(i, j)] = point_index
                index_to_point[point_index] = (x, y)
                point_index += 1
            else:
                invalid_points[(i, j)] = (x, y)
    
    graph = {}
    graph_invalid = {}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for grid_pos in valid_points.keys():
        graph[grid_pos] = []
    for grid_pos in invalid_points.keys():
        graph_invalid[grid_pos] = []
    
    distance_dict = {node: -1 for node in valid_points.keys()}
    distance_dict[target_node] = 0
    
    for (i, j), world_coord in valid_points.items():
        for di, dj in directions:
            ni, nj = i + di, j + dj
            neighbor_pos = (ni, nj)
            
            if neighbor_pos in valid_points:
                # check if the edge is blocked by an obstacle
                p1 = valid_points[(i, j)]
                p2 = valid_points[neighbor_pos]

                # check if the edge intersects with any obstacle boundaries
                blocked_by_obstacle = False
                mid_point = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                mid_mid_point_1 = (p1[0] + mid_point[0]) / 2, (p1[1] + mid_point[1]) / 2
                mid_mid_point_2 = (mid_point[0] + p2[0]) / 2, (mid_point[1] + p2[1]) / 2
                if h(mid_point[0], mid_point[1], obstacle_edges) > 0 or h(mid_mid_point_1[0], mid_mid_point_1[1], obstacle_edges) > 0 or h(mid_mid_point_2[0], mid_mid_point_2[1], obstacle_edges) > 0:
                    blocked_by_obstacle = True
                
                # Add only when the edge is not blocked
                if  not blocked_by_obstacle:
                    graph[(i, j)].append(neighbor_pos)
    # BFS queue
    queue = deque([target_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if distance_dict[neighbor] == -1:  # 未访问过
                distance_dict[neighbor] = distance_dict[current_node] + 1
                queue.append(neighbor)

    if distance_dict[init_node] == -1:
        raise ValueError("No path found")
    # For valid points that are still at a distance of 1, change them to invalid points.
    keys_to_delete = []
    for node in distance_dict:
        if distance_dict[node] == -1:
            invalid_points[node] = valid_points[node]
            graph_invalid[node] = []
            keys_to_delete.append(node) # Record the valid key points that need to be deleted
    for key in keys_to_delete:
        del valid_points[key]
        del graph[key]
        del distance_dict[key]
    
    shortest_path = [init_node]
    directions_8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] #, (1, 1), (1, -1), (-1, 1), (-1, -1)
    current_node = init_node
    while current_node != target_node:
        min_distance = float('inf')
        best_node = None
        for di, dj in directions_8:
            ni, nj = current_node[0] + di, current_node[1] + dj
            neighbor_pos = (ni, nj)
            if neighbor_pos in distance_dict and distance_dict[neighbor_pos] != -1:
                if distance_dict[neighbor_pos] < min_distance:
                    min_distance = distance_dict[neighbor_pos]
                    best_node = neighbor_pos
        shortest_path.append(best_node)
        current_node = best_node
    
    distance_from_shortest_path = {node: -1 for node in valid_points.keys()}
    for node in shortest_path:
        distance_from_shortest_path[node] = 0
    q = deque(shortest_path)
    while q:
        u = q.popleft()
        for neighbor_pos in graph[u]:
            if distance_from_shortest_path[neighbor_pos] == -1:
                distance_from_shortest_path[neighbor_pos] = distance_from_shortest_path[u] + 1
                q.append(neighbor_pos)
    
    for node in valid_points.keys():
        distance_dict[node] = distance_from_shortest_path[node]*2 + distance_dict[node]
        
    distance_map_invalid = {}
    q = deque()
    for (i, j), world_coord in invalid_points.items():
        for di, dj in directions:
            ni, nj = i + di, j + dj
            neighbor_pos = (ni, nj)
            distance_map_invalid[(i, j)] = -1 # Unvisited Flag
            
            if neighbor_pos in valid_points:
                graph_invalid[(i, j)].append(neighbor_pos)
                if not neighbor_pos in graph_invalid: # Valid points adjacent to invalid points that have not been visited before
                    graph_invalid[neighbor_pos] = []
                    distance_map_invalid[neighbor_pos] = distance_dict[neighbor_pos]
                    q.append(neighbor_pos)
                    for di, dj in directions:
                        mi, mj = ni + di, nj + dj
                        neighbor_pos_2 = (mi, mj)
                        if neighbor_pos_2 in invalid_points:
                            graph_invalid[neighbor_pos].append(neighbor_pos_2)
            elif neighbor_pos in invalid_points:
                graph_invalid[(i, j)].append(neighbor_pos)
        
    while q:
        u = q.popleft()
        delta = 10 if u in valid_points else 20
        for neighbor_pos in graph_invalid[u]:
            if distance_map_invalid[neighbor_pos] == -1:
                distance_map_invalid[neighbor_pos] = distance_map_invalid[u] + delta
                q.append(neighbor_pos)
        
    # Delete the valid points in distance map invalid
    distance_map_invalid = {k: v for k, v in distance_map_invalid.items() 
                       if k not in valid_points}

    return graph, valid_points, invalid_points, distance_dict, distance_map_invalid, W, H, shortest_path

def visualize_navigation_graph(obstacles_vertices, graph, valid_points, blocking_lines, 
                             car_width, resolution, terminal_point, initial_point=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for i, vertices in enumerate(obstacles_vertices):
        polygon = Polygon(vertices)
        x, y = polygon.exterior.xy
        ax1.fill(x, y, alpha=0.3, color='pink', label='Obstacles' if i == 0 else "")
        ax1.plot(x, y, 'r--', linewidth=1)
        ax1.text(np.mean(x), np.mean(y), str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.fill(x, y, alpha=0.3, color='pink', label='Obstacles' if i == 0 else "")
        ax2.plot(x, y, 'r--', linewidth=1)
        ax2.text(np.mean(x), np.mean(y), str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i, line in enumerate(blocking_lines):
        x, y = line.xy
        ax1.plot(x, y, 'g-', linewidth=3, label='Blocking Lines' if i == 0 else "")
        ax2.plot(x, y, 'g-', linewidth=3, label='Blocking Lines' if i == 0 else "")
    
    all_points = list(valid_points.values())
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    ax1.scatter(xs, ys, color='blue', s=10, alpha=0.6, label='Grid Points')
    ax2.scatter(xs, ys, color='blue', s=10, alpha=0.6, label='Grid Points')
    
    edge_count = 0
    for grid_pos, neighbors in graph.items():
        p1 = valid_points[grid_pos]
        for neighbor_grid in neighbors:
            p2 = valid_points[neighbor_grid]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.5, linewidth=0.5)
            edge_count += 1
    
    if initial_point:
        ax1.plot(initial_point[0], initial_point[1], 'gs', markersize=10, label='Initial Position')
        ax2.plot(initial_point[0], initial_point[1], 'gs', markersize=10, label='Initial Position')
    
    ax1.plot(terminal_point[0], terminal_point[1], 'b^', markersize=10, label='Terminal Position')
    ax2.plot(terminal_point[0], terminal_point[1], 'b^', markersize=10, label='Terminal Position')
    
    ax1.set_xlim(globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax)
    ax1.set_ylim(globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Grid Points and Obstacles\n(Car Width: {car_width}m, Resolution: {resolution}m)')
    ax1.legend()
    
    ax2.set_xlim(globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax)
    ax2.set_ylim(globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Navigation Graph\n({len(valid_points)} nodes, {edge_count} edges)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
def graph_to_adjacency_list(graph, valid_points):
    adjacency_list = {}
    
    for grid_pos, neighbors in graph.items():
        world_coord = valid_points[grid_pos]
        adjacency_list[world_coord] = []
        for neighbor_grid in neighbors:
            neighbor_world = valid_points[neighbor_grid]
            adjacency_list[world_coord].append(neighbor_world)
    
    return adjacency_list

def compute_shortest_distances(graph, valid_points, target_node):
    distance_dict = {node: -1 for node in valid_points.keys()}
    distance_dict[target_node] = 0
    
    # BFS queue
    queue = deque([target_node])
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in graph[current_node]:
            if distance_dict[neighbor] == -1:  # Not visited
                distance_dict[neighbor] = distance_dict[current_node] + 1
                queue.append(neighbor)
    
    return distance_dict


def dict_map_to_array(grid_dict, W, H, default_value=np.inf):
    """
    将单个字典映射转换为 2D 数组 [H, W]
    Args:
        grid_dict: Dict[Tuple[int,int], float], (i,j) -> distance 的映射
        H, W: 网格的高度和宽度
        default_value: 默认填充值（用于字典中不存在的键）
    Returns:
        distance_array: ndarray [H, W], dtype=np.float32
    """
    distance_array = np.full((W, H), default_value, dtype=np.float32)
    for (i, j), dist in grid_dict.items():
        if 0 <= i < W and 0 <= j < H:
            distance_array[i, j] = dist
    # 检查是否所有位置都被填充
    if np.any(distance_array == default_value):
        # for i in range(H):
        #     for j in range(W):
        #         if distance_array[i, j] == default_value:
        #             print(f"Warning: Position ({i}, {j}) was not filled in the distance array.")
        raise ValueError("Some grid positions were not filled in the distance array.")
    return distance_array

def ensure_clockwise_vertices(vertices):
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 2):
        raise ValueError("shape should be [k, 4, 2]")
    
    k = vertices.shape[0]
    
    # Extract four vertices
    p1 = vertices[:, 0]  # [k, 2]
    p2 = vertices[:, 1]  # [k, 2]
    p3 = vertices[:, 2]  # [k, 2]
    p4 = vertices[:, 3]  # [k, 2]

    # Compute the signed area (2 times the area, ignore the 1/2 factor)
    # Formula: sum = (x1y2 - x2y1) + (x2y3 - x3y2) + (x3y4 - x4y3) + (x4y1 - x1y4)
    sum_val = (p1[:, 0] * p2[:, 1] - p2[:, 0] * p1[:, 1] + 
               p2[:, 0] * p3[:, 1] - p3[:, 0] * p2[:, 1] + 
               p3[:, 0] * p4[:, 1] - p4[:, 0] * p3[:, 1] + 
               p4[:, 0] * p1[:, 1] - p1[:, 0] * p4[:, 1])

    # Determine orientation: sum_val < 0 means clockwise, sum_val > 0 means counterclockwise
    # We need clockwise, so if sum_val > 0 (counterclockwise), we need to flip
    need_flip = sum_val > 0  # [k]

    # Create corrected vertices
    corrected_vertices = vertices.copy()
    
    # For quadrilaterals that need to be flipped, reverse the vertex order from [p1, p2, p3, p4] to [p1, p4, p3, p2]
    if np.any(need_flip):
        # Build the reversed order
        flipped_order = np.stack([p1, p4, p3, p2], axis=1)  # [k, 4, 2]
        
        # Use Boolean indexing to select the quadrilaterals to flip
        corrected_vertices[need_flip] = flipped_order[need_flip]
    
    return corrected_vertices

def generate_map_data():
    theta0 = np.random.uniform(0, math.pi/2.)
    planning_scale = {
        'xmin': globalvar.planning_scale_.xmin, 'xmax': globalvar.planning_scale_.xmax,
        'ymin': globalvar.planning_scale_.ymin, 'ymax': globalvar.planning_scale_.ymax
    }
    
    vehicle_TPBV = {
        'x0': globalvar.vehicle_TPBV_.x0, 'y0': globalvar.vehicle_TPBV_.y0, 'theta0': theta0,
        'xtf': globalvar.vehicle_TPBV_.xtf, 'ytf': globalvar.vehicle_TPBV_.ytf, 'thetatf': globalvar.vehicle_TPBV_.thetatf
    }
    
    vehicle_geometrics = {
        'vehicle_length': globalvar.vehicle_geometrics_.vehicle_length,
        'vehicle_width': globalvar.vehicle_geometrics_.vehicle_width
    }
    
    Nobs = 8  # Number of obstacles to generate

    terminal_point = np.array([vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf']])
    initial_point = np.array([vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0']])
    obstacles = GenerateStaticObstacles_unstructured(
        planning_scale, vehicle_TPBV, vehicle_geometrics, Nobs
    )
    
    obstacles_vertices = [
        list(zip(obs['x'], obs['y'])) for obs in obstacles
    ]
    obstacles_numpy = np.array(obstacles_vertices).reshape(-1, 4, 2)
    car_width = globalvar.vehicle_geometrics_.vehicle_width
    resolution = globalvar.planning_scale_.resolution
    target_node_grid = world_to_grid(terminal_point[0], terminal_point[1])
    initial_node_grid = world_to_grid(initial_point[0], initial_point[1])
    
    graph, valid_points, invalid_points, distance_dict, distance_map_invalid, W, H, shortest_path = generate_navigation_graph(
        obstacles_vertices, obstacles_numpy, car_width*1.1, resolution, target_node_grid, initial_node_grid
    )
    dict1 = distance_dict
    dict2 = distance_map_invalid
    grid_dict = {**dict1, **dict2}
    distance_map = dict_map_to_array(grid_dict, W, H, default_value=np.inf)
    obstacles_pure = chain.from_iterable(obstacles_vertices)
    obstacles_pure = list(chain.from_iterable(obstacles_pure))
    obstacles_pure = np.array(obstacles_pure).reshape(-1, 4, 2)
    obstacles_pure = ensure_clockwise_vertices(obstacles_pure)
    data={
        'obstacles_vertices': obstacles_pure,
        'distance_map': distance_map,
        'target': terminal_point[:2],
    }
    return data

def deal_single_frame(index):
    np.random.seed(index)
    random.seed(index)
    save_path = f'./dataset/{index}.npz'
    globalvar.vehicle_TPBV_.xtf = globalvar.planning_scale_.target_x_min + rand() * (globalvar.planning_scale_.target_x_max - globalvar.planning_scale_.target_x_min)
    globalvar.vehicle_TPBV_.ytf = globalvar.planning_scale_.target_y_min + rand() * (globalvar.planning_scale_.target_y_max - globalvar.planning_scale_.target_y_min)
    target = np.array([globalvar.vehicle_TPBV_.xtf, globalvar.vehicle_TPBV_.ytf])
    
    if os.path.exists(save_path):
        return
    success = False
    while not success:
        try:
            data = generate_map_data()
            data['target'] = target
            np.savez(save_path, **data)
            success = True
        except Exception as e:
            if e is KeyboardInterrupt:
                raise e
            print(f"Error processing index {index}: {e}")
            # 重新生成随机种子
            new_index = index + 10000000 if index < 10000000 else index + randint(1, 10000000)
            index = new_index
            np.random.seed(index)
            random.seed(index)


if __name__ == "__main__":
    num = 200000
    cpu_num = multiprocessing.cpu_count()
    indexs = [randint(0, num) for _ in range(cpu_num - 1)]
    with multiprocessing.Pool(processes=cpu_num - 1) as pool:
        pool.map(deal_single_frame, indexs)