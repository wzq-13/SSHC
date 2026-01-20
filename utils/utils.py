import numpy as np
import torch
import matplotlib.pyplot as plt
import globalvar
from utils.prob import xy2xy_heading
import os

planning_scale_ = globalvar.planning_scale_
Nobs = globalvar.Nobs
vehicle_TPBV_ = globalvar.vehicle_TPBV_
vehicle_geometrics_ = globalvar.vehicle_geometrics_
vehicle_kinematics_ = globalvar.vehicle_kinematics_
margin_obs_ = globalvar.margin_obs_ 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'


def xy2xy_heading_numpy(xy):
    headings = []
    for i in range(len(xy)):
        if i == 0:
            dx = xy[i+1, 0] - xy[i, 0]
            dy = xy[i+1, 1] - xy[i, 1]
        elif i == len(xy) - 1:
            dx = xy[i, 0] - xy[i-1, 0]
            dy = xy[i, 1] - xy[i-1, 1]
        else:
            dx = xy[i+1, 0] - xy[i-1, 0]
            dy = xy[i+1, 1] - xy[i-1, 1]
        heading = np.arctan2(dy, dx)
        headings.append(heading)
        
    headings = np.array(headings)
    xy_heading = np.hstack([xy, headings.reshape(-1, 1)])
    return xy_heading

def get_rect_points_vectorized(xy_heading, width=0.5, length=1.0):
    if not isinstance(xy_heading, torch.Tensor):
        xy_heading = torch.tensor(xy_heading, dtype=torch.float32)
    
    N = xy_heading.shape[0]
    cos_h = torch.cos(xy_heading[:, 2])  # (N,)
    sin_h = torch.sin(xy_heading[:, 2])  # (N,)

    half_w = width / 2.0
    half_l = length / 2.0

    offsets = torch.tensor([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w]
    ], dtype=xy_heading.dtype, device=xy_heading.device)

    rotated_offsets = torch.zeros((N, 4, 2), dtype=xy_heading.dtype, device=xy_heading.device)
    
    for i in range(4):
        dx, dy = offsets[i]
        rotated_offsets[:, i, 0] = dx * cos_h - dy * sin_h
        rotated_offsets[:, i, 1] = dx * sin_h + dy * cos_h

    centers = xy_heading[:, :2].unsqueeze(1)  # (N, 1, 2)
    rectangles = centers + rotated_offsets  # (N, 4, 2)
    
    return rectangles

def path_smoothness(path):
    """
    calculate the smoothness score of the path
    path: numpy array of shape (N, 2)
    """
    path = np.array(path)
    if len(path) < 3:
        
        return 1.0, 1.0
    curvatures = []
    for i in range(1, len(path) - 1):
        p1 = path[i - 1]
        p2 = path[i]
        p3 = path[i + 1]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a == 0 or b == 0 or c == 0:
            curvature = 0

        elif abs(a + b - c) < 1e-6 or abs(b + c - a) < 1e-6 or abs(c + a - b) < 1e-6:
            curvature = 0
        else:
            curvature = (np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) / (a * b * c)
        
        curvatures.append(curvature)
    curvatures = np.array(curvatures)

    curvature_changes = np.abs(np.diff(curvatures))
    smoothness = np.mean(curvature_changes)
    
    min_turning_radius = vehicle_kinematics_.min_turning_radius
    max_curvature = 1.0 / min_turning_radius
    
    radius = 1.0 / (curvatures + 1e-6)
    score_per_point = np.clip(radius / min_turning_radius, 0, 1)
    score = np.mean(score_per_point)

    return smoothness, score


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
    if poly1.shape == (4,2):
        poly1 = poly1.T
    if poly2.shape == (4,2):
        poly2 = poly2.T
    
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("input must be a 2x4 numpy array")

    if check_edges_intersection(poly1, poly2):
        return True
    
    if check_containment(poly1, poly2):
        return True
    
    return False

def h(x, y, polygons_edges, rho=10.0):
    all_distances = []
    for edge_set in polygons_edges:
        distances = []
        for a, b, c in edge_set:
            d = (a * x + b * y + c) / np.sqrt(a**2 + b**2)
            distances.append(d)
        all_distances.append(np.min(np.array(distances)))
    h1 = np.max(np.array(all_distances))
    return h1 + 1.5

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