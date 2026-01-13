import math
import numpy as np

class vehicle_geometrics_Set:
    def __init__(self) -> None:
        self.vehicle_wheelbase = 2.8  # L_W,wheelbase of the ego vehicle (m)
        self.vehicle_front_hang = 0.96 # L_F,front hang length of the ego vehicle (m)
        self.vehicle_rear_hang = 0.93 # L_R,rear hang length of the ego vehicle (m)
        self.vehicle_width = 1.9 # width of the ego vehicle (m)
        self.vehicle_half_width = 0.5*self.vehicle_width
        self.vehicle_length = self.vehicle_wheelbase + self.vehicle_front_hang + self.vehicle_rear_hang # length of the ego vehicle (m)
        self.Safety_margin = 1.25

# vehicle kinematics settings
class vehicle_kinematics_Set:
    def __init__(self) -> None:
        self.vehicle_v_max = 10.0  # upper and lower bounds of v(t) (m/s)
        self.vehicle_v_min = -2.5 # upper and lower bounds of v(t) (m/s)
        self.vehicle_a_max = 2.0
        self.vehicle_a_min = -0.5 # upper and lower bounds of a(t) (m/s^2)
        self.vehicle_jerk_max = 0.5
        self.vehicle_jerk_min = -0.5 # upper and lower bounds of jerk(t) (m/s^3) 
        self.vehicle_phi_max = 0.7
        self.vehicle_phi_min = -0.7 # upper and lower bounds of phi(t) (rad)
        self.vehicle_omega_max = 0.5
        self.vehicle_omega_min = -0.5 # upper and lower bounds of omega(t) (rad/s)

        global vehicle_geometrics_ 
        self.min_turning_radius = vehicle_geometrics_.vehicle_wheelbase/math.tan(self.vehicle_phi_max)
        
class planning_scale_Set:
    def __init__(self) -> None:
        self.xmin=-4
        self.xmax=36
        self.ymin=-10
        self.ymax=10  # space is a rectange, [lx,ux],[ly,uy]
        self.x_scale = self.xmax - self.xmin
        self.y_scale = self.ymax - self.ymin
        self.resolution = 0.2
        self.target_x_min = 30
        self.target_x_max = 34
        self.target_y_min = -8
        self.target_y_max = 8
        self.obs_x_min = 4
        self.obs_x_max = 28
        self.obs_y_min = -10
        self.obs_y_max = 10
class vehicle_TPBV_Set:
    def __init__(self) -> None:
        self.x0=0
        self.y0=0
        self.theta0=0
        self.v0=0
        self.phi0=0
        self.a0=0
        self.omega0=0

        self.xtf=32
        self.ytf=32
        self.thetatf=0
        self.vtf=0
        self.phitf=0
        self.atf=0
        self.omegatf=0
class vclass:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.A = 0

global vehicle_geometrics_
vehicle_geometrics_ = vehicle_geometrics_Set()

global vehicle_kinematics_
vehicle_kinematics_ = vehicle_kinematics_Set()

global vehicle_TPBV_
vehicle_TPBV_ = vehicle_TPBV_Set()

global planning_scale_
planning_scale_ = planning_scale_Set()

global Nobs 
global margin_obs_ # margin_obs_ for dilated obstacles
margin_obs_=0.5
Nobs = 8