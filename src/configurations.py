#!/usr/bin/env python3

# Imports
import numpy as np
from numba import cuda
import enum
import torch
import scipy.stats as stats

class PlannerModel(enum.Enum):
    """Dynamics used for the motion planner"""
    UNICYCLE_DYNAMICS = 0
    EDD5 = 1
    GAUSSIAN_PROCESS = 2

class Config:
    """Configuration parameters"""
    
    def __init__(self):
        
        # Sanity check for GPU
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot run MPPI, GPU needed for sampling trajectories")
        else:
            self.device = "cuda"
        
        # Information about GPU
        gpu = cuda.get_current_device()
        
        # Dynamics used in MPPI planner
        self.planner_model = PlannerModel.UNICYCLE_DYNAMICS
        
        # Maximum threads per block
        self.max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
        
        # Maximum dimension of a square grid
        self.max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
        
        # Maximum number of blocks in x direction per grid
        self.max_blocks = gpu.MAX_GRID_DIM_X
        
        # Maximum recommended number of blocks and the number of control rollouts
        self.max_rec_blocks = self.rec_max_control_rollouts = int(1e6)
        
        # Minimum recommended number of control rollouts
        self.rec_min_control_rollouts = 100
        
        # Seed for random number generator
        self.seed = 1
        
        # MPC horizon in seconds
        self.T = 4
        
        # MPC discretization
        self.dt = 0.1
        
        # Number of MPC timesteps
        self.num_steps = int(self.T / self.dt)
        
        # Number of control rollouts
        self.num_control_rollouts = 1024
        
        # For visualization of state rollouts
        self.num_vis_state_rollouts = 25
        
        # Obstacle avoidance weight
        self.collision_cost = 1e6
        
        # Stage cost for distance to goal
        self.dist_weight = 10.
        
        # Terminal cost if not at the goal
        self.terminal_cost_if_not_at_goal = 1
        
        # MPPI temperature parameter
        self.lambda_weight = 1.0
        
        # Commanded linear velocity standard deviation
        self.linear_velocity_variance = 1.
        
        # Commanded angular velocity standard deviation
        self.angular_velocity_variance = 1.
        
        # Control variance 
        self.u_std = np.array([self.linear_velocity_variance,self.angular_velocity_variance])
        
        # Linear velocity range
        self.vrange = np.array([0.0, 2.0])
        
        # Angular velocity range
        self.wrange = np.array([-np.pi, np.pi])
        
        # Number of obstacles. Set to None to disable obstacles
        self.number_of_obstacles = 5
        
        # Positions of obstacles
        self.obstacle_positions = np.array([[2.,1.], [1.,4.], [5.,5.], [6.,1.], [6.,4.]])

        # Radius of obstacles
        self.obstacle_radius = np.array([0.7, 0.9,0.75,0.85,0.85])

        assert len(self.obstacle_positions) == len(self.obstacle_radius) == self.number_of_obstacles
    
        # Based on propagated variance along the control horizon, 
        # increase the distance robot must stay away from object.
        self.obstacle_safety_factor = np.array([0.0]*self.number_of_obstacles)
        self.obstacle_safety_factor = np.tile(self.obstacle_safety_factor,(self.num_steps,1))
                
        # Cap the obstacle inflation
        self.max_obs_safety_factor = 1.5
            
        # Initial position
        self.x0 = np.array([0,0,np.pi/4])
         
        # Initial velocity
        self.v0 = np.array([0,0])
        
        # Goal position
        self.xgoal = np.array([7,5])
        
        # If within 0.5 meters of goal, consider reached
        self.goal_tolerance = 0.5
        
        # Maximum number of MPPI solves
        self.max_steps = 1000
        
        # Visualization limits x
        self.vis_xlim = [-1,8]
        
        # Visualization limits y
        self.vis_ylim = [-1,6]
        
        # Plot once every this many iterations
        self.plot_every_n = 200
        
        # The number of data points in each SGF polynomial fit
        # To disable set to 3
        self.sgf_window_length = 11 #3 to disable/minimize effect
        
        # Order of polynomial to fit within each SGF fit
        self.sgf_polynomial_order = 2
        
        # Weights for each terrain in simulation
        # Final GP output is weighted sum of all GPs
        # Instantiate all terrains with the same weight
        
        self.terrain_weights = np.array([0.34,0.33,0.33])
        
        self.num_terrains = len(self.terrain_weights)
        
        # Number of GPs/tasks to train, separate for lin and ang error per terrain
        self.num_tasks = self.num_terrains * 2
        
        # Dimension of GP input - Note order: cmd lin, cmd ang, cur lin, cur ang
        self.gp_ip_dim = 4
        
        # Linear Velocity variance cost
        self.linear_velocity_variance_cost = 1000.0
        
        # Angular Velocity variance cost
        self.angular_velocity_variance_cost = 1000.0
        
        # Number of iterations for which to increase distance to maintain
        # from obstacle. This operation is computationally expensive so we do not do
        # this for the whole horizon
        self.obstacle_inflation_horizon = 10
        assert self.obstacle_inflation_horizon <= self.num_steps
        
        # Precompute the constraint tightening multiplier based on number of obstacles
        self.probability_of_obstacle_avoidance = 0.95 #p_x
        
        delta_x = (1.-self.probability_of_obstacle_avoidance) / (self.number_of_obstacles)
        
        self.constraint_tightening_multiplier = stats.norm.ppf(1-delta_x)
        
        # Ground truth estimation using neural network
        # Each terrain's properties are encapsulated inside the outputs of nn
        # By chosing the terrain here, the appropriate nn output is extracted
        self.terrain_dict = {"asphalt":0, "grass":1, "tile":2}
        self.terrain = "grass"
        
        # Horizon to look back at for solving the terrain optimizer problem
        self.lookback_horizon = 15 # past 1.5 seconds of data
        
        # Penalizing the deviation between solutions iterations
        self.regularization_eps = 0.001
        
        # Wheelbase - clearpath jackal
        self.wheelbase_m = 0.277
        
        # Half track width approximated for a robot size of jackal 
        self.half_track_width_m = 3.5/2.
        
        # Minimum half track width. Since this is modified based on
        # propagated variance, make sure we dont shrink too much
        self.max_track_shrinking = 0.2
        
        # Radius of center line circle (midpoint of the track)
        self.center_radius_m = 8.25
        
        ##### Circular path tracking costs #####
        
        # Cost 1 - Penalize deviation from lane center at each MPC iteration
        self.path_tracking_lane_center_deviation_penalty = 1e4
        
        # Cost 2 - Penalize leaving the lane. Early on in the 
        # MPC horizon, higher penalty is accrued
        self.path_tracking_lane_departure_exponential_factor = 0.9
        
        # Multiplier for leaving the lane penalty
        self.path_tracking_lane_departure_high_penalty = 0.0
        
        # Cost 3 - Penalize deviation from desired (high) speed
        self.path_tracking_desired_speed = 1.75
        
        # Deviation from desired speed penalty
        self.path_tracking_desired_speed_penalty = 5e2
        
        # Cost 4 - Penalize robot side slip
        
        # Non zero slip values generally must accrue
        # penalty scaled by this multiplier
        self.path_tracking_nominal_slip_penalty = 1.0
        
        # Cost 5 -  Beyond a certain slip accrue massive penalty
        
        # Slip values higher than this are considered catastrophic so 
        # accrue very high penalty
        self.path_tracking_side_slip_abs_threshold = 0.2
        
        # The high penalty for side slip magnitude 
        # greater than threshold
        self.path_tracking_side_slip_high_penalty = 1e4
        
        # Number of iterations for which to shrink track radius 
        # based on propagated variance. This operation is computationally
        # expenside so we do not do this for the whole horizon
        self.track_radius_reduction_horizon = 10
        assert self.track_radius_reduction_horizon <= self.num_steps
        
        # Value of chai squared for two degrees of freedom
        # Acts as a multiplier to maximum eigen value of XY covariance
        
        _df = 2
        _probability_of_constraint_satisfaction = 0.4 #0.4 makes multiplier around 1
        
        self.kai_squared_mutliplier = stats.chi2.ppf(_probability_of_constraint_satisfaction, _df)
        
        self.track_radius_reduction_values_default = np.array([0.0]*self.num_steps)
        
        self.center_square_side_len_m = 20