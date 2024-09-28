#!/usr/bin/env python3

import numpy as np
import numba
from numba import cuda
import casadi as ca
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32
import torch
import torch.nn as nn
import math
import os

from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import fast_pred_var


# Define a Batch Independent Multitask GP Model
class BatchIndependentMultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_tasks,input_dimension):
        super().__init__(train_x, train_y, likelihood)
        
        # Define the mean module with batch shape for multitasking
        self.mean_module = ZeroMean(batch_shape=torch.Size([num_tasks]))
        
        # Define the RBF kernel with batch shape for multitasking and ARD for each input dimension
        self.covar_module =\
            ScaleKernel(RBFKernel(ard_num_dims=input_dimension, batch_shape=torch.Size([num_tasks])), batch_shape=torch.Size([num_tasks]))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, covar_x))

def nominal_dynamics_ode(x,u):
    """
    Summary:
        ODE representation of the nominal robot dynamics without GP compensation
    Args:
        x(ca.MX) -- Current robot state -- [X,Y,psi,v,omega]
        u(ca.MX) -- Currnet robot controls -- [v_ref,omega_ref]
    Returns:
        nom_dyn (ca.MX) -- Nominal dynamics in continuous form
    """
    parameters_file_path = os.path.join(os.getcwd(), "models","theta","theta.npy")
    global_theta = np.load(parameters_file_path)
    
    # Precomputed value of theta
    theta1 = global_theta[0]; theta2 = global_theta[1]; theta3 = global_theta[2]
    theta4 = global_theta[3]; theta5 = global_theta[4]; theta6 = global_theta[5]
    
    # Robot state space
    X = x[0]; Y = x[1] ; psi = x[2] ; v = x[3] ; omega = x[4]
    
    # Robot control actions
    v_ref = u[0] ; omega_ref = u[1]
    
    # Equations of motion
    Xdot = v * ca.cos(psi) 
    Ydot = v * ca.sin(psi)
    psidot = omega
    vdot = (theta3/theta1) * omega**2 - (theta4/theta1) * v + (1./theta1) * v_ref
    omegadot = (-theta5/theta2) * v * omega - (theta6/theta2) * omega + (1./theta2) * omega_ref
    
    # Collate the vectors
    nom_dyn = [Xdot,
            Ydot,
            psidot,
            vdot,
            omegadot]
    
    return ca.vertcat(*nom_dyn)

# Define the fully connected neural network model class
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 6)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

"""Integration functions for forward simulation"""

def nominal_dynamics_func(t, vel, *params):
    """
    Dynamics equations considering theta 1 to theta 6.
    Does not consider GP learnt errors.

    Args:
        t (float): Time of evaluation of dynamics
        vel (array): Current linear and angular velocity of the robot
        params (tuple of list) -- params[0] - theta, params[1] - commanded velocities

    Returns:
        veldot (array) : current derivative of velocities based on nominal dynamics
    """

    # Extract linear and angular velocities
    u = vel[0]; omega = vel[1]

    # Extract Parameters θ1 to θ6
    theta1 = params[0][0];theta2=params[0][1];theta3=params[0][2];theta4=params[0][3];theta5=params[0][4];theta6=params[0][5]

    # Extract commanded  velocities
    u_ref = params[1][0]; omega_ref = params[1][1]

    # Nominal velocity dynamics
    vdot = (theta3/theta1)*omega**2 - (theta4/theta1)*u + (1./theta1)*u_ref, \
            (-theta5/theta2)*u*omega - (theta6/theta2)*omega + (1./theta2)*omega_ref

    return vdot



############ CUDA Kernels from here on ############




# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
	return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
	return (1-np.float32(goal_reached))*dist2

# Sample control noise
@cuda.jit(fastmath=True)
def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
    """
    Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
    noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    Assume each thread corresponds to one time step
    For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.
    This will not work if time steps are more than max_threads_per_block (usually 1024)
    """
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    abs_thread_id = cuda.grid(1)
    
    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)

@cuda.jit(fastmath=True)
def get_state_rollout_across_control_noise(
        state_rollout_batch_d, # where to store results
        x0_d, 
        dt_d,
        noise_samples_d,
        vrange_d,
        wrange_d,
        u_prev_d,
        u_cur_d):
    """
    Do a fixed number of rollouts for visualization across blocks.
    Assume kernel is launched as get_state_rollout_across_control_noise[num_blocks, 1]
    The block with id 0 will always visualize the best control sequence. Other blocks will visualize random samples.
    """

    # Use block id
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    timesteps = len(u_cur_d)


    if bid==0:
        # Visualize the current best 
        # Explicit unicycle update and map lookup
        # From here on we assume grid is properly padded so map lookup remains valid
        x_curr = cuda.local.array(3, numba.float32)
        for i in range(3): 
            x_curr[i] = x0_d[i]
            state_rollout_batch_d[bid,0,i] = x0_d[i]
        
        for t in range(timesteps):
            # Nominal noisy control
            v_nom = u_cur_d[t, 0]
            w_nom = u_cur_d[t, 1]
            
            # Forward simulate
            x_curr[0] += dt_d*v_nom*math.cos(x_curr[2])
            x_curr[1] += dt_d*v_nom*math.sin(x_curr[2])
            x_curr[2] += dt_d*w_nom

            # Save state
            state_rollout_batch_d[bid,t+1,0] = x_curr[0]
            state_rollout_batch_d[bid,t+1,1] = x_curr[1]
            state_rollout_batch_d[bid,t+1,2] = x_curr[2]
    else:
        
        # Explicit unicycle update and map lookup
        # From here on we assume grid is properly padded so map lookup remains valid
        x_curr = cuda.local.array(3, numba.float32)
        for i in range(3): 
            x_curr[i] = x0_d[i]
            state_rollout_batch_d[bid,0,i] = x0_d[i]

        
        for t in range(timesteps):
            # Nominal noisy control
            v_nom = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
            w_nom = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
            v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
            w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))

            # # Nominal noisy control
            v_nom = u_prev_d[t, 0]
            w_nom = u_prev_d[t, 1]
            
            # Forward simulate
            x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
            x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
            x_curr[2] += dt_d*w_noisy

            # Save state
            state_rollout_batch_d[bid,t+1,0] = x_curr[0]
            state_rollout_batch_d[bid,t+1,1] = x_curr[1]
            state_rollout_batch_d[bid,t+1,2] = x_curr[2]
            
@cuda.jit
def create_gp_inputs(noise_samples_d, cur_vel_d,u_cur_d,t_d,vrange_d,wrange_d,gp_input_d):
    """
    CUDA kernel to combine 1024,2 slices from u_std and u_curr into 1024,4 in the combined array.
    Launched as [bpg:no of control samples (1024), tpb:1]
    
    Args:
        noise_samples_d: Noise samples from MPPI sampling of shape (1024,20,2)
        cur_vel_d: Current velocity of the robot shape (1024,20,2)
        u_cur_d: Output of previous MPPi solve applied to the robot shape (20,2)
        t_d: Which timestep are loading the 1024,2 samples from
        vrange_d: Range of allowable linear velocities
        wrange_d: Range of allowable angular velocities
        gp_input_d: Actual output of this kernel. These 1024,4 values are sent to GP solver
    """
    # Global thread id
    idx = cuda.grid(1)
    
    if idx < gp_input_d.shape[0]:
        # Calculate noisy commanded velocities based on the inputs
        vnom = u_cur_d[t_d, 0] + noise_samples_d[idx, t_d, 0]
        wnom = u_cur_d[t_d, 1] + noise_samples_d[idx, t_d, 1]
        
        # Clamp the commanded values within the given range
        vnoisy = max(vrange_d[0], min(vrange_d[1], vnom))
        wnoisy = max(wrange_d[0], min(wrange_d[1], wnom))
        
        # Write the results to the output array
        gp_input_d[idx, 0] = vnoisy #commanded linear velocity 
        gp_input_d[idx, 1] = wnoisy #commanded angular velocity 
        gp_input_d[idx, 2] = cur_vel_d[idx, t_d, 0] #current linear velocity 
        gp_input_d[idx, 3] = cur_vel_d[idx, t_d, 1] #current angular velocity

@cuda.jit    
def update_cur_vel(noise_samples_d, cur_vel_d, t, u_cur_d, vrange_d, wrange_d, mean_d, variance_d,terrain_weights_d,
                   linear_velocity_variance_cost_d, angular_velocity_variance_cost_d,theta_d,dt_d,costs_d):
    """
    Update next velocity based on noise samples, gp outputs. Also start populating the costs for each trajectory
    based on the variance of the GP. Kernel launched as [num_control_samples,1]~[1024,1]
    """
    
    # Get block index i.e. trajectory number
    bid = cuda.blockIdx.x
    
    # Initialize costs for this trajectory
    costs_d[bid] = 0.0
    
    # Begin by computing weighted sum of GP mean and variances
    mean_cur_traj_d = mean_d[bid]
    
    variance_cur_traj_d = variance_d[bid]

    final_linear_mean = 0.0 ; final_linear_variance = 0.0
    final_angular_mean = 0.0; final_angular_variance = 0.0

    for terrain_no in range(len(terrain_weights_d)):
        # Each terrain has 2 GPs
        # First one is linear vel, second is ang vel
        final_linear_mean  += (terrain_weights_d[terrain_no]  * mean_cur_traj_d[terrain_no*2])
        final_angular_mean += (terrain_weights_d[terrain_no]  * mean_cur_traj_d[(terrain_no*2)+1])
        
        final_linear_variance  += (terrain_weights_d[terrain_no]**2  * variance_cur_traj_d[terrain_no*2])
        final_angular_variance += (terrain_weights_d[terrain_no]**2  * variance_cur_traj_d[(terrain_no*2)+1])
    
    # Output 1 : Add cost for GP variance
    costs_d[bid] += \
        (linear_velocity_variance_cost_d * final_linear_variance +  angular_velocity_variance_cost_d * final_angular_variance)
        
    ### Forward simulate nominal and GP dynamics ###
    
    # Nominal noisy control
    v_nom = u_cur_d[t,0] + noise_samples_d[bid,t,0]
    w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
    v_ref = max(vrange_d[0], min(vrange_d[1], v_nom))
    w_ref = max(wrange_d[0], min(wrange_d[1], w_nom))
    
    v_cur = cur_vel_d[bid,t,0]
    w_cur = cur_vel_d[bid,t,1]
    
    theta1 = theta_d[0]; theta2 = theta_d[1]; theta3 = theta_d[2]
    theta4 = theta_d[3]; theta5 = theta_d[4]; theta6 = theta_d[5]
    
    # Next actual velocities
    v_next = v_cur + dt_d * ( (theta3/theta1)*w_cur**2  - (theta4/theta1)*v_cur + (1./theta1)*v_ref ) + final_linear_mean
    w_next = w_cur + dt_d * ( (-theta5/theta2)*v_cur*w_cur - (theta6/theta2)*w_cur + (1./theta2)*w_ref ) + final_angular_mean
    
    # Output 2: Actual next velocities
    cur_vel_d[bid,t+1,0] = max(vrange_d[0], min(vrange_d[1], v_next))
    cur_vel_d[bid,t+1,1] = max(wrange_d[0], min(wrange_d[1], w_next))
    
@cuda.jit(fastmath=True)
def rollout_gp_numba_obstacle_avoidance(
    xgoal_d,
    obs_cost_d,
    obs_pos_d,
    obs_r_d,
    goal_tolerance_d,
    lambda_weight_d,
    u_std_d,
    x0_d,
    dt_d,
    dist_weight_d,
    noise_samples_d,
    u_cur_d,
    cur_vel_d,
    obs_safety_factor_d,
    costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    
    # Index of block, trajectory number
    bid = cuda.blockIdx.x   
    
    # Explicit euler unicycle update
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9

    curr_obs_pos = obs_pos_d
    for t in range(timesteps):

        # Forward simulate
        linear_velocity  = cur_vel_d[bid, t, 0]
        angular_velocity = cur_vel_d[bid, t, 1]
        
        x_curr[0] += dt_d*linear_velocity*math.cos(x_curr[2])
        x_curr[1] += dt_d*linear_velocity*math.sin(x_curr[2])
        x_curr[2] += dt_d*angular_velocity
        
        # Stage cost
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)
        
        # Obstacle cost
        num_obs = len(obs_pos_d)
        
        for obs_i in range(num_obs):
            op_x = curr_obs_pos[obs_i][0] 
            op_y = curr_obs_pos[obs_i][1]
            dist_diff = (x_curr[0]-op_x)**2 +\
                        (x_curr[1]-op_y)**2 - obs_r_d[obs_i]**2
            
            costs_d[bid] += (1-numba.float32(dist_diff>obs_safety_factor_d[t][obs_i]))*obs_cost_d
        
            curr_obs_pos[obs_i][0] = op_x
            curr_obs_pos[obs_i][1] = op_y

        if dist_to_goal2<= goal_tolerance_d2:
            goal_reached = True
            break
    
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)
    
    for t in range(timesteps):
        costs_d[bid] += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + \
            (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])
        
@cuda.jit(fastmath=True)
def rollout_unicycle_numba_obstacle_avoidance(
    vrange_d,
    wrange_d,
    xgoal_d,
    obs_cost_d,
    obs_pos_d,
    obs_r_d,
    goal_tolerance_d,
    lambda_weight_d,
    u_std_d,
    x0_d,
    dt_d,
    dist_weight_d,
    noise_samples_d,
    u_cur_d,
    costs_d):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    
    # Get block id
    bid = cuda.blockIdx.x   # index of block
    costs_d[bid] = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom = v_noisy = w_nom = w_noisy = 0.0

    for t in range(timesteps):
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        # Stage cost
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)
        
        # Obstacle cost
        num_obs = len(obs_pos_d)
        for obs_i in range(num_obs):
            op = obs_pos_d[obs_i]
            dist_diff = (x_curr[0]-op[0])**2+(x_curr[1]-op[1])**2-obs_r_d[obs_i]**2
            costs_d[bid] += (1-numba.float32(dist_diff>0))*obs_cost_d
        
        if dist_to_goal2<= goal_tolerance_d2:
            goal_reached = True
            break
    
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)
    
    for t in range(timesteps):
        costs_d[bid] += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])
        
@cuda.jit(fastmath=True)
def rollout_edd5_obstacle_avoidance_numba(
    vrange_d,
    wrange_d,
    xgoal_d,
    obs_cost_d,
    obs_pos_d,
    obs_r_d,
    goal_tolerance_d,
    lambda_weight_d,
    u_std_d,
    x0_d,
    dt_d,
    dist_weight_d,
    noise_samples_d,
    u_cur_d,
    alphar_d, alphal_d, xv_d, yr_d, yl_d,\
    b_m_d, r_m_d,\
    # Output
    costs_d
    ):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    
    # Get block index i.e. trajectory number
    bid = cuda.blockIdx.x

    # Cost for this trajectory
    costs_d[bid] = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    v_nom = v_noisy = w_nom = w_noisy = 0.0

    
    for t in range(timesteps):
        
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
    
        # Commanded right wheel and left wheel speed 
        # Formula based on : https://en.wikipedia.org/wiki/Differential_wheeled_robot
        cmd_w_r  = (v_noisy + 0.5 * w_noisy * b_m_d) / r_m_d
        cmd_w_l  = (v_noisy - 0.5 * w_noisy * b_m_d) / r_m_d
    
        # Pre multiplier
        pre_multiplier = r_m_d / (yl_d - yr_d)
        
        v_noisy = pre_multiplier * ( -yr_d * alphal_d* cmd_w_l + yl_d * alphar_d * cmd_w_r )
        _ = pre_multiplier * (  xv_d * alphal_d* cmd_w_l - xv_d * alphar_d * cmd_w_r )
        w_noisy = pre_multiplier * ( -1. * alphal_d* cmd_w_l + 1. * alphar_d * cmd_w_r )
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        # Stage cost
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)
        
        # Obstacle cost
        num_obs = len(obs_pos_d)
        for obs_i in range(num_obs):
            op = obs_pos_d[obs_i]
            dist_diff = (x_curr[0]-op[0])**2+(x_curr[1]-op[1])**2-obs_r_d[obs_i]**2
            costs_d[bid] += (1-numba.float32(dist_diff>0))*obs_cost_d
        
        if dist_to_goal2<= goal_tolerance_d2:
            goal_reached = True
            break
    
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)
    
    for t in range(timesteps):
        costs_d[bid] += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

@cuda.jit(fastmath=True)
def update_useq_numba(
    lambda_weight_d,
    costs_d,
    noise_samples_d,
    weights_d,
    vrange_d,
    wrange_d,
    u_cur_d):
    
    """ 
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """
    
    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    numel = len(noise_samples_d)
    gap = int(math.ceil(numel / num_threads))
    
    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    
    if starti<numel:
        weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
        weights_d[starti] = min(weights_d[starti], costs_d[i])
    cuda.syncthreads()
    
    s = gap
    
    while s < numel:
        if (starti % (2 * s) == 0) and ((starti + s) < numel):
            # Stride by `s` and add
            weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
        s *= 2
        cuda.syncthreads()
    
    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
        weights_d[i] = math.exp(-1./lambda_weight_d*(costs_d[i]-beta))
    
    cuda.syncthreads()
    
    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
        costs_d[i] = weights_d[i]
    cuda.syncthreads()
    
    for i in range(starti+1, endi):
        costs_d[starti] += costs_d[i]
    
    cuda.syncthreads()
    s = gap
    
    while s < numel:
        if (starti % (2 * s) == 0) and ((starti + s) < numel):
            # Stride by `s` and ad
            costs_d[starti] += costs_d[starti + s]
        s *= 2
        cuda.syncthreads()
    
    for i in range(starti, endi):
        weights_d[i] /= costs_d[0]
    cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
        for i in range(starti, endi):
            cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*noise_samples_d[i, t, 0])
            cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*noise_samples_d[i, t, 1])
    
    cuda.syncthreads()
    
@cuda.jit(fastmath=True)
def rollout_gp_circular_path_tracking_numba(lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    cur_vel_d,track_radius_reduction_values_d,costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """

    # Get block index
    bid = cuda.blockIdx.x

    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]

    timesteps = len(u_cur_d)

    for t in range(timesteps):
        # Forward simulate
        linear_velocity  = cur_vel_d[bid, t, 0]
        angular_velocity = cur_vel_d[bid, t, 1]
        
        x_curr[0] += dt_d*linear_velocity*math.cos(x_curr[2])
        x_curr[1] += dt_d*linear_velocity*math.sin(x_curr[2])
        x_curr[2] += dt_d*angular_velocity
        
        ''' Costs specific to path tracking '''

        # Cost 1 : Deviation from lane center cost
        
        # Calculate the closest point on the center line
        angle = math.atan2(x_curr[1],x_curr[0])
        
        closest_center_x = center_radius_m_d * math.cos(angle)
        closest_center_y = center_radius_m_d * math.sin(angle)
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )
        
        # Normalize the distance
        half_track_width_m_d -= track_radius_reduction_values_d[t] # Assumed to be appropriately capped
        
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( linear_velocity - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(linear_velocity) / abs(angular_velocity+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost
                    
@cuda.jit(fastmath=True)
def rollout_unicycle_circular_path_tracking_numba(vrange_d,wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    
    # Get block index
    bid = cuda.blockIdx.x
    
    # Cost for this trajectory
    costs_d[bid] = 0.0 
    
    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    
    v_nom = v_noisy = w_nom = w_noisy = 0.0
    
    for t in range(timesteps):
        
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        ''' Costs specific to path tracking '''

        # Cost 1 : Deviation from lane center cost
        
        # Calculate the closest point on the center line
        angle = math.atan2(x_curr[1],x_curr[0])
        
        closest_center_x = center_radius_m_d * math.cos(angle)
        closest_center_y = center_radius_m_d * math.sin(angle)
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )
        
        # Normalize the distance
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( v_noisy - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(v_noisy) / abs(w_noisy+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])
    
    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost
                    
@cuda.jit(fastmath=True)
def rollout_edd5_circular_path_tracking_numba(vrange_d,wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    alphar_d, alphal_d, xv_d, yr_d, yl_d,\
    b_m_d, r_m_d,\
    # Output
    costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    # Get block index
    bid = cuda.blockIdx.x

    # Cost for this trajectory
    costs_d[bid] = 0.0 
    
    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    
    v_nom = v_noisy = w_nom = w_noisy = 0.0
    
    for t in range(timesteps):
        
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
    
        # Commanded right wheel and left wheel speed 
        # Formula based on : https://en.wikipedia.org/wiki/Differential_wheeled_robot
        cmd_w_r  = (v_noisy + 0.5 * w_noisy * b_m_d) / r_m_d
        cmd_w_l  = (v_noisy - 0.5 * w_noisy * b_m_d) / r_m_d
    
        # Pre multiplier
        pre_multiplier = r_m_d / (yl_d - yr_d)
        
        v_noisy = pre_multiplier * ( -yr_d * alphal_d* cmd_w_l + yl_d * alphar_d * cmd_w_r )
        _ = pre_multiplier * (  xv_d * alphal_d* cmd_w_l - xv_d * alphar_d * cmd_w_r )
        w_noisy = pre_multiplier * ( -1. * alphal_d* cmd_w_l + 1. * alphar_d * cmd_w_r )
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        ''' Costs specific to path tracking '''
        # Cost 1 : Deviation from lane center cost
        
        # Calculate the closest point on the center line
        angle = math.atan2(x_curr[1],x_curr[0])
        
        closest_center_x = center_radius_m_d * math.cos(angle)
        closest_center_y = center_radius_m_d * math.sin(angle)
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )
        
        # Normalize the distance
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( v_noisy - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(v_noisy) / abs(w_noisy+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

    # print("--")
    # print("deviation_from_lane_center_cost",deviation_from_lane_center_cost)
    # print("leaving_lane_boundaries_cost",leaving_lane_boundaries_cost)
    # print("deviation_from_desired_speed_cost",deviation_from_desired_speed_cost)
    # print("nominal_side_slip_cost",nominal_side_slip_cost)
    # print("severe_side_slip_cost",severe_side_slip_cost)
    # print("mppi_cost",mppi_cost)
    # print("--")

    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost


@cuda.jit(fastmath=True)
def rollout_gp_square_path_tracking_numba(lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d, center_square_side_len_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    cur_vel_d,track_radius_reduction_values_d,costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """

    # Get block index
    bid = cuda.blockIdx.x

    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]

    timesteps = len(u_cur_d)

    for t in range(timesteps):
        # Forward simulate
        linear_velocity  = cur_vel_d[bid, t, 0]
        angular_velocity = cur_vel_d[bid, t, 1]
        
        x_curr[0] += dt_d*linear_velocity*math.cos(x_curr[2])
        x_curr[1] += dt_d*linear_velocity*math.sin(x_curr[2])
        x_curr[2] += dt_d*angular_velocity
        
        ''' Costs specific to path tracking '''

        # Cost 1 : Deviation from lane center cost
        
        # Calculate the closest point on the center line
        #########################
        """
        Square Track Calculation
        """
        #########################
        closest_center_x = max(-center_square_side_len_m_d/2, min(x_curr[0], center_square_side_len_m_d/2))
        closest_center_y = max(-center_square_side_len_m_d/2, min(x_curr[1], center_square_side_len_m_d/2))

        if -center_square_side_len_m_d/2 < x_curr[0] < center_square_side_len_m_d/2 and\
           -center_square_side_len_m_d/2 < x_curr[1] < center_square_side_len_m_d/2:
        # Closest edge: choose the one with the minimum distance
            if min(x_curr[0], center_square_side_len_m_d/2 - x_curr[0]) <\
               min(x_curr[1], center_square_side_len_m_d/2 - x_curr[1]):
                closest_center_x = -center_square_side_len_m_d/2 if x_curr[0] < 0 else center_square_side_len_m_d/2 
                closest_center_y = x_curr[1]  # Left or right edge
            else:
                closest_center_x = x_curr[0]
                closest_center_y = -center_square_side_len_m_d/2 if x_curr[1] < 0 else center_square_side_len_m_d/2  # Bottom or top edge
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )
        
        # Normalize the distance
        half_track_width_m_d -= track_radius_reduction_values_d[t] # Assumed to be appropriately capped
        
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( linear_velocity - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(linear_velocity) / abs(angular_velocity+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

    # print("--")
    # print("deviation_from_lane_center_cost",deviation_from_lane_center_cost)
    # print("leaving_lane_boundaries_cost",leaving_lane_boundaries_cost)
    # print("deviation_from_desired_speed_cost",deviation_from_desired_speed_cost)
    # print("nominal_side_slip_cost",nominal_side_slip_cost)
    # print("severe_side_slip_cost",severe_side_slip_cost)
    # print("mppi_cost",mppi_cost)
    # print("--")

    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost
                    
                    
@cuda.jit(fastmath=True)
def rollout_unicycle_square_path_tracking_numba(vrange_d,wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,center_square_side_len_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    
    # Get block index
    bid = cuda.blockIdx.x
    
    # Cost for this trajectory
    costs_d[bid] = 0.0 
    
    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    
    v_nom = v_noisy = w_nom = w_noisy = 0.0
    
    for t in range(timesteps):
        
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        ''' Costs specific to path tracking '''
        
        #########################
        """
        Square Track Calculation
        """
        #########################
        closest_center_x = max(-center_square_side_len_m_d/2, min(x_curr[0], center_square_side_len_m_d/2))
        closest_center_y = max(-center_square_side_len_m_d/2, min(x_curr[1], center_square_side_len_m_d/2))

        if -center_square_side_len_m_d/2 < x_curr[0] < center_square_side_len_m_d/2 and\
           -center_square_side_len_m_d/2 < x_curr[1] < center_square_side_len_m_d/2:
        # Closest edge: choose the one with the minimum distance
            if min(x_curr[0], center_square_side_len_m_d/2 - x_curr[0]) <\
               min(x_curr[1], center_square_side_len_m_d/2 - x_curr[1]):
                closest_center_x = -center_square_side_len_m_d/2 if x_curr[0] < 0 else center_square_side_len_m_d/2 
                closest_center_y = x_curr[1]  # Left or right edge
            else:
                closest_center_x = x_curr[0]
                closest_center_y = -center_square_side_len_m_d/2 if x_curr[1] < 0 else center_square_side_len_m_d/2  # Bottom or top edge
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )

        # Normalize the distance
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( v_noisy - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(v_noisy) / abs(w_noisy+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost
                    
@cuda.jit(fastmath=True)
def rollout_edd5_square_path_tracking_numba(vrange_d,wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,noise_samples_d,u_cur_d,\
    robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,center_square_side_len_m_d,\
    lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
    desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
    alphar_d, alphal_d, xv_d, yr_d, yl_d,\
    b_m_d, r_m_d,\
    # Output
    costs_d):
    
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    # Get block index
    bid = cuda.blockIdx.x

    # Cost for this trajectory
    costs_d[bid] = 0.0 
    
    # Enumerate costs
    deviation_from_lane_center_cost = 0.0
    leaving_lane_boundaries_cost = 0.0
    deviation_from_desired_speed_cost = 0.0
    nominal_side_slip_cost = 0.0
    severe_side_slip_cost = 0.0
    mppi_cost = 0.0
    
    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(3, numba.float32)
    for i in range(3):
        x_curr[i] = x0_d[i]
    
    timesteps = len(u_cur_d)
    
    v_nom = v_noisy = w_nom = w_noisy = 0.0
    
    for t in range(timesteps):
        
        # Nominal noisy control
        v_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
    
        # Commanded right wheel and left wheel speed 
        # Formula based on : https://en.wikipedia.org/wiki/Differential_wheeled_robot
        cmd_w_r  = (v_noisy + 0.5 * w_noisy * b_m_d) / r_m_d
        cmd_w_l  = (v_noisy - 0.5 * w_noisy * b_m_d) / r_m_d
    
        # Pre multiplier
        pre_multiplier = r_m_d / (yl_d - yr_d)
        
        v_noisy = pre_multiplier * ( -yr_d * alphal_d* cmd_w_l + yl_d * alphar_d * cmd_w_r )
        _ = pre_multiplier * (  xv_d * alphal_d* cmd_w_l - xv_d * alphar_d * cmd_w_r )
        w_noisy = pre_multiplier * ( -1. * alphal_d* cmd_w_l + 1. * alphar_d * cmd_w_r )
        
        # Forward simulate
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
        x_curr[2] += dt_d*w_noisy
        
        ''' Costs specific to path tracking '''
        # Cost 1 : Deviation from lane center cost
        
        # Calculate the closest point on the center line
        #########################
        """
        Square Track Calculation
        """
        #########################
        closest_center_x = max(-center_square_side_len_m_d/2, min(x_curr[0], center_square_side_len_m_d/2))
        closest_center_y = max(-center_square_side_len_m_d/2, min(x_curr[1], center_square_side_len_m_d/2))

        if -center_square_side_len_m_d/2 < x_curr[0] < center_square_side_len_m_d/2 and\
           -center_square_side_len_m_d/2 < x_curr[1] < center_square_side_len_m_d/2:
        # Closest edge: choose the one with the minimum distance
            if min(x_curr[0], center_square_side_len_m_d/2 - x_curr[0]) <\
               min(x_curr[1], center_square_side_len_m_d/2 - x_curr[1]):
                closest_center_x = -center_square_side_len_m_d/2 if x_curr[0] < 0 else center_square_side_len_m_d/2 
                closest_center_y = x_curr[1]  # Left or right edge
            else:
                closest_center_x = x_curr[0]
                closest_center_y = -center_square_side_len_m_d/2 if x_curr[1] < 0 else center_square_side_len_m_d/2  # Bottom or top edge
        
        # Calculate the Euclidean distance between the robot's position and the center line
        distance = math.sqrt( (x_curr[0] - closest_center_x)**2 + (x_curr[1] - closest_center_y)**2 )
        
        # Normalize the distance
        normalized_distance = distance / half_track_width_m_d
        
        # Ensure the value does not exceed 1.0
        normalized_distance = min( normalized_distance, 1.0 )
        
        deviation_from_lane_center_cost += (lane_center_deviation_penalty_d * normalized_distance)
        
        # Cost 2 : Leaving the lane boundaries cost
        normalized_distance = 1.0 if normalized_distance > 0.99 else 0.0
            
        leaving_lane_boundaries_cost += \
            ( math.pow( lane_departure_exponential_factor_d,t ) * lane_departure_high_penalty_d * normalized_distance )
            
        # Cost 3 : Deviation from desired speed
        desired_speed_deviation_squared = ( v_noisy - desired_speed_d )**2
        deviation_from_desired_speed_cost += (desired_speed_deviation_squared * desired_speed_penalty_d)

        # Cost 4 : Nominal side slip cost
        radius_of_circular_motion = abs(v_noisy) / abs(w_noisy+1e-6)
        current_slip = robot_wheelbase_m_d / radius_of_circular_motion
        
        nominal_side_slip_cost += (current_slip*nominal_slip_penalty_d)
        
        # Cost 5 : Extreme side slip cost
        if current_slip > side_slip_abs_threshold_d:
            severe_side_slip_cost += (current_slip*side_slip_high_penalty_d)
        
    # Cost 6 : MPPI specific cost
    for t in range(timesteps):
        mppi_cost += lambda_weight_d*(
            (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1])

    # print("--")
    # print("deviation_from_lane_center_cost",deviation_from_lane_center_cost)
    # print("leaving_lane_boundaries_cost",leaving_lane_boundaries_cost)
    # print("deviation_from_desired_speed_cost",deviation_from_desired_speed_cost)
    # print("nominal_side_slip_cost",nominal_side_slip_cost)
    # print("severe_side_slip_cost",severe_side_slip_cost)
    # print("mppi_cost",mppi_cost)
    # print("--")

    costs_d[bid] = deviation_from_lane_center_cost + leaving_lane_boundaries_cost + deviation_from_desired_speed_cost + \
                    nominal_side_slip_cost + severe_side_slip_cost + mppi_cost