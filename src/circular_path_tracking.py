#!/usr/bin/env python3
""" 
Obstacle avoidance using different MPPI variants for skid steer robot Jackal
Change the PlannerModel enum in the configurations file to try out obstacle 
avoidance using unicycle, EDD5(learned kinematic model) and GP based MPPI
The corresponding MPPI kernel functions are in the utility.py file
"""

# Python specific imports.
import numpy as np
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgf
import pandas as pd
import os
import casadi as ca
import torch
import torch.nn as nn

import pickle
import cvxpy
from collections import deque
from scipy.integrate import solve_ivp
import copy

from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import fast_pred_var

# Local imports
from utility import *
from configurations import *
from mppi import *

class CircularPathTracking(MPPI):
    
    def __init__(self):
        super().__init__()
        
        ### Circular path tracking specific variables ###

        # Initialize the track radius reduction values for safe motion planning
        self.track_radius_reduction_values = self.cfg.track_radius_reduction_values_default
        
        # Modify some configs
        self.cfg.vis_xlim = [-12,12]
        self.cfg.vis_ylim = [-12,12]
        self.cfg.sgf_window_length = 15
        self.cfg.sgf_polynomial_order = 2
        self.cfg.x0 = np.array([0,-8.25,0.0])

    def solve(self):
        """Entry point for different algoritims"""
        # Move MPPI variables to GPU
        vrange_d, wrange_d, lambda_weight_d, u_std_d, x0_d, dt_d,\
        robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
        lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,\
        lane_departure_high_penalty_d,desired_speed_d,desired_speed_penalty_d,\
        nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
        terrain_weights_d,track_radius_reduction_values_d = self.move_mppi_task_vars_to_device()
        
        # Sample control noise
        sample_noise_numba[self.num_control_rollouts,self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
        
        if self.cfg.planner_model == PlannerModel.GAUSSIAN_PROCESS:
            # Loop through all commanded velocities, get actual velocities before integrating them
            # to get robot position rollouts
            for t in range(self.num_steps-1):
                # Step 1 : Using the sampled control noise and current velocities, get inputs for GP
                create_gp_inputs[self.num_control_rollouts,1](
                    self.noise_samples_d,self.cur_vel_d,self.u_cur_d, t, vrange_d, wrange_d, self.gp_input_d)
                
                
                # Step 2 : Convert the device array to torch tensor for GP querying
                gp_inputs_tensor = torch.as_tensor(self.gp_input_d,device=self.device)
                
                
                # Step 3 : Get output from Batch GP for model residue
                with torch.no_grad(), fast_pred_var():
                    op = self.likelihood(self.gp_model(gp_inputs_tensor))
                
                mean_tensors = op.mean.to(self.device)
                variance_tensors = op.variance.to(self.device)
                
                mean_d = cuda.as_cuda_array(mean_tensors); variance_d = cuda.as_cuda_array(variance_tensors)
                
                # Step 4: Use the GP mean and predict next velocity. Use GP variance and add it to the cost
                update_cur_vel[self.num_control_rollouts,1](
                    self.noise_samples_d, self.cur_vel_d, t, self.u_cur_d, \
                    vrange_d, wrange_d, mean_d, variance_d, \
                    terrain_weights_d, self.linear_velocity_variance_cost_d,\
                    self.angular_velocity_variance_cost_d,self.theta_d,dt_d,self.costs_d)
                
            # Find costs for each trajectory
            rollout_gp_circular_path_tracking_numba[self.num_control_rollouts,1](
                lambda_weight_d,u_std_d,x0_d,dt_d,self.noise_samples_d,self.u_cur_d,\
                robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
                
                #cost parameters
                lane_center_deviation_penalty_d,
                lane_departure_exponential_factor_d,
                lane_departure_high_penalty_d,
                desired_speed_d,
                desired_speed_penalty_d,
                nominal_slip_penalty_d,
                side_slip_abs_threshold_d,
                side_slip_high_penalty_d,
                self.cur_vel_d,
                track_radius_reduction_values_d,
                
                #results
                self.costs_d)
        
        elif self.cfg.planner_model == PlannerModel.UNICYCLE_DYNAMICS:
            # Find costs for each trajectory
            rollout_unicycle_circular_path_tracking_numba[self.num_control_rollouts,1](
                vrange_d, wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,self.noise_samples_d,self.u_cur_d,\
                robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
                
                #cost parameters
                lane_center_deviation_penalty_d,
                lane_departure_exponential_factor_d,
                lane_departure_high_penalty_d,
                desired_speed_d,
                desired_speed_penalty_d,
                nominal_slip_penalty_d,
                side_slip_abs_threshold_d,
                side_slip_high_penalty_d,\
                
                #results
                self.costs_d)
            
        elif self.cfg.planner_model == PlannerModel.EDD5:
            # Find costs for this trajectory
            rollout_edd5_circular_path_tracking_numba[self.num_control_rollouts,1](
                vrange_d,wrange_d,lambda_weight_d,u_std_d,x0_d,dt_d,self.noise_samples_d,self.u_cur_d,\
                robot_wheelbase_m_d,half_track_width_m_d,center_radius_m_d,\
                lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,lane_departure_high_penalty_d,\
                desired_speed_d,desired_speed_penalty_d,nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,\
                self.alphar_d, self.alphal_d, self.xv_d, self.yr_d, self.yl_d,\
                self.b_m_d, self.r_m_d,\
                # results
                self.costs_d)
        
        else:
            raise NotImplementedError("Planner not implemented")
        
        # Save current control action before finding the next optimal control action
        self.u_prev_d = self.u_cur_d
        
        # Compute cost and update the optimal control action on device
        update_useq_numba[1, 32](
            lambda_weight_d, 
            self.costs_d, 
            self.noise_samples_d, 
            self.weights_d, 
            vrange_d,
            wrange_d,
            # output:
            self.u_cur_d) #unfiltered
        
        u_after_noise = self.u_cur_d.copy_to_host() # control signals after weighted noise is added
        
        return self.sgf_filtering_post(u_after_noise)

    def move_mppi_task_vars_to_device(self):
        vrange_d = cuda.to_device(self.cfg.vrange.astype(np.float32))
        wrange_d = cuda.to_device(self.cfg.wrange.astype(np.float32))
        lambda_weight_d = np.float32(self.cfg.lambda_weight)
        u_std_d = cuda.to_device(self.cfg.u_std.astype(np.float32))
        x0_d = cuda.to_device(self.cfg.x0.astype(np.float32))
        dt_d = np.float32(self.cfg.dt)
        
        # Deploy robot and track parameters
        robot_wheelbase_m = np.float32(self.cfg.wheelbase_m)
        
        half_track_width_m = np.float32(self.cfg.half_track_width_m)
        center_radius_m = np.float32(self.cfg.center_radius_m)
        
        
        # Deploy cost function parameters
        lane_center_deviation_penalty_d = np.float32(self.cfg.path_tracking_lane_center_deviation_penalty)
        lane_departure_exponential_factor_d = np.float32(self.cfg.path_tracking_lane_departure_exponential_factor)
        lane_departure_high_penalty_d = np.float32(self.cfg.path_tracking_lane_departure_high_penalty)
        desired_speed_d = np.float32(self.cfg.path_tracking_desired_speed)   
        desired_speed_penalty_d = np.float32(self.cfg.path_tracking_desired_speed_penalty) 
        nominal_slip_penalty_d = np.float32(self.cfg.path_tracking_nominal_slip_penalty)  
        side_slip_abs_threshold_d = np.float32(self.cfg.path_tracking_side_slip_abs_threshold)
        side_slip_high_penalty_d = np.float32(self.cfg.path_tracking_side_slip_high_penalty)
        
        # Terrain weights
        terrain_weights_d = cuda.to_device(self.terrain_weights.astype(np.float32))
        
        # Track radius reduction factors
        track_radius_reduction_values_d = cuda.to_device(self.track_radius_reduction_values.astype(np.float32))
        
        return vrange_d, wrange_d, lambda_weight_d, u_std_d, x0_d, dt_d,\
            robot_wheelbase_m,half_track_width_m,center_radius_m,\
            lane_center_deviation_penalty_d,lane_departure_exponential_factor_d,\
            lane_departure_high_penalty_d,desired_speed_d,desired_speed_penalty_d,\
            nominal_slip_penalty_d,side_slip_abs_threshold_d,side_slip_high_penalty_d,terrain_weights_d,\
            track_radius_reduction_values_d
    
    def propagate_variance_and_shrink_track(self,useq):
        
        """
        Propagate variance of robot states based on GP outputs using 
        Taylor series expansion. Based on the propagated state variance,
        shrink the track radius for safe motion planning
        https://arxiv.org/abs/1705.10702
        """
        
        # Initialize with measured state
        
        # Current robot pose
        curr_x = self.cfg.x0[0]
        curr_y = self.cfg.x0[1]
        curr_heading = self.cfg.x0[2]
        
        # Current robot velocity
        curr_lin_vel = self.cfg.v0[0]
        curr_ang_vel = self.cfg.v0[1]
        
        # Initialize state variance to zero
        sigma_x = np.zeros( (5,5) )
        
        # Selection matrix for what sub state space is obtained using GP
        Bd = np.array([ [0,0],[0,0],[0,0],[1,0],[0,1] ])

        # Initialize all the half track radius to default values that is all zero
        self.track_radius_reduction_values = self.cfg.track_radius_reduction_values_default
        
        for iter in range(self.cfg.track_radius_reduction_horizon):

            # Compute obstacle avoidance safety factor based on signed distance function
            sigma_XY = sigma_x[:2,:2]
            
            # Compute safety factor
            eigen_values, _ = np.linalg.eig(sigma_XY)
            max_eigen_value = np.max(eigen_values)

            if max_eigen_value > 0.01:
                track_radius_reduction_value = math.sqrt(self.cfg.kai_squared_mutliplier * max_eigen_value)  
            
                track_radius_reduction_value = \
                    min(track_radius_reduction_value,self.cfg.max_track_shrinking)
            
            else: #catch case for negative eigen values
                track_radius_reduction_value = 0.0
                
            # Append it back
            self.track_radius_reduction_values[iter] = track_radius_reduction_value

            # Initialize with the measured initial state of the robot and the commanded values
            curr_cmd_lin_vel = useq[iter,0]
            curr_cmd_ang_vel = useq[iter,1]
            
            ############ Compute the contribution of nominal dynamics to the next robot state mean ############
            # Initial robot state
            state   = [curr_x,curr_y,curr_heading,curr_lin_vel,curr_ang_vel]
            # Applied control action
            control = [curr_cmd_lin_vel,curr_cmd_ang_vel]
            nominal_next_state = np.array(self.nominal_dynamics_func(state=state , control = control)["next_state"]).reshape(5,) 
            
            ############### Compute the contribution of nominal dynamics to the next robot state variance ##############
            # Derivative of the nominal casadi dynamics
            casadi_derivative = self.nominal_dynamics_casadi_derivative(state=state,control=control)["jac_x"]
            # Nominal dynamics derivative - dimension 5 by 5
            nominal_dyn_derivative = np.array(casadi_derivative) #grad_f
            
            #########################################################################################################

            ############### Compute the contribution of GP dynamics to the next robot state mean ##############
            
            gp_next_state_mean = np.zeros( (5) )
            gp_next_state_variance = self.sigma_noise
            
            gp_input_values = [curr_cmd_lin_vel, curr_cmd_ang_vel, curr_lin_vel, curr_ang_vel]
            
            # Convert the list to a torch tensor
            gp_input = torch.tensor(gp_input_values, dtype=torch.float32).reshape(1, 4).to(self.device)

            with torch.no_grad(), fast_pred_var():
                op = self.likelihood(self.gp_model(gp_input))
            
            mean_numpy = op.mean.cpu().numpy().reshape(-1)
            variance_numpy = op.variance.cpu().numpy().reshape(-1)
            
            for terrain_no in range(len(self.terrain_weights)):
                
                gp_next_state_mean[3] += (self.terrain_weights[terrain_no] * mean_numpy[terrain_no*2])
                gp_next_state_mean[4] += (self.terrain_weights[terrain_no] * mean_numpy[(terrain_no*2)+1])
                
                gp_next_state_variance[0][0] += (self.terrain_weights[terrain_no]**2 * variance_numpy[terrain_no*2])
                gp_next_state_variance[1][1] += (self.terrain_weights[terrain_no]**2 * variance_numpy[(terrain_no*2)+1])

            
            #########Output for this iteration###########
            
            # Effective next state prediction is the sum total of the contributions
            final_next_state = nominal_next_state + gp_next_state_mean
            
            # Update the current state for the next prediction step
            curr_x = final_next_state[0]
            curr_y = final_next_state[1]
            curr_heading = final_next_state[2]
            curr_lin_vel = final_next_state[3]
            curr_ang_vel = final_next_state[4]
            
            total_size = sigma_x.shape[0] + gp_next_state_variance.shape[0]
            
            # Initialize the larger matrix
            sigma_i = np.zeros((total_size, total_size))
            
            # Place sigma_x in the top-left corner
            sigma_i[:sigma_x.shape[0], :sigma_x.shape[0]] = sigma_x
            
            # Place gp_next_state_variance in the bottom-right corner
            sigma_i[sigma_x.shape[0]:, sigma_x.shape[0]:] = gp_next_state_variance        
            
            # Gradiennt composite matrix
            composite_matrix = np.hstack((nominal_dyn_derivative, Bd))
            
            # Update state covariance
            sigma_x = np.dot(composite_matrix, np.dot(sigma_i, composite_matrix.T))

    
"""Main function to generate MPPI based motion plans"""
def initiate_planner():
    mppi = CircularPathTracking()
    cfg = mppi.cfg
    
    # Ground truth measurements
    xhist = np.zeros((cfg.max_steps+1, 3))*np.nan # positions
    vhist = np.zeros((cfg.max_steps+1, 2))*np.nan # velocities
    
    # Commanded velocities i.e. MPPI outputs
    uhist = np.zeros((cfg.max_steps, 2))*np.nan
    
    # Initial state
    xhist[0] = cfg.x0
    vhist[0] = cfg.v0
    
    absolute_init = cfg.x0
    
    # Track parameters
    outer_radius = cfg.center_radius_m + cfg.half_track_width_m #outer radius of track
    track_width = cfg.half_track_width_m*2.0  # Total width of the track
    
    # Calculate the radius of the inner circle
    inner_radius = outer_radius - track_width
    
    # Calculate the radius of the center line circle (midpoint of the track)
    center_radius = cfg.center_radius_m
    
    # Create an array of angles for the circles
    angles = np.linspace(0, 2 * np.pi, 100)
    
    # Calculate the points for the outer circle
    outer_x = outer_radius * np.cos(angles)
    outer_y = outer_radius * np.sin(angles)
    
    # Calculate the points for the inner circle
    inner_x = inner_radius * np.cos(angles)
    inner_y = inner_radius * np.sin(angles)
    
    # Calculate the points for the center line circle (midpoint of the track)
    center_x = center_radius * np.cos(angles)
    center_y = center_radius * np.sin(angles)
    
    # Run MPPI planner for a maximum number of steps
    for t in range(cfg.max_steps):
        # Solve optimal control problem
        useq = mppi.solve()
        
        u_curr = useq[0]
        uhist[t] = u_curr
    
        # Simulate state forward
        mppi.current_gt_lin_vel = vhist[t][0]
        mppi.current_gt_ang_vel = vhist[t][1]   
    
        # Find next robot state in response to commanded velocities based
        # on ground truth neural network
        xhist[t+1], vhist[t+1] = \
            mppi.neural_network_dynamics_ground_truth(xhist[t],vhist[t],u_curr)
        
        mppi.next_gt_lin_vel = vhist[t+1][0]
        mppi.next_gt_ang_vel = vhist[t+1][1]
    
        if cfg.planner_model == PlannerModel.GAUSSIAN_PROCESS:
            # Based on estimated next state and ground truth next state, adjust terrain weights
            mppi.compute_terrain_weights(u_curr)

        if t % cfg.plot_every_n == 0:
            # Visualize the basic setup
            fig, ax = plt.subplots()
            
            # Plot the circular track
            ax.plot(outer_x, outer_y, color='orange', label='Outer Lane Lines',linewidth=4)
            ax.plot(inner_x, inner_y, color='orange', linestyle='-', label='Inner Lane Lines',linewidth=4)
            ax.plot(center_x, center_y, color='black', linestyle='--', label='Center Line')

            # Plot start position and current state
            ax.plot([absolute_init[0]], [absolute_init[1]], 'ro', markersize=10, markerfacecolor='none', label="Start")
            ax.plot([xhist[t+1, 0]], [xhist[t+1, 1]], 'ro', markersize=10, label="Curr. State", zorder=5)

            # Get rollout states from a subset of maps for visualization (e.g., 50)
            rollout_states_vis = mppi.get_state_rollout()

            # Plot past states and rollouts
            ax.plot(xhist[:, 0], xhist[:, 1], 'r', label="Past State")
            ax.plot(rollout_states_vis[:, :, 0].T, rollout_states_vis[:, :, 1].T, 'k', alpha=0.5, zorder=3)
            ax.plot(rollout_states_vis[0, :, 0], rollout_states_vis[0, :, 1], 'k', alpha=0.5, label="Rollouts")

            # Set visualization limits
            ax.set_xlim(cfg.vis_xlim)
            ax.set_ylim(cfg.vis_ylim)

            # Set aspect ratio and other plot settings
            ax.legend(bbox_to_anchor = (1.01, 1), loc="upper left")
            fig.subplots_adjust(right=0.5)
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()
   
        # Update MPPI state (x0, useq)
        mppi.shift_and_update(xhist[t+1], vhist[t+1],useq, num_shifts=1)
    
    # Plot optimal control actions
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(uhist[:,0], label='v')
    ax.plot(uhist[:,1], label='w')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    initiate_planner()