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

class ObstacleAvoidance(MPPI):
    
    def __init__(self):
        super().__init__()
        
        # Obstacle specific configs
        self.number_of_obstacles = self.cfg.number_of_obstacles
        self.obstacle_positions = self.cfg.obstacle_positions
        self.obstacle_radius =  self.cfg.obstacle_radius
        self.obstacle_safety_factor = self.cfg.obstacle_safety_factor

    def solve(self):
        """Entry point for different algoritims""" 
        vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d, \
            u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d,obs_safety_factor_d ,\
            terrain_weights_d = self.move_mppi_task_vars_to_device()
    
        # Weight for distance cost
        dist_weight = self.cfg.dist_weight
        
        # Sample control noise
        sample_noise_numba[self.num_control_rollouts,self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
        
        if self.cfg.planner_model == PlannerModel.GAUSSIAN_PROCESS:
        
            # Loop through all commanded velocities, get actual velocities 
            # before integrating them to get robot position rollouts
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
            rollout_gp_numba_obstacle_avoidance[self.num_control_rollouts,1](
                xgoal_d,obs_cost_d,obs_pos_d,obs_r_d,goal_tolerance_d,lambda_weight_d,
                u_std_d,x0_d,dt_d,dist_weight,self.noise_samples_d,self.u_cur_d,self.cur_vel_d,obs_safety_factor_d,
                self.costs_d)# results
            
        elif self.cfg.planner_model == PlannerModel.UNICYCLE_DYNAMICS:
            # Find costs for each trajectory
            rollout_unicycle_numba_obstacle_avoidance[self.num_control_rollouts,1](
            vrange_d,wrange_d,xgoal_d,obs_cost_d,obs_pos_d,obs_r_d,goal_tolerance_d,lambda_weight_d,
            u_std_d,x0_d,dt_d,dist_weight,self.noise_samples_d,self.u_cur_d,
            self.costs_d# results
            )
            
        elif self.cfg.planner_model == PlannerModel.EDD5:
            # Find costs for each trajectory
            rollout_edd5_obstacle_avoidance_numba[self.num_control_rollouts,1](
            vrange_d,wrange_d, xgoal_d, obs_cost_d, obs_pos_d, obs_r_d, goal_tolerance_d, lambda_weight_d, u_std_d, x0_d,
            dt_d, dist_weight,  self.noise_samples_d, self.u_cur_d,
            self.alphar_d, self.alphal_d, self.xv_d, self.yr_d, self.yl_d,
            self.b_m_d, self.r_m_d,\
            # results
            self.costs_d)
            
        else:
            raise NotImplementedError("Planner not implemented")    
            
        # Save current control action before finding the next optimal control action
        self.u_prev_d = self.u_cur_d
        u_before_noise = self.u_prev_d.copy_to_host() # control signals before weighted noise is added

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

        return self.sgf_filtering(u_before_noise, u_after_noise)

    def move_mppi_task_vars_to_device(self):
        vrange_d = cuda.to_device(self.cfg.vrange.astype(np.float32))
        wrange_d = cuda.to_device(self.cfg.wrange.astype(np.float32))
        xgoal_d = cuda.to_device(self.cfg.xgoal.astype(np.float32))
        goal_tolerance_d = np.float32(self.cfg.goal_tolerance)
        lambda_weight_d = np.float32(self.cfg.lambda_weight)
        u_std_d = cuda.to_device(self.cfg.u_std.astype(np.float32))
        x0_d = cuda.to_device(self.cfg.x0.astype(np.float32))
        dt_d = np.float32(self.cfg.dt)
        obs_cost_d = np.float32(self.cfg.collision_cost)
        obs_pos_d = cuda.to_device(self.cfg.obstacle_positions.astype(np.float32))
        obs_r_d = cuda.to_device(self.cfg.obstacle_radius.astype(np.float32))
        obs_safety_factor_d = cuda.to_device(self.obstacle_safety_factor.astype(np.float32))
        terrain_weights_d = cuda.to_device(self.terrain_weights.astype(np.float32))
        
        return vrange_d, wrange_d, xgoal_d,goal_tolerance_d, lambda_weight_d,\
            u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d,obs_safety_factor_d,terrain_weights_d

    def propagate_variance_and_inflate_obstacles(self,useq):
        
        """
        Propagate variance of robot states based on GP outputs using 
        Taylor series expansion. Based on the propagated state variance,
        increase the minimum distance needed for obstacle avoidance 
        https://ieeexplore.ieee.org/abstract/document/9143595
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
        
        # Initialize all obstacle safety factors to zero
        self.obstacle_safety_factor = self.cfg.obstacle_safety_factor
        
        for iter in range(self.cfg.obstacle_inflation_horizon):

            # Compute obstacle avoidance safety factor based on signed distance function
            sigma_XY = sigma_x[:2,:2]
            
            # Compute safety factor
            for obs_num in range(self.cfg.number_of_obstacles):
                self.obstacle_safety_factor[iter][obs_num] = \
                    self.compute_safety_factor(self.obstacle_positions[obs_num][0],
                                            self.obstacle_positions[obs_num][1],
                                            self.obstacle_radius[obs_num],
                                            curr_x,
                                            curr_y,
                                            sigma_XY,
                                            self.cfg.constraint_tightening_multiplier)

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
            
    def compute_safety_factor(self,x_c, y_c, r, x_p, y_p, sigma_xy, phi_inv):
        """
        Compute the scalar value: phi_inv * sqrt(ni^T * sigma_xy * ni)

        Parameters:
        - x_c (float): x-coordinate of the circle center (obstacle center)
        - y_c (float): y-coordinate of the circle center (obstacle center)
        - r (float): Radius of the circular obstacle
        - x_p (float): x-coordinate of the robot's position
        - y_p (float): y-coordinate of the robot's position
        - sigma_xy (numpy array): 2x2 covariance matrix
        - phi_inv (float): A scalar multiplier

        Returns:
        - safety_factor (float): The computed safety factor for this obstacle
        """

        # Step 1: Compute the vector v from the circle center to the robot's position
        v = np.array([x_p - x_c, y_p - y_c])

        # Step 2: Normalize the vector v to get the direction vector (unit vector)
        v_unit = v / np.linalg.norm(v)

        # Step 3: Find the closest point on the circle (obstacle) to the robot
        closest_point = np.array([x_c, y_c]) + r * v_unit

        # Step 4: Calculate the signed distance function di(x)
        d_i = np.linalg.norm(v) - r

        # Step 5: Compute the vector ni
        ni = (np.array([x_p, y_p]) - closest_point) / d_i

        # Step 6: Compute the safety factor
        safety_factor = phi_inv * np.sqrt(np.dot(ni.T, np.dot(sigma_xy, ni)))

        # Step 7: Cap the safety factor
        safety_factor = min(safety_factor,self.cfg.max_obs_safety_factor)

        return safety_factor

    
"""Main function to generate MPPI based motion plans"""
def initiate_planner():
    mppi = ObstacleAvoidance()
    cfg = mppi.cfg
    
    # Ground truth measurements
    xhist = np.zeros((cfg.max_steps+1, 3))*np.nan # positions
    vhist = np.zeros((cfg.max_steps+1, 2))*np.nan # velocities
    
    # Commanded velocities i.e. MPPI outputs
    uhist = np.zeros((cfg.max_steps, 2))*np.nan
    
    # Initial state
    xhist[0] = cfg.x0
    vhist[0] = cfg.v0
    
    # Run MPPI planner for a maximum number of steps
    for t in range(cfg.max_steps):
        # Solve optimal control problem
        useq = mppi.solve()
        
        if cfg.planner_model == PlannerModel.GAUSSIAN_PROCESS:
            # Tighten safety buffers for obstacle avoidance based
            mppi.propagate_variance_and_inflate_obstacles(useq)
        
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

        if t%cfg.plot_every_n==0:
            # Visualize the basic set up
            fig, ax = plt.subplots()
            ax.plot([cfg.x0[0]], [cfg.x0[1]], 'ro', markersize=10, markerfacecolor='none', label="Start")
            ax.plot([xhist[t+1, 0]], [xhist[t+1, 1]], 'ro', markersize=10, label="Curr. State", zorder=5)
            c1 = plt.Circle(cfg.xgoal, cfg.goal_tolerance, color='b', linewidth=3, fill=False, label="Goal", zorder=7)
            ax.add_patch(c1)
            
            # Show obstacles
            for obs_pos, obs_r in zip(cfg.obstacle_positions, cfg.obstacle_radius):
                obs = plt.Circle(obs_pos, obs_r, color='k', fill=True, zorder=6)
                ax.add_patch(obs)

            # Get rollout states from subset of maps for visualization
            rollout_states_vis = mppi.get_state_rollout()
            
            ax.plot(xhist[:,0], xhist[:,1], 'r', label="Past State")
            ax.plot(rollout_states_vis[:,:,0].T, rollout_states_vis[:,:,1].T, 'k', alpha=0.5, zorder=3)
            ax.plot(rollout_states_vis[0,:,0], rollout_states_vis[0,:,1], 'k', alpha=0.5, label="Rollouts")
            ax.set_xlim(cfg.vis_xlim)
            ax.set_ylim(cfg.vis_ylim)
            
            ax.legend(loc="upper left")
            ax.set_aspect("equal")
            plt.tight_layout()

            plt.show()
        
        # Update MPPI state (x0, useq)
        mppi.shift_and_update(xhist[t+1], vhist[t+1],useq, num_shifts=1)
        
        # Goal check
        if np.linalg.norm(xhist[t+1, :2] - cfg.xgoal) <= cfg.goal_tolerance:
            print("Goal reached at t={:.2f}s".format(t*cfg.dt))
            break
    
    # Plot optimal control actions
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(uhist[:,0], label='v')
    ax.plot(uhist[:,1], label='w')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    initiate_planner()