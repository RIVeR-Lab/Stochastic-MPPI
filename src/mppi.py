""" Base class for all MPPI planner implementations"""

# Python specific imports
from abc import ABC, abstractmethod
import numpy as np
import os
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import pickle
import casadi as ca
import cvxpy
from scipy.integrate import solve_ivp
from collections import deque
import pandas as pd
from scipy.signal import savgol_filter as sgf

# Local imports
from utility import *
from configurations import *

#### A few global variables for ease of integration functions#####
# Distance between robot center of mass and center of the rear axle
a_m = 0.145

# Robot width in meters
b_m = 0.377

# Wheel radius in meters
r_m = 0.09

class MPPI:
    
    def __init__(self):
        
        # Fixed configs
        self.cfg = Config()
        self.T = self.cfg.T
        self.dt = self.cfg.dt
        self.num_steps = self.cfg.num_steps
        self.num_control_rollouts = self.cfg.num_control_rollouts
        self.num_vis_state_rollouts = self.cfg.num_vis_state_rollouts
        self.seed = self.cfg.seed
        self.max_threads_per_block = self.cfg.max_threads_per_block
        self.device = self.cfg.device
        
        # Initialize reuseable device variables
        self.noise_samples_d = None
        self.u_cur_d = None
        self.u_prev_d = None
        self.costs_d = None
        self.weights_d = None
        self.rng_states_d = None
        self.state_rollout_batch_d = None
        self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
        
        self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, 2), dtype=np.float32)
        self.u_cur_d = cuda.to_device(self.u_seq0)
        self.u_prev_d = cuda.to_device(self.u_seq0)
        self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
        self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
        self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
        self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)
        
        ##### Initialize components specific to GP #####
        
        # Number of GPs and input dimensionality
        self.num_tasks = self.cfg.num_tasks
        self.gp_ip_dim = self.cfg.gp_ip_dim
        
        # Number of GPs and input dimensionality
        self.num_tasks = self.cfg.num_tasks
        self.gp_ip_dim = self.cfg.gp_ip_dim 
        
        # Velocities computed using GP and nominal dynamics.
        self.cur_vel_d = cuda.device_array((self.num_control_rollouts, self.num_steps, 2), dtype=np.float32)
        
        # Input to GP solver at each timestep for all trajectories
        self.gp_input_d = cuda.device_array((self.num_control_rollouts, self.cfg.gp_ip_dim), dtype=np.float32)
        
        # Initialize the weights for each terrain
        self.terrain_weights = self.cfg.terrain_weights
        
        # Ground truth estimation using fully connected neural network
        self.terrain_dict  = self.cfg.terrain_dict
        self.terrain_index = self.cfg.terrain_dict[self.cfg.terrain]
        
        # Cost parameters
        self.linear_velocity_variance_cost_d = np.float32(self.cfg.linear_velocity_variance_cost)
        self.angular_velocity_variance_cost_d = np.float32(self.cfg.angular_velocity_variance_cost)
        
        # Robot parameters needed to convert linear and angular velocities to wheel speeds
        # Robot width in meters
        self.b_m_d = np.float32(b_m)
        
        # Wheel radius in meters
        self.r_m_d = np.float32(r_m)
        
        # Load all trained models
        self.load_models()
        
        # Also create casadi models for uncertainty propagation
        self.create_nominal_dynamics_casadi()
        
        # Finally set up the convex optimization problem for computing terrain weights
        self.setup_terrain_weights_optimization()
        
        print("MPPI planner base initialized")
            
    def load_models(self):
        """
        Setup GP pre trained Batch model and nominal dynamics parameters
        """
        
        # GP Model load
        gp_model_path = os.path.join(os.getcwd(), "models","batch_gp")
        
        # Training inputs
        train_df = pd.read_csv(os.path.join(gp_model_path,"Train.csv"))
        
        train_x = train_df[['gp_ip_cmd_lin', 'gp_ip_cmd_ang', 'gp_ip_curr_lin', 'gp_ip_curr_ang']].values
        train_x = torch.tensor(train_x,dtype=torch.float32).to(self.device)
        
        train_y = train_df[['Asphalt_Lin_Error', 'Asphalt_Ang_Error',
                    'Grass_Lin_Error', 'Grass_Ang_Error',
                    'Tile_Lin_Error', 'Tile_Ang_Error']].values
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        # Initialize the model and likelihood

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.cfg.num_tasks).to(self.device)
        self.gp_model = \
            BatchIndependentMultitaskGPModel(train_x, train_y, self.likelihood,self.cfg.num_tasks,input_dimension=self.cfg.gp_ip_dim).to(self.device)
                
        # Load the model's saved state
        self.gp_model.load_state_dict(torch.load( os.path.join(gp_model_path,"batch_gp_model_hyperparameters.pth") , weights_only=True))
        
        # Set the model and likelihood to evaluation mode
        self.gp_model.eval(); self.likelihood.eval()
        
        # Setup nominal dynamics parameters
        parameters_file_path = os.path.join(os.getcwd(), "models","theta","theta.npy")
        self.theta = np.load(parameters_file_path).astype(np.float32)
        self.theta_d = cuda.to_device(np.load(parameters_file_path).astype(np.float32))
        
        # Save signal variance for uncertainty propagation - size (6,)
        self.sigma_noise = np.zeros((2,2))
        self.sigma_noise[0][0] = self.likelihood.task_noises.cpu().detach().numpy()[0]
        self.sigma_noise[1][1] = self.likelihood.task_noises.cpu().detach().numpy()[1]
        
        # Finally load the full connected neural network for ground truth estimation
        nn_model_path = os.path.join(os.getcwd(), "models","neural_network","fcnn_model.pth")
        input_scaler_pickle_path =  os.path.join(os.getcwd(), "models","neural_network","input_scaler.pkl")
        output_scaler_pickle_path = os.path.join(os.getcwd(), "models","neural_network","output_scaler.pkl")
        
        self.nn_model = FCNN().to(self.device)
        self.nn_model.load_state_dict(torch.load(nn_model_path, weights_only=True))
        self.nn_model.eval() #Set the model to evaluation mode
        
        # Load the scalers
        with open(input_scaler_pickle_path, 'rb') as f:
            self.input_scaler = pickle.load(f)
        with open(output_scaler_pickle_path, 'rb') as f:
            self.output_scaler = pickle.load(f)
            
        # Extract individual EDD5 params
        
        # Define the base path for the EDD5 models
        edd5_models_base_path = os.path.join(os.getcwd(), "models", "edd5")
        
        # Load the numpy arrays from the files and ensure they are of type np.float32
        asphalt_edd5_params = np.load(os.path.join(edd5_models_base_path, "edd_5_Asphalt.npy")).astype(np.float32)
        grass_edd5_params = np.load(os.path.join(edd5_models_base_path, "edd_5_Grass.npy")).astype(np.float32)
        tile_edd5_params = np.load(os.path.join(edd5_models_base_path, "edd_5_Tile.npy")).astype(np.float32)

        # Stack the numpy arrays
        alledd5_params = np.stack((asphalt_edd5_params, grass_edd5_params, tile_edd5_params), axis=0)

        # Finally extract the parameters for the actual terrain
        edd5_params = alledd5_params[self.terrain_index]
        
        self.alphar_d,self.alphal_d,self.xv_d,self.yr_d,self.yl_d = edd5_params
        
    def neural_network_dynamics_ground_truth(self,x_curr, v_curr, u_cmd):
        
        '''
        Based on a fully connected neural network, estimate next robot position
        and velocity
        https://ieeexplore.ieee.org/document/7989202
        
        x_curr - Current robot pose
        v_curr - Current velocities 
        u_cmd -  Commanded velocities
        
        x_next,v_next - Next robot pose and velocity
        '''
        
        # Initial robot state
        state = [x_curr[0], x_curr[1], x_curr[2], v_curr[0], v_curr[1]]
        
        # Applied control action
        control = [u_cmd[0], u_cmd[1]]
        
        # Nominal dynamics
        nominal_next_state = np.array(self.nominal_dynamics_func(state=state , control = control)["next_state"]).reshape(5,) 
        
        # Input to neural network
        nn_input_values = np.array([u_cmd[0],u_cmd[1],v_curr[0],v_curr[1]]).reshape(-1,4)
        # Normalize the input data
        nn_input_values = self.input_scaler.transform(nn_input_values)
        
        nn_input_values = torch.tensor(nn_input_values,dtype=torch.float32).to(self.device)

        # Residual dynamics
        residual_next_state = np.zeros(5,)

        with torch.no_grad():
            nn_next_state = self.nn_model(nn_input_values).cpu().numpy()
        
        # Inverse transform the predictions to the original scale
        nn_next_state = self.output_scaler.inverse_transform(nn_next_state).reshape(-1)
        
        nn_linear_velocity  = nn_next_state[self.terrain_index*2]
        nn_angular_velocity = nn_next_state[(self.terrain_index*2) + 1]
        residual_next_state[3] = nn_linear_velocity
        residual_next_state[4] = nn_angular_velocity 
            
        ground_truth_next_state = nominal_next_state + residual_next_state

        ground_truth_pose = ground_truth_next_state[:3]
        ground_truth_velocity = ground_truth_next_state[3:]
        
        return ground_truth_pose, ground_truth_velocity
    
    def get_state_rollout(self):
        """
        Generate state sequences based on the current optimal control sequence.
        """
        
        # Move things to GPU
        vrange_d = cuda.to_device(self.cfg.vrange.astype(np.float32))
        wrange_d = cuda.to_device(self.cfg.wrange.astype(np.float32))
        x0_d = cuda.to_device(self.cfg.x0.astype(np.float32))
        dt_d = np.float32(self.cfg.dt)

        get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
            self.state_rollout_batch_d, # where to store results
            x0_d, 
            dt_d,
            self.noise_samples_d,
            vrange_d,
            wrange_d,
            self.u_prev_d,
            self.u_cur_d,
            )
        
        return self.state_rollout_batch_d.copy_to_host()
    
    def create_nominal_dynamics_casadi(self):
        # Create CasADi function to compute the derivative of the nominal dynamics
        # wrt robot state for linearizatio based uncertainty propagation
        x = ca.MX.sym('x',5)
        u = ca.MX.sym('u',2)
        
        # Dictionary to set up the nominal dynamics integrator
        ode_dict = {'x':x, 'p':u  , 'ode': nominal_dynamics_ode(x,u)}
        
        # Create the discrete time representation of the nominal robot dynamics
        # Last two arguments represent the time duration for which to integrate
        self.nominal_dynamics_integrator = ca.integrator('nominal_dynamics_integrator','cvodes',ode_dict,0.0,self.dt)
        
        # CasADi function to compute nominal dynamics based next robot state
        self.nominal_dynamics_func = ca.Function("nominal_dynamics_func",[x,u],[self.nominal_dynamics_integrator(x0=x,p=u)['xf']],\
                                                    ["state","control"] , ["next_state"]  )
        
        # Derivative of nominal robot dynamics wrt robot state, dimension (5X5)
        self.nominal_dynamics_casadi_derivative = \
            ca.Function('nominal_dynamics_casadi_derivative',[x,u],[ca.jacobian( self.nominal_dynamics_integrator(x0=x,p=u)['xf'],x ) ], \
                                                       ["state","control"] , ["jac_x"])
    
    def setup_terrain_weights_optimization(self):
        
        # Previous Weight
        self.prev_w_var = cvxpy.Parameter((self.cfg.num_terrains,1))
        
        # Decision variable -- Weights for each terrain
        self.w_var = cvxpy.Variable((self.cfg.num_terrains,1))
        
        # Ground truth for each terrain  for both the GPs.
        # Row vector -- Stacked vertically as:
        # gt_lin_gp_t0,gt_ang_gp_t0,...gt_lin_gp_tK,gt_ang_gp_tK
        self.Y_par = cvxpy.Parameter((2*self.cfg.lookback_horizon,1))
        
        # Pooled mean estimates for all the terrains and all their GP types
        # Each row corresponds to one terrain. Each row tacked vertically as 
        # terrain_1_lin_pred_t0,terrain_1_ang_pred_t0,..,terrain_1_lin_pred_tK,terrain_1_ang_pred_tK
        self.F_par = cvxpy.Parameter((2*self.cfg.lookback_horizon,self.cfg.num_terrains))
        
        # Sum of squares error
        objective = cvxpy.sum_squares(self.Y_par - self.F_par @ self.w_var)
        
        # Penalize deviation between previous and current w
        objective += self.cfg.regularization_eps * cvxpy.norm( self.w_var - self.prev_w_var ,1 )
        
        constraints = [self.w_var >=0.0, self.w_var <=1.0 , cvxpy.sum(self.w_var) == 1.0]
        
        self.w_prob = cvxpy.Problem(cvxpy.Minimize(objective) , constraints )
        
        # Keep track of current and next ground truth velocities for use in optimizer
        self.current_gt_lin_vel = None; self.next_gt_lin_vel = None
        self.current_gt_ang_vel = None; self.next_gt_ang_vel = None
        
        # Variables used in weights optimizer cost function
        self.deque_Y = deque(maxlen=2*self.cfg.lookback_horizon)
        self.deque_F = deque(maxlen=2*self.cfg.lookback_horizon)
        
        # Initialize weights for each terrain as having uniform probability for all terrains
        default_weight = 1./self.cfg.num_terrains
        
        self.terrain_weights = np.array([default_weight] * self.cfg.num_terrains)
        
        self.prev_w = np.array([default_weight] * self.cfg.num_terrains).reshape(-1,1)
    
    def compute_Y_and_F(self,cmd_vel):
        
        """
        Cost function for weights computation is 2-norm squared of ||Y-Fw||
        """
        
        # Create ground truth linear and angular error
        cmd_lin_vel = cmd_vel[0]; cmd_ang_vel = cmd_vel[1]
        params = (self.theta,[cmd_lin_vel,cmd_ang_vel])
        v0 = (self.current_gt_lin_vel,self.current_gt_ang_vel)

        sol = solve_ivp(nominal_dynamics_func, (0,self.dt), v0, args=params)        
        
        if sol.success == True:
            nominal_next_lin_vel = sol.y[0][-1]
            nominal_next_ang_vel = sol.y[1][-1]
        else:
            raise RuntimeError ("Integration unsuccessful, unable to apply nominal unicycle dynamic model")
        
        gt_lin_error = self.next_gt_lin_vel - nominal_next_lin_vel
        gt_ang_error = self.next_gt_ang_vel - nominal_next_ang_vel

        gp_input_values = [cmd_lin_vel, cmd_ang_vel, self.current_gt_lin_vel, self.current_gt_ang_vel]

        # Convert the list to a torch tensor
        gp_input = torch.tensor(gp_input_values, dtype=torch.float32).reshape(1, 4).to(self.device)
                
        with torch.no_grad(), fast_pred_var():
            op = self.likelihood(self.gp_model(gp_input))
        
        mean_numpy = op.mean.cpu().numpy().reshape(-1).astype(np.float64)
        
        reordered_mean_residue_linear = [mean_numpy[i] for i in [0, 2, 4]]
        
        reordered_mean_residue_angular = [mean_numpy[i] for i in [1, 3, 5]]
        
        self.deque_Y.append(gt_lin_error)
        self.deque_Y.append(gt_ang_error)
        
        self.deque_F.append(reordered_mean_residue_linear)
        self.deque_F.append(reordered_mean_residue_angular)
    
    def compute_terrain_weights(self, u_curr):
        
        self.compute_Y_and_F(u_curr)
        
        if len(self.deque_Y) < 2*self.cfg.lookback_horizon:
            return
        
        self.Y_par.value = np.array(self.deque_Y).reshape(-1,1)
        self.F_par.value = np.array(self.deque_F)
        
        self.prev_w_var.value = self.prev_w.reshape(-1,1)
        
        self.w_prob.solve(solver=cvxpy.OSQP, polish=True,eps_abs=0.001, adaptive_rho=True, eps_rel=0.001, verbose=False, warm_start=False)
        
        self.terrain_weights = self.w_var.value.reshape(-1)
        
        self.prev_w = self.w_var.value
    
    def shift_and_update(self, new_x0, new_v0, u_cur, num_shifts=1):
        self.cfg.x0 = new_x0.copy()
        self.cfg.v0 = new_v0.copy()
        
        # Initialize current velocity for next MPC solve
        initial_array = np.tile(self.cfg.v0,(self.num_control_rollouts,self.num_steps,1))
        self.cur_vel_d = cuda.to_device(initial_array.astype(np.float32))
        
        self.shift_optimal_control_sequence(u_cur, num_shifts)
    
    def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
        u_cur_shifted = u_cur.copy()
        u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
        self.u_cur_d = cuda.to_device(u_cur_shifted.astype(np.float32))
        
    def sgf_filtering_post(self,u_after_noise):

        unfiltered_linear  = u_after_noise[:,0].flatten()
        unfiltered_angular = u_after_noise[:,1].flatten()
        
        # Apply SGF filtering
        filtered_linear  = sgf(unfiltered_linear,  self.cfg.sgf_window_length,self.cfg.sgf_polynomial_order)
        filtered_angular = sgf(unfiltered_angular, self.cfg.sgf_window_length,self.cfg.sgf_polynomial_order)
        
        filtered_control = np.column_stack((filtered_linear,filtered_angular))

        # Control saturation
        for ti in range(self.cfg.num_steps):
            filtered_control[ti,0] = max(self.cfg.vrange[0], min(self.cfg.vrange[1], filtered_control[ti,0]))
            filtered_control[ti,1] = max(self.cfg.wrange[0], min(self.cfg.wrange[1], filtered_control[ti,1]))

        # Transfer back filtered control sequence
        self.u_cur_d = cuda.to_device(filtered_control)
        
        return filtered_control
    
    def sgf_filtering(self, u_before_noise, u_after_noise):
        
        """
        SGF filtering and capping
        """
        
        # Extract out unfiltered mppi control noise contribution
        u_noise = u_after_noise - u_before_noise

        unfiltered_noise_linear  = u_noise[:,0].flatten()
        unfiltered_noise_angular = u_noise[:,1].flatten()
        
        # Apply SGF filtering
        filtered_noise_linear  = sgf(unfiltered_noise_linear,  self.cfg.sgf_window_length,self.cfg.sgf_polynomial_order)
        filtered_noise_angular = sgf(unfiltered_noise_angular, self.cfg.sgf_window_length,self.cfg.sgf_polynomial_order)
        
        u_noise_filtered = np.column_stack((filtered_noise_linear,filtered_noise_angular))

        filtered_control = u_before_noise + u_noise_filtered

        # Control saturation
        for ti in range(self.cfg.num_steps):
            filtered_control[ti,0] = max(self.cfg.vrange[0], min(self.cfg.vrange[1], filtered_control[ti,0]))
            filtered_control[ti,1] = max(self.cfg.wrange[0], min(self.cfg.wrange[1], filtered_control[ti,1]))

        # Transfer back filtered control sequence
        self.u_cur_d = cuda.to_device(filtered_control)
        
        return filtered_control
    
    @abstractmethod
    def solve(self):
        raise NotImplementedError ("Solve function override needed for each planner")
    
    @abstractmethod
    def move_mppi_task_vars_to_device(self):
        raise NotImplementedError ("Move to device function override needed for each planner")