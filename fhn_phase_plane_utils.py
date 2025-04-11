#!/usr/bin/env python3
# fhn_phase_plane_utils.py - Utilities for FitzHugh-Nagumo model phase plane analysis

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, cm
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve

def fhn_model(X, t, params):
    """
    FitzHugh-Nagumo model for phase plane analysis.
    
    Args:
        X: State vector [x, y]
        t: Time (unused but required by odeint)
        params: Dictionary of parameters {'a': value, 'b': value, 'tau': value, 'I_ext': value}
    
    Returns:
        List [dx/dt, dy/dt]
    """
    x, y = X
    
    a = params.get('a', 0.7)
    b = params.get('b', 0.8)
    tau = params.get('tau', 12.5)
    I_ext = params.get('I_ext', 0.5)
    
    dxdt = x - (x**3 / 3.0) - y + I_ext
    dydt = (x + a - b * y) / tau
    
    return [dxdt, dydt]

def get_nullclines(params, x_range, y_range, num_points=100):
    """
    Calculate the nullclines of the FHN model.
    
    Args:
        params: Dictionary of parameters
        x_range: Range for x variable [min, max]
        y_range: Range for y variable [min, max]
        num_points: Number of points for discretization
    
    Returns:
        Dictionary with nullcline points
    """
    a = params.get('a', 0.7)
    b = params.get('b', 0.8)
    tau = params.get('tau', 12.5)
    I_ext = params.get('I_ext', 0.5)
    
    # X-nullcline: y = x - x^3/3 + I_ext
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_x_nullcline = x_vals - x_vals**3/3.0 + I_ext
    
    # Y-nullcline: y = (x + a)/b
    y_y_nullcline = (x_vals + a) / b
    
    return {
        'x_vals': x_vals,
        'x_nullcline': y_x_nullcline,
        'y_nullcline': y_y_nullcline
    }
def find_fixed_points(params):
    """
    Find the fixed points of the FHN model.
    
    Args:
        params: Dictionary of parameters
    
    Returns:
        List of fixed points [(x1,y1), (x2,y2), ...]
    """
    a = params.get('a', 0.7)
    b = params.get('b', 0.8)
    tau = params.get('tau', 12.5)
    I_ext = params.get('I_ext', 0.5)
    
    # Function to find roots of
    def fixed_point_eqs(X):
        x, y = X
        dx = x - (x**3 / 3.0) - y + I_ext
        dy = (x + a - b * y) / tau
        return [dx, dy]
    
    # Try different initial guesses to find all fixed points
    guesses = [
        [-3.0, -3.0], [-3.0, 0.0], [-3.0, 3.0],
        [0.0, -3.0], [0.0, 0.0], [0.0, 3.0],
        [3.0, -3.0], [3.0, 0.0], [3.0, 3.0],
        [-1.5, -1.5], [-1.5, 1.5],
        [1.5, -1.5], [1.5, 1.5]
    ]
    fixed_points = []
    
    for guess in guesses:
        sol = fsolve(fixed_point_eqs, guess)
        # Check if it's actually a fixed point and not already in our list
        if np.allclose(fixed_point_eqs(sol), [0, 0], atol=1e-6):
            # Check if this point is already in our list
            if not any(np.allclose(sol, fp, atol=1e-6) for fp in fixed_points):
                fixed_points.append(sol)
    
    return fixed_points

def plot_phase_plane(params, x_range=[-2.5, 2.5], y_range=[-1.0, 2.5], grid_size=20, 
                    trajectories=None, title=None, figsize=(10, 8), fixed_points=True):
    """
    Plot phase plane of FHN model with nullclines, vector field, and optional trajectories.
    
    Args:
        params: Dictionary of parameters
        x_range: Range for x variable [min, max]
        y_range: Range for y variable [min, max]
        grid_size: Size of the grid for vector field
        trajectories: List of trajectories to plot [traj1, traj2, ...] 
                      where each traj is an array of shape (times, 2)
        title: Plot title
        figsize: Figure size
        fixed_points: Whether to plot fixed points
    
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid for vector field
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Compute vector field
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            dx, dy = fhn_model([X[i, j], Y[i, j]], 0, params)
            U[i, j] = dx
            V[i, j] = dy
    
    # Normalize arrows for better visualization
    speed = np.sqrt(U**2 + V**2)
    U_norm = U / (speed + 1e-10)  # Avoid division by zero
    V_norm = V / (speed + 1e-10)
    
    # Plot vector field
    ax.streamplot(X, Y, U, V, density=1.0, color='gray', linewidth=0.8, arrowsize=0.8)
    
    # Plot nullclines
    nullclines = get_nullclines(params, x_range, y_range)
    ax.plot(nullclines['x_vals'], nullclines['x_nullcline'], 'b-', label='x-nullcline')
    ax.plot(nullclines['x_vals'], nullclines['y_nullcline'], 'r-', label='y-nullcline')
    
    # Plot fixed points
    if fixed_points:
        fps = find_fixed_points(params)
        for fp in fps:
            ax.plot(fp[0], fp[1], 'ko', markersize=8)
            
    # Plot trajectories if provided
    if trajectories is not None:
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5, alpha=0.7)
            ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=6)  # Mark start point
    
    # Set labels and title
    ax.set_xlabel('Voltage (x)')
    ax.set_ylabel('Recovery (y)')
    if title:
        ax.set_title(title)
    else:
        param_str = ", ".join([f"{k}={v:.2f}" for k, v in params.items()])
        ax.set_title(f'FHN Phase Plane ({param_str})')
    
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    return fig, ax

def simulate_trajectory(params, initial_condition, t_span, n_points=1000):
    """
    Simulate a trajectory of the FHN model.
    
    Args:
        params: Dictionary of parameters
        initial_condition: Initial state [x0, y0]
        t_span: Time span [t_start, t_end]
        n_points: Number of time points
    
    Returns:
        Dictionary with time and state arrays
    """
    t = np.linspace(t_span[0], t_span[1], n_points)
    
    # Solve ODE
    solution = odeint(fhn_model, initial_condition, t, args=(params,))
    
    # Calculate derivatives for each point
    derivatives = np.array([fhn_model(state, 0, params) for state in solution])
    
    return {
        't': t,
        'states': solution,
        'derivatives': derivatives
    }

def plot_time_series(simulation_data, params=None, figsize=(12, 6)):
    """
    Plot time series of FHN model simulation.
    
    Args:
        simulation_data: Dictionary from simulate_trajectory function
        params: Dictionary of parameters (for title)
        figsize: Figure size
    
    Returns:
        Figure and axes
    """
    t = simulation_data['t']
    states = simulation_data['states']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot voltage variable
    ax1.plot(t, states[:, 0], 'b-', linewidth=2)
    ax1.set_ylabel('Voltage (x)')
    ax1.grid(True)
    
    # Plot recovery variable
    ax2.plot(t, states[:, 1], 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Recovery (y)')
    ax2.grid(True)
    
    # Set title
    if params:
        param_str = ", ".join([f"{k}={v:.2f}" for k, v in params.items()])
        fig.suptitle(f'FHN Model Time Series ({param_str})')
    else:
        fig.suptitle('FHN Model Time Series')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def calculate_jacobian(x, y, params):
    """
    Calculate the Jacobian matrix at a point for the FHN model.
    
    Args:
        x: x-coordinate (voltage)
        y: y-coordinate (recovery)
        params: Dictionary of parameters
    
    Returns:
        2x2 Jacobian matrix
    """
    a = params.get('a', 0.7)
    b = params.get('b', 0.8)
    tau = params.get('tau', 12.5)
    
    # Partial derivatives
    df1_dx = 1 - x**2
    df1_dy = -1
    df2_dx = 1/tau
    df2_dy = -b/tau
    
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])

def get_eigenvalues(jac):
    """
    Calculate eigenvalues from Jacobian matrix.
    
    Args:
        jac: 2x2 Jacobian matrix
    
    Returns:
        Numpy array of eigenvalues
    """
    return np.linalg.eigvals(jac)

def classify_fixed_point(jac):
    """
    Classify fixed point based on eigenvalues of Jacobian.
    
    Args:
        jac: 2x2 Jacobian matrix
    
    Returns:
        String indicating stability type: "stable", "unstable", "saddle", or "hopf"
    """
    eigenvalues = get_eigenvalues(jac)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Check for complex eigenvalues with real parts very close to zero
    # Use a more generous tolerance for Hopf detection
    near_zero_real = np.any(np.abs(real_parts) < 1e-4)
    
    # Check if all eigenvalues have negative real parts
    all_negative = np.all(real_parts < 0)
    
    # Check if all eigenvalues have positive real parts
    all_positive = np.all(real_parts > 0)
    
    # Check if eigenvalues have imaginary parts (complex eigenvalues)
    has_imag = np.any(np.abs(imag_parts) > 1e-6)
    
    if near_zero_real and has_imag:
        return "hopf"  # Hopf bifurcation
    elif all_negative:
        return "stable"  # Stable node/focus
    elif all_positive:
        return "unstable"  # Unstable node/focus
    else:
        return "saddle"  # Saddle point

def detect_hopf_bifurcation(param_values, fixed_points, params, param_name):
    """
    Detect Hopf bifurcations by tracking eigenvalue sign changes.
    
    Args:
        param_values: Array of parameter values
        fixed_points: List of fixed points for each parameter value
        params: Base parameter dictionary
        param_name: Name of the parameter being varied
    
    Returns:
        List of (param_value, fixed_point) where Hopf bifurcations occur
    """
    hopf_points = []
    eigenvalue_history = {}
    
    # Group fixed points by their location (to track the same fixed point across parameter values)
    for i, val in enumerate(param_values):
        if i >= len(fixed_points) or not fixed_points[i]:
            continue
            
        for fp in fixed_points[i]:
            # Create a key based on rounded fixed point coordinates (for tracking)
            fp_key = (round(fp[0], 1), round(fp[1], 1))
            
            # Calculate eigenvalues
            p = params.copy()
            p[param_name] = val
            jac = calculate_jacobian(fp[0], fp[1], p)
            eigenvals = get_eigenvalues(jac)
            
            # Add to history
            if fp_key not in eigenvalue_history:
                eigenvalue_history[fp_key] = []
            eigenvalue_history[fp_key].append((val, fp, eigenvals))
    
    # Process each fixed point's history to detect sign changes in real parts
    for fp_key, history in eigenvalue_history.items():
        if len(history) < 2:
            continue
            
        # Sort by parameter value
        history.sort(key=lambda x: x[0])
        
        for i in range(len(history) - 1):
            val1, fp1, eig1 = history[i]
            val2, fp2, eig2 = history[i + 1]
            
            # Check for sign change in real part of eigenvalues
            real1 = np.real(eig1)
            real2 = np.real(eig2)
            imag1 = np.imag(eig1)
            imag2 = np.imag(eig2)
            
            # Must have complex eigenvalues and a sign change in real parts
            if (np.any(np.abs(imag1) > 1e-6) and np.any(np.abs(imag2) > 1e-6)):
                # Check if real parts cross zero (sign change)
                sign_change = False
                for r1, r2 in zip(real1, real2):
                    if (r1 * r2 <= 0) and not np.isclose(r1, 0) and not np.isclose(r2, 0):
                        sign_change = True
                        break
                
                if sign_change:
                    # Interpolate to find approximate bifurcation point
                    interp_val = val1 + (val2 - val1) * (0 - np.max(real1)) / (np.max(real2) - np.max(real1))
                    interp_fp = [fp1[0] + (fp2[0] - fp1[0]) * (interp_val - val1) / (val2 - val1),
                                fp1[1] + (fp2[1] - fp1[1]) * (interp_val - val1) / (val2 - val1)]
                    
                    hopf_points.append((interp_val, interp_fp))
    
    return hopf_points

def simulate_limit_cycle(params, t_max=1000, transient=800, dt=0.01, initial_condition=None):
    """
    Simulate to find limit cycle and return its amplitude.
    
    Args:
        params: Dictionary of model parameters
        t_max: Maximum simulation time (increased for better convergence)
        transient: Time to discard as transient (increased to ensure system reaches steady state)
        dt: Time step
        initial_condition: Optional initial state [x, y]
        
    Returns:
        Dict with min/max values or None if no limit cycle found
    """
    # Create time points
    t = np.arange(0, t_max, dt)
    
    # Set initial condition near a typical equilibrium point
    if initial_condition is None:
        initial_condition = [0.5, 0.5]
    
    # Simulate
    solution = odeint(fhn_model, initial_condition, t, args=(params,))
    
    # Discard transient
    trans_idx = int(transient / dt)
    steady_sol = solution[trans_idx:, :]
    
    # Check if we have a limit cycle by looking at range
    x_range = np.max(steady_sol[:, 0]) - np.min(steady_sol[:, 0])
    y_range = np.max(steady_sol[:, 1]) - np.min(steady_sol[:, 1])
    
    # If the range is very small, probably not a limit cycle
    if x_range < 0.1 and y_range < 0.1:
        return None
    
    # Return amplitude data
    return {
        'x_min': np.min(steady_sol[:, 0]),
        'x_max': np.max(steady_sol[:, 0]),
        'y_min': np.min(steady_sol[:, 1]),
        'y_max': np.max(steady_sol[:, 1]),
        'x_amplitude': x_range,
        'y_amplitude': y_range
    }

def check_if_limit_cycle_exists(params, initial_conditions):
    """
    Determine if a limit cycle exists for the given parameters.
    Tries multiple initial conditions and returns the largest amplitude.
    
    Args:
        params: Dictionary of model parameters
        initial_conditions: List of initial conditions to try
    
    Returns:
        Limit cycle data or None if none found
    """
    largest_cycle = None
    max_amplitude = 0
    
    for ic in initial_conditions:
        lc = simulate_limit_cycle(params, initial_condition=ic)
        if lc and lc['x_amplitude'] > 0.2:  # Only consider significant oscillations
            if largest_cycle is None or lc['x_amplitude'] > max_amplitude:
                largest_cycle = lc
                max_amplitude = lc['x_amplitude']
    
    return largest_cycle

def plot_bifurcation_diagram(param_name, param_range, other_params, x_range=[-2.5, 2.5], 
                            num_points=1000, figsize=(12, 6), detect_hopf=True, y_lim=[-5, 5],
                            track_limit_cycles=True, max_plot_points=300):
    """
    Plot bifurcation diagram for FHN model with stability analysis.
    
    Args:
        param_name: Name of parameter to vary
        param_range: Range of parameter values [min, max]
        other_params: Dictionary of other fixed parameters
        x_range: Range for x variable [min, max]
        num_points: Number of parameter values to compute
        figsize: Figure size
        detect_hopf: Whether to use enhanced Hopf detection
        y_lim: Y-axis limits for both plots
        track_limit_cycles: Whether to track limit cycle amplitudes
        max_plot_points: Maximum number of points to plot (for performance)
    
    Returns:
        Figure and axes
    """
    # Generate parameter values
    param_values = np.linspace(param_range[0], param_range[1], num_points)
    
    # Arrays to store fixed points
    fixed_points_x = []
    fixed_points_y = []
    param_for_points = []
    stability_types = []  # Store the stability of each fixed point
    all_fixed_points = []  # Store all fixed points for each parameter
    
    # Compute fixed points for each parameter value
    for val in param_values:
        params = other_params.copy()
        params[param_name] = val
        
        # Get fixed points
        fps = find_fixed_points(params)
        all_fixed_points.append(fps)
        
        for fp in fps:
            x, y = fp
            jac = calculate_jacobian(x, y, params)
            stability = classify_fixed_point(jac)
            
            fixed_points_x.append(x)
            fixed_points_y.append(y)
            param_for_points.append(val)
            stability_types.append(stability)
    
    # Detect Hopf bifurcations via eigenvalue tracking
    hopf_bifurcations = []
    if detect_hopf:
        hopf_bifurcations = detect_hopf_bifurcation(param_values, all_fixed_points, other_params, param_name)
    
    # Track limit cycles if requested
    limit_cycle_data = []
    if track_limit_cycles:
        # Define the parameter values at which to check for limit cycles
        start_check = param_range[0]
        if hopf_bifurcations:
            # Start just before the first Hopf bifurcation
            first_hopf = min([h[0] for h in hopf_bifurcations])
            start_check = max(param_range[0], first_hopf - 0.1)
        
        # Sample densely through the parameter range
        lc_param_values = np.linspace(start_check, param_range[1], 60)
        
        print(f"Tracking limit cycles for {len(lc_param_values)} parameter values...")
        
        # Define standard set of initial conditions to try
        initial_conditions = [
            [1.0, 0.3], [-1.0, -0.3], [1.5, 0.5], [-1.5, -0.5],
            [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0],
            [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]
        ]
        
        # Try additional initial conditions around equilibrium points
        for fps in all_fixed_points:
            for fp in fps:
                # Add initial conditions near each fixed point
                for dx, dy in [(0.1, 0.1), (0.2, 0.0), (0.0, 0.2), (-0.1, -0.1)]:
                    ic = [fp[0] + dx, fp[1] + dy]
                    if ic not in initial_conditions:
                        initial_conditions.append(ic)
        
        # For each parameter value, find the largest limit cycle
        for val in lc_param_values:
            params = other_params.copy()
            params[param_name] = val
            
            lc = check_if_limit_cycle_exists(params, initial_conditions)
            if lc:
                limit_cycle_data.append({
                    'param': val,
                    'x_min': lc['x_min'],
                    'x_max': lc['x_max'],
                    'y_min': lc['y_min'],
                    'y_max': lc['y_max'],
                    'amplitude': lc['x_amplitude']
                })
        
        print(f"Found {len(limit_cycle_data)} limit cycle points")
    
    # Convert to numpy arrays for easier filtering
    fixed_points_x = np.array(fixed_points_x)
    fixed_points_y = np.array(fixed_points_y)
    param_for_points = np.array(param_for_points)
    stability_types = np.array(stability_types)
    
    # Sample points if there are too many (for performance and cleaner plots)
    if len(param_for_points) > max_plot_points:
        # Group points by stability type and sample each group separately
        stable_idx = np.where(stability_types == "stable")[0]
        unstable_idx = np.where(stability_types == "unstable")[0]
        saddle_idx = np.where(stability_types == "saddle")[0]
        
        # Calculate number of points to sample from each group
        n_stable = min(len(stable_idx), int(max_plot_points * len(stable_idx) / len(param_for_points)))
        n_unstable = min(len(unstable_idx), int(max_plot_points * len(unstable_idx) / len(param_for_points)))
        n_saddle = min(len(saddle_idx), int(max_plot_points * len(saddle_idx) / len(param_for_points)))
        
        # Sample indices
        if len(stable_idx) > n_stable:
            stable_sample = np.sort(np.random.choice(stable_idx, n_stable, replace=False))
        else:
            stable_sample = stable_idx
            
        if len(unstable_idx) > n_unstable:
            unstable_sample = np.sort(np.random.choice(unstable_idx, n_unstable, replace=False))
        else:
            unstable_sample = unstable_idx
            
        if len(saddle_idx) > n_saddle:
            saddle_sample = np.sort(np.random.choice(saddle_idx, n_saddle, replace=False))
        else:
            saddle_sample = saddle_idx
        
        # Create masks for sampled points
        stable_mask = np.zeros_like(param_for_points, dtype=bool)
        stable_mask[stable_sample] = True
        
        unstable_mask = np.zeros_like(param_for_points, dtype=bool)
        unstable_mask[unstable_sample] = True
        
        saddle_mask = np.zeros_like(param_for_points, dtype=bool)
        saddle_mask[saddle_sample] = True
    else:
        # No sampling needed, use all points
        stable_mask = stability_types == "stable"
        unstable_mask = stability_types == "unstable"
        saddle_mask = stability_types == "saddle"
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors (consistent across plots)
    stable_color = '#DDA853'  # Gold/amber
    unstable_color = '#27548A'  # Blue
    saddle_color = '#034C53'  # Teal
    hopf_color = '#F7374F'  # Red
    amplitude_color = '#F7374F'  # Red
    
    # Plot x-coordinate with different markers by stability
    ax1.scatter(param_for_points[stable_mask], fixed_points_x[stable_mask], 
               color=stable_color, marker='o', s=20, label='Stable')
    ax1.scatter(param_for_points[unstable_mask], fixed_points_x[unstable_mask], 
               color=unstable_color, marker='^', s=20, label='Unstable')
    ax1.scatter(param_for_points[saddle_mask], fixed_points_x[saddle_mask], 
               color=saddle_color, marker='s', s=20, label='Saddle')
    
    # Add the detected Hopf bifurcation points (use * instead of x)
    for hopf_param, hopf_fp in hopf_bifurcations:
        ax1.scatter(hopf_param, hopf_fp[0], color=hopf_color, marker='*', s=150, 
                   edgecolor='k', linewidth=1, label='Hopf bifurcation')
        ax2.scatter(hopf_param, hopf_fp[1], color=hopf_color, marker='*', s=150, 
                  edgecolor='k', linewidth=1, label='_nolegend_')
        # Only add to legend once
        break
        
    # Plot limit cycle amplitudes if available
    if limit_cycle_data:
        print(f"Processing {len(limit_cycle_data)} limit cycle data points for plotting")
        
        # Sort by parameter value and interpolate to get smoother curves
        limit_cycle_data_sorted = sorted(limit_cycle_data, key=lambda x: x['param'])
        
        # Extract data
        lc_params = np.array([lc['param'] for lc in limit_cycle_data_sorted])
        lc_x_min = np.array([lc['x_min'] for lc in limit_cycle_data_sorted])
        lc_x_max = np.array([lc['x_max'] for lc in limit_cycle_data_sorted])
        lc_y_min = np.array([lc['y_min'] for lc in limit_cycle_data_sorted])
        lc_y_max = np.array([lc['y_max'] for lc in limit_cycle_data_sorted])
        
        # Add data padding to extend curves at both ends if needed
        if len(lc_params) > 1:
            # Calculate rate of change to project endpoints
            first_idx, last_idx = 0, len(lc_params) - 1
            
            # Add point at the beginning if needed
            if lc_params[first_idx] > param_range[0] + 0.05:
                # Project back using the slope at the beginning
                dx_min = (lc_x_min[first_idx+1] - lc_x_min[first_idx]) / (lc_params[first_idx+1] - lc_params[first_idx])
                dx_max = (lc_x_max[first_idx+1] - lc_x_max[first_idx]) / (lc_params[first_idx+1] - lc_params[first_idx])
                dy_min = (lc_y_min[first_idx+1] - lc_y_min[first_idx]) / (lc_params[first_idx+1] - lc_params[first_idx])
                dy_max = (lc_y_max[first_idx+1] - lc_y_max[first_idx]) / (lc_params[first_idx+1] - lc_params[first_idx])
                
                back_param = param_range[0]
                back_x_min = lc_x_min[first_idx] - dx_min * (lc_params[first_idx] - back_param)
                back_x_max = lc_x_max[first_idx] - dx_max * (lc_params[first_idx] - back_param)
                back_y_min = lc_y_min[first_idx] - dy_min * (lc_params[first_idx] - back_param)
                back_y_max = lc_y_max[first_idx] - dy_max * (lc_params[first_idx] - back_param)
                
                lc_params = np.insert(lc_params, 0, back_param)
                lc_x_min = np.insert(lc_x_min, 0, back_x_min)
                lc_x_max = np.insert(lc_x_max, 0, back_x_max)
                lc_y_min = np.insert(lc_y_min, 0, back_y_min)
                lc_y_max = np.insert(lc_y_max, 0, back_y_max)
                
            # Add point at the end if needed
            if lc_params[last_idx] < param_range[1] - 0.05:
                # Project forward using the slope at the end
                dx_min = (lc_x_min[last_idx] - lc_x_min[last_idx-1]) / (lc_params[last_idx] - lc_params[last_idx-1])
                dx_max = (lc_x_max[last_idx] - lc_x_max[last_idx-1]) / (lc_params[last_idx] - lc_params[last_idx-1])
                dy_min = (lc_y_min[last_idx] - lc_y_min[last_idx-1]) / (lc_params[last_idx] - lc_params[last_idx-1])
                dy_max = (lc_y_max[last_idx] - lc_y_max[last_idx-1]) / (lc_params[last_idx] - lc_params[last_idx-1])
                
                forward_param = param_range[1]
                forward_x_min = lc_x_min[last_idx] + dx_min * (forward_param - lc_params[last_idx])
                forward_x_max = lc_x_max[last_idx] + dx_max * (forward_param - lc_params[last_idx])
                forward_y_min = lc_y_min[last_idx] + dy_min * (forward_param - lc_params[last_idx])
                forward_y_max = lc_y_max[last_idx] + dy_max * (forward_param - lc_params[last_idx])
                
                lc_params = np.append(lc_params, forward_param)
                lc_x_min = np.append(lc_x_min, forward_x_min)
                lc_x_max = np.append(lc_x_max, forward_x_max)
                lc_y_min = np.append(lc_y_min, forward_y_min)
                lc_y_max = np.append(lc_y_max, forward_y_max)
        
        # Plot the curves
        ax1.plot(lc_params, lc_x_max, color=amplitude_color, linewidth=2, label='Max amplitude')
        ax1.plot(lc_params, lc_x_min, color=amplitude_color, linewidth=2, label='Min amplitude')
        ax2.plot(lc_params, lc_y_max, color=amplitude_color, linewidth=2, label='_nolegend_')
        ax2.plot(lc_params, lc_y_min, color=amplitude_color, linewidth=2, label='_nolegend_')
    
    ax1.set_xlabel(f'Parameter {param_name}')
    ax1.set_ylabel('Fixed Point (x)')
    ax1.set_title(f'Bifurcation Diagram for x-coordinate')
    ax1.set_ylim(y_lim)
    ax1.grid(True)
    
    # Plot y-coordinate with same colors and markers for consistency
    ax2.scatter(param_for_points[stable_mask], fixed_points_y[stable_mask], 
               color=stable_color, marker='o', s=20, label='_nolegend_')
    ax2.scatter(param_for_points[unstable_mask], fixed_points_y[unstable_mask], 
               color=unstable_color, marker='^', s=20, label='_nolegend_')
    ax2.scatter(param_for_points[saddle_mask], fixed_points_y[saddle_mask], 
               color=saddle_color, marker='s', s=20, label='_nolegend_')
    
    ax2.set_xlabel(f'Parameter {param_name}')
    ax2.set_ylabel('Fixed Point (y)')
    ax2.set_title(f'Bifurcation Diagram for y-coordinate')
    ax2.set_ylim(y_lim)
    ax2.grid(True)
    
    # Create a unified legend at the bottom of the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01), frameon=False)
    
    # Remove individual legends
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for the legend
    return fig, (ax1, ax2)

def plot_eigenvalue_movement(param_name, param_range, point_of_interest, other_params, num_points=200, figsize=(10, 6)):
    """
    Plot the movement of eigenvalues in the complex plane as a parameter changes.
    Useful for identifying Hopf bifurcations.
    
    Args:
        param_name: Name of parameter to vary
        param_range: Range of parameter values [min, max]
        point_of_interest: Initial fixed point to track [x, y]
        other_params: Dictionary of other fixed parameters
        num_points: Number of parameter values to compute
        figsize: Figure size
    
    Returns:
        Figure and axes
    """
    # Generate parameter values
    param_values = np.linspace(param_range[0], param_range[1], num_points)
    
    # Arrays to store eigenvalues
    eig_real_parts = []
    eig_imag_parts = []
    params_for_eigs = []
    
    # Track a specific fixed point as the parameter changes
    x_prev, y_prev = point_of_interest
    
    for val in param_values:
        params = other_params.copy()
        params[param_name] = val
        
        # Find the fixed point closest to our previous point
        fps = find_fixed_points(params)
        if len(fps) == 0:
            continue
            
        # Find the closest fixed point to our previous one
        distances = [np.sqrt((fp[0] - x_prev)**2 + (fp[1] - y_prev)**2) for fp in fps]
        closest_idx = np.argmin(distances)
        fp = fps[closest_idx]
        
        # Update for next iteration
        x_prev, y_prev = fp
        
        # Calculate eigenvalues
        jac = calculate_jacobian(fp[0], fp[1], params)
        eigenvals = np.linalg.eigvals(jac)
        
        # Store real and imaginary parts of both eigenvalues
        for eigenval in eigenvals:
            eig_real_parts.append(np.real(eigenval))
            eig_imag_parts.append(np.imag(eigenval))
            params_for_eigs.append(val)
    
    # Convert to numpy arrays
    eig_real_parts = np.array(eig_real_parts)
    eig_imag_parts = np.array(eig_imag_parts)
    params_for_eigs = np.array(params_for_eigs)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot eigenvalues in complex plane
    scatter = ax1.scatter(eig_real_parts, eig_imag_parts, c=params_for_eigs, cmap='viridis')
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Eigenvalue Movement in Complex Plane')
    ax1.grid(True)
    fig.colorbar(scatter, ax=ax1, label=f'Parameter {param_name}')
    
    # Plot real parts of eigenvalues vs parameter
    ax2.scatter(params_for_eigs, eig_real_parts, c=eig_imag_parts, cmap='coolwarm')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f'Parameter {param_name}')
    ax2.set_ylabel('Real Part of Eigenvalues')
    ax2.set_title('Real Parts vs Parameter')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def parameter_sweep_phase_planes(param_name, param_values, other_params, 
                                x_range=[-2.5, 2.5], y_range=[-1.0, 2.5], 
                                grid_size=20, figsize=(15, 10)):
    """
    Create a grid of phase plane plots for different parameter values.
    
    Args:
        param_name: Name of parameter to vary
        param_values: List of parameter values to use
        other_params: Dictionary of other fixed parameters
        x_range: Range for x variable [min, max]
        y_range: Range for y variable [min, max]
        grid_size: Size of the grid for vector field
        figsize: Figure size
    
    Returns:
        Figure
    """
    n_plots = len(param_values)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, val in enumerate(param_values):
        if i < len(axes):
            params = other_params.copy()
            params[param_name] = val
            
            # Create grid for vector field
            x = np.linspace(x_range[0], x_range[1], grid_size)
            y = np.linspace(y_range[0], y_range[1], grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Compute vector field
            U = np.zeros_like(X)
            V = np.zeros_like(X)
            for j in range(grid_size):
                for k in range(grid_size):
                    dx, dy = fhn_model([X[j, k], Y[j, k]], 0, params)
                    U[j, k] = dx
                    V[j, k] = dy
            
            # Plot vector field
            axes[i].streamplot(X, Y, U, V, density=1.0, color='gray', linewidth=0.8, arrowsize=0.8)
            
            # Plot nullclines
            nullclines = get_nullclines(params, x_range, y_range)
            axes[i].plot(nullclines['x_vals'], nullclines['x_nullcline'], 'b-', label='x-nullcline')
            axes[i].plot(nullclines['x_vals'], nullclines['y_nullcline'], 'r-', label='y-nullcline')
            
            # Plot fixed points
            fps = find_fixed_points(params)
            for fp in fps:
                axes[i].plot(fp[0], fp[1], 'ko', markersize=6)
            
            # Set title and labels
            axes[i].set_title(f'{param_name}={val:.2f}')
            axes[i].set_xlabel('Voltage (x)')
            axes[i].set_ylabel('Recovery (y)')
            axes[i].grid(True)
            axes[i].set_xlim(x_range)
            axes[i].set_ylim(y_range)
            
            if i == 0:
                axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(param_values), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig 