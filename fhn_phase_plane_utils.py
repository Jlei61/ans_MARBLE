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
    guesses = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
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

def plot_bifurcation_diagram(param_name, param_range, other_params, x_range=[-2.5, 2.5], 
                            num_points=100, figsize=(10, 6)):
    """
    Plot bifurcation diagram for FHN model.
    
    Args:
        param_name: Name of parameter to vary
        param_range: Range of parameter values [min, max]
        other_params: Dictionary of other fixed parameters
        x_range: Range for x variable [min, max]
        num_points: Number of parameter values to compute
        figsize: Figure size
    
    Returns:
        Figure and axes
    """
    # Generate parameter values
    param_values = np.linspace(param_range[0], param_range[1], num_points)
    
    # Arrays to store fixed points
    fixed_points_x = []
    fixed_points_y = []
    param_for_points = []
    
    # Compute fixed points for each parameter value
    for val in param_values:
        params = other_params.copy()
        params[param_name] = val
        
        fps = find_fixed_points(params)
        for fp in fps:
            fixed_points_x.append(fp[0])
            fixed_points_y.append(fp[1])
            param_for_points.append(val)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot x-coordinate of fixed points vs parameter
    ax1.scatter(param_for_points, fixed_points_x, color='b', s=10)
    ax1.set_xlabel(f'Parameter {param_name}')
    ax1.set_ylabel('Fixed Point (x)')
    ax1.set_title(f'Bifurcation Diagram for x-coordinate')
    ax1.grid(True)
    
    # Plot y-coordinate of fixed points vs parameter
    ax2.scatter(param_for_points, fixed_points_y, color='r', s=10)
    ax2.set_xlabel(f'Parameter {param_name}')
    ax2.set_ylabel('Fixed Point (y)')
    ax2.set_title(f'Bifurcation Diagram for y-coordinate')
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