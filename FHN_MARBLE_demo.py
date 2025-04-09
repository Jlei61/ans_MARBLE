#!/usr/bin/env python3
# FHN_MARBLE_demo.py - Simplified FitzHugh-Nagumo model analyzed with MARBLE

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch

import MARBLE
from MARBLE import plotting, geometry
import MARBLE.preprocessing as mprep

# Create directories for results
os.makedirs('temp_Figures', exist_ok=True)
os.makedirs('temp_Data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Default FHN model parameters
FHN_PARAMS = {
    "a": 0.7,
    "b": 0.8,
    "tau": 12.5,
    "I_ext": 0.5
}

# MARBLE parameters
MARBLE_PARAMS = {
    'batch_size': 64,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'diffusion': True,
    'hidden_channels': [64, 32],
    'out_channels': 2,
    'batch_norm': True,
    'seed': 42
}

def fun_fhn(par=None):
    """FitzHugh-Nagumo oscillator function definition"""
    if par is None:
        par = FHN_PARAMS.copy()

    def f(_, X):
        x, y = X
        dx = x - (x**3)/3 - y + par["I_ext"]
        dy = (x + par["a"] - par["b"] * y) / par["tau"]
        return [dx, dy]

    def jac(_, X):
        x, y = X
        df1 = [1 - x**2, -1]
        df2 = [1/par["tau"], -par["b"]/par["tau"]]
        return [df1, df2]

    return f, jac

def solve_ode(f, jac, t, x0):
    """Solve ODE system"""
    x = odeint(f, x0, t, Dfun=jac, tfirst=True)
    xprime = np.array([f(t_, x_) for t_, x_ in zip(t, x)])
    return x, xprime

def sample_2d(N=100, interval=None, method="random", seed=0):
    """Sample N points in a 2D area."""
    if interval is None:
        interval = [[-2.0, -0.8], [2.0, 2.0]]
    
    np.random.seed(seed)
    x = np.random.uniform(
        (interval[0][0], interval[0][1]), 
        (interval[1][0], interval[1][1]), 
        (N, 2)
    )
    return x

def initial_conditions(n, reps, area=None, seed=42):
    """Generate initial conditions"""
    if area is None:
        area = [[-2.0, -0.8], [2.0, 2.0]]
    X0_range = [sample_2d(1, area, "random", seed=i + seed) for i in range(n * reps)]
    return X0_range

def simulate_fhn(params, X0_list, t, vary_param=None, param_range=None):
    """Simulate FHN model for multiple initial conditions"""
    pos_list, vel_list, param_list = [], [], []
    
    for i, X0 in enumerate(X0_list):
        # If varying a parameter, assign unique value to this trajectory
        if vary_param and param_range:
            traj_params = params.copy()
            param_val = np.random.uniform(param_range[0], param_range[1])
            traj_params[vary_param] = param_val
            param_list.append(param_val)
        else:
            traj_params = params
            param_list.append(0)  # Default value if not varying
            
        # Get system equations
        f, jac = fun_fhn(traj_params)
        
        # Solve ODE
        X0_flat = X0.flatten()  # Ensure X0 is 1D array
        pos, vel = solve_ode(f, jac, t, X0_flat)
        
        pos_list.append(pos)
        vel_list.append(vel)

    return pos_list, vel_list, np.array(param_list)

def prepare_marble_dataset(pos_list, vel_list, param_list, k=100):
    """Prepare data for MARBLE training"""
    print("Preparing MARBLE dataset...")
    
    # Construct the dataset
    data = MARBLE.construct_dataset(
        anchor=pos_list, 
        vector=vel_list,
        k=k,  # Number of nearest neighbors
        # number_of_eigenvectors=5,
        number_of_resamples=0,
    )

    # Store parameter values
    param_values = np.zeros(data.x.shape[0])
    traj_indices = np.zeros(data.x.shape[0], dtype=int)

    start_idx = 0
    for i, sample_indices in enumerate(data.sample_ind.split(data.num_nodes.tolist())):
        param_val = param_list[i]
        end_idx = start_idx + len(sample_indices)
        param_values[start_idx:end_idx] = param_val
        traj_indices[start_idx:end_idx] = i
        start_idx = end_idx

    data.param_values = torch.tensor(param_values, dtype=torch.float32)
    data.traj_indices = torch.tensor(traj_indices, dtype=torch.long)
    
    print(f"Created dataset with {data.x.shape[0]} points and {data.edge_index.shape[1]} edges")
    return data

def train_marble_model(data):
    """Train MARBLE model on dataset"""
    print("Training MARBLE model...")
    
    # Create and train model
    model = MARBLE.net(data, params=MARBLE_PARAMS, verbose=True)
    model.fit(data, verbose=True)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(model.losses['train_loss'], label='Training')
    plt.plot(model.losses['val_loss'], label='Validation')
    plt.title('MARBLE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('temp_Figures/marble_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

def visualize_embeddings(model, data):
    """Visualize MARBLE embeddings"""
    print("Visualizing embeddings...")
    
    # Transform data to get embeddings
    embedded_data = model.transform(data)
    embeddings = embedded_data.emb.numpy()
    param_values = data.param_values.numpy()
    
    # Plot embeddings colored by parameter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=param_values, cmap='viridis', alpha=0.5, s=5)
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.colorbar(scatter, label='Parameter value')
    plt.title('MARBLE Embeddings of FHN Dynamics')
    plt.grid(True)
    plt.savefig('temp_Figures/fhn_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot original phase space colored by first embedding dimension
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), 
                         c=embeddings[:, 0], cmap='coolwarm', s=5, alpha=0.5)
    plt.xlabel('Voltage (x)')
    plt.ylabel('Recovery (y)')
    plt.title('State Space Colored by Embedding Dim 1')
    plt.grid(True)
    plt.colorbar(scatter, label='Embedding Dim 1')
    plt.savefig('temp_Figures/state_space_colored.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return embeddings

# Main execution
if __name__ == "__main__":
    print("Simplified FitzHugh-Nagumo MARBLE Demo")
    print("=" * 40)
    
    # Setup
    n_trajectories = 100  # Number of trajectories to simulate
    t_span = np.linspace(0, 100, 1000)  # Time points
    
    # Parameter to vary (set to None for fixed parameters)
    vary_param = 'b'
    param_range = [0.5, 1.5]
    
    print(f"Generating {n_trajectories} trajectories...")
    if vary_param:
        print(f"Varying parameter: {vary_param} in range {param_range}")
    
    # Generate initial conditions
    X0_list = initial_conditions(n_trajectories, 1)
    
    # Simulate FHN model
    pos_list, vel_list, param_list = simulate_fhn(
        FHN_PARAMS, X0_list, t_span, 
        vary_param=vary_param, 
        param_range=param_range
    )
    
    # Visualize sample trajectories
    plt.figure(figsize=(8, 6))
    for i in range(min(10, n_trajectories)):
        plt.plot(pos_list[i][:, 0], pos_list[i][:, 1], alpha=0.7)
    plt.title('Sample FHN Trajectories')
    plt.xlabel('Voltage (x)')
    plt.ylabel('Recovery (y)')
    plt.grid(True)
    plt.savefig('temp_Figures/fhn_sample_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare dataset and train model
    data = prepare_marble_dataset(pos_list, vel_list, param_list)
    model = train_marble_model(data)
    
    # Visualize results
    embeddings = visualize_embeddings(model, data)
    
    print("=" * 40)
    print("Demo completed. Results saved to temp_Figures/ directory.") 