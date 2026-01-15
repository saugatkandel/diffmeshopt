import numpy as np
import matplotlib.pyplot as plt
import torch
from src.loss_2d import BiGaussianLoss
from src.optimize_2d import compute_normals

def plot_prior_and_landscape(x, y_signal, peak_dist=6.0, sigma=1.0):
    """
    Visualizes the BiGaussian prior and the resulting loss landscape.
    This addresses 'Objective 1: Validate Loss Landscape' from plan_2d.md.
    
    Args:
        x (np.array): 1D coordinates along the normal vector.
        y_signal (np.array): The observed 1D intensity profile.
        peak_dist (float): Distance between peaks.
        sigma (float): Width of peaks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Visualize the Template (Prior) ---
    x_tensor = torch.from_numpy(x).float()
    y_template = BiGaussianLoss.get_bigaussian_profile(x_tensor, peak_dist, sigma).numpy()

    # For visualization, normalize the observed signal to compare shapes
    y_signal_plot = (y_signal - np.mean(y_signal)) / (np.std(y_signal) + 1e-8)
    
    ax1.plot(x, y_template, label='Template $T$ (BiGaussian)', color='blue', linewidth=2)
    ax1.plot(x, y_signal_plot, label='Observed Signal', color='orange', alpha=0.7, linestyle='--')
    ax1.fill_between(x, y_template, color='blue', alpha=0.1)
    ax1.set_title("1D BiGaussian Prior vs Signal")
    ax1.set_xlabel("Distance along Normal ($k$)")
    ax1.set_ylabel("Intensity")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate the structure
    ax1.annotate('Peak 1', xy=(-peak_dist/2, 1.0), xytext=(-10, 1.2), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.annotate('Peak 2', xy=(peak_dist/2, 1.0), xytext=(5, 1.2), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.legend(loc='upper right')

    # --- 2. Visualize the Loss Landscape ---
    # Calculate Cross-Correlation for various shifts
    shifts = np.linspace(-8, 8, 100)
    correlations = []
    
    # Normalize template for correlation calculation
    T_norm = (y_template - np.mean(y_template)) / (np.std(y_template) + 1e-6)

    for s in shifts:
        # Shift the signal by -s (equivalent to sampling at x+s)
        # We use interpolation to simulate sampling at non-integer coordinates
        y_sampled = np.interp(x + s, x, y_signal)
        S_norm = (y_sampled - np.mean(y_sampled)) / (np.std(y_sampled) + 1e-6)
        
        # Cross-correlation
        corr = np.mean(S_norm * T_norm)
        correlations.append(corr)

    ax2.plot(shifts, correlations, color='red', label='Cross-Correlation')
    ax2.axvline(0, color='green', linestyle='--', label='Ideal Shift (0)')
    ax2.set_title("Loss Landscape (Objective 1)")
    ax2.set_xlabel("Shift Parameter $s$")
    ax2.set_ylabel("Correlation Score")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_profile_statistics(profiles, x=None, title="Profile Statistics", ax=None, template=None):
    """
    Visualizes the mean and spread of a batch of profiles.
    
    Args:
        profiles (np.array or torch.Tensor): Shape (N, L) where N is batch size, L is profile length.
        x (np.array): Optional x-axis coordinates.
        title (str): Plot title.
        ax (matplotlib.axes.Axes): Optional axes to plot on. If None, creates a new figure.
        template (np.array or torch.Tensor): Optional template profile to overlay.
    """
    if isinstance(profiles, torch.Tensor):
        profiles = profiles.detach().cpu().numpy()
        
    mean_profile = np.mean(profiles, axis=0)
    std_profile = np.std(profiles, axis=0)
    
    if x is None:
        x = np.arange(profiles.shape[1])
        
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        show_plot = True
        
    ax.plot(x, mean_profile, label='Mean Profile', color='blue', linewidth=2)
    ax.fill_between(x, mean_profile - std_profile, mean_profile + std_profile, 
                     color='blue', alpha=0.2, label='Standard Deviation')
    
    if template is not None:
        if isinstance(template, torch.Tensor):
            template = template.detach().cpu().numpy()
        ax.plot(x, template, label='Template', color='red', linestyle='--', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("Sample Index / Distance")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if show_plot:
        plt.show()

def plot_contour_normals(image, contour, profile_len=21, num_lines=50, ax=None):
    """
    Visualizes the contour and its normals on top of the image.
    Useful for verifying normal calculation and sampling direction.
    
    Args:
        image (np.array): 2D image array.
        contour (np.array): (N, 2) array of (row, col) coordinates.
        profile_len (int): Length of the profile line to visualize centered at vertex.
        num_lines (int): Number of normal lines to plot.
        ax (matplotlib.axes.Axes): Optional axes.
    """
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
        
    ax.imshow(image, cmap='gray')
    ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1, label='Contour')
    
    # Calculate normals
    contour_tensor = torch.from_numpy(contour).float()
    normals = compute_normals(contour_tensor).numpy()
    
    N_points = len(contour)
    # Ensure we don't try to plot more lines than points
    num_lines = min(num_lines, N_points)
    if num_lines > 0:
        indices = np.linspace(0, N_points - 1, num_lines, dtype=int)
        
        for i in indices:
            nr, nc = normals[i]
            
            # Center point
            r0, c0 = contour[i]
            
            # Define line segment for profile
            # From -profile_len/2 to +profile_len/2 along normal
            half_len = (profile_len - 1) / 2.0
            
            r_start = r0 - nr * half_len
            c_start = c0 - nc * half_len
            r_end = r0 + nr * half_len
            c_end = c0 + nc * half_len
            
            # Plot line (x=col, y=row)
            ax.plot([c_start, c_end], [r_start, r_end], 'y-', alpha=0.8, linewidth=1)
        
    ax.set_title(f"Contour Normals (showing {num_lines} of {N_points})")
    ax.legend()
    
    if show_plot:
        plt.show()
