import os
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch 


# Plot dx distribution
def plt_dx_distribution(num_frames_to_process, traj1, traj2):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # dx trajectory
    dx_trajectory1 = np.zeros([num_frames_to_process - 1, traj1[0].dx.shape[0], traj1[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory1[i] = traj1[i].dx

    dx_trajectory2 = np.zeros([num_frames_to_process - 1, traj2[0].dx.shape[0], traj2[0].dx.shape[1]])
    for i in range(num_frames_to_process - 1):
        dx_trajectory2[i] = traj2[i].dx

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create 3D scatter plots for the dx trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(traj1[0].dx.shape[0]):
        x1 = dx_trajectory1[:, i, 0]
        y1 = dx_trajectory1[:, i, 1]
        z1 = dx_trajectory1[:, i, 2]
        ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', s=10, marker='o', label='Atom {}'.format(i + 1))
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D dx Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(traj2[0].dx.shape[0]):
        x2 = dx_trajectory2[:, i, 0]
        y2 = dx_trajectory2[:, i, 1]
        z2 = dx_trajectory2[:, i, 2]
        ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', s=10, marker='^', label='Atom {}'.format(i + 1))
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D dx Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add colorbars to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()

# Plot point distribution
def plt_point_distribution(num_frames_to_process, traj1, traj2, idx=0):
    # Generate coordinates over time
    time = np.arange(0, num_frames_to_process - 1)

    # point trajectory
    x1 = np.zeros(num_frames_to_process - 1)
    y1 = np.zeros(num_frames_to_process - 1)
    z1 = np.zeros(num_frames_to_process - 1)

    x2 = np.zeros(num_frames_to_process - 1)
    y2 = np.zeros(num_frames_to_process - 1)
    z2 = np.zeros(num_frames_to_process - 1)

    for i in range(num_frames_to_process - 1):
        x1[i] = traj1[i].dx[idx][0]
        y1[i] = traj1[i].dx[idx][1]
        z1[i] = traj1[i].dx[idx][2]
        x2[i] = traj2[i].dx[idx][0]
        y2[i] = traj2[i].dx[idx][1]
        z2[i] = traj2[i].dx[idx][2]

    # Create a colormap based on time index
    colors = cm.viridis(np.linspace(0, 1, len(time)))

    # Create a 3D scatter plot for the point trajectory with a colormap
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x1, y1, z1, c=colors, cmap='viridis', label='Trajectory')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('3D Point Trajectory with Time Index - Scale=1.0')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x2, y2, z2, c=colors, cmap='viridis', label='Trajectory')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title('3D Point Trajectory with Time Index - Scale=2.0')
    ax2.legend()

    # Add a colorbar to show the time index
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax1)
    cbar.set_label('Time Index')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
    cbar.set_label('Time Index')

    # Show the plot
    plt.show()