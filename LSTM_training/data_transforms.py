"""
Minimal data transforms for LSTM training: EMA smoothing and LiDAR augmentation.

Usage:
  - EMA: smooth_steering_targets(targets, alpha=0.2)
  - Augmentation: LiDARNoise (PyTorch transform for DataLoader)
"""

import numpy as np
import torch
import torch.nn as nn


def smooth_steering_targets(expert_actions, alpha=0.2, episode_ids=None):
    """
    Apply exponential moving average (EMA) to steering targets only.
    Leaves speed targets unchanged. Causal (no lookahead).
    
    Args:
        expert_actions: (T, 2) array [steering, speed]
        alpha: EMA smoothing factor (0 = no smoothing, 1 = current only)
        episode_ids: (T,) optional array to prevent cross-episode smoothing
    
    Returns:
        smoothed_actions: (T, 2) with steering smoothed, speed unchanged
    """
    smoothed = expert_actions.copy()
    steering = expert_actions[:, 0].copy()
    
    if episode_ids is None:
        # Single episode: causal EMA
        smoothed_steering = np.zeros_like(steering)
        smoothed_steering[0] = steering[0]
        for t in range(1, len(steering)):
            smoothed_steering[t] = alpha * steering[t] + (1 - alpha) * smoothed_steering[t - 1]
        smoothed[:, 0] = smoothed_steering
    else:
        # Multi-episode: reset at boundaries
        smoothed_steering = np.zeros_like(steering)
        unique_eps = np.unique(episode_ids)
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_indices = np.where(mask)[0]
            ep_steering = steering[mask]
            
            ep_smoothed = np.zeros_like(ep_steering)
            ep_smoothed[0] = ep_steering[0]
            for i in range(1, len(ep_steering)):
                ep_smoothed[i] = alpha * ep_steering[i] + (1 - alpha) * ep_smoothed[i - 1]
            
            smoothed_steering[ep_indices] = ep_smoothed
        
        smoothed[:, 0] = smoothed_steering
    
    return smoothed


class LiDARNoise(nn.Module):
    """
    Training-only LiDAR augmentation: Gaussian range noise + random beam dropout.
    Assumes LiDAR occupancy is in the last 60 features of input.
    
    Args:
        gaussian_sigma_m: Std dev of range noise in meters (default 0.1)
        dropout_prob: Probability of zeroing each beam (default 0.05)
    """
    
    def __init__(self, gaussian_sigma_m=0.1, dropout_prob=0.05):
        super().__init__()
        self.gaussian_sigma_m = gaussian_sigma_m
        self.dropout_prob = dropout_prob
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 63) where last 60 are LiDAR bins
        
        Returns:
            x_aug: (batch, seq_len, 63) with augmented LiDAR
        """
        x_aug = x.clone()
        
        # Extract LiDAR part (last 60 features)
        lidar_aug = x_aug[:, :, 3:]  # Shape: (batch, seq_len, 60)
        
        # Gaussian range noise (clipped to [0, 1] after normalization)
        if self.gaussian_sigma_m > 0:
            noise = torch.randn_like(lidar_aug) * self.gaussian_sigma_m
            lidar_aug = lidar_aug + noise
            lidar_aug = torch.clamp(lidar_aug, 0, 1)
        
        # Beam dropout: randomly zero out beams
        if self.dropout_prob > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(lidar_aug, 1 - self.dropout_prob)
            )
            lidar_aug = lidar_aug * dropout_mask
        
        x_aug[:, :, 3:] = lidar_aug
        return x_aug
    
    def __repr__(self):
        return (
            f"LiDARNoise(gaussian_sigma_m={self.gaussian_sigma_m}, "
            f"dropout_prob={self.dropout_prob})"
        )
