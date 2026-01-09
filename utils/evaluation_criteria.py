import torch
import gc
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def mae_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    return torch.mean(torch.abs(y_true - y_pred))


def mse_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    return torch.mean((y_true - y_pred) ** 2)


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    y_true, y_pred = y_true.float().to(device), y_pred.float().to(device)
    y_true_mean = torch.mean(y_true, dim=0)
    ss_total = torch.sum((y_true - y_true_mean) ** 2, dim=0)
    ss_residual = torch.sum((y_true - y_pred) ** 2, dim=0)
    r2 = 1 - (ss_residual / (ss_total + 1e-8))
    return torch.mean(r2)


def mape_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    epsilon = 1e-10
    denominator = torch.clamp(denominator, min=epsilon)
    return 100 * torch.mean(numerator / denominator)


def r_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    y_true = y_true.float()
    y_pred = y_pred.float()

    if y_true.dim() == 1:
        # Single-output case
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
        denominator = torch.sqrt(torch.sum((y_true - y_true_mean) ** 2) * torch.sum((y_pred - y_pred_mean) ** 2))
        r = numerator / (denominator + 1e-8)
    else:
        # Multi-output case
        y_true_mean = torch.mean(y_true, dim=0)
        y_pred_mean = torch.mean(y_pred, dim=0)
        numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=0)
        denominator = torch.sqrt(torch.sum((y_true - y_true_mean) ** 2, dim=0) * torch.sum((y_pred - y_pred_mean) ** 2, dim=0))
        r = numerator / (denominator + 1e-8)
        r = torch.mean(r)  # Average across features

    return r


def masked_mse_loss(pred_response, target_response, ground_motion):
    """
    Args:
        pred_response: [B, T, F]
        target_response: [B, T, F]
        ground_motion: [B, T] (used to detect padding)
    Returns:
        Scalar masked MSE loss
    """
    # Build mask: True where ground motion is not zero
    mask = (ground_motion != 0).unsqueeze(-1)  # [B, T, 1]
    mask = mask.expand_as(pred_response)  # [B, T, F]

    # Select only valid elements
    pred_valid = pred_response[mask]
    target_valid = target_response[mask]

    # Compute MSE over valid elements
    return F.mse_loss(pred_valid, target_valid)


def compute_masked_loss(pred_response, target_response, ground_motion, criterion=F.mse_loss, eps=1e-8):
    """
    Compute masked loss (e.g., MSE) with device and dtype consistency.

    Args:
        pred_response (torch.Tensor): Predicted values, shape [B, T, F].
        target_response (torch.Tensor): Target values, shape [B, T, F].
        ground_motion (torch.Tensor): Mask, shape [B, T], non-zero for valid timesteps.
        criterion: Loss function (default: F.mse_loss).
        eps (float): Threshold to determine valid timesteps.

    Returns:
        torch.Tensor: Masked loss as a single-element tensor with requires_grad=True.
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)  # [B]
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)  # [B, T, F]

    pred_valid = pred_response[mask]
    target_valid = target_response[mask]

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    return criterion(pred_valid, target_valid)


def compute_masked_mae(pred_response, target_response, ground_motion, eps=1e-8):
    """
    Compute masked Mean Absolute Error (MAE) with device and dtype consistency.

    Args:
        pred_response (torch.Tensor): Predicted values, shape [B, T, F].
        target_response (torch.Tensor): Target values, shape [B, T, F].
        ground_motion (torch.Tensor): Mask, shape [B, T], non-zero for valid timesteps.

    Returns:
        torch.Tensor: Masked MAE as a single-element tensor.
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)

    pred_valid = pred_response[mask]
    target_valid = target_response[mask]

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.mean(torch.abs(pred_valid - target_valid))

# compute_masked_pearson_r2
def compute_masked_pearson_r2(pred_response, target_response, ground_motion, eps=1e-8):
    """
    Compute masked Pearson R-squared coefficient.

    Args:
        pred_response (torch.Tensor): Predicted values, shape [B, T, F].
        target_response (torch.Tensor): Target values, shape [B, T, F].
        ground_motion (torch.Tensor): Mask, shape [B, T].

    Returns:
        torch.Tensor: Masked Pearson R-squared as a single-element tensor (not differentiable).
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)

    pred_valid = pred_response[mask]
    target_valid = target_response[mask]

    if pred_valid.numel() < 2:
        return torch.tensor(0.0, device=device, dtype=dtype)

    pred_mean = torch.mean(pred_valid)
    target_mean = torch.mean(target_valid)

    cov = torch.mean((pred_valid - pred_mean) * (target_valid - target_mean))
    pred_std = torch.std(pred_valid)
    target_std = torch.std(target_valid)

    if pred_std.item() == 0.0 or target_std.item() == 0.0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    pearson_r = cov / (pred_std * target_std)
    pearson_r = torch.clamp(pearson_r, -1.0, 1.0)
    pearson_r2 = pearson_r ** 2
    return pearson_r2


def compute_masked_pearson_r(pred_response, target_response, ground_motion, eps=1e-8):
    """
    Compute masked Pearson correlation coefficient.

    Args:
        pred_response (torch.Tensor): Predicted values, shape [B, T, F].
        target_response (torch.Tensor): Target values, shape [B, T, F].
        ground_motion (torch.Tensor): Mask, shape [B, T].

    Returns:
        torch.Tensor: Masked Pearson R as a single-element tensor (not differentiable).
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)

    pred_valid = pred_response[mask]
    target_valid = target_response[mask]

    if pred_valid.numel() < 2:
        return torch.tensor(0.0, device=device, dtype=dtype)

    pred_mean = torch.mean(pred_valid)
    target_mean = torch.mean(target_valid)

    cov = torch.mean((pred_valid - pred_mean) * (target_valid - target_mean))
    pred_std = torch.std(pred_valid)
    target_std = torch.std(target_valid)

    if pred_std.item() == 0.0 or target_std.item() == 0.0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    pearson_r = cov / (pred_std * target_std)
    return torch.clamp(pearson_r, -1.0, 1.0)


def compute_peak_response_error(pred_response, target_response, ground_motion, eps=1e-6):
    """
    Compute peak response error between predicted and true responses in percentage
    Uses the same masking logic as compute_masked_loss and compute_masked_mae
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)  # [B]
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)  # [B, T, F]

    peak_errors = []
    for batch_idx in range(B):
        if lengths[batch_idx] > 0:
            # Get valid data for this batch
            batch_mask = mask[batch_idx]  # [T, F]
            pred_valid = pred_response[batch_idx][batch_mask].view(-1, F)  # [valid_timesteps, F]
            true_valid = target_response[batch_idx][batch_mask].view(-1, F)  # [valid_timesteps, F]

            if pred_valid.numel() > 0:
                # Calculate peak values for each channel
                pred_peaks = torch.max(torch.abs(pred_valid), dim=0)[0]  # [F]
                true_peaks = torch.max(torch.abs(true_valid), dim=0)[0]  # [F]

                # Calculate relative error in percentage
                peak_error = (torch.abs(pred_peaks - true_peaks) / (torch.abs(true_peaks) + eps)) * 100.0
                peak_errors.append(peak_error.mean())

    if not peak_errors:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.stack(peak_errors).mean()


def compute_phase_error(pred_response, target_response, ground_motion, dt=0.02, eps=1e-6, plot_sample=False, batch_to_plot=1, channel_to_plot=1):
    """
    Compute average phase error between predicted and true responses
    Uses the same masking logic as compute_masked_loss and compute_masked_mae
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)  # [B]
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)  # [B, T, F]

    phase_errors = []
    for batch_idx in range(B):
        if lengths[batch_idx] > 0:
            # Get valid data for this batch
            batch_mask = mask[batch_idx]  # [T, F]
            pred_valid = pred_response[batch_idx][batch_mask].view(-1, F).detach().cpu().numpy() # [valid_timesteps, F]
            true_valid = target_response[batch_idx][batch_mask].view(-1, F).detach().cpu().numpy()   # [valid_timesteps, F]

            if pred_valid.size > 0:
                channel_phase_errors = []
                for channel in range(F):
                    pred_channel = pred_valid[:, channel]
                    true_channel = true_valid[:, channel]

                    if len(pred_channel) > 1:  # Need at least 2 points for FFT
                        # Compute FFT for both signals
                        pred_fft = fft(pred_channel)
                        true_fft = fft(true_channel)

                        # Compute phase
                        pred_phase = np.angle(pred_fft)
                        true_phase = np.angle(true_fft)

                        # Calculate phase difference (wrapped to [-π, π])
                        phase_diff = np.angle(np.exp(1j * (pred_phase - true_phase)))

                        # Average phase error (weighted by magnitude)
                        freqs = fftfreq(len(pred_phase), dt)
                        weights = np.abs(true_fft) / (np.sum(np.abs(true_fft)) + eps)

                        # Focus on meaningful frequencies (avoid DC and very high freq)
                        valid_freq_mask = (np.abs(freqs) > 0.2) & (np.abs(freqs) < 15)
                        if np.any(valid_freq_mask):
                            weighted_phase_error = np.sum(np.abs(phase_diff[valid_freq_mask]) *
                                                          weights[valid_freq_mask])
                            channel_phase_errors.append(weighted_phase_error)
                            # --- PLOTTING LOGIC ---
                            if plot_sample and batch_idx == batch_to_plot and channel == channel_to_plot:
                                fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                                # Plot Original Signals (Time Domain)
                                time_axis = np.arange(len(pred_channel)) * dt
                                axs[0].plot(time_axis, pred_channel, label='Predicted Response')
                                axs[0].plot(time_axis, true_channel, label='True Response', linestyle='--')
                                axs[0].set_title(f'Time Domain Signal (Batch {batch_idx}, Channel {channel})')
                                axs[0].set_xlabel('Time (s)')
                                axs[0].set_ylabel('Amplitude')
                                axs[0].legend()
                                axs[0].grid(True)

                                # Plot Phase in Frequency Domain
                                # Plotting phase requires careful consideration of frequency range
                                # Let's plot only valid frequencies for clarity
                                plot_freqs = freqs[valid_freq_mask]
                                plot_pred_phase = pred_phase[valid_freq_mask]
                                plot_true_phase = true_phase[valid_freq_mask]
                                plot_phase_diff = phase_diff[valid_freq_mask]

                                axs[1].plot(plot_freqs, plot_true_phase, 'o-', label='True Phase')
                                axs[1].plot(plot_freqs, plot_pred_phase, 'x-', label='Predicted Phase')
                                # axs[1].plot(plot_freqs, plot_phase_diff, '.-', label='Phase Difference (pred-true)') # Optionally plot difference
                                axs[1].set_title(f'Phase in Frequency Domain (Batch {batch_idx}, Channel {channel})')
                                axs[1].set_xlabel('Frequency (Hz)')
                                axs[1].set_ylabel('Phase (radians)')
                                axs[1].legend()
                                axs[1].grid(True)
                                axs[1].set_ylim([-np.pi, np.pi]) # Phase is wrapped to -pi to pi

                                plt.tight_layout()
                                plt.show()
                        else:
                            # Handle case where no valid frequencies found for plotting
                            if plot_sample and batch_idx == batch_to_plot and channel == channel_to_plot:
                                print(f"Warning: No valid frequencies found for plotting for Batch {batch_idx}, Channel {channel}")

                if channel_phase_errors:
                    phase_errors.append(np.mean(channel_phase_errors))

    if not phase_errors:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.tensor(np.mean(phase_errors), device=device, dtype=dtype)


def compute_amplitude_ratio(pred_response, target_response, ground_motion, eps=1e-8):
    """
    Compute amplitude ratio between predicted and true responses
    Uses the same masking logic as compute_masked_loss and compute_masked_mae
    """
    device = ground_motion.device
    dtype = pred_response.dtype

    pred_response = pred_response.to(device=device, dtype=dtype)
    target_response = target_response.to(device=device, dtype=dtype)

    lengths = (ground_motion.abs() > eps).sum(dim=1).clamp(min=1)  # [B]
    B, T, F = pred_response.shape

    time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    mask = (time_indices < lengths.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, F)  # [B, T, F]

    amplitude_ratios = []
    for batch_idx in range(B):
        if lengths[batch_idx] > 0:
            # Get valid data for this batch
            batch_mask = mask[batch_idx]  # [T, F]
            pred_valid = pred_response[batch_idx][batch_mask].view(-1, F)  # [valid_timesteps, F]
            true_valid = target_response[batch_idx][batch_mask].view(-1, F)  # [valid_timesteps, F]

            if pred_valid.numel() > 0:
                # Calculate RMS amplitude for each channel
                pred_rms = torch.sqrt(torch.mean(pred_valid ** 2, dim=0))  # [F]
                true_rms = torch.sqrt(torch.mean(true_valid ** 2, dim=0))  # [F]

                # Calculate amplitude ratio
                ratio = pred_rms / (true_rms + eps)
                amplitude_ratios.append(ratio.mean())

    if not amplitude_ratios:
        return torch.tensor(1.0, device=device, dtype=dtype)

    return torch.stack(amplitude_ratios).mean()
