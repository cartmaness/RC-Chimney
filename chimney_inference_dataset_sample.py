import torch
import torch.nn as nn
import numpy as np
import sys
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from chimney_data_preprocessing import *
from utils.functions import *
from utils.evaluation_criteria import *
import chimney_transformer

# Create module alias so the checkpoint can find the old module name
sys.modules['Chimeny_GPT'] = chimney_transformer

# --- Configuration ---
DATA_FOLDER = 'Chimney_Data/ProcessedData_Reduced/'
RESPONSE_TYPE = "acceleration"  # or "displacement" or "both"
PARAMETERS_FEATURES = 14        # Number of structural parameters per chimney
RESPONSE_FEATURES = 5           # Predict response at 5 height levels
NUM_FREQUENCIES = 5             # Natural frequencies
SEQUENCE_LENGTH = 4500          # Length of ground motion time series Dataset has maximum of 4500
# BATCH_SIZE = 1                # Loading only one sample for inference
SCALER_FOLDER = Path(DATA_FOLDER) / "Scaler"

# Paths to the saved scalers (used during training to normalize data)
CHIMNEY_SCALER_FILE = SCALER_FOLDER / "param_scaler.joblib"
FREQ_SCALER_FILE = SCALER_FOLDER / "freq_scaler.joblib"
MOTION_SCALER_FILE = SCALER_FOLDER / "motion_scaler.joblib"
ACC_SCALER_FILE = SCALER_FOLDER / "acceleration_scaler.joblib"
DISP_SCALER_FILE = SCALER_FOLDER / "displacement_scaler.joblib"


def dataset_sample_inference(sample_idx=0, target_key='displacement'):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f" RUNNING INFERENCE ON A REAL DATASET SAMPLE ")
    print(f"{'=' * 60}")
    print(f"PyTorch version: {torch.__version__} --- Using device: {device}")
    print(f"Loading sample index: {sample_idx}")

    # --- Load the trained model ---
    if target_key == "acceleration":
        model_path = f'checkpoints/{target_key}/Chimney_Transformer_Adaptive_acceleration_full_0.9807_256_3.pth'
    elif target_key == "displacement":
        model_path = f'checkpoints/{target_key}/Chimney_Transformer_Adaptive_displacement_full_0.9933_256_3.pth'
    else:
        raise ValueError(f"Invalid target_key: {target_key}")

    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval() # Set to evaluation mode (no dropout, batch norm frozen, etc.)

    # --- Load the full test dataset and pick one random sample ---
    print(f"\nLoading all processed data from {DATA_FOLDER}...")
    param_file = os.path.join(DATA_FOLDER, 'Parameters.parquet')
    params_df = pd.read_parquet(param_file)
    total_samples = len(params_df)
    print(f"Total samples in dataset: {total_samples}")

    # Load all data
    data, normalizer = load_data(
        total_samples,  # Load all data
        PARAMETERS_FEATURES,
        SEQUENCE_LENGTH,
        RESPONSE_FEATURES,
        DATA_FOLDER,
        response_type=target_key,
        normalize_data=False,  # Keep it false
        verbose=True
    )

    # Convert to PyTorch tensors and move to GPU if available
    data_tensors = {key: torch.tensor(value, dtype=torch.float32).to(device)
                    for key, value in data.items()}

    dataset = ChimneySequenceDataset(data_tensors, verbose=True)

    # Verify sample index is valid
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} out of range. Dataset has {len(dataset)} samples.")

    # Get single sample
    print(f"\nExtracting sample {sample_idx} from dataset...")
    sample = dataset[sample_idx]

    # Unpack sample data
    chimney_params_tensor = sample['chimney_params'].unsqueeze(0).to(device, non_blocking=True)
    frequencies_tensor = sample['frequencies'].unsqueeze(0).to(device, non_blocking=True)
    ground_motion_tensor = sample['ground_motion'].unsqueeze(0).to(device, non_blocking=True)
    target_response_tensor = sample[target_key].unsqueeze(0).to(device, non_blocking=True)

    print(f"\nTensor shapes:")
    print(f"  Chimney params: {chimney_params_tensor.shape}")
    print(f"  Frequencies:    {frequencies_tensor.shape}")
    print(f"  Ground motion:  {ground_motion_tensor.shape}")
    print(f"  True response:  {target_response_tensor.shape}")

    # --- Inference ---
    print("\nRunning inference...")
    with torch.no_grad():
        pred_response = model(chimney_params_tensor, frequencies_tensor, ground_motion_tensor)

        # Compute performance metrics (masked to ignore zero-padding)
        loss = compute_masked_loss(pred_response, target_response_tensor, ground_motion_tensor)
        mae = compute_masked_mae(pred_response, target_response_tensor, ground_motion_tensor)
        r = compute_masked_pearson_r(pred_response, target_response_tensor, ground_motion_tensor)

    # === Denormalize ===
    SCALER_MAP = {
        'acceleration': ACC_SCALER_FILE,
        'displacement': DISP_SCALER_FILE,
    }
    chimney_params = chimney_params_tensor[0].cpu().numpy().reshape(1, -1)
    chimney_params_denorm = denormalize(chimney_params, scaler_filename=CHIMNEY_SCALER_FILE, sequence=False).flatten()
    ground_motion_denorm = denormalize(ground_motion_tensor[0].cpu().numpy(), scaler_filename=MOTION_SCALER_FILE, sequence=True)
    pred_response_denorm = denormalize(pred_response[0].cpu().numpy(),scaler_filename=SCALER_MAP[target_key], sequence=True)
    true_response_denorm = denormalize(target_response_tensor[0].cpu().numpy(), scaler_filename=SCALER_MAP[target_key], sequence=True)

    # Print sample information
    print(f"\n{'=' * 60}")
    print(f"SAMPLE INFORMATION")
    print(f"{'=' * 60}")
    print(f"Sample Index: {sample_idx}")
    print(f"Chimney Parameters (denormalized):")
    param_names = ['h', 'd0', 'd1', 't0', 't1', 'rou_v', 'fc', 'fy', 'taper_rate', 'taper_angle', 'slenderness', 'element_length', 'mass', 'volume']
    for i, name in enumerate(param_names):
        if i < len(chimney_params_denorm):
            print(f"  {name}: {chimney_params_denorm[i]:.4f}")

    # --- Visualize the results ---
    desc = f"Sample_{sample_idx}"
    loss_val, mae_val, r_val = plot_generated_sample(
        ground_motion=ground_motion_denorm,
        pred_response=pred_response_denorm,
        true_response=true_response_denorm,
        target_key=target_key,
        desc=desc,
        save_dir='figures_dataset',
        loss=loss.item() if loss is not None else None,
        mae=mae.item() if mae is not None else None,
        r=r.item() if r is not None else None,
        dt=0.02
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"INFERENCE RESULTS")
    print(f"{'=' * 60}")
    if loss is not None:
        print(f"Loss:      {loss.item():.6f}")
        print(f"MAE:       {mae.item():.6f}")
        print(f"Pearson R: {r.item():.6f}")
    else:
        print("Inference completed (no ground truth provided).")
    print(f"{'=' * 60}\n")

    return model, chimney_params_denorm, pred_response_denorm, true_response_denorm


def plot_generated_sample(ground_motion, pred_response, true_response, desc,
                          target_key='displacement', save_dir='figures',
                          loss=None, mae=None, r=None, dt=0.02):
    """Plot inference results for a single sample."""
    if loss is not None:
        print(f"\n📊 Model performance for this sample:")
        print(f"Loss: {loss:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Pearson R: {r:.6f}")

    # --- Crop trailing zeros ---
    def crop_zeros(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if arr.ndim == 1:
            non_zero = np.where(arr != 0)[0]
            return arr[:non_zero[-1] + 1] if len(non_zero) > 0 else arr
        else:
            non_zero = np.where(np.any(arr != 0, axis=1))[0]
            return arr[:non_zero[-1] + 1] if len(non_zero) > 0 else arr

    ground_motion = crop_zeros(ground_motion)
    pred_response = crop_zeros(pred_response)
    true_response = crop_zeros(true_response)

    # Get minimum length and scale data
    min_length = min(len(ground_motion), len(pred_response), len(true_response))

    ground_motion = ground_motion[:min_length] / 9806.65  # mm/s² to g
    if target_key == 'acceleration':
        pred_response = pred_response[:min_length] / 1000.0  # mm/s² → m/s²
        true_response = true_response[:min_length] / 1000.0  # mm/s² → m/s²
    else:  # displacement
        pred_response = pred_response[:min_length]  # mm
        true_response = true_response[:min_length]  # mm

    time = np.arange(min_length) * dt

    # === Plotting ===
    plt.rcParams['font.family'] = 'Times New Roman'
    n_features = pred_response.shape[1] if pred_response.ndim > 1 else 1
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(7, 1.5 * (n_features + 1)),
                             sharex=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Calculate y-limits for response plots
    y_min = min(np.min(true_response), np.min(pred_response))
    y_max = max(np.max(true_response), np.max(pred_response))
    y_margin = (y_max - y_min) * 0.05
    y_limits = [y_min - y_margin, y_max + y_margin]

    # Plot ground motion
    axes[0].plot(time, ground_motion, label=f'Ground Motion Acceleration ({desc})',
                 color='black', linewidth=1)
    axes[0].set_ylabel('Ground Acceleration (g)', fontsize=9)
    axes[0].legend(loc='upper right', fontsize=9, frameon=True)
    axes[0].grid(True)

    # Plot response channels
    colors = ['#2B6B9E', '#D83132']  # Blue for true, red for predicted
    units = {'acceleration': 'm/s$^2', 'displacement': 'mm'}
    label_name = target_key.capitalize()
    full_label = f'{label_name} ({units.get(target_key, "Unit")})'

    for channel in range(n_features):
        ax = axes[channel + 1]
        ax.plot(time, true_response[:, channel],
                label=f'True {label_name} (Height {(channel + 1) * 20}% )',
                color=colors[0], linestyle='-', linewidth=2)
        ax.plot(time, pred_response[:, channel],
                label=f'Predicted {label_name} (Height {(channel + 1) * 20}% )',
                color=colors[1], linestyle='-', linewidth=1)
        ax.set_ylabel(full_label, fontsize=9)
        ax.legend(loc='upper right', fontsize=9, frameon=True)
        ax.grid(True)
        ax.set_ylim(y_limits)

        # Add metrics text
        r_h = np.corrcoef(true_response[:, channel], pred_response[:, channel])[0, 1]
        mse_h = np.mean((true_response[:, channel] - pred_response[:, channel]) ** 2)
        mae_h = np.mean(np.abs(true_response[:, channel] - pred_response[:, channel]))
        metrics_text = f'R={r_h:.4f}'
        ax.text(0.990, 0.040, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlim([0, time[-1]])
    axes[-1].set_xlabel('Time (s)', fontsize=10)

    # === Save to SVG ===
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'infer_response_{target_key}_{desc}_{timestamp}.svg')
    plt.savefig(save_path, format='svg', dpi=300)
    plt.show()

    return (loss if loss is not None else None,
            mae if mae is not None else None,
            r if r is not None else None)


if __name__ == "__main__":
    # Run inference on a specific sample from the dataset
    # Change sample_idx to load different samples (0-indexed)
    dataset_sample_inference(sample_idx=100, target_key=RESPONSE_TYPE)