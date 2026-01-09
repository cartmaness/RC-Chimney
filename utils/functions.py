import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import torch

# ------------------------------------------------------------------------
# Unit System: Base units in { mm, N, MPa, sec }
# ------------------------------------------------------------------------
# Base Units
mm = 1.0  # 1 millimeter
N = 1.0  # 1 Newton
sec = 1.0  # 1 second

m = 1000.0 * mm  # 1 meter is 1000 millimeters
cm = 10.0 * mm  # 1 centimeter is 10 millimeters
kN = 1000.0 * N  # 1 kilo-Newton is 1000 Newtons
m2 = m * m  # Square meter
cm2 = cm * cm  # Square centimeter
mm2 = mm * mm  # Square millimeter
MPa = N / mm2  # MegaPascal (Pressure)
kPa = 0.001 * MPa  # KiloPascal (Pressure)
GPa = 1000 * MPa  # GigaPascal (Pressure)
G_TO_MM_S2 = 9810.0

def save_model(model, response_type, save_mode='full', additional_info=None):
    """
    Save model in full or state_dict mode with metadata, including accuracy in file name if present in additional_info.

    Args:
        model: PyTorch model to save
        response_type (str): Type of response for organizing save path
        save_mode (str): 'full' to save entire model, 'state_dict' to save only weights
        additional_info (dict, optional): Additional metadata to save, may include accuracy
    Returns:
        str or None: Path to saved model or None if saving fails
    """
    model_name = type(model).__name__
    mode_suffix = 'full' if save_mode == 'full' else 'state_dict'

    # Check for 'R' key in additional_info for accuracy
    accuracy = None
    if additional_info and 'R' in additional_info:
        if isinstance(additional_info['R'], (int, float)):
            accuracy = additional_info['R']
        elif isinstance(additional_info['R'], str):
            try:
                accuracy = float(additional_info['R'])
            except ValueError:
                print(f"Warning: 'R' value '{additional_info['R']}' is not a valid number, skipping accuracy in file name")

    # Format accuracy for filename if found
    accuracy_str = ''
    if accuracy is not None:
        # Format accuracy to 4 decimal places, removing decimal point for filename compatibility
        accuracy_str = f'_acc{int(accuracy * 10000):05d}'

    # Construct model path with accuracy in filename if available
    model_path = f'weights/{response_type}/{model_name}_{response_type}_{mode_suffix}{accuracy_str}.pth'

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Move to CPU for saving (better compatibility)
    device = next(model.parameters()).device
    model_to_save = model.cpu() if device.type == 'cuda' else model

    # Prepare save dictionary
    save_dict = {
        'model_class': model_name,
        'device': str(device),
        'pytorch_version': torch.__version__,
    }

    if save_mode == 'full':
        save_dict['model'] = model_to_save
    elif save_mode == 'state_dict':
        save_dict['state_dict'] = model_to_save.state_dict()
    else:
        raise ValueError("save_mode must be 'full' or 'state_dict'")

    if additional_info:
        save_dict.update(additional_info)

    try:
        torch.save(save_dict, model_path, pickle_protocol=4)
        print(f"Model saved successfully: {model_path}")

        # Move model back to original device
        if device.type == 'cuda':
            model.to(device)

        return model_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return None


def load_model(model_path, model_class=None, device='cpu'):
    """
    Load a PyTorch model saved in either full or state_dict mode.

    Args:
        model_path (str): Path to the saved model file (.pth)
        model_class (type, optional): Model class for state_dict mode (required if state_dict mode)
        device (str or torch.device): Device to load the model onto (default: 'cpu')
    Returns:
        tuple: (model, additional_info)
            - model: Loaded PyTorch model
            - additional_info: Dictionary of metadata (or empty dict if none)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize additional_info as empty dict if not present
        additional_info = {k: v for k, v in checkpoint.items() if k not in ['model', 'state_dict', 'model_class', 'device', 'pytorch_version']}

        # Determine save mode based on checkpoint contents
        if 'model' in checkpoint:
            # Full mode: model is directly stored
            model = checkpoint['model']
            model.to(device)
            print(f"Loaded full model: {checkpoint['model_class']} to {device}")
        elif 'state_dict' in checkpoint:
            # State dict mode: requires model_class to instantiate
            if model_class is None:
                raise ValueError("model_class must be provided for state_dict mode")
            model = model_class()  # Instantiate the model
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            print(f"Loaded state_dict for model: {checkpoint['model_class']} to {device}")
        else:
            raise ValueError("Checkpoint does not contain 'model' or 'state_dict'")

        return model, additional_info

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def plotting(data, x_label, y_label, title, dt=None):
    plt.rcParams.update({'font.size': 11, "font.family": "Times New Roman"})
    plt.figure(figsize=(10, 3))

    x_values = None
    if dt is not None:
        # Calculate time values for the x-axis
        # Assuming the first dimension of data represents time steps
        num_samples = data.shape[0]
        x_values = np.arange(num_samples) * dt
        x_label = f"Time ({x_label})"  # Update x_label to reflect time

    if data.ndim == 2:
        if x_values is not None:
            plt.plot(x_values, data[:, -1], 'r-', linewidth=0.8, label='Top Node')
        else:
            plt.plot(data[:, -1], 'r-', linewidth=0.8, label='Top Node')
    elif data.ndim == 1:
        if x_values is not None:
            plt.plot(x_values, data, 'b-', linewidth=0.8, label='Data')
        else:
            plt.plot(data, 'b-', linewidth=0.8, label='Data')
    else:
        raise ValueError(f"Data has an unsupported number of dimensions: {data.ndim}")

    plt.title(title, fontdict={'fontname': 'Times New Roman', 'size': 11})
    plt.xlabel(x_label, fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.ylabel(y_label, fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.legend()

    # Set x-axis to start from 0
    plt.xlim(left=0)

    # Set x-axis margin to 0
    plt.margins(x=0)
    # Add grid
    plt.grid(True)  # This line adds the grid
    plt.tight_layout()
    plt.show()


def plot_max_responses(max_data, data_label):
    plt.rcParams.update({'font.size': 11, "font.family": "Times New Roman"})

    num_nodes = len(max_data)
    plt.figure(figsize=(4, 8))
    plt.plot(max_data, range(num_nodes), 'r-', linewidth=0.8, label=f'Max {data_label}')
    plt.title(f'Maximum {data_label}', fontdict={'fontname': 'Times New Roman', 'size': 11})
    plt.xlabel(f'{data_label} (mm/s^2)' if data_label == 'Acceleration' else f'{data_label} (mm)',
               fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.ylabel('Node', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.yticks(range(num_nodes))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_max_forces(forces):
    plt.rcParams.update({'font.size': 11, "font.family": "Times New Roman"})

    num_nodes = len(forces)
    plt.figure(figsize=(4, 8))
    plt.plot(forces, range(num_nodes), 'b-', linewidth=0.8, label='Max Force')
    plt.title('Maximum Force', fontdict={'fontname': 'Times New Roman', 'size': 11})
    plt.xlabel('Force (N)', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.ylabel('Node', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.yticks(range(num_nodes))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_shear_distribution(shear_forces, title="Shear Force Distribution"):
    import matplotlib.pyplot as plt

    num_nodes = len(shear_forces)
    plt.rcParams.update({'font.size': 11, "font.family": "Times New Roman"})

    plt.figure(figsize=(4, 8))
    plt.plot(shear_forces, range(num_nodes), 'b-', linewidth=0.8, label='Shear Force')
    plt.title(title, fontdict={'fontname': 'Times New Roman', 'size': 11})
    plt.xlabel('Force (N)', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.ylabel('Node', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12})
    plt.yticks(range(num_nodes))
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_modal_analysis(eigenvalues):
    """
    Print natural periods and frequencies from eigenvalues.

    Parameters:
    eigenvalues: Array of eigenvalues from modal analysis
    """
    print("\nMode | T [sec] | f [Hz]")
    print("-" * 25)

    for i, eigenvalue in enumerate(eigenvalues):
        omega = np.sqrt(eigenvalue)
        if omega > 1e-10:
            period = 2 * math.pi / omega
            freq_hz = omega / (2 * math.pi)
        else:
            period = float('inf')
            freq_hz = 0.0

        print(f"{i + 1:4d} | {period:7.4f} | {freq_hz:7.4f}")


def plot_chimney(D0, D1, H, t0, t1):
    fig, ax = plt.subplots()
    x_outer = [D0 / 2, D1 / 2]
    x_inner = [D0 / 2 - t0, D1 / 2 - t1]
    y = [0, H]

    # Outer wall
    ax.plot(x_outer, y, color='black', label='Outer Wall')
    ax.plot([-x for x in x_outer], y, color='black')

    # Inner wall
    ax.plot(x_inner, y, color='blue', linestyle='--', label='Inner Wall')
    ax.plot([-x for x in x_inner], y, color='blue', linestyle='--')

    ax.set_xlabel('Radius (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('RC Chimney Geometry')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


def print_parameters_table(parameters):
    print("\n" + "=" * 80)
    print("RC CHIMNEY PARAMETERS TABLE")
    print("=" * 80)
    print(f"{'Parameter':<30} {'Value':<15} {'Unit':<10}")
    print("-" * 80)
    print(f"{'Height (H)':<30} {parameters[0] / 1000:<15,.1f} {'m':<10}")
    print(f"{'Base Diameter (D0)':<30} {parameters[1] / 1000:<15,.1f} {'m':<10}")
    print(f"{'Top Diameter (D1)':<30} {parameters[2] / 1000:<15,.1f} {'m':<10}")
    print(f"{'Base Wall Thickness (t0)':<30} {parameters[3]:<15,.0f} {'mm':<10}")
    print(f"{'Top Wall Thickness (t1)':<30} {parameters[4]:<15,.0f} {'mm':<10}")
    print(f"{'Vertical Reinforcement (pv)':<30} {parameters[5]:<15,.4f} {'':<10}")  # Assuming no unit for ratio
    print(f"{'Concrete Strength (fc)':<30} {parameters[6]:<15,.0f} {'MPa':<10}")
    print(f"{'Steel Strength (fy)':<30} {parameters[7]:<15,.0f} {'MPa':<10}")
    print(f"{'Taper Rate':<30} {parameters[8]:<15,.4f} {'':<10}")  # Assuming no unit for rate per mm
    print(f"{'Taper Angle':<30} {parameters[9]:<15,.1f} {'degrees':<10}")
    print(f"{'Slenderness (H/D0)':<30} {parameters[10]:<15,.2f} {'':<10}")
    print(f"{'Mass Factor':<30} {parameters[11]:<15,.2f} {'%':<10}")
    print("=" * 80)


def plot_chimney_response_prediction(pred_response, true_response=None, ground_motion=None, sample_idx=0, dt=0.02, title_prefix="Sample", show=False):
    """
    Plot ground motion and predicted vs. true response for a single sample.

    Parameters:
    - pred_response: ndarray of shape (1, T, C)
    - true_response: ndarray of shape (1, T, C) or None
    - ground_motion: ndarray of shape (1, T) or None
    - sample_idx: int, for title annotation
    - dt: float, time step (e.g., 0.02s)
    - title_prefix: str, prefix in the plot title
    """
    pred = np.array(pred_response[0])
    true = np.array(true_response[0]) if true_response is not None else None
    gm = np.array(ground_motion[0]) if ground_motion is not None else None

    min_length = pred.shape[0]
    if true is not None:
        min_length = min(min_length, true.shape[0])
    if gm is not None:
        min_length = min(min_length, gm.shape[0])

    pred = pred[:min_length]
    if true is not None:
        true = true[:min_length]
    if gm is not None:
        gm = gm[:min_length]

    time = np.arange(min_length) * dt
    RESPONSE_FEATURES = pred.shape[1] if pred.ndim > 1 else 1

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(RESPONSE_FEATURES + 1, 1, figsize=(10, 2.5 * (RESPONSE_FEATURES + 1)), sharex=True, constrained_layout=True)

    if RESPONSE_FEATURES == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]

    y_min = np.min(pred)
    y_max = np.max(pred)
    if true is not None:
        y_min = min(y_min, np.min(true))
        y_max = max(y_max, np.max(true))

    y_margin = (y_max - y_min) * 0.05
    y_limits = [y_min - y_margin, y_max + y_margin]

    # Plot ground motion
    if gm is not None:
        axes[0].plot(time, gm, label='Ground Motion Acceleration', color='black', linewidth=1)
        axes[0].set_ylabel('Ground Acc. (g)')
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].axis('off')

    # Plot each response channel
    for ch in range(RESPONSE_FEATURES):
        ax = axes[ch + 1]
        if true is not None:
            ax.plot(time, true[:, ch], label='True Response', color='#2B6B9E', linewidth=2)
        ax.plot(time, pred[:, ch], label='Predicted Response', color='#D83132', linewidth=1)

        ax.set_title(f'Response at {(ch + 1) * 20}% Height')
        ax.set_ylim(y_limits)
        ax.legend()
        ax.grid(True)

        # Channel metrics
        if true is not None:
            r = np.corrcoef(true[:, ch], pred[:, ch])[0, 1]
            mse = np.mean((true[:, ch] - pred[:, ch]) ** 2)
            mae = np.mean(np.abs(true[:, ch] - pred[:, ch]))
            metrics_text = f'MSE={mse:.4f}\nMAE={mae:.4f}\nR={r:.4f}'
            ax.text(0.995, 0.035, metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{title_prefix} {sample_idx + 1} - Prediction', fontsize=13)
    if show:
        plt.show()


def save_training_summary(model, folder, file_name, hyperparameters, metrics, best_epoch, test_metrics):
    training_folder = folder
    training_dir = os.path.join(training_folder)
    os.makedirs(training_dir, exist_ok=True)
    file_path = os.path.join(training_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Model Training Summary ===\n\n")

        # Save hyperparameters
        f.write("=== Hyperparameters ===\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Model Type: {type(model).__name__}\n\n")

        # Save training and validation metrics
        f.write("=== Training and Validation Metrics ===\n")
        f.write("| Epoch | Train Loss | Train MAE | Train R | Val Loss | Val MAE | Val R  |\n")
        f.write("|-------|------------|-----------|---------|----------|---------|--------|\n")
        for epoch in range(len(metrics["train_losses"])):
            # Ensure indices are within bounds
            current_train_loss = metrics["train_losses"][epoch]
            current_train_mae = metrics["train_maes"][epoch]
            current_train_r = metrics["train_r_scores"][epoch]
            current_val_loss = metrics["val_losses"][epoch]
            current_val_mae = metrics["val_maes"][epoch]
            current_val_r = metrics["val_r_scores"][epoch]

            f.write(
                f"| {epoch + 1:^5} | {current_train_loss:^10.4f} | {current_train_mae:^9.4f} | "
                f"{current_train_r:^7.4f} | {current_val_loss:^8.4f} | "
                f"{current_val_mae:^7.4f} | {current_val_r:^6.4f} |\n"
            )
        f.write(f"\nBest Epoch (based on validation loss): {best_epoch}\n\n")

        # Save test metrics
        f.write("=== Test Metrics ===\n")
        f.write(f"Test Loss: {test_metrics['test_loss']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['test_mae']:.4f}\n")
        f.write(f"Test Pearson R: {test_metrics['test_r']:.4f}\n\n")

        # Save best validation metrics and corresponding training metrics
        best_epoch_index = best_epoch - 1
        # Check if best_epoch_index is valid
        if 0 <= best_epoch_index < len(metrics["val_losses"]):
            f.write("=== Best Validation Metrics ===\n")
            f.write(f"Best Validation Loss: {metrics['val_losses'][best_epoch_index]:.4f}\n")
            f.write(f"Best Validation MAE: {metrics['val_maes'][best_epoch_index]:.4f}\n")
            f.write(f"Best Validation Pearson R: {metrics['val_r_scores'][best_epoch_index]:.4f}\n\n")

            f.write("=== Training Metrics at Best Validation Epoch ===\n")
            f.write(f"Training Loss: {metrics['train_losses'][best_epoch_index]:.4f}\n")
            f.write(f"Training MAE: {metrics['train_maes'][best_epoch_index]:.4f}\n")
            f.write(f"Training Pearson R: {metrics['train_r_scores'][best_epoch_index]:.4f}\n\n")
        else:
            f.write("Could not retrieve metrics for best epoch. Index out of bounds.\n\n")

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save model summary
        f.write("=== Model Summary ===\n")
        f.write(f"Total Parameters: {total_params:,}\n")

    print(f"Model training summary saved to '{file_path}'")


def save_training_summary_2(model, folder, file_name, hyperparameters, metrics, best_epoch, test_metrics, inference_metrics=None):
    training_folder = folder
    training_dir = os.path.join(training_folder)
    os.makedirs(training_dir, exist_ok=True)
    file_path = os.path.join(training_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Model Training Summary ===\n\n")

        # Save hyperparameters
        f.write("=== Hyperparameters ===\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Model Type: {type(model).__name__}\n\n")

        # Save training and validation metrics with all 6 metrics
        f.write("=== Training and Validation Metrics ===\n")
        f.write("| Epoch | Train Loss | Train MAE | Train R | Train Peak | Train Phase | Train Amp | Val Loss | Val MAE | Val R | Val Peak | Val Phase | Val Amp |\n")
        f.write("|-------|------------|-----------|---------|------------|-------------|-----------|----------|---------|-------|----------|-----------|----------|\n")
        for epoch in range(len(metrics["train_losses"])):
            # Ensure indices are within bounds
            current_train_loss = metrics["train_losses"][epoch]
            current_train_mae = metrics["train_maes"][epoch]
            current_train_r = metrics["train_r_scores"][epoch]
            current_train_peak = metrics["train_peak_errors"][epoch]
            current_train_phase = metrics["train_phase_errors"][epoch]
            current_train_amp = metrics["train_amp_ratios"][epoch]
            current_val_loss = metrics["val_losses"][epoch]
            current_val_mae = metrics["val_maes"][epoch]
            current_val_r = metrics["val_r_scores"][epoch]
            current_val_peak = metrics["val_peak_errors"][epoch]
            current_val_phase = metrics["val_phase_errors"][epoch]
            current_val_amp = metrics["val_amp_ratios"][epoch]

            f.write(
                f"| {epoch + 1:^5} | {current_train_loss:^10.4f} | {current_train_mae:^9.4f} | "
                f"{current_train_r:^7.4f} | {current_train_peak:^10.4f} | {current_train_phase:^11.4f} | "
                f"{current_train_amp:^9.4f} | {current_val_loss:^8.4f} | {current_val_mae:^7.4f} | "
                f"{current_val_r:^5.4f} | {current_val_peak:^8.4f} | {current_val_phase:^9.4f} | "
                f"{current_val_amp:^7.4f} |\n"
            )
        f.write(f"\nBest Epoch (based on validation loss): {best_epoch}\n\n")

        # Save test metrics
        f.write("=== Test Metrics ===\n")
        f.write(f"Test Loss: {test_metrics['test_loss']:.7f}\n")
        f.write(f"Test MAE: {test_metrics['test_mae']:.7f}\n")
        f.write(f"Test Pearson R: {test_metrics['test_r']:.6f}\n")
        f.write(f"Test Peak Error: {test_metrics['test_peak_error']:.6f}\n")
        f.write(f"Test Phase Error: {test_metrics['test_phase_error']:.6f}\n")
        f.write(f"Test Amplitude Ratio: {test_metrics['test_amp_ratio']:.6f}\n\n")

        # Save inference metrics if provided
        if inference_metrics:
            f.write("=== Inference Sample Metrics ===\n")
            f.write(f"Inference Loss: {inference_metrics['loss']:.7f}\n")
            f.write(f"Inference MAE: {inference_metrics['mae']:.7f}\n")
            f.write(f"Inference Pearson R: {inference_metrics['r']:.6f}\n")
            f.write(f"Inference Peak Error: {inference_metrics['peak_error']:.6f}\n")
            f.write(f"Inference Phase Error: {inference_metrics['phase_error']:.6f}\n")
            f.write(f"Inference Amplitude Ratio: {inference_metrics['amp_ratio']:.6f}\n\n")

        # Save best validation metrics and corresponding training metrics
        best_epoch_index = best_epoch - 1
        # Check if best_epoch_index is valid
        if 0 <= best_epoch_index < len(metrics["val_losses"]):
            f.write("=== Best Validation Metrics ===\n")
            f.write(f"Best Validation Loss: {metrics['val_losses'][best_epoch_index]:.7f}\n")
            f.write(f"Best Validation MAE: {metrics['val_maes'][best_epoch_index]:.7f}\n")
            f.write(f"Best Validation Pearson R: {metrics['val_r_scores'][best_epoch_index]:.6f}\n")
            f.write(f"Best Validation Peak Error: {metrics['val_peak_errors'][best_epoch_index]:.6f}\n")
            f.write(f"Best Validation Phase Error: {metrics['val_phase_errors'][best_epoch_index]:.6f}\n")
            f.write(f"Best Validation Amplitude Ratio: {metrics['val_amp_ratios'][best_epoch_index]:.6f}\n\n")

            f.write("=== Training Metrics at Best Validation Epoch ===\n")
            f.write(f"Training Loss: {metrics['train_losses'][best_epoch_index]:.7f}\n")
            f.write(f"Training MAE: {metrics['train_maes'][best_epoch_index]:.7f}\n")
            f.write(f"Training Pearson R: {metrics['train_r_scores'][best_epoch_index]:.6f}\n")
            f.write(f"Training Peak Error: {metrics['train_peak_errors'][best_epoch_index]:.6f}\n")
            f.write(f"Training Phase Error: {metrics['train_phase_errors'][best_epoch_index]:.6f}\n")
            f.write(f"Training Amplitude Ratio: {metrics['train_amp_ratios'][best_epoch_index]:.6f}\n\n")
        else:
            f.write("Could not retrieve metrics for best epoch. Index out of bounds.\n\n")

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save model summary
        f.write("=== Model Summary ===\n")
        f.write(f"Total Parameters: {total_params:,}\n")

        # Save metric statistics
        f.write("\n=== Training Metric Statistics ===\n")
        f.write(f"Final Train Loss: {metrics['train_losses'][-1]:.7f}\n")
        f.write(f"Final Train MAE: {metrics['train_maes'][-1]:.7f}\n")
        f.write(f"Final Train R: {metrics['train_r_scores'][-1]:.6f}\n")
        f.write(f"Final Train Peak Error: {metrics['train_peak_errors'][-1]:.6f}\n")
        f.write(f"Final Train Phase Error: {metrics['train_phase_errors'][-1]:.6f}\n")
        f.write(f"Final Train Amplitude Ratio: {metrics['train_amp_ratios'][-1]:.6f}\n\n")

        f.write("=== Validation Metric Statistics ===\n")
        f.write(f"Final Val Loss: {metrics['val_losses'][-1]:.7f}\n")
        f.write(f"Final Val MAE: {metrics['val_maes'][-1]:.7f}\n")
        f.write(f"Final Val R: {metrics['val_r_scores'][-1]:.6f}\n")
        f.write(f"Final Val Peak Error: {metrics['val_peak_errors'][-1]:.6f}\n")
        f.write(f"Final Val Phase Error: {metrics['val_phase_errors'][-1]:.6f}\n")
        f.write(f"Final Val Amplitude Ratio: {metrics['val_amp_ratios'][-1]:.6f}\n")

    print(f"Model training summary saved to '{file_path}'")



def plot_chimney_response_prediction(pred_response, true_response=None, ground_motion=None, sample_idx=0, dt=0.02, title_prefix="Sample", show=False,
                                     plot_channels=None, plot_ground_motion=True):
    """
    Plot ground motion and predicted vs. true response for a single sample.

    Parameters:
    - pred_response: ndarray of shape (1, T, C)
    - true_response: ndarray of shape (1, T, C) or None
    - ground_motion: ndarray of shape (1, T) or None
    - sample_idx: int, for title annotation
    - dt: float, time step (e.g., 0.02s)
    - title_prefix: str, prefix in the plot title
    - plot_channels: list of int or None, specific channels to plot (0-indexed). If None, plot all channels
    - plot_ground_motion: bool, whether to plot ground motion
    """
    pred = np.array(pred_response[0])
    true = np.array(true_response[0]) if true_response is not None else None
    gm = np.array(ground_motion[0]) if ground_motion is not None else None

    min_length = pred.shape[0]
    if true is not None:
        min_length = min(min_length, true.shape[0])
    if gm is not None:
        min_length = min(min_length, gm.shape[0])

    pred = pred[:min_length]
    if true is not None:
        true = true[:min_length]
    if gm is not None:
        gm = gm[:min_length]

    time = np.arange(min_length) * dt
    RESPONSE_FEATURES = pred.shape[1] if pred.ndim > 1 else 1

    # Determine which channels to plot
    if plot_channels is None:
        channels_to_plot = list(range(RESPONSE_FEATURES))
    else:
        channels_to_plot = [ch for ch in plot_channels if 0 <= ch < RESPONSE_FEATURES]

    # Calculate number of subplots needed
    num_response_plots = len(channels_to_plot)
    num_plots = num_response_plots + (1 if plot_ground_motion and gm is not None else 0)

    if num_plots == 0:
        print("Warning: No plots to generate (no ground motion or response channels selected)")
        return

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True, constrained_layout=True)

    if num_plots == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]

    # Calculate y-limits for response plots
    if num_response_plots > 0:
        y_min = np.min(pred[:, channels_to_plot])
        y_max = np.max(pred[:, channels_to_plot])
        if true is not None:
            y_min = min(y_min, np.min(true[:, channels_to_plot]))
            y_max = max(y_max, np.max(true[:, channels_to_plot]))

        y_margin = (y_max - y_min) * 0.05
        y_limits = [y_min - y_margin, y_max + y_margin]

    plot_idx = 0

    # Plot ground motion
    if plot_ground_motion and gm is not None:
        axes[plot_idx].plot(time, gm, label='Ground Motion Acceleration', color='black', linewidth=1)
        axes[plot_idx].set_ylabel('Ground Acc. (g)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        axes[plot_idx].set_title('Ground Motion')
        plot_idx += 1

    # Plot selected response channels
    for ch in channels_to_plot:
        ax = axes[plot_idx]
        if true is not None:
            ax.plot(time, true[:, ch], label='True Response', color='#2B6B9E', linewidth=2)
        ax.plot(time, pred[:, ch], label='Predicted Response', color='#D83132', linewidth=1)

        ax.set_title(f'Response at {(ch + 1) * 20}% Height (Channel {ch + 1})')
        ax.set_ylim(y_limits)
        ax.legend()
        ax.grid(True)

        # Channel metrics
        if true is not None:
            r = np.corrcoef(true[:, ch], pred[:, ch])[0, 1]
            mse = np.mean((true[:, ch] - pred[:, ch]) ** 2)
            mae = np.mean(np.abs(true[:, ch] - pred[:, ch]))
            metrics_text = f'MSE={mse:.4f}\nMAE={mae:.4f}\nR={r:.4f}'
            ax.text(0.995, 0.035, metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    axes[-1].set_xlabel('Time (s)')

    # Create descriptive title
    channel_desc = f"Channels: {[ch + 1 for ch in channels_to_plot]}" if plot_channels else "All Channels"
    gm_desc = " + GM" if plot_ground_motion and gm is not None else ""
    plt.suptitle(f'{title_prefix} {sample_idx + 1} - {channel_desc}{gm_desc}', fontsize=13)

    if show:
        plt.show()
