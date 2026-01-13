import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, Normalizer, PowerTransformer
import os
# from RCWall_Cyclic_chimney_params import *
import joblib
from pathlib import Path  # For path handling
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


# =================================================================================================================================================================
def normalize(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, scaling_strategy='minmax', fit=False, save_scaler_path=None):
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    # Ensure input is a NumPy array
    data = np.asarray(data, dtype=np.float32)

    # Load or create scaler
    if scaler is None:
        if scaler_filename:
            if os.path.exists(scaler_filename):
                scaler = joblib.load(scaler_filename)
            else:
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
        else:
            # Define scaler mapping
            scaler_mapping = {
                'minmax': MinMaxScaler(feature_range=range),
                'robust': RobustScaler(),
                'standard': StandardScaler(),
                'maxabs': MaxAbsScaler(),
                'quantile': QuantileTransformer(output_distribution='normal'),
                'power': PowerTransformer(method='yeo-johnson')
            }
            if scaling_strategy in scaler_mapping:
                scaler = scaler_mapping[scaling_strategy]
            else:
                raise ValueError(f"Unknown scaling strategy: {scaling_strategy}. "
                                 f"Available strategies: {list(scaler_mapping.keys())}")

    # Handle sequence data
    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        original_shape = data.shape
        data_reshaped = data

    # Scale data
    if fit:
        data_scaled = scaler.fit_transform(data_reshaped)
    else:
        data_scaled = scaler.transform(data_reshaped)

    # Reshape back if sequence
    if sequence:
        data_scaled = data_scaled.reshape(original_shape)

    # Save the scaler if specified
    if save_scaler_path and fit:
        joblib.dump(scaler, save_scaler_path)

    # Always return data and scaler
    return data_scaled, scaler


def denormalize(data, scaler=None, scaler_filename=None, sequence=False):
    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    data = np.asarray(data, dtype=np.float32)

    if scaler is None:
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
        else:
            raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")

    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data

    try:
        data_restored = scaler.inverse_transform(data_reshaped)
    except Exception as e:
        raise ValueError(f"Error during inverse transformation: {str(e)}")

    if sequence:
        data_restored = data_restored.reshape(original_shape)

    return data_restored


# =================================================================================================================================================================
def load_data(data_size=500, chimney_params=14, sequence_length=3000, response_features=5, data_folder="Chimney_Data/ProcessedData", response_type=None, normalize_data=True, verbose=True):
    if response_type is None:
        response_type = ["displacement"]
    data_folder = Path(data_folder)
    scaler_folder = data_folder / "Scaler"
    scaler_folder.mkdir(parents=True, exist_ok=True)

    data_dict = {}
    scalers = {}

    # Load chimney_params
    chimney_params = pd.read_parquet(data_folder / "Parameters.parquet").iloc[:data_size, :chimney_params].to_numpy(dtype=np.float32)
    data_dict['chimney_params'] = chimney_params

    frequencies = pd.read_parquet(data_folder / "Eigenvalues.parquet").iloc[:data_size].to_numpy(dtype=np.float32)
    data_dict['frequencies'] = frequencies

    # Load Ground Motions (common to all responses)
    if "displacement" in response_type or "acceleration" in response_type or "ground_motion" in response_type:
        ground_motion = pd.read_parquet(data_folder / "GMA.parquet").iloc[:data_size, :sequence_length].fillna(0).to_numpy(dtype=np.float32)
        data_dict['ground_motion'] = ground_motion

    if "displacement" in response_type:
        raw_displacement = pd.read_parquet(data_folder / "Displacements.parquet").iloc[:data_size * 5, :sequence_length].fillna(0).to_numpy(dtype=np.float32)
        full_features = raw_displacement.shape[0] // len(chimney_params)
        if not (1 <= response_features <= full_features):
            raise ValueError(f"response_features must be between 1 and {full_features}, got {response_features}")
        displacement = raw_displacement.reshape(len(chimney_params), full_features, sequence_length)[:, :response_features, :].transpose(0, 2, 1)
        data_dict['displacement'] = displacement

    if "acceleration" in response_type:
        raw_acceleration = pd.read_parquet(data_folder / "Accelerations.parquet").iloc[:data_size * 5, :sequence_length].fillna(0).to_numpy(dtype=np.float32)
        full_features = raw_acceleration.shape[0] // len(chimney_params)
        if not (1 <= response_features <= full_features):
            raise ValueError(f"response_features must be between 1 and {full_features}, got {response_features}")
        acceleration = raw_acceleration.reshape(len(chimney_params), full_features, sequence_length)[:, :response_features, :].transpose(0, 2, 1)
        data_dict['acceleration'] = acceleration

    # Print shapes
    if verbose:
        print("\n📦 Data Structure:")
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                print(f"  {key:<15}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key:<15}: type={type(value)}")

    # Normalize
    if normalize_data:
        norm_params, param_scaler = normalize(
            chimney_params, sequence=False, range=(0, 1), scaling_strategy='minmax',
            fit=True, save_scaler_path=scaler_folder / "param_scaler.joblib"
        )
        data_dict['chimney_params'] = norm_params
        scalers['chimney_params'] = param_scaler

        norm_freqs, freq_scaler = normalize(
            data_dict['frequencies'], sequence=True, range=(0, 1), scaling_strategy='minmax',
            fit=True, save_scaler_path=scaler_folder / "freq_scaler.joblib"
        )
        data_dict['frequencies'] = norm_freqs
        scalers['frequencies'] = freq_scaler

        if "ground_motion" in response_type or "displacement" in response_type or "acceleration" in response_type:
            norm_ground_motion, motion_scaler = normalize(
                data_dict['ground_motion'], sequence=True, range=(-1, 1), scaling_strategy='maxabs',
                fit=True, save_scaler_path=scaler_folder / "motion_scaler.joblib"
            )
            data_dict['ground_motion'] = norm_ground_motion
            scalers['ground_motion'] = motion_scaler

        if "displacement" in response_type:
            norm_disp, disp_scaler = normalize(
                data_dict['displacement'], sequence=True, range=(-1, 1), scaling_strategy='maxabs',
                fit=True, save_scaler_path=scaler_folder / "displacement_scaler.joblib"
            )
            data_dict['displacement'] = norm_disp
            scalers['displacement'] = disp_scaler

        if "acceleration" in response_type:
            norm_acc, acc_scaler = normalize(
                data_dict['acceleration'], sequence=True, range=(-1, 1), scaling_strategy='maxabs',
                fit=True, save_scaler_path=scaler_folder / "acceleration_scaler.joblib"
            )
            data_dict['acceleration'] = norm_acc
            scalers['acceleration'] = acc_scaler

        # Extra verbose block for data statistics
        if verbose:
            print(f"\n🔢 Dataset Max and Min values:")
            print("  Chimney chimney_params:")
            print(f"    Max  : {', '.join(f'{val:.2f}' for val in np.max(chimney_params, axis=0))}")
            print(f"    Min  : {', '.join(f'{val:.2f}' for val in np.min(chimney_params, axis=0))}")
            print("  Normalized chimney_params:")
            print(f"    Max  : {', '.join(f'{val:.2f}' for val in np.max(norm_params, axis=0))}")
            print(f"    Min  : {', '.join(f'{val:.2f}' for val in np.min(norm_params, axis=0))}")
            print(f"  Dominant Frequencies:")
            print(f"    Max  : {np.max(frequencies):.3f}  | Norm Max: {np.max(norm_freqs):.3f}")
            print(f"    Min  : {np.min(frequencies):.3f} | Norm Min: {np.min(norm_freqs):.3f}")
            if "ground_motion" in response_type or "acceleration" in response_type or "displacement" in response_type:
                print("  Ground Motions:")
                print(f"    Max  : {np.max(ground_motion):.3f} | Norm Max: {np.max(norm_ground_motion):.3f}")
                print(f"    Min  : {np.min(ground_motion):.3f} | Norm Min: {np.min(norm_ground_motion):.3f}")
            if "displacement" in response_type:
                print(f"  Displacement Responses:")
                print(f"    Max  : {np.max(displacement):.3f}  | Norm Max: {np.max(norm_disp):.3f}")
                print(f"    Min  : {np.min(displacement):.3f} | Norm Min: {np.min(norm_disp):.3f}")
            if "acceleration" in response_type:
                print(f"  Acceleration Responses:")
                print(f"    Max  : {np.max(acceleration):.3f}  | Norm Max: {np.max(norm_acc):.3f}")
                print(f"    Min  : {np.min(acceleration):.3f} | Norm Min: {np.min(norm_acc):.3f}")

            print("\n🧪 Normalizer Structure:")
            for key, scaler in scalers.items():
                print(f"  {key:<15}: type={type(scaler)}")

        return data_dict, scalers

    else:
        data_folder = Path("Chimney_Data/ProcessedData")
        scaler_folder = data_folder / "Scaler"
        # Load and apply existing scalers
        norm_params, param_scaler = normalize(
            chimney_params, sequence=False, scaler_filename=scaler_folder / "param_scaler.joblib"
        )
        data_dict['chimney_params'] = norm_params
        scalers['chimney_params'] = param_scaler

        norm_freqs, freq_scaler = normalize(
            data_dict['frequencies'], sequence=True, scaler_filename=scaler_folder / "freq_scaler.joblib"
        )
        data_dict['frequencies'] = norm_freqs
        scalers['frequencies'] = freq_scaler

        if "ground_motion" in response_type or "displacement" in response_type or "acceleration" in response_type:
            norm_ground_motion, motion_scaler = normalize(
                data_dict['ground_motion'], sequence=True, scaler_filename=scaler_folder / "motion_scaler.joblib"
            )
            data_dict['ground_motion'] = norm_ground_motion
            scalers['ground_motion'] = motion_scaler

        if "displacement" in response_type:
            norm_disp, disp_scaler = normalize(
                data_dict['displacement'], sequence=True, scaler_filename=scaler_folder / "displacement_scaler.joblib"
            )
            data_dict['displacement'] = norm_disp
            scalers['displacement'] = disp_scaler

        if "acceleration" in response_type:
            norm_acc, acc_scaler = normalize(
                data_dict['acceleration'], sequence=True, scaler_filename=scaler_folder / "acceleration_scaler.joblib"
            )
            data_dict['acceleration'] = norm_acc
            scalers['acceleration'] = acc_scaler

        # Print normalizer structure when verbose
        if verbose:
            print("\n🧪 Normalizer Structure:")
            for key, scaler in scalers.items():
                print(f"  {key:<15}: type={type(scaler)}")

    return data_dict, scalers


def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device=None, verbose=True):
    if isinstance(data, dict):
        keys = list(data.keys())
        arrays = [data[k] for k in keys]
    elif isinstance(data, (list, tuple)):
        arrays = list(data)
        keys = [f"feature_{i}" for i in range(len(arrays))]
    else:
        raise TypeError("Input 'data' must be a dict, list, or tuple of arrays.")

    n_samples = len(arrays[0])
    for i, array in enumerate(arrays):
        if len(array) != n_samples:
            raise ValueError(f"All input arrays must have the same number of samples. Mismatch at index {i}.")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tensors = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in arrays]

    # Split into train+val and test sets
    splits = train_test_split(*tensors, test_size=test_size, random_state=random_state)
    train_val_splits = splits[::2]
    test_splits = splits[1::2]

    # Split train+val into train and val
    splits2 = train_test_split(*train_val_splits, test_size=val_size / (1 - test_size), random_state=random_state)
    train_splits = splits2[::2]
    val_splits = splits2[1::2]

    train_dict = {k: v for k, v in zip(keys, train_splits)}
    val_dict = {k: v for k, v in zip(keys, val_splits)}
    test_dict = {k: v for k, v in zip(keys, test_splits)}

    if verbose:
        print("\n📊 Split Tensor Structures:")
        for name, splits in zip(["Train", "Val", "Test"], [train_splits, val_splits, test_splits]):
            print(f"  {name} Set:")
            for key, tensor in zip(keys, splits):
                print(f"    {key:<15}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

    return train_dict, val_dict, test_dict


# For Flexible Model ================================================================================
class ChimneySequenceDataset(Dataset):
    def __init__(self, data_dict, required_key=None, verbose=True):
        """
        Args:
            data_dict (dict): Dictionary of named tensors (e.g., {"chimney_params": tensor, "displacement": tensor, ...})
            required_key (str): Key to use for determining dataset length and valid indices (e.g., "chimney_params")
            verbose (bool): Whether to print dataset structure summary
        """
        self.data = data_dict
        self.keys = list(data_dict.keys())

        if required_key is None:
            required_key = self.keys[0]  # fallback to first key

        self.required_key = required_key
        num_samples = data_dict[required_key].shape[0]

        # Filter valid indices
        self.index_map = list(range(num_samples))
        for key, tensor in data_dict.items():
            self.index_map = [i for i in self.index_map if tensor[i].size(0) > 0]

        if verbose:
            print(f"\n📦 Initialized Dataset")
            if len(self.index_map) > 0:
                sample = self.__getitem__(0)
                for key, val in sample.items():
                    print(f"   {key:<20}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
            else:
                print("  (No valid samples)")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        i = self.index_map[idx]
        return {key: tensor[i] for key, tensor in self.data.items()}


def collate_fn_general(batch):
    batch_dict = {}
    keys = batch[0].keys()

    for key in keys:
        # Stack tensors for this key, if tensors
        values = [item[key] for item in batch]

        # If they are tensors, stack; otherwise just gather in a list
        if torch.is_tensor(values[0]):
            batch_dict[key] = torch.stack(values)
        else:
            batch_dict[key] = values  # e.g. sequence_index, start_index might be int

    return batch_dict


# For Frequency Model # For Simple Model ============================================================
class ChimneyStaticDataset(Dataset):
    def __init__(self, chimney_params, frequencies):
        self.chimney_params = chimney_params  # Shape: [num_samples, chimney_params_FEATURES]
        self.frequencies = frequencies  # Shape: [num_samples, NUM_FREQUENCIES]

    def __len__(self):
        return len(self.chimney_params)

    def __getitem__(self, idx):
        return {
            'chimney_params': self.chimney_params[idx],
            'frequencies': self.frequencies[idx]
        }


def collate_fn_static(batch):
    chimney_params = torch.stack([item['chimney_params'] for item in batch])
    frequencies = torch.stack([item['frequencies'] for item in batch])
    return {
        'chimney_params': chimney_params,
        'frequencies': frequencies
    }


# ===================================================================================================
class ChimneyFullSequenceCombined(Dataset):
    def __init__(self, chimney_params, dominant_frequencies, ground_motion, displacement_responses, acceleration_responses):
        assert len(ground_motion) == len(displacement_responses) == len(acceleration_responses) == chimney_params.shape[0]
        self.P = chimney_params
        self.F = dominant_frequencies
        self.G_list = ground_motion
        self.D_list = displacement_responses
        self.A_list = acceleration_responses
        self.index_map = [n for n in range(len(ground_motion)) if ground_motion[n].size(0) > 0]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        n = self.index_map[idx]
        return {
            'chimney_params': self.P[n],
            'frequencies': self.F[n],
            'ground_motion': self.G_list[n],
            'target_displacement': self.D_list[n],
            'target_acceleration': self.A_list[n],
        }


def collate_fn(batch):
    """
    Custom collate function to properly batch the data for the model.
    """
    # Stack all the tensors
    chimney_params = torch.stack([item['chimney_params'] for item in batch])
    frequencies = torch.stack([item['frequencies'] for item in batch])
    ground_motion = torch.stack([item['ground_motion'] for item in batch])
    hist_displacement = torch.stack([item['hist_displacement'] for item in batch])
    hist_acceleration = torch.stack([item['hist_acceleration'] for item in batch])
    target_displacement = torch.stack([item['target_displacement'] for item in batch])
    target_acceleration = torch.stack([item['target_acceleration'] for item in batch])
    sequence_index = torch.tensor([item['sequence_index'] for item in batch], dtype=torch.long)
    start_index = torch.tensor([item['start_index'] for item in batch], dtype=torch.long)

    return {
        'chimney_params': chimney_params,
        'frequencies': frequencies,
        'ground_motion': ground_motion,
        'hist_displacement': hist_displacement,
        'hist_acceleration': hist_acceleration,
        'target_displacement': target_displacement,
        'target_acceleration': target_acceleration,
        'sequence_index': sequence_index,
        'start_index': start_index
    }
