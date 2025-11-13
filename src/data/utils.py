"""Utilities for time grids, noise, and batching."""

import numpy as np
import torch
from typing import Tuple, Optional, List


def create_time_grid(
    t_span: Tuple[float, float], n_points: int, spacing: str = "uniform"
) -> np.ndarray:
    """
    Create time grid with various spacing options.

    Args:
        t_span: (t_start, t_end)
        n_points: number of points
        spacing: 'uniform', 'log', 'chebyshev'

    Returns:
        time array
    """
    t_start, t_end = t_span

    if spacing == "uniform":
        return np.linspace(t_start, t_end, n_points)
    elif spacing == "log":
        # Logarithmic spacing (useful for capturing fast initial dynamics)
        if t_start == 0:
            t_start = 1e-6
        return np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    elif spacing == "chebyshev":
        # Chebyshev nodes mapped to [t_start, t_end]
        k = np.arange(1, n_points + 1)
        cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * n_points))
        # Map from [-1, 1] to [t_start, t_end]
        return t_start + (t_end - t_start) * (1 - cheb_nodes) / 2
    else:
        raise ValueError(f"Unknown spacing: {spacing}")


def add_noise(
    data: np.ndarray, noise_type: str = "gaussian", noise_level: float = 0.01, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to data.

    Args:
        data: input data
        noise_type: 'gaussian', 'uniform', 'relative'
        noise_level: noise magnitude
        seed: random seed

    Returns:
        noisy data
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, data.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, data.shape)
    elif noise_type == "relative":
        # Relative noise: noise_level * |data| * random
        noise = noise_level * np.abs(data) * np.random.normal(0, 1, data.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return data + noise


def create_batches(
    data: torch.Tensor, batch_size: int, shuffle: bool = True, seed: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Create batches from data.

    Args:
        data: input tensor [N, ...]
        batch_size: batch size
        shuffle: whether to shuffle data
        seed: random seed

    Returns:
        list of batches
    """
    n_samples = data.shape[0]

    if shuffle:
        if seed is not None:
            torch.manual_seed(seed)
        indices = torch.randperm(n_samples)
        data = data[indices]

    batches = []
    for i in range(0, n_samples, batch_size):
        batch = data[i : i + batch_size]
        batches.append(batch)

    return batches


def numpy_to_torch(
    *arrays: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, ...]:
    """
    Convert numpy arrays to torch tensors.

    Args:
        arrays: numpy arrays
        device: torch device
        dtype: torch dtype

    Returns:
        tuple of torch tensors
    """
    tensors = []
    for arr in arrays:
        tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        tensors.append(tensor)

    return tuple(tensors) if len(tensors) > 1 else tensors[0]


def torch_to_numpy(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """
    Convert torch tensors to numpy arrays.

    Args:
        tensors: torch tensors

    Returns:
        tuple of numpy arrays
    """
    arrays = []
    for tensor in tensors:
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.squeeze(-1)
        arrays.append(arr)

    return tuple(arrays) if len(arrays) > 1 else arrays[0]


def split_train_test(
    *arrays: np.ndarray, train_ratio: float = 0.8, seed: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """
    Split arrays into train and test sets.

    Args:
        arrays: input arrays (must have same length)
        train_ratio: ratio of training data
        seed: random seed

    Returns:
        train and test arrays interleaved (train1, test1, train2, test2, ...)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = arrays[0].shape[0]
    n_train = int(n_samples * train_ratio)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return tuple(result)


def subsample_data(
    *arrays: np.ndarray, n_samples: int, method: str = "uniform", seed: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """
    Subsample data arrays.

    Args:
        arrays: input arrays
        n_samples: number of samples to keep
        method: 'uniform', 'random'
        seed: random seed

    Returns:
        subsampled arrays
    """
    n_total = arrays[0].shape[0]

    if n_samples >= n_total:
        return arrays

    if method == "uniform":
        indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
    elif method == "random":
        if seed is not None:
            np.random.seed(seed)
        indices = np.sort(np.random.choice(n_total, n_samples, replace=False))
    else:
        raise ValueError(f"Unknown method: {method}")

    result = tuple(arr[indices] for arr in arrays)
    # If single array, return as tuple to maintain consistency
    return result if len(result) > 1 else result

