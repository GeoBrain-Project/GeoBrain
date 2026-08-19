"""
Dataset classes for geological data.

Provides PyTorch Dataset implementations for loading and transforming
geological property data.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import torch
from geobrain.core.errors import GeoBrainError
from torch.utils.data import Dataset
from typing import Optional, Callable, Union, Tuple


class SimpleTensorDataset(Dataset):
    """
    Simple dataset for accessing a single tensor.

    This lightweight dataset implementation provides direct access to elements
    of a tensor through standard indexing, without additional transformations.
    Useful for working with pre-processed data or model parameters in PyTorch.

    Args:
        data: Source tensor data of shape [n_samples, ...].

    Example:
        >>> data = torch.randn(100, 3, 64, 64)
        >>> dataset = SimpleTensorDataset(data)
        >>> sample = dataset[0]  # shape: [3, 64, 64]
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the sample at the specified index."""
        return self.data[index]


class GeoDataset(Dataset):
    """
    Dataset for geological property fields with transform support.

    Supports loading paired data (e.g., properties and labels) and
    applying transform pipelines for data augmentation and preprocessing.

    Args:
        data: Primary data tensor of shape [n_samples, ...].
        labels: Optional label tensor of shape [n_samples, ...].
        transform: Transform to apply to data samples.
        target_transform: Transform to apply to labels.

    Example:
        >>> from geobrain.core.transforms import Compose, Normalize, Clamp
        >>> transform = Compose([Normalize(), Clamp()])
        >>> dataset = GeoDataset(
        ...     data=properties,
        ...     labels=seismic,
        ...     transform=transform
        ... )
        >>> props, seis = dataset[0]
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        # Validate shapes
        if labels is not None and len(data) != len(labels):
            raise GeoBrainError(
                "data and labels must have the same length",
                object_name="GeoDataset", field="labels",
                expected=len(data), actual=len(labels),
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(
        self,
        index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return sample (and label if available) at index.

        Args:
            index: Sample index.

        Returns:
            If labels provided: (data, label) tuple.
            Otherwise: data tensor only.
        """
        x = self.data[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.labels is None:
            return x

        y = self.labels[index]
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    @classmethod
    def from_files(
        cls,
        data_path: str,
        labels_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> 'GeoDataset':
        """
        Create dataset from saved tensor files.

        Args:
            data_path: Path to data tensor file (.pt or .npy).
            labels_path: Optional path to labels tensor file.
            transform: Transform to apply to data.
            target_transform: Transform to apply to labels.

        Returns:
            GeoDataset instance.

        Example:
            >>> from geobrain.core.transforms import Normalize
            >>> dataset = GeoDataset.from_files(
            ...     'data/properties.pt',
            ...     'data/seismic.pt',
            ...     transform=Normalize()
            ... )
        """

        # Load data
        if data_path.endswith('.npy'):
            import numpy as np
            data = torch.from_numpy(np.load(data_path))
        else:
            # weights_only: tensors/containers only: never unpickle code.
            data = torch.load(data_path, weights_only=True)

        # Load labels if provided
        labels = None
        if labels_path is not None:
            if labels_path.endswith('.npy'):
                import numpy as np
                labels = torch.from_numpy(np.load(labels_path))
            else:
                labels = torch.load(labels_path, weights_only=True)

        return cls(
            data=data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple['GeoDataset', 'GeoDataset']:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Fraction of data for training.
            shuffle: Whether to shuffle before splitting.
            seed: Random seed for reproducibility.

        Returns:
            (train_dataset, val_dataset) tuple.

        Example:
            >>> train_ds, val_ds = dataset.split(train_ratio=0.8, seed=42)
        """
        n = len(self)
        indices = torch.arange(n)

        if shuffle:
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = None
            indices = indices[torch.randperm(n, generator=generator)]

        n_train = int(n * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_data = self.data[train_indices]
        val_data = self.data[val_indices]

        train_labels = self.labels[train_indices] if self.labels is not None else None
        val_labels = self.labels[val_indices] if self.labels is not None else None

        train_ds = GeoDataset(
            data=train_data,
            labels=train_labels,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        val_ds = GeoDataset(
            data=val_data,
            labels=val_labels,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        return train_ds, val_ds

    def __repr__(self) -> str:
        info = [f"GeoDataset(n_samples={len(self)}"]
        info.append(f"data_shape={tuple(self.data.shape[1:])}")
        if self.labels is not None:
            info.append(f"labels_shape={tuple(self.labels.shape[1:])}")
        if self.transform is not None:
            info.append(f"transform={self.transform}")
        return ", ".join(info) + ")"
