import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    feature_names = wine_data.feature_names
    return X, y, feature_names


def load_mnist_dataset(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        indices = np.random.choice(len(x_train), min(num_samples, len(x_train)), replace=False)
        x_flat = x_train[indices].reshape(len(indices), -1).astype(np.float32) / 255.0
        y = y_train[indices]
        return x_flat, y
    except ImportError:
        try:
            import torch
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            indices = np.random.choice(len(mnist_data), min(num_samples, len(mnist_data)), replace=False)
            samples = [mnist_data[i] for i in indices]
            x_flat = np.array([s[0].numpy().flatten() for s in samples])
            y = np.array([s[1] for s in samples])
            return x_flat, y
        except:
            np.random.seed(42)
            x_flat = np.random.randn(num_samples, 784).astype(np.float32)
            y = np.random.randint(0, 10, num_samples)
            return x_flat, y


def generate_synthetic_data(
    num_parties: int,
    num_samples: int,
    num_features: int,
    intersection_ratio: float
) -> List[Tuple[List[str], np.ndarray]]:
    np.random.seed(42)
    total_samples = int(num_samples / intersection_ratio)
    all_ids = [f"sample_{i}" for i in range(total_samples)]
    intersection_size = int(total_samples * intersection_ratio)
    shared_ids = all_ids[:intersection_size]

    data_parts = []
    for party_idx in range(num_parties):
        party_ids = shared_ids + [
            f"party{party_idx}_unique_{i}"
            for i in range(total_samples - intersection_size)
        ]
        features = np.random.randn(len(party_ids), num_features).astype(np.float32)
        data_parts.append((party_ids, features))

    return data_parts


def split_data_vertically(
    X: np.ndarray,
    num_parties: int,
    feature_names: List[str] = None
) -> List[np.ndarray]:
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    features_per_party = X.shape[1] // num_parties
    splits = []
    for i in range(num_parties):
        start = i * features_per_party
        end = (i + 1) * features_per_party if i < num_parties - 1 else X.shape[1]
        splits.append(X[:, start:end])
    return splits
