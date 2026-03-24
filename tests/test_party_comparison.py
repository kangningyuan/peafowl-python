import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server


def load_wine_data():
    wine = load_wine()
    X, y = wine.data, wine.target
    return X, y


def load_breast_cancer_data():
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    return X, y


def load_mnist_sample(num_samples=1000):
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        indices = np.random.choice(len(x_train), min(num_samples, len(x_train)), replace=False)
        x_flat = x_train[indices].reshape(len(indices), -1).astype(np.float32) / 255.0
        y = y_train[indices]
        return x_flat, y
    except ImportError:
        np.random.seed(42)
        x_flat = np.random.randn(num_samples, 784).astype(np.float32)
        y = np.random.randint(0, 10, num_samples)
        return x_flat, y


def simulate_peafowl_alignment(X, y, num_parties=3):
    sample_ids = [f"sample_{j}" for j in range(len(y))]
    id_to_label = {sample_ids[i]: y[i] for i in range(len(y))}

    features_per_party = X.shape[1] // num_parties
    X_splits = []
    for i in range(num_parties):
        start = i * features_per_party
        end = (i + 1) * features_per_party if i < num_parties - 1 else X.shape[1]
        X_splits.append(X[:, start:end])

    config = {
        'num_parties': num_parties,
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'secret_modulus': 2**64,
        'precision_bits': 16,
        'shprg_d': 8,
        'shprg_q': 2**128,
        'shprg_p': 2**64,
        'prf_key_bytes': 16,
    }

    data_providers = []
    for i in range(num_parties):
        ids = sample_ids.copy()
        features = X_splits[i]
        dp = DataProvider(f"P{i}", config, ids, features)
        dp.prf_key = b'0' * 16
        data_providers.append(dp)

    server = Server("S", config)
    peafowl = PEAFOWL(config)
    aligned_features_dict = peafowl.run_protocol(data_providers, server)

    X_aligned = None
    aligned_ids = None
    for party_id in sorted(aligned_features_dict.keys()):
        if X_aligned is None:
            X_aligned = aligned_features_dict[party_id]
            aligned_ids = data_providers[int(party_id[1:])].get_aligned_ids()
        else:
            X_aligned = np.hstack([X_aligned, aligned_features_dict[party_id]])

    y_aligned = np.array([id_to_label[sid] for sid in aligned_ids])

    return X_aligned, y_aligned


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1


def test_different_parties(X, y, party_numbers, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    results = {
        'original': {'accuracy': 0, 'f1': 0},
        'parties': {}
    }
    
    acc_orig, f1_orig = train_and_evaluate(X_train, X_test, y_train, y_test)
    results['original']['accuracy'] = acc_orig
    results['original']['f1'] = f1_orig
    
    for num_parties in party_numbers:
        print(f"Testing with {num_parties} party/parties...")
        
        X_aligned, y_aligned = simulate_peafowl_alignment(X, y, num_parties=num_parties)
        
        X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned = train_test_split(
            X_aligned, y_aligned, test_size=test_size, random_state=random_state
        )
        
        acc_peafowl, f1_peafowl = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned)
        
        results['parties'][num_parties] = {
            'accuracy': acc_peafowl,
            'f1': f1_peafowl,
            'diff_accuracy': acc_orig - acc_peafowl,
            'diff_f1': f1_orig - f1_peafowl
        }
        
        print(f"  Accuracy: {acc_orig:.4f} -> {acc_peafowl:.4f} (diff: {acc_orig - acc_peafowl:.4f})")
        print(f"  F1 Score: {f1_orig:.4f} -> {f1_peafowl:.4f} (diff: {f1_orig - f1_peafowl:.4f})")
    
    return results


def run_wine_experiment():
    print("=" * 80)
    print("Wine Dataset - Different Party Numbers Comparison")
    print("=" * 80)
    
    X, y = load_wine_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {X.shape[1]}")
    
    party_numbers = [2, 3, 4, 5, 6]
    
    print("\n--- Testing different party numbers ---")
    results = test_different_parties(X, y, party_numbers)
    
    return {
        'dataset': 'wine',
        'results': results,
        'party_numbers': party_numbers
    }


def run_breast_cancer_experiment():
    print("\n" + "=" * 80)
    print("Breast Cancer Dataset - Different Party Numbers Comparison")
    print("=" * 80)
    
    X, y = load_breast_cancer_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {X.shape[1]}")
    
    party_numbers = [2, 3, 4, 5, 6]
    
    print("\n--- Testing different party numbers ---")
    results = test_different_parties(X, y, party_numbers)
    
    return {
        'dataset': 'breast_cancer',
        'results': results,
        'party_numbers': party_numbers
    }


def run_mnist_experiment():
    print("\n" + "=" * 80)
    print("MNIST Dataset - Different Party Numbers Comparison")
    print("=" * 80)
    
    X, y = load_mnist_sample(num_samples=1000)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {X.shape[1]}")
    
    party_numbers = [2, 3, 4, 5, 7, 8]
    
    print("\n--- Testing different party numbers ---")
    results = test_different_parties(X, y, party_numbers)
    
    return {
        'dataset': 'mnist',
        'results': results,
        'party_numbers': party_numbers
    }


def print_summary(results_list):
    print("\n" + "=" * 80)
    print("SUMMARY - Impact of Party Numbers on Training Performance")
    print("=" * 80)
    
    for result in results_list:
        dataset = result['dataset']
        party_numbers = result['party_numbers']
        original_acc = result['results']['original']['accuracy']
        
        print(f"\n{dataset.upper()}:")
        print(f"  Original Accuracy: {original_acc:.4f}")
        print(f"  {'Parties':<10} {'Accuracy':<12} {'F1 Score':<12} {'Acc Diff':<12} {'F1 Diff':<12}")
        print(f"  {'-'*58}")
        
        for num_parties in party_numbers:
            party_result = result['results']['parties'][num_parties]
            print(f"  {num_parties:<10} {party_result['accuracy']:<12.4f} {party_result['f1']:<12.4f} "
                  f"{party_result['diff_accuracy']:<12.4f} {party_result['diff_f1']:<12.4f}")


if __name__ == '__main__':
    np.random.seed(42)
    
    results = []
    
    results.append(run_wine_experiment())
    results.append(run_breast_cancer_experiment())
    
    try:
        results.append(run_mnist_experiment())
    except Exception as e:
        print(f"\nMNIST experiment skipped due to error: {e}")
    
    print_summary(results)
