import sys
sys.path.insert(0, '/home/yuank/peafowl-2')

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server


def load_wine_data():
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    return X, y, feature_names


def load_breast_cancer_data():
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    feature_names = bc.feature_names
    return X, y, feature_names


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


def simulate_peafowl_alignment(X, y, num_parties=3, feature_split=True):
    sample_ids = [f"sample_{j}" for j in range(len(y))]
    id_to_label = {sample_ids[i]: y[i] for i in range(len(y))}

    if feature_split:
        features_per_party = X.shape[1] // num_parties
        X_splits = []
        for i in range(num_parties):
            start = i * features_per_party
            end = (i + 1) * features_per_party if i < num_parties - 1 else X.shape[1]
            X_splits.append(X[:, start:end])
    else:
        samples_per_party = X.shape[0] // num_parties
        X_splits = []
        indices = np.random.permutation(X.shape[0])
        for i in range(num_parties):
            start = i * samples_per_party
            end = (i + 1) * samples_per_party if i < num_parties - 1 else X.shape[0]
            X_splits.append(X[indices[start:end]])

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


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='logistic'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, model


def run_wine_experiment():
    print("=" * 60)
    print("Wine Dataset Experiment")
    print("=" * 60)

    X, y, feature_names = load_wine_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Original Data ---")
    acc_orig, f1_orig, _ = train_and_evaluate(X_train, X_test, y_train, y_test, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_orig:.4f}, F1={f1_orig:.4f}")

    acc_orig_rf, f1_orig_rf, _ = train_and_evaluate(X_train, X_test, y_train, y_test, 'rf')
    print(f"Random Forest: Accuracy={acc_orig_rf:.4f}, F1={f1_orig_rf:.4f}")

    print("\n--- PEAFOWL Processed Data (Vertical Split, 3 parties) ---")
    X_aligned, y_aligned = simulate_peafowl_alignment(X, y, num_parties=3, feature_split=True)
    print(f"Aligned dataset shape: X={X_aligned.shape}, y={y_aligned.shape}")

    X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )

    acc_peafowl, f1_peafowl, _ = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_peafowl:.4f}, F1={f1_peafowl:.4f}")

    acc_peafowl_rf, f1_peafowl_rf, _ = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned, 'rf')
    print(f"Random Forest: Accuracy={acc_peafowl_rf:.4f}, F1={f1_peafowl_rf:.4f}")

    print("\n--- Comparison ---")
    print(f"Logistic Regression: Original vs PEAFOWL = {acc_orig:.4f} vs {acc_peafowl:.4f} (diff: {abs(acc_orig - acc_peafowl):.4f})")
    print(f"Random Forest: Original vs PEAFOWL = {acc_orig_rf:.4f} vs {acc_peafowl_rf:.4f} (diff: {abs(acc_orig_rf - acc_peafowl_rf):.4f})")

    return {
        'dataset': 'wine',
        'original_lr_acc': acc_orig, 'peafowl_lr_acc': acc_peafowl,
        'original_rf_acc': acc_orig_rf, 'peafowl_rf_acc': acc_peafowl_rf,
    }


def run_breast_cancer_experiment():
    print("\n" + "=" * 60)
    print("Breast Cancer Dataset Experiment")
    print("=" * 60)

    X, y, feature_names = load_breast_cancer_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Original Data ---")
    acc_orig, f1_orig, _ = train_and_evaluate(X_train, X_test, y_train, y_test, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_orig:.4f}, F1={f1_orig:.4f}")

    acc_orig_rf, f1_orig_rf, _ = train_and_evaluate(X_train, X_test, y_train, y_test, 'rf')
    print(f"Random Forest: Accuracy={acc_orig_rf:.4f}, F1={f1_orig_rf:.4f}")

    print("\n--- PEAFOWL Processed Data (Vertical Split, 3 parties) ---")
    X_aligned, y_aligned = simulate_peafowl_alignment(X, y, num_parties=3, feature_split=True)
    print(f"Aligned dataset shape: X={X_aligned.shape}, y={y_aligned.shape}")

    X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )

    acc_peafowl, f1_peafowl, _ = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_peafowl:.4f}, F1={f1_peafowl:.4f}")

    acc_peafowl_rf, f1_peafowl_rf, _ = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned, 'rf')
    print(f"Random Forest: Accuracy={acc_peafowl_rf:.4f}, F1={f1_peafowl_rf:.4f}")

    print("\n--- Comparison ---")
    print(f"Logistic Regression: Original vs PEAFOWL = {acc_orig:.4f} vs {acc_peafowl:.4f} (diff: {abs(acc_orig - acc_peafowl):.4f})")
    print(f"Random Forest: Original vs PEAFOWL = {acc_orig_rf:.4f} vs {acc_peafowl_rf:.4f} (diff: {abs(acc_orig_rf - acc_peafowl_rf):.4f})")

    return {
        'dataset': 'breast_cancer',
        'original_lr_acc': acc_orig, 'peafowl_lr_acc': acc_peafowl,
        'original_rf_acc': acc_orig_rf, 'peafowl_rf_acc': acc_peafowl_rf,
    }


def run_mnist_experiment():
    print("\n" + "=" * 60)
    print("MNIST Dataset Experiment (Sample)")
    print("=" * 60)

    X, y = load_mnist_sample(num_samples=1000)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Original Data ---")
    acc_orig, f1_orig, _ = train_and_evaluate(X_train, X_test, y_train, y_test, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_orig:.4f}, F1={f1_orig:.4f}")

    print("\n--- PEAFOWL Processed Data (Vertical Split, 4 parties) ---")
    X_aligned, y_aligned = simulate_peafowl_alignment(X, y, num_parties=4, feature_split=True)
    print(f"Aligned dataset shape: X={X_aligned.shape}, y={y_aligned.shape}")

    X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )

    acc_peafowl, f1_peafowl, _ = train_and_evaluate(X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned, 'logistic')
    print(f"Logistic Regression: Accuracy={acc_peafowl:.4f}, F1={f1_peafowl:.4f}")

    print("\n--- Comparison ---")
    print(f"Logistic Regression: Original vs PEAFOWL = {acc_orig:.4f} vs {acc_peafowl:.4f} (diff: {abs(acc_orig - acc_peafowl):.4f})")

    return {
        'dataset': 'mnist',
        'original_lr_acc': acc_orig, 'peafowl_lr_acc': acc_peafowl,
    }


if __name__ == '__main__':
    np.random.seed(42)

    results = []

    results.append(run_wine_experiment())
    results.append(run_breast_cancer_experiment())

    try:
        results.append(run_mnist_experiment())
    except Exception as e:
        print(f"\nMNIST experiment skipped due to error: {e}")

    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)
    for r in results:
        print(f"\n{r['dataset']}:")
        print(f"  Logistic Regression: Original={r.get('original_lr_acc', 'N/A'):.4f}, PEAFOWL={r.get('peafowl_lr_acc', 'N/A'):.4f}")
        if 'original_rf_acc' in r:
            print(f"  Random Forest: Original={r['original_rf_acc']:.4f}, PEAFOWL={r['peafowl_rf_acc']:.4f}")
