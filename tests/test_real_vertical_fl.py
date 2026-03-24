import sys
sys.path.insert(0, '/home/yuank/peafowl-2')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server


def generate_vertical_fl_data(num_samples_per_party=(100, 80, 60), 
                               num_common_features=10,
                               num_unique_features_per_party=(5, 4, 3),
                               num_classes=2,
                               noise_level=0.1):
    """
    生成模拟垂直联邦学习场景的数据
    
    参数:
        num_samples_per_party: 每个参与方的样本数量
        num_common_features: 共同特征数量（所有参与方都有）
        num_unique_features_per_party: 每个参与方的独有特征数量
        num_classes: 类别数量
        noise_level: 噪声水平
    
    返回:
        X_list: 每个参与方的特征矩阵列表
        y_list: 每个参与方的标签列表（相同样本的标签相同）
        sample_ids_list: 每个参与方的样本ID列表
    """
    num_parties = len(num_samples_per_party)
    
    # 生成所有样本的共同特征
    total_samples = max(num_samples_per_party)
    X_common = np.random.randn(total_samples, num_common_features)
    
    # 生成标签（基于共同特征）
    y = np.zeros(total_samples, dtype=int)
    for i in range(total_samples):
        score = np.sum(X_common[i, :num_common_features//2])
        y[i] = int(score > 0)
    
    # 确保类别平衡
    while len(np.unique(y)) < num_classes:
        y = np.random.randint(0, num_classes, total_samples)
    
    X_list = []
    sample_ids_list = []
    
    for party_idx in range(num_parties):
        num_samples = num_samples_per_party[party_idx]
        num_unique = num_unique_features_per_party[party_idx]
        
        base_ids = [f"sample_{j}" for j in range(total_samples)]
        
        np.random.seed(42 + party_idx)
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        party_sample_ids = [base_ids[i] for i in selected_indices]
        
        X_party_common = X_common[selected_indices, :]
        
        np.random.seed(100 + party_idx)
        X_party_unique = np.random.randn(len(selected_indices), num_unique)
        
        X_party = np.hstack([X_party_common, X_party_unique])
        
        X_list.append(X_party)
        sample_ids_list.append(party_sample_ids)
    
    return X_list, y, sample_ids_list


def load_mnist_data(num_samples=1000):
    """加载MNIST数据集"""
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


def generate_mnist_vertical_fl_data(num_samples_per_party=(400, 350, 300),
                                     num_common_features=100,
                                     num_unique_features_per_party=(50, 40, 30)):
    """
    为MNIST数据集生成垂直联邦学习场景
    每个参与方有不同的样本集和特征集
    """
    num_parties = len(num_samples_per_party)
    
    # 加载MNIST数据
    X_mnist, y_mnist = load_mnist_data(num_samples=max(num_samples_per_party) + 100)
    
    X_list = []
    sample_ids_list = []
    
    for party_idx in range(num_parties):
        num_samples = num_samples_per_party[party_idx]
        num_unique = num_unique_features_per_party[party_idx]
        
        total_samples = num_samples + 100
        
        base_ids = [f"sample_{j}" for j in range(total_samples)]
        
        np.random.seed(42 + party_idx)
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        party_sample_ids = [base_ids[i] for i in selected_indices]
        party_labels = y_mnist[selected_indices]
        
        # 选择共同特征（MNIST的某些列）
        np.random.seed(100 + party_idx)
        common_feature_indices = np.random.choice(784, num_common_features, replace=False)
        common_feature_indices = np.sort(common_feature_indices)
        
        # 选择独有特征
        remaining_indices = np.setdiff1d(np.arange(784), common_feature_indices)
        np.random.seed(200 + party_idx)
        unique_feature_indices = np.random.choice(remaining_indices, num_unique, replace=False)
        unique_feature_indices = np.sort(unique_feature_indices)
        
        # 提取特征
        X_party_common = X_mnist[selected_indices][:, common_feature_indices]
        X_party_unique = X_mnist[selected_indices][:, unique_feature_indices]
        
        X_party = np.hstack([X_party_common, X_party_unique])
        
        X_list.append(X_party)
        sample_ids_list.append(party_sample_ids)
    
    return X_list, y_mnist[:max(num_samples_per_party) + 100], sample_ids_list


def simulate_vertical_fl(X_list, y, sample_ids_list, num_parties):
    """
    模拟垂直联邦学习场景
    """
    config = {
        'num_parties': num_parties,
        'num_samples': max([len(ids) for ids in sample_ids_list]),
        'num_features': sum([X.shape[1] for X in X_list]),
        'secret_modulus': 2**64,
        'precision_bits': 16,
        'shprg_d': 8,
        'shprg_q': 2**128,
        'shprg_p': 2**64,
        'prf_key_bytes': 16,
    }
    
    data_providers = []
    for i in range(num_parties):
        dp = DataProvider(f"P{i}", config, sample_ids_list[i].copy(), X_list[i])
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
    
    id_to_label = {f"sample_{j}": y[j] for j in range(len(y))}
    y_aligned = np.array([id_to_label[sid] for sid in aligned_ids if sid in id_to_label])
    
    valid_indices = [i for i, sid in enumerate(aligned_ids) if sid in id_to_label]
    X_aligned = X_aligned[valid_indices]
    y_aligned = y_aligned[:len(valid_indices)]
    
    return X_aligned, y_aligned


def train_and_evaluate_logistic(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, model


def build_cnn_model(input_shape, num_classes):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except ImportError:
        return None


def train_and_evaluate_cnn(X_train, X_test, y_train, y_test):
    try:
        from tensorflow.keras.utils import to_categorical

        num_classes = len(np.unique(y_train))
        
        # 检查特征是否是28x28的图像
        if X_train.shape[1] == 784:
            X_train_cnn = X_train.reshape(-1, 28, 28, 1)
            X_test_cnn = X_test.reshape(-1, 28, 28, 1)
            input_shape = (28, 28, 1)
        else:
            # 对于垂直联邦学习的特征，使用1D卷积
            X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)
            input_shape = (X_train.shape[1], 1)
        
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        model = build_cnn_model_1d(input_shape, num_classes)
        
        if model is None:
            return None, None, None
        
        model.fit(X_train_cnn, y_train_cat, epochs=5, batch_size=32, verbose=0)
        
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        
        y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return test_acc, f1, model
    except Exception as e:
        print(f"CNN training failed: {e}")
        return None, None, None


def build_cnn_model_1d(input_shape, num_classes):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical

        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except ImportError:
        return None


def run_mnist_vertical_fl_test():
    print("=" * 80)
    print("MNIST垂直联邦学习场景测试")
    print("=" * 80)
    
    num_samples_per_party = (400, 350, 300)
    num_common_features = 100
    num_unique_features_per_party = (50, 40, 30)
    
    X_list, y, sample_ids_list = generate_mnist_vertical_fl_data(
        num_samples_per_party=num_samples_per_party,
        num_common_features=num_common_features,
        num_unique_features_per_party=num_unique_features_per_party
    )
    
    num_parties = len(X_list)
    
    print(f"\n参与方数量: {num_parties}")
    print(f"每个参与方的样本数量: {[len(ids) for ids in sample_ids_list]}")
    print(f"每个参与方的特征数量: {[X.shape[1] for X in X_list]}")
    print(f"  - 共同特征: {num_common_features}")
    print(f"  - 独有特征: {num_unique_features_per_party}")
    
    total_features_if_no_overlap = sum([X.shape[1] for X in X_list])
    print(f"\n如果无重叠的总特征数: {total_features_if_no_overlap}")
    
    print("\n--- PEAFOWL对齐后 ---")
    X_aligned, y_aligned = simulate_vertical_fl(X_list, y, sample_ids_list, num_parties)
    print(f"对齐后样本数量: {len(y_aligned)}")
    print(f"对齐后特征数量: {X_aligned.shape[1]}")
    print(f"特征压缩率: {(1 - X_aligned.shape[1] / total_features_if_no_overlap) * 100:.2f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    print("\n--- 逻辑回归分类结果 ---")
    acc_lr, f1_lr, _ = train_and_evaluate_logistic(X_train, X_test, y_train, y_test)
    print(f"准确率: {acc_lr:.4f}")
    print(f"F1分数: {f1_lr:.4f}")
    
    print("\n--- CNN分类结果 ---")
    acc_cnn, f1_cnn, _ = train_and_evaluate_cnn(X_train, X_test, y_train, y_test)
    if acc_cnn is not None:
        print(f"准确率: {acc_cnn:.4f}")
        print(f"F1分数: {f1_cnn:.4f}")
    else:
        print("CNN训练失败")
    
    return {
        'num_parties': num_parties,
        'samples_per_party': [len(ids) for ids in sample_ids_list],
        'features_per_party': [X.shape[1] for X in X_list],
        'aligned_samples': len(y_aligned),
        'aligned_features': X_aligned.shape[1],
        'lr_accuracy': acc_lr,
        'lr_f1': f1_lr,
        'cnn_accuracy': acc_cnn,
        'cnn_f1': f1_cnn
    }


def run_comparison_with_different_parties():
    print("\n" + "=" * 80)
    print("不同参与方数量的对比测试")
    print("=" * 80)
    
    results = []
    
    for num_parties in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"测试 {num_parties} 个参与方")
        print(f"{'='*60}")
        
        num_samples_per_party = tuple([400 - i*50 for i in range(num_parties)])
        num_common_features = 100
        num_unique_features_per_party = tuple([50 - i*10 for i in range(num_parties)])
        
        X_list, y, sample_ids_list = generate_mnist_vertical_fl_data(
            num_samples_per_party=num_samples_per_party,
            num_common_features=num_common_features,
            num_unique_features_per_party=num_unique_features_per_party
        )
        
        config = {
            'num_parties': num_parties,
            'num_samples': max([len(ids) for ids in sample_ids_list]),
            'num_features': sum([X.shape[1] for X in X_list]),
            'secret_modulus': 2**64,
            'precision_bits': 16,
            'shprg_d': 8,
            'shprg_q': 2**128,
            'shprg_p': 2**64,
            'prf_key_bytes': 16,
        }
        
        data_providers = []
        for i in range(num_parties):
            dp = DataProvider(f"P{i}", config, sample_ids_list[i].copy(), X_list[i])
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
        
        id_to_label = {f"sample_{j}": y[j] for j in range(len(y))}
        y_aligned = np.array([id_to_label[sid] for sid in aligned_ids if sid in id_to_label])
        
        valid_indices = [i for i, sid in enumerate(aligned_ids) if sid in id_to_label]
        X_aligned = X_aligned[valid_indices]
        y_aligned = y_aligned[:len(valid_indices)]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y_aligned, test_size=0.2, random_state=42
        )
        
        acc_lr, f1_lr, _ = train_and_evaluate_logistic(X_train, X_test, y_train, y_test)
        acc_cnn, f1_cnn, _ = train_and_evaluate_cnn(X_train, X_test, y_train, y_test)
        
        total_features = sum([X.shape[1] for X in X_list])
        
        print(f"  参与方数量: {num_parties}")
        print(f"  每方样本数: {[len(ids) for ids in sample_ids_list]}")
        print(f"  每方特征数: {[X.shape[1] for X in X_list]}")
        print(f"  对齐后样本数: {len(y_aligned)}")
        print(f"  对齐后特征数: {X_aligned.shape[1]}")
        print(f"  逻辑回归准确率: {acc_lr:.4f}")
        print(f"  逻辑回归F1: {f1_lr:.4f}")
        if acc_cnn is not None:
            print(f"  CNN准确率: {acc_cnn:.4f}")
            print(f"  CNNF1: {f1_cnn:.4f}")
        
        results.append({
            'num_parties': num_parties,
            'samples_per_party': [len(ids) for ids in sample_ids_list],
            'features_per_party': [X.shape[1] for X in X_list],
            'aligned_samples': len(y_aligned),
            'aligned_features': X_aligned.shape[1],
            'lr_accuracy': acc_lr,
            'lr_f1': f1_lr,
            'cnn_accuracy': acc_cnn,
            'cnn_f1': f1_cnn
        })
    
    return results


def print_summary(results):
    print("\n" + "=" * 80)
    print("SUMMARY - MNIST垂直联邦学习测试")
    print("=" * 80)
    
    print(f"\n{'Parties':<10} {'Samples':<25} {'Features':<25} {'Aligned':<15} {'LR Acc':<10} {'CNN Acc':<10}")
    print("-" * 90)
    
    for r in results:
        samples_str = str(r['samples_per_party'])
        features_str = str(r['features_per_party'])
        cnn_acc_str = f"{r['cnn_accuracy']:.4f}" if r['cnn_accuracy'] is not None else "N/A"
        print(f"{r['num_parties']:<10} {samples_str:<25} {features_str:<25} "
              f"{r['aligned_samples']:<15} {r['lr_accuracy']:<10.4f} {cnn_acc_str:<10}")


if __name__ == '__main__':
    np.random.seed(42)
    
    result = run_mnist_vertical_fl_test()
    
    results = run_comparison_with_different_parties()
    
    print_summary(results)
