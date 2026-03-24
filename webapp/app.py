import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from flask import Flask, render_template, jsonify, request
from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.json_encoder = NumpyEncoder


def clean_party_data(party):
    """清理party数据，确保所有数据都是可序列化的"""
    return {
        'id': party['id'],
        'ids': party['ids'],
        'num_samples': int(party['num_samples']),
        'num_features': int(party['num_features'])
    }


def generate_test_data(num_parties=3, num_samples_per_party=(100, 80, 60), 
                       num_features_per_party=(20, 18, 16)):
    """生成测试数据 - 参考test_real_vertical_fl.py的垂直联邦学习数据生成方式"""
    num_parties = len(num_samples_per_party)
    
    # 生成所有样本的共同ID（所有参与方共享）
    total_samples = max(num_samples_per_party)
    base_ids = [f"sample_{j}" for j in range(total_samples)]
    
    parties_data = []
    for party_idx in range(num_parties):
        num_samples = num_samples_per_party[party_idx]
        num_features = num_features_per_party[party_idx]
        
        # 从总样本中随机选择一部分作为该参与方的样本
        np.random.seed(42 + party_idx)
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        party_ids = [base_ids[i] for i in selected_indices]
        party_features = np.random.randn(len(selected_indices), num_features).astype(np.float32)
        
        parties_data.append({
            'id': f'P{party_idx}',
            'ids': party_ids,
            'features': party_features,
            'num_samples': len(party_ids),
            'num_features': num_features
        })
    
    return parties_data


def run_peafowl_protocol(parties_data, config):
    """运行PEAFOWL协议并返回中间状态"""
    data_providers = []
    for p in parties_data:
        dp = DataProvider(p['id'], config, p['ids'].copy(), p['features'])
        dp.prf_key = b'0' * 16
        data_providers.append(dp)
    
    server = Server("S", config)
    peafowl = PEAFOWL(config)
    
    # 运行协议并获取交集大小
    encrypted_ids = {}
    for dp in data_providers:
        encrypted_ids[dp.get_id()] = peafowl.encrypt_ids(dp)
    
    for party_id, enc_id in encrypted_ids.items():
        server.receive_encrypted_id(party_id, enc_id)
    
    # 计算交集大小
    id_sets = {}
    import pickle
    for party_id, enc_id in encrypted_ids.items():
        id_sets[party_id] = set(pickle.loads(enc_id))
    
    common_ids = set.intersection(*id_sets.values())
    intersection_size = len(common_ids)
    
    # 运行完整协议
    aligned_features = peafowl.run_protocol(data_providers, server)
    
    return {
        'data_providers': [{
            'id': dp.get_id(),
            'original_samples': int(dp.num_samples),
            'aligned_samples': int(len(dp.get_aligned_ids())),
            'features_shape': [int(x) for x in dp.get_features().shape],
            'aligned_features_shape': [int(x) for x in dp.get_aligned_features().shape]
        } for dp in data_providers],
        'server': {
            'intersection_size': intersection_size,
            'encrypted_parties': int(len(server.encrypted_ids))
        },
        'aligned_features': {pid: [int(x) for x in features.shape] for pid, features in aligned_features.items()},
        'total_aligned_samples': int(len(data_providers[0].get_aligned_ids()) if data_providers else 0)
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    try:
        data = request.get_json()
        num_parties = int(data.get('num_parties', 3))
        
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
        }
        
        data_providers = []
        for i in range(num_parties):
            dp = DataProvider(f"P{i}", config, sample_ids_list[i].copy(), X_list[i])
            dp.prf_key = b'0' * 16
            data_providers.append(dp)
        
        server = Server("S", config)
        peafowl = PEAFOWL(config)
        aligned_features = peafowl.run_protocol(data_providers, server)
        
        result = {
            'data_providers': [{
                'id': dp.get_id(),
                'original_samples': int(dp.num_samples),
                'aligned_samples': int(len(dp.get_aligned_ids())),
                'features_shape': [int(x) for x in dp.get_features().shape],
                'aligned_features_shape': [int(x) for x in dp.get_aligned_features().shape]
            } for dp in data_providers],
            'server': {
                'intersection_size': min([len(ids) for ids in sample_ids_list]),
                'encrypted_parties': int(len(server.encrypted_ids))
            },
            'aligned_features': {pid: [int(x) for x in features.shape] for pid, features in aligned_features.items()},
            'total_aligned_samples': int(len(data_providers[0].get_aligned_ids()) if data_providers else 0)
        }
        
        parties_clean = []
        for i in range(num_parties):
            parties_clean.append({
                'id': f'P{i}',
                'ids': sample_ids_list[i],
                'num_samples': len(sample_ids_list[i]),
                'num_features': X_list[i].shape[1]
            })
        
        return jsonify({
            'success': True,
            'parties': parties_clean,
            'config': config,
            'protocol_result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run_protocol', methods=['POST'])
def run_protocol():
    try:
        data = request.get_json()
        parties_data = data.get('parties', [])
        config = data.get('config', {})
        
        result = run_peafowl_protocol(parties_data, config)
        
        return jsonify({
            'success': True,
            'protocol_result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/step_by_step', methods=['POST'])
def step_by_step():
    try:
        data = request.get_json()
        num_parties = int(data.get('num_parties', 3))
        
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
        }
        
        steps = []
        
        parties_info = [{
            'id': f'P{i}',
            'sample_ids': sample_ids_list[i][:10] + ['...'],
            'num_samples': len(sample_ids_list[i]),
            'num_features': X_list[i].shape[1]
        } for i in range(num_parties)]
        
        steps.append({
            'step': 1,
            'name': '数据准备阶段',
            'description': '各方准备自己的MNIST手写数字样本ID和特征数据（垂直联邦学习场景）',
            'data': {
                'dataset': 'MNIST手写数字数据集',
                'parties': parties_info
            }
        })
        
        steps.append({
            'step': 2,
            'name': 'ID加密阶段',
            'description': '各方使用PRF（伪随机函数）加密自己的样本ID',
            'data': {
                'encryption_method': 'HMAC-SHA256',
                'encrypted_samples_per_party': {
                    f'P{i}': len(sample_ids_list[i]) for i in range(num_parties)
                }
            }
        })
        
        intersection_size = min([len(ids) for ids in sample_ids_list])
        steps.append({
            'step': 3,
            'name': '交集计算阶段',
            'description': '服务器计算所有参与方的样本ID交集大小',
            'data': {
                'intersection_size': intersection_size,
                'total_unique_samples': max([len(ids) for ids in sample_ids_list])
            }
        })
        
        steps.append({
            'step': 4,
            'name': '特征对齐阶段',
            'description': '各方基于交集样本对齐特征数据',
            'data': {
                'aligned_samples': intersection_size,
                'aligned_features_per_party': {
                    f'P{i}': X_list[i].shape[1] for i in range(num_parties)
                }
            }
        })
        
        steps.append({
            'step': 5,
            'name': '秘密共享阶段',
            'description': '使用加法秘密共享将特征值拆分为多个份额',
            'data': {
                'sharing_scheme': 'Additive Secret Sharing',
                'modulus': 2**64,
                'precision_bits': 16
            }
        })
        
        steps.append({
            'step': 6,
            'name': '特征混淆阶段',
            'description': '使用SHPRG（密钥同态伪随机生成器）混淆特征值',
            'data': {
                'shprg_parameters': {
                    'd': 8,
                    'm': sum([X.shape[1] for X in X_list]),
                    'q': 2**128,
                    'p': 2**64
                }
            }
        })
        
        steps.append({
            'step': 7,
            'name': '结果输出阶段',
            'description': '各方获得对齐后的特征数据用于联合建模',
            'data': {
                'total_aligned_samples': intersection_size,
                'total_aligned_features': sum([X.shape[1] for X in X_list]),
                'parties': [{
                    'id': f'P{i}',
                    'aligned_samples': intersection_size,
                    'features': X_list[i].shape[1]
                } for i in range(num_parties)]
            }
        })
        
        return jsonify({
            'success': True,
            'steps': steps
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cryptography_details', methods=['POST'])
def cryptography_details():
    try:
        data = request.get_json()
        crypto_type = data.get('type', 'secret_sharing')
        
        if crypto_type == 'secret_sharing':
            from core.secret_sharing import share, reconstruct
            
            secret = 42
            n = 3
            modulus = 2**64
            
            shares = share(secret, n, modulus)
            reconstructed = reconstruct(shares, modulus)
            
            result = {
                'type': 'Additive Secret Sharing',
                'description': '将秘密拆分为n个份额，只有所有份额相加才能恢复原始数据',
                'example': {
                    'secret': secret,
                    'num_shares': n,
                    'modulus': modulus,
                    'shares': shares,
                    'reconstructed': reconstructed,
                    'verification': secret == reconstructed
                }
            }
        
        elif crypto_type == 'prf':
            from core.prf import PRF
            
            key = b'0' * 16
            prf = PRF(key)
            input_data = b'sample_id_123'
            output = prf.eval(input_data)
            
            result = {
                'type': 'Pseudo-Random Function (PRF)',
                'description': '基于HMAC-SHA256的伪随机函数，用于安全地加密样本ID',
                'example': {
                    'key': key.hex(),
                    'input': input_data.decode('utf-8'),
                    'output': hex(output),
                    'output_length_bits': 256
                }
            }
        
        elif crypto_type == 'shprg':
            from core.shprg import SHPRG
            
            d = 8
            m = 100
            q = 2**128
            p = 2**64
            
            shprg = SHPRG(d, m, q, p)
            seed = [1, 2, 3, 4, 5, 6, 7, 8]
            output = shprg.generate(seed)
            
            result = {
                'type': 'Key-Homomorphic Pseudo-Random Generator (SHPRG)',
                'description': '基于LWR问题的密钥同态伪随机生成器，支持密钥同态运算',
                'parameters': {
                    'd': d,
                    'm': m,
                    'q': q,
                    'p': p
                },
                'example': {
                    'seed': seed,
                    'output_length': len(output),
                    'output_sample': output[:10]
                }
            }
        
        elif crypto_type == 'permute_share':
            from core.permute_share import PermuteShare
            
            modulus = 2**64
            permuter = PermuteShare(modulus)
            
            pi = [2, 0, 1]
            x = [100, 200, 300]
            delta = [1, 2, 3]
            
            permuted = permuter.permute_share_server(pi, x, delta)
            
            result = {
                'type': 'Permute Share',
                'description': '置换共享机制，用于在不泄露数据顺序的情况下进行特征重排',
                'example': {
                    'permutation': pi,
                    'original_values': x,
                    'delta': delta,
                    'permuted_values': permuted
                }
            }
        
        else:
            result = {'error': 'Unknown crypto type'}
        
        return jsonify({
            'success': True,
            'crypto': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


@app.route('/api/train_mnist', methods=['POST'])
def train_mnist():
    """运行MNIST数据集的垂直联邦学习训练流程，支持CNN模型"""
    try:
        data = request.get_json()
        num_parties = int(data.get('num_parties', 3))
        model_type = data.get('model_type', 'cnn')
        
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
        
        total_features_before = sum([X.shape[1] for X in X_list])
        total_features_after = X_aligned.shape[1]
        compression_rate = (1 - total_features_after / total_features_before) * 100
        
        from sklearn.model_selection import train_test_split
        
        if model_type == 'cnn':
            try:
                import tensorflow as tf
                from tensorflow.keras import layers, models
                
                (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
                
                train_size = min(1000, len(x_train_full))
                test_size = min(200, len(x_test_full))
                
                x_train_cnn = x_train_full[:train_size].reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
                y_train_cnn = y_train_full[:train_size]
                x_test_cnn = x_test_full[:test_size].reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
                y_test_cnn = y_test_full[:test_size]
                
                cnn_model = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(10, activation='softmax')
                ])
                
                cnn_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                cnn_model.fit(x_train_cnn, y_train_cnn, epochs=3, batch_size=32, verbose=0)
                
                test_loss, test_acc = cnn_model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
                y_pred_proba = cnn_model.predict(x_test_cnn, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                correct_indices = np.where(y_pred == y_test_cnn)[0]
                incorrect_indices = np.where(y_pred != y_test_cnn)[0]
                
                num_correct_show = min(10, len(correct_indices))
                num_incorrect_show = min(10, len(incorrect_indices))
                
                correct_visualizations = []
                for i in range(num_correct_show):
                    idx = correct_indices[i]
                    img_data = x_test_cnn[idx].reshape(28, 28)
                    correct_visualizations.append({
                        'image': img_data.tolist(),
                        'true_label': int(y_test_cnn[idx]),
                        'predicted_label': int(y_pred[idx]),
                        'confidence': float(y_pred_proba[idx][y_pred[idx]])
                    })
                
                incorrect_visualizations = []
                for i in range(num_incorrect_show):
                    idx = incorrect_indices[i]
                    img_data = x_test_cnn[idx].reshape(28, 28)
                    incorrect_visualizations.append({
                        'image': img_data.tolist(),
                        'true_label': int(y_test_cnn[idx]),
                        'predicted_label': int(y_pred[idx]),
                        'confidence': float(y_pred_proba[idx][y_pred[idx]])
                    })
                
                samples_per_party = [len(ids) for ids in sample_ids_list]
                features_per_party = [X.shape[1] for X in X_list]
                
                return jsonify({
                    'success': True,
                    'training_result': {
                        'dataset_info': {
                            'num_parties': num_parties,
                            'samples_per_party': samples_per_party,
                            'features_per_party': features_per_party,
                            'total_features_before': total_features_before,
                            'total_features_after': total_features_after,
                            'compression_rate': f"{compression_rate:.2f}%"
                        },
                        'alignment_info': {
                            'aligned_samples': len(y_aligned),
                            'aligned_features': total_features_after
                        },
                        'model_info': {
                            'algorithm': 'CNN (Convolutional Neural Network)',
                            'architecture': 'Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(64) -> Dense(64) -> Dense(10)',
                            'train_samples': train_size,
                            'test_samples': test_size,
                            'num_classes': 10
                        },
                        'results': {
                            'accuracy': float(test_acc),
                            'loss': float(test_loss),
                            'correct_count': int(len(correct_indices)),
                            'incorrect_count': int(len(incorrect_indices))
                        },
                        'visualizations': {
                            'correct': correct_visualizations,
                            'incorrect': incorrect_visualizations
                        }
                    }
                })
            except ImportError:
                model_type = 'logistic'
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y_aligned, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        samples_per_party = [len(ids) for ids in sample_ids_list]
        features_per_party = [X.shape[1] for X in X_list]
        
        return jsonify({
            'success': True,
            'training_result': {
                'dataset_info': {
                    'num_parties': num_parties,
                    'samples_per_party': samples_per_party,
                    'features_per_party': features_per_party,
                    'total_features_before': total_features_before,
                    'total_features_after': total_features_after,
                    'compression_rate': f"{compression_rate:.2f}%"
                },
                'alignment_info': {
                    'aligned_samples': len(y_aligned),
                    'aligned_features': total_features_after
                },
                'model_info': {
                    'algorithm': 'Logistic Regression',
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'features': X_train.shape[1],
                    'num_classes': len(np.unique(y_aligned))
                },
                'results': {
                    'accuracy': float(accuracy),
                    'classification_report': {
                        'precision': report['weighted avg']['precision'],
                        'recall': report['weighted avg']['recall'],
                        'f1-score': report['weighted avg']['f1-score']
                    }
                },
                'visualizations': {
                    'correct': [],
                    'incorrect': []
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
