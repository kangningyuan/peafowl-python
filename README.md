# PEAFOWL - Privacy-Preserving Entity Alignment for Vertical Federated Learning

[English](#english) | [中文](#中文)

## 中文

PEAFOWL 是一个为垂直联邦学习设计的隐私保护实体对齐协议框架。它允许不同数据方在不泄露各自数据的情况下，安全地找出共同样本（交集），并基于对齐后的数据联合训练机器学习模型。

### 核心特性

- **隐私保护**：使用加法秘密共享、伪随机函数、密钥同态伪随机生成器等密码学技术，确保交集信息不被泄露
- **高效协议**：通信和计算复杂度为线性级别 O(n)，适合大规模数据集
- **多方支持**：支持多个数据参与方的实体对齐
- **灵活模型**：支持逻辑回归、CNN、SVM等多种机器学习模型

### 技术原理

PEAFOWL 协议的核心思想是通过**置换（Permutation）**和**秘密共享（Secret Sharing）**来实现安全对齐：

1. **实体对齐**：各方加密自己的样本ID并发送给服务器，服务器计算交集大小，但不知道具体交集ID
2. **特征组合**：各方基于交集样本，使用PEAFOWL协议组合各自的特征
3. **安全计算**：通过秘密共享和SHPRG（密钥同态伪随机生成器）混淆特征值，确保服务器无法获取原始数据

### 项目结构

```
peafowl-2/
├── core/                      # 基础密码学原语
│   ├── secret_sharing.py      # 加法秘密共享
│   ├── prf.py                 # 伪随机函数（基于HMAC-SHA256）
│   ├── shprg.py               # 密钥同态伪随机生成器（基于LWR）
│   ├── ot.py                  # 不经意传输扩展
│   └── polynomial.py          # 多项式运算（拉格朗日插值）
├── protocol/                  # PEAFOWL协议实现
│   ├── przs.py                # 伪随机零共享
│   └── peafowl.py             # 主协议
├── party/                     # 角色实现
│   ├── data_provider.py       # 数据提供方
│   └── server.py              # 云服务器
├── network/                   # 网络模拟
│   └── channel.py             # 通信通道
├── utils/                     # 工具函数
│   └── data_loader.py         # 数据加载
├── tests/                     # 测试文件
│   ├── test_core.py           # 核心模块测试
│   ├── test_protocol.py       # 协议测试
│   ├── test_end_to_end.py     # 端到端测试
│   ├── test_ml_comparison.py  # 机器学习对比测试
│   └── test_real_vertical_fl.py # 垂直联邦学习场景测试
├── develop.md                 # 开发文档
└── README.md                  # 项目说明
```

### 快速开始

#### 环境要求

- Python 3.8+
- TensorFlow (可选，用于CNN测试)
- scikit-learn
- numpy

#### 安装依赖

```bash
cd /home/yuank/peafowl-2
pip install -r requirements.txt
```

#### 运行测试

```bash
# 运行所有单元测试
python -m pytest tests/ -v

# 运行垂直联邦学习场景测试
python tests/test_real_vertical_fl.py

# 运行机器学习对比测试
python tests/test_ml_comparison.py
```

#### 使用示例

```python
from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server

# 配置
config = {
    'num_parties': 3,
    'num_samples': 1000,
    'num_features': 100,
    'secret_modulus': 2**64,
    'precision_bits': 16,
}

# 创建数据提供方
data_providers = []
for i in range(num_parties):
    dp = DataProvider(f"P{i}", config, sample_ids[i], features[i])
    dp.prf_key = b'0' * 16
    data_providers.append(dp)

# 创建服务器
server = Server("S", config)

# 运行PEAFOWL协议
peafowl = PEAFOWL(config)
aligned_features = peafowl.run_protocol(data_providers, server)

# 使用对齐后的特征进行训练
# ...
```

### 测试结果

#### 垂直联邦学习场景测试

| 参与方数 | 样本数 | 特征数 | 对齐后样本数 | 逻辑回归准确率 | CNN准确率 |
|---------|-------|--------|-------------|--------------|----------|
| 2       | [400, 350] | [150, 140] | 280 | 80.36% | 73.21% |
| 3       | [400, 350, 300] | [150, 140, 130] | 188 | 65.79% | 60.53% |
| 4       | [400, 350, 300, 250] | [150, 140, 130, 120] | 106 | 63.64% | 54.55% |

### 核心模块说明

#### 1. 秘密共享 (Secret Sharing)

使用加法秘密共享将数据拆分为多个份额，只有所有份额相加才能恢复原始数据。

```python
from core.secret_sharing import share, reconstruct

# 分享
shares = share(secret=42, n=3, modulus=2**64)

# 重建
reconstructed = reconstruct(shares, modulus=2**64)
```

#### 2. 伪随机函数 (PRF)

基于HMAC-SHA256实现，用于安全地加密样本ID。

```python
from core.prf import PRF

prf = PRF(key=b'0' * 16)
encrypted = prf.eval(b'sample_id')
```

#### 3. 密钥同态伪随机生成器 (SHPRG)

基于LWR（Learning With Errors）问题实现，支持密钥同态运算。

```python
from core.shprg import SHPRG

shprg = SHPRG(d=8, m=1024, q=2**128, p=2**64)
output = shprg.generate(seed=[1, 2, 3, 4, 5, 6, 7, 8])
```

### 许可证

本项目仅用于学术研究和教学目的。



## English

PEAFOWL is a privacy-preserving entity alignment protocol framework designed for vertical federated learning. It allows different data parties to securely find common samples (intersection) without exposing their respective data, and jointly train machine learning models based on the aligned data.

### Key Features

- **Privacy Preservation**: Uses cryptographic techniques including additive secret sharing, pseudo-random functions, and key-homomorphic pseudo-random generators to protect intersection information
- **Efficient Protocol**: Linear complexity O(n) for communication and computation, suitable for large-scale datasets
- **Multi-Party Support**: Supports entity alignment with multiple data parties
- **Flexible Models**: Supports various machine learning models including logistic regression, CNN, SVM, etc.

### Technical Principles

The core idea of the PEAFOWL protocol is to achieve secure alignment through **permutation** and **secret sharing**:

1. **Entity Alignment**: Each party encrypts their sample IDs and sends them to the server, which computes the intersection size without knowing the specific intersection IDs
2. **Feature Combination**: Based on the intersection samples, parties combine their features using the PEAFOWL protocol
3. **Secure Computation**: Features are obfuscated through secret sharing and SHPRG (Key-Homomorphic Pseudo-Random Generator) to ensure the server cannot access raw data

### Project Structure

```
peafowl-2/
├── core/                      # Core cryptographic primitives
│   ├── secret_sharing.py      # Additive secret sharing
│   ├── prf.py                 # Pseudo-random function (based on HMAC-SHA256)
│   ├── shprg.py               # Key-homomorphic PRG (based on LWR)
│   ├── ot.py                  # Oblivious transfer extension
│   └── polynomial.py          # Polynomial operations (Lagrange interpolation)
├── protocol/                  # PEAFOWL protocol implementation
│   ├── przs.py                # Pseudo-random zero sharing
│   └── peafowl.py             # Main protocol
├── party/                     # Party implementations
│   ├── data_provider.py       # Data provider party
│   └── server.py              # Cloud server
├── network/                   # Network simulation
│   └── channel.py             # Communication channel
├── utils/                     # Utility functions
│   └── data_loader.py         # Data loading
├── tests/                     # Test files
│   ├── test_core.py           # Core module tests
│   ├── test_protocol.py       # Protocol tests
│   ├── test_end_to_end.py     # End-to-end tests
│   ├── test_ml_comparison.py  # ML comparison tests
│   └── test_real_vertical_fl.py # Real vertical FL scenario tests
├── develop.md                 # Development documentation
└── README.md                  # Project README
```

### Quick Start

#### Requirements

- Python 3.8+
- TensorFlow (optional, for CNN testing)
- scikit-learn
- numpy

#### Installation

```bash
cd /home/yuank/peafowl-2
pip install -r requirements.txt
```

#### Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run vertical federated learning scenario tests
python tests/test_real_vertical_fl.py

# Run ML comparison tests
python tests/test_ml_comparison.py
```

#### Usage Example

```python
from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server

# Configuration
config = {
    'num_parties': 3,
    'num_samples': 1000,
    'num_features': 100,
    'secret_modulus': 2**64,
    'precision_bits': 16,
}

# Create data providers
data_providers = []
for i in range(num_parties):
    dp = DataProvider(f"P{i}", config, sample_ids[i], features[i])
    dp.prf_key = b'0' * 16
    data_providers.append(dp)

# Create server
server = Server("S", config)

# Run PEAFOWL protocol
peafowl = PEAFOWL(config)
aligned_features = peafowl.run_protocol(data_providers, server)

# Train model with aligned features
# ...
```

### Test Results

#### Vertical Federated Learning Scenario Tests

| Parties | Samples | Features | Aligned Samples | LR Accuracy | CNN Accuracy |
|---------|---------|----------|-----------------|-------------|--------------|
| 2       | [400, 350] | [150, 140] | 280 | 80.36% | 73.21% |
| 3       | [400, 350, 300] | [150, 140, 130] | 188 | 65.79% | 60.53% |
| 4       | [400, 350, 300, 250] | [150, 140, 130, 120] | 106 | 63.64% | 54.55% |

### Core Modules

#### 1. Secret Sharing

Additive secret sharing splits data into multiple shares, where all shares must be combined to reconstruct the original data.

```python
from core.secret_sharing import share, reconstruct

# Share
shares = share(secret=42, n=3, modulus=2**64)

# Reconstruct
reconstructed = reconstruct(shares, modulus=2**64)
```

#### 2. Pseudo-Random Function (PRF)

Based on HMAC-SHA256, used for securely encrypting sample IDs.

```python
from core.prf import PRF

prf = PRF(key=b'0' * 16)
encrypted = prf.eval(b'sample_id')
```

#### 3. Key-Homomorphic Pseudo-Random Generator (SHPRG)

Based on the LWR (Learning With Errors) problem, supporting key-homomorphic operations.

```python
from core.shprg import SHPRG

shprg = SHPRG(d=8, m=1024, q=2**128, p=2**64)
output = shprg.generate(seed=[1, 2, 3, 4, 5, 6, 7, 8])
```

### License

This project is for academic research and educational purposes only.


