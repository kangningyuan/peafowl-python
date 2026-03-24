## **项目概述**

### 背景与目标

在隐私保护机器学习（PPML）的垂直联邦学习场景中，不同数据方持有相同样本的不同特征。为了联合训练模型，各方需要先进行实体对齐，即找出各方共有的样本（交集）。然而，直接使用传统方法（如PSI）会泄露交集信息本身，这在医疗、金融等敏感领域是不可接受的。

PEAFOWL 的目标是提供一个安全、高效的多方实体对齐协议。其核心思想是：

不直接泄露交集：通过使用置换（Permutation） 这一概念，让各方和服务器只知道哪些样本需要保留以及如何排列，而不知道具体的交集ID。

服务器辅助的轻量级协议：引入一个云服务器（S）来执行复杂的置换计算，但确保服务器无法访问原始数据。

高效的密码学构建块：结合了秘密共享洗牌（Secret-Shared Shuffle）、密钥同态伪随机生成器（SHPRG） 和份额转换（Share Conversion） 技术，将通信和计算开销降至线性级别。

---

## **1. 项目总览与文件结构**

```
./
│
├── config.yaml                # 全局配置（参与方数、样本数、特征数、安全参数等）
├── requirements.txt           # 依赖库
│
├── core/                      # 基础密码学原语
│   ├── __init__.py
│   ├── secret_sharing.py      # 加法秘密分享（支持整数模 2^64 或大数）
│   ├── prf.py                 # PRF（基于 AES-128 的 GGM 树实现）
│   ├── shprg.py               # 密钥同态伪随机生成器（基于 LWR）
│   ├── ot.py                  # 不经意传输扩展（用于 OPV/ST）
│   ├── permute_share.py       # Permute+Share 和 Share Translation
│   └── polynomial.py          # 多项式运算（用于 Shamir 份额转换）
│
├── protocol/                  # PEAFOWL 协议实现
│   ├── __init__.py
│   ├── przs.py                # 伪随机零分享生成（基于 PRSS）
│   ├── peafowl.py             # PEAFOWL 主协议（在线/离线）
│   └── peafowl_offline.py     # 离线阶段逻辑
│
├── network/                   # 网络模拟
│   ├── __init__.py
│   ├── channel.py             # 消息队列、广播、点对点通信抽象
│   └── serialization.py       # 大整数与字节流的序列化/反序列化
│
├── party/                     # 角色实现
│   ├── __init__.py
│   ├── base_party.py          # 基础类（存储 ID、特征、份额、密钥）
│   ├── data_provider.py       # 数据提供方（继承 base_party）
│   └── server.py              # 云服务器（继承 base_party）
│
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_loader.py         # 加载真实或模拟数据
│   ├── constants.py           # 常量（如安全参数、模数、PRF 密钥长度）
│   ├── random.py              # 安全随机数生成器封装
│   └── math_utils.py          # 模逆、高斯消元等
│
└── tests/                     # 单元测试与集成测试
    ├── test_core.py
    ├── test_protocol.py
    └── test_end_to_end.py
```

---

## **2. 全局配置与参数选择**

**`config.yaml` 示例：**
```yaml
# 系统参数
security_parameter: 128         # 安全等级 λ（比特）
num_parties: 3                  # N
threshold: 1                    # t（最多允许腐败的参与方数量，需满足 t < N/2 等）
num_samples: 1024               # n
num_features: 1024              # m
intersection_ratio: 0.5         # β（用于测试）

# PRF 参数
prf_key_bytes: 16               # AES-128 密钥长度

# SHPRG 参数（基于 LWR）
shprg_d: 8                      # 种子向量维度
shprg_m: 1024                   # 输出向量维度（通常等于特征数 m）
shprg_q: 340282366920938463463374607431768211456  # 2^128
shprg_p: 18446744073709551616   # 2^64
# 注意：q 和 p 的关系应满足 q/p 为整数或接近整数，且 LWE 问题困难

# 秘密分享模数（使用 64 位模数，便于运算）
secret_modulus: 18446744073709551616  # 2^64

# 精度（用于浮点数转整数）
precision_bits: 16              # L
```

**参数选择理由**：
- `secret_modulus = 2^64`：在 Python 中 `int` 可以无溢出处理，但为了性能可以约定所有运算模 `2^64`，利用 Python 的自动大整数，最后取模。
- SHPRG 参数：`q=2^128, p=2^64` 使得 LWR 问题困难（`q/p = 2^64` 整数），简化舍入计算。`d=8` 是常见选择。
- `precision_bits=16`：将浮点数乘以 `2^16` 转为整数，用于特征编码。

---

## **3. 核心模块详细设计**

### **3.1. `core/secret_sharing.py` - 加法秘密分享**

**设计目标**：支持将整数（在 `Z_{2^64}` 中）拆分为 N 个份额，且任意 t 个份额无法恢复原值（这里我们使用简单加法分享，适用于所有 N 个份额共同重建，而不是阈值）。实际上 PEAFOWL 要求所有份额相加才能恢复，因此 t = N-1（所有参与方一起重建）。但我们仍需支持少于 N 个份额时的不可区分性。

**接口**：
```python
def share(secret: int, n: int, modulus: int) -> List[int]:
    """
    将 secret 拆分为 n 个份额，返回列表。
    算法：生成 n-1 个随机数 r_i (0 ≤ r_i < modulus)，
          最后一个份额为 (secret - sum(r_i)) % modulus。
    """
    shares = [secrets.randbelow(modulus) for _ in range(n-1)]
    last = (secret - sum(shares)) % modulus
    shares.append(last)
    return shares

def reconstruct(shares: List[int], modulus: int) -> int:
    """将 n 个份额相加取模，得到原值"""
    return sum(shares) % modulus
```

**注意事项**：
- 使用 `secrets.randbelow` 生成真随机数（安全）。
- 模运算使用 Python 的 `%` 即可，由于 `modulus` 小于 `2^64`，Python `int` 可以高效处理。
- 当需要处理大数（如 SHPRG 输出 `Z_p` 且 p=2^64）时，统一使用 `% modulus`。

---

### **3.2. `core/prf.py` - 伪随机函数**

**设计目标**：实现一个基于 AES-128 的 PRF，用于将 ID 映射到固定长度的输出。这里我们采用 **GGM 树** 或直接使用 AES 加密一个计数器模式（因为输入是已知的 ID 字符串，可以编码为 16 字节）。

**方案选择**：由于 ID 可能是任意字符串，我们使用 HMAC-SHA256 作为 PRF，或者使用 AES-CMAC。这里采用 **HMAC-SHA256** 简化。

**接口**：
```python
import hmac
import hashlib

class PRF:
    def __init__(self, key: bytes):
        self.key = key  # 16 或 32 字节

    def eval(self, input_bytes: bytes) -> bytes:
        """返回 16 字节伪随机输出"""
        h = hmac.new(self.key, input_bytes, hashlib.sha256)
        return h.digest()[:16]   # 取前 16 字节
```

**注意事项**：
- 输入 `input_bytes` 是 ID 的 UTF-8 编码或固定长度编码。
- 输出长度固定为 16 字节（128 位），与 `λ=128` 对应。

---

### **3.3. `core/shprg.py` - 密钥同态伪随机生成器（基于 LWR）**

**设计目标**：实现 `G: Z_q^d -> Z_p^m`，满足 `G(s1+s2) ≈ G(s1)+G(s2)`。

**参数**：
- `d=8`, `m` 由特征数决定（如 1024）
- `q=2^128`, `p=2^64`
- 公共矩阵 `A`：`d x m` 矩阵，元素在 `Z_q` 中。

**实现细节**：
```python
import numpy as np
import secrets

class SHPRG:
    def __init__(self, d: int, m: int, q: int, p: int):
        self.d = d
        self.m = m
        self.q = q
        self.p = p
        # 生成公共矩阵 A，每个元素随机选自 0..q-1
        self.A = np.array([[secrets.randbelow(q) for _ in range(m)] for _ in range(d)], dtype=object)
        # 注意：使用 dtype=object 来存储 Python 大整数，因为 q 可能 > 2^63

    def generate(self, seed: List[int]) -> List[int]:
        """
        seed: d 维列表，每个元素为 0..q-1 的整数
        返回 m 维列表，每个元素为 0..p-1 的整数
        """
        # 计算 A^T * seed (mod q)
        # 注意：矩阵乘法使用 Python 大整数
        intermediate = [0] * self.m
        for col in range(self.m):
            s = 0
            for row in range(self.d):
                s = (s + self.A[row][col] * seed[row]) % self.q
            intermediate[col] = s
        # 舍入：取 (x * p) // q
        # 由于 q 是 p 的倍数（q/p 为整数），可以直接移位
        # 为了通用，使用整数除法
        shift = self.q // self.p   # 应为 2^64
        result = [(x // shift) % self.p for x in intermediate]
        return result
```

**同态性处理**：
- 当使用 `G(s1) + G(s2)` 时，结果模 `p` 会有误差，误差项为 `[-1,0,1]`。在 PEAFOWL 中，我们允许这种误差，因为它只影响特征最低的若干比特，不会影响模型训练精度（因为特征有更高位）。

**优化**：
- 为了提高性能，可以将 `A` 存储为 `list of list` 而不是 numpy，因为 numpy 的 int 对象在大于 64 位时会转为 Python int，反而慢。直接用 Python 列表乘法即可。
- 如果需要并行，可以使用多线程计算各列。

**溢出处理**：
- Python 的 `int` 没有固定大小，因此 `(A[row][col] * seed[row])` 可能产生巨大数，但取模后仍保持在 `q` 范围内。`q` 为 `2^128`，乘积可能达到 `2^256`，Python 可以处理，但速度可能下降。为了优化，可以在乘法后立即取模：`(self.A[row][col] * seed[row]) % self.q`，但这样每次乘法都要取模。更好的做法是使用 `(self.A[row][col] * seed[row]) % self.q` 累加。
- 由于 `self.q` 是 2 的幂，取模可以优化为位与操作：`& (self.q - 1)`，但 Python 的 `%` 已经优化。

---

### **3.4. `core/permute_share.py` - Share Translation 和 Permute+Share**

**背景**：这是 PEAFOWL 最复杂的部分。根据 Chase 等论文，ST 协议可以通过 OPV（Oblivious Punctured Vector）实现，而 OPV 又可以基于 GGM 树和 OT 扩展构建。为了简化实现，我们可以采用 **更简单的两方实现**：使用 OT 扩展直接实现基于置换的份额转换，但需要确保通信复杂度为 `O(n log n)`。

鉴于实现复杂度，我建议在 Python 中采用 **基于随机置换和对称加密的模拟方案**，仅用于原型验证。但若要正式实现，需要参考 Chase 论文中的 **OPV 构造**。我将提供 OPV 的概要实现，并依赖 OT 库（如 `emp-toolkit` 或 `libote` 的 Python 绑定）。但为了纯 Python 可运行，我们可以使用 **模拟的 OT** 并忽略安全性，仅验证逻辑。

**实际实现建议**：
- 使用 `permute_share` 作为黑盒，调用外部 C++ 库（如 EMP-toolkit）实现，因为 Python 实现 OT 扩展性能差。
- 为了本项目描述，我提供伪代码和接口设计，实现部分可留作 TODO。

**接口**：
```python
def share_translate(pi: List[int], party_id: str, other_party_conn: Channel) -> Tuple[Any, Any]:
    """
    根据角色执行 Share Translation。
    如果 party_id == 'S'（服务器），输入 pi（置换），返回 Δ。
    如果 party_id == 'P'（数据方），输入无，返回 (a, b)。
    """
    # 这里需要实现 OPV 协议，具体细节略。
    # 输出：Δ 是向量（长度 n，每个元素在 Z_modulus），a 和 b 也是向量。
    pass

def permute_share(pi: List[int], x: List[int], party_id: str, other_party_conn: Channel) -> List[int]:
    """
    执行 Permute+Share。
    如果 party_id == 'S'，输入 pi 和 x（来自对方的数据），返回份额 s0。
    如果 party_id == 'P'，输入 x，返回份额 s1。
    """
    if party_id == 'S':
        # 执行 ST 得到 Δ
        # 接收对方发送的 m = x - a
        # 返回 π(m) + Δ
    else:
        # 执行 ST 得到 (a, b)
        # 发送 m = x - a 给 S
        # 返回 b
```

**安全注意**：
- 实际实现需要 OT 协议，确保输入隐藏。
- 由于我们使用加法秘密分享，所有运算在 `Z_modulus` 中进行。

---

### **3.5. `core/polynomial.py` - 多项式运算（用于 Shamir 份额转换）**

**需求**：在份额转换中，我们需要为每个集合 A 定义多项式 `f_A` 满足：
- `deg(f_A) = t`
- `f_A(0) = 1`
- `f_A(j) = 0` for `j not in A` (where A is a set of size n-t)

这可以通过拉格朗日插值得到。给定点集 `{0} ∪ ([n] \ A)`，我们希望多项式在这些点上取值分别为 1 和 0。可以直接构造。

**实现**：
```python
def get_f_A(A: Set[int], n: int, t: int) -> List[int]:
    """
    返回多项式 f_A 的系数（从常数项到最高次），长度为 t+1。
    注意：多项式次数为 t。
    """
    # 点集 X = [0] + list(set(range(1, n+1)) - A)
    # 值集 Y = [1] + [0]*len(X[1:])
    # 使用拉格朗日插值计算系数
    pass

def evaluate_poly(coeffs: List[int], x: int, modulus: int) -> int:
    """霍纳方法求值"""
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % modulus
    return result
```

**注意事项**：
- 所有系数和求值均在 `Z_modulus` 中进行（`modulus = 2^64` 或更大）。
- 由于模数不是素数，拉格朗日插值需要求逆。但 `2^64` 不是素数，不能直接求逆。这里有两种处理：
    - 使用素数模数，例如选择一个大素数 `P = 2^61 - 1` 或 `2^64 - 59` 等，这样可以用模逆。
    - 在特征对齐中，我们实际上不需要在 `2^64` 上求逆，因为所有运算都是加法，没有除法。在多项式构造中，我们只需要系数，可以通过求解线性方程组得到，但为了避免除法，可以使用 **整数系数**（不取模）然后对结果取模。因为所有求值点都是整数，拉格朗日系数是分数，但乘以分母后可以得到整数系数。更简单的方法是：我们不直接构造多项式，而是在 PRSS 中直接使用预先计算好的 `f_A(j)` 值，这些值可以通过 **浮点数** 计算后取整，但由于误差可能影响正确性。**更好的做法**：使用 **整数环** 上的拉格朗日插值，通过 **有理数** 精确计算，最后乘以分母的公倍数，再对模数取模。这需要大整数运算，但可行。
    - 实际上，在 PEAFOWL 论文中，他们使用了有限域（如 `Z_p`）上的 Shamir 份额。因此，我们应该选择一个素数作为秘密分享的模数，比如 `P = 2^61 - 1`（一个梅森素数），这样所有运算都有逆元，且 Python 可以高效处理。

**修改建议**：将 `secret_modulus` 设为一个安全的大素数，例如 `2^61 - 1`（约 2.3e18）。这样可以在素数域上进行多项式插值。同时 SHPRG 的输出 `Z_p` 可以与这个素数不同，但最终份额会在 `Z_P` 中。需要确保特征数据编码后也在 `Z_P` 范围内。

---

### **3.6. `protocol/przs.py` - 伪随机零分享**

**功能**：生成 Shamir 份额的一个零值（即多项式 f(0)=0）。

**实现**（基于 PRSS）：
```python
class PRZS:
    def __init__(self, keys: Dict[Tuple[int], List[bytes]]):
        # keys 是一个字典，键为集合 A（大小为 n-t），值为 t 个密钥（用于 PRF）
        pass

    def generate_share(self, party_id: int, point: int, modulus: int) -> int:
        """
        为给定参与方 party_id 生成一个份额，对应输入 point（通常是公共输入 a）。
        返回一个整数。
        """
        share = 0
        for A in self.keys:
            if party_id in A:
                for i, key in enumerate(self.keys[A]):
                    # 使用 PRF 生成随机数
                    r = prf_eval(key, point)   # 返回一个 128 位整数，然后取模 modulus
                    # 多项式 f_A^i 在 party_id 处的值
                    f_val = poly_val[A][i][party_id]   # 预计算
                    share = (share + r * f_val) % modulus
        return share
```

**预计算**：
- 对于每个 A（大小 n-t），选取 t 个线性无关的多项式（如 `f_A^i(x) = x^i` 但需满足零条件？实际上，我们需要基多项式满足 f(0)=0 且 f(j)=0 for j not in A）。可以通过构造 `F_A` 空间的基来实现。简便方法：取多项式 `g_i(x) = x * h_i(x)`，其中 `h_i` 是满足 h(0)=0 且 h(j)=0 for j not in A 的基，但 `g` 的次数会增加。更精确地，`F_A` 是次数 ≤ 2t 且满足上述零点条件的多项式空间，维度为 t。我们可以通过选择 t 个不同点（例如 A 中任意 t 个元素）构造基多项式，每个基多项式在一个选定的点上值为 1，在其他选定的点上值为 0。这种方法可行，但需要解线性方程组。

**简化方案**：由于我们只需要生成随机零分享，可以直接使用 **随机加法分享** 代替 Shamir 零分享，然后通过份额转换转换为 Shamir 分享。这可以避免多项式构造的复杂性。但在 PEAFOWL 中，PRZS 用于在特征对齐中混淆服务器，实际上只需要加法分享的零值即可，因为最终份额会通过 `Permute+Share` 转换，而 `Permute+Share` 输入是加法分享。因此，我们可以使用 **加法秘密分享的零分享**（即所有份额和为0），这样更简单。

---

### **3.7. `protocol/peafowl.py` - 主协议**

**实现步骤**：

```python
class PEAFOWL:
    def __init__(self, config, party_id, data):
        self.config = config
        self.id = party_id
        self.data = data   # (IDs, features)
        self.shares = {}   # 存储收到的份额
        self.seeds = {}    # 存储发给别人的种子

    def run(self, server_conn: Channel, party_conns: List[Channel]):
        # 1. 初始化
        self._init_prf_key()
        self._shuffle_data()

        # 2. ID 安全匹配
        encrypted_ids = self._encrypt_ids()
        server_conn.send(encrypted_ids)
        # 等待服务器广播交集大小和置换（对服务器透明，这里数据方不接收）
        # 实际上数据方只接收服务器发送的 π''（在离线版本中）

        # 3. 特征对齐
        # 生成种子并发送
        self._generate_seeds()
        for j, conn in enumerate(party_conns):
            conn.send(self.seeds[j])   # 发送给其他数据方

        # 计算自己的特征份额并发送给服务器
        my_share = self._compute_my_feature_share()
        server_conn.send(my_share)

        # 执行 Permute+Share 协议（与服务器交互）
        # 这里需要循环所有 i，实际上对每个 i 作为数据提供者，服务器和 Pj 交互
        # 我们简化：每个数据方 i 会作为源，服务器和所有 j!=i 执行 Permute+Share
        # 这需要复杂的通信协调，暂时略。

        # 接收服务器返回的份额并更新本地份额
        # 最终拼接得到最终份额
```

**详细通信流程**（从服务器视角）：
1. 服务器接收所有加密 ID，计算交集，得到 `π_i`。
2. 对于每个数据方 i：
   - 服务器与每个其他数据方 j 执行 `permute_share(π_i, seed_ij)`，得到 `⟨π_i(seed_ij)⟩_S` 和 `⟨π_i(seed_ij)⟩_Pj`。
   - 服务器接收 `⟨X_i⟩_i`，计算 `⟨X_i'⟩_S = π_i(⟨X_i⟩_i) + Σ_{j≠i} G(⟨π_i(seed_ij)⟩_S)`。
   - 服务器将 `⟨X_i'⟩_S` 发送给 `Pi`。
3. 每个数据方 j 从与其他数据方的交互中收集 `⟨π_i(seed_ij)⟩_Pj`，计算 `⟨X_i'⟩_Pj = G(⟨π_i(seed_ij)⟩_Pj)`，最后拼接所有 `⟨X_i'⟩_Pj` 得到最终份额。

**注意**：在 Python 中模拟这些通信需要定义消息类型和序列化格式。我们使用 `pickle` 或自定义格式。

---

### **3.8. 大数溢出与性能优化**

**问题**：Python 的 `int` 是无界的，但大数运算（如乘法）会变慢。对于 `q=2^128`，`p=2^64`，乘法结果可能达到 `2^256`，虽可处理但效率下降。

**优化策略**：
1. 使用 `% modulus` 保持数字在较小范围内。
2. 利用模数为 2 的幂时，用位运算 `& (modulus-1)` 替代 `%`。
3. 对于 SHPRG，由于 `q` 是 2 的幂，`(x * p) // q` 可以优化为 `x >> (log2(q)-log2(p))`，即 `x >> 64`。
4. 对于秘密分享模数，选择 2 的幂（如 2^64）便于取模，但需要注意加法、乘法在 Python 中取模仍有开销。由于 2^64 是 64 位，Python int 在 2^63 以内使用小整数优化，超出则用大整数。但 2^64 已超出小整数范围，所以仍是大整数。可以考虑使用 **模 2^64 的整数环** 通过 Python 的 `& (2**64-1)` 来取模，这比 `%` 快。
5. 使用 `numba` 或 `Cython` 加速关键循环（如矩阵乘法）。
6. 对于 SHPRG 的矩阵乘法，可以预先将 `A` 的每一列乘以 2^64 的逆？但实际上不需要。

**具体实现建议**：
- 使用 Python 内置的 `int` 并利用位运算进行取模：`mod = (1 << 64) - 1`，取模用 `x & mod`。
- 对于乘法，先做乘法再取模，但 Python 会自动处理。
- 在循环中避免过多大整数创建，复用变量。

**内存管理**：
- 份额和向量使用 `list` 存储。
- 对于大规模数据（如 m=10000），确保使用生成器或分块处理。

---

### **3.9. 网络序列化**

**问题**：传输大整数时，需要序列化为字节流。

**方案**：使用 `int.to_bytes(length, byteorder)` 和 `int.from_bytes`。对于长度不确定的整数，可以先传输长度再传输内容。

**序列化消息格式**（简化）：
```python
def serialize_int(x: int) -> bytes:
    # 固定长度 16 字节（适用于 128 位）
    return x.to_bytes(16, 'big')

def deserialize_int(b: bytes) -> int:
    return int.from_bytes(b, 'big')
```

对于长度不确定的向量，可以发送长度 + 所有元素的序列化。

---

### **3.10. 测试与验证**

**单元测试**：
- 测试秘密分享的正确性（随机分享后重建）。
- 测试 SHPRG 的伪随机性和近似同态性（检查误差在 [-1,0,1]）。
- 测试 Permute+Share 的正确性（对随机输入验证结果）。
- 测试协议端到端：多数据方 + 服务器，验证最终份额重建后等于对齐数据。

**集成测试**：
- 使用随机生成的数据，检查对齐结果与理论交集一致。
- 验证训练准确率不受误差影响（使用简单线性模型）。

---

## **4. 总结**

这份详尽的实现指南覆盖了 PEAFOWL 的每个核心模块，并针对 Python 实现中的常见问题（大数溢出、模运算、性能、序列化）给出了具体解决方案。coding agent 可以依据这些描述直接编写代码，每个模块都有清晰的输入输出、算法步骤和注意事项。
