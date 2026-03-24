import numpy as np

print("=" * 60)
print("图像矩阵表示示例")
print("=" * 60)

print("\n1. 灰度图像 (单通道)")
print("-" * 40)
print("尺寸: 4×4 像素")
gray_image = np.array([
    [0, 50, 100, 150],
    [50, 100, 150, 200],
    [100, 150, 200, 255],
    [150, 200, 255, 255]
])
print(f"形状: {gray_image.shape}")
print("矩阵表示 (每个值代表亮度，0=黑, 255=白):")
print(gray_image)

print("\n\n2. RGB彩色图像 (三通道)")
print("-" * 40)
print("尺寸: 2×2 像素")
print("\n每个像素有3个值: [R, G, B]")

red_channel = np.array([
    [255, 0],
    [0, 255]
])
green_channel = np.array([
    [0, 255],
    [255, 0]
])
blue_channel = np.array([
    [0, 0],
    [255, 255]
])

rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
print(f"\n形状: {rgb_image.shape} (高, 宽, 通道数)")
print("\n彩色图像矩阵 (三维数组):")
for i in range(2):
    for j in range(2):
        print(f"  像素[{i},{j}]: R={rgb_image[i,j,0]}, G={rgb_image[i,j,1]}, B={rgb_image[i,j,2]}")

print("\n\n3. 展平成特征向量")
print("-" * 40)
gray_flat = gray_image.flatten()
print(f"灰度图像展平: {gray_image.shape} → {gray_flat.shape}")
print(f"特征数: {gray_flat.shape[0]}")

rgb_flat = rgb_image.reshape(-1)
print(f"RGB图像展平: {rgb_image.shape} → {rgb_flat.shape}")
print(f"特征数: {rgb_flat.shape[0]} (是灰度图像的3倍)")

print("\n\n4. MNIST数据集对比")
print("-" * 40)
print("MNIST灰度图: 28×28×1 = 784个特征")
print("彩色MNIST:   28×28×3 = 2352个特征 (×3倍)")
print("常见彩色图:  224×224×3 = 150,528个特征")

print("\n\n5. 手写数字1的矩阵表示")
print("-" * 40)
print("28×28像素的数字'1' (简化示例: 10×10):")
digit_1 = np.zeros((10, 10), dtype=int)
digit_1[:, 4:6] = 255
print(digit_1)
print("说明: 中间两列设为255(白色)，形成数字'1'的形状")

print("\n\n6. 颜色值含义")
print("-" * 40)
print("0   = 完全黑色 (无亮度/无颜色)")
print("128  = 中等灰度 (50%亮度)")
print("255  = 完全白色 (最大亮度/最大颜色强度)")
print("\nRGB组合示例:")
print("  [255, 0, 0]   = 红色")
print("  [0, 255, 0]   = 绿色")
print("  [0, 0, 255]   = 蓝色")
print("  [255, 255, 255] = 白色")
print("  [0, 0, 0]     = 黑色")
print("  [255, 255, 0] = 黄色 (红+绿)")

print("\n" + "=" * 60)
