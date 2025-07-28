import numpy as np

# 激活函数和导数
def relu(x):
    return np.maximum(0, x)

# ReLU 的导数：ReLU 在正区间导数为 1，负区间为 0（用于反向传播时链式求导）。
def relu_derivative(x):
    return (x > 0).astype(float)

# 初始化参数
np.random.seed(42)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

# 训练数据：简单的 XOR 类问题
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 学习率
lr = 0.1

# 训练 1000 次
for epoch in range(1000):
    # === 前向传播 ===
    z1 = X.dot(W1) + b1         # 输入 -> 隐藏层线性
    a1 = relu(z1)               # ReLU 激活
    z2 = a1.dot(W2) + b2        # 隐藏层 -> 输出层
    y_pred = z2                 # 线性输出

    # === 损失 ===
    loss = np.mean((y - y_pred) ** 2)

    # === 反向传播 ===
    dL_dy = 2 * (y_pred - y) / y.shape[0]    # 对输出求导

    dL_dW2 = a1.T.dot(dL_dy)                 # 隐藏层输出对W2的导数
    dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)

    da1 = dL_dy.dot(W2.T)                    # 链式法则传播回去
    dz1 = da1 * relu_derivative(z1)          # ReLU 的导数

    dL_dW1 = X.T.dot(dz1)
    dL_db1 = np.sum(dz1, axis=0, keepdims=True)

    # === 参数更新 ===
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    # 每 100 次打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")
