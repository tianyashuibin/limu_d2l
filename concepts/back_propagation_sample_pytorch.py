import torch
import torch.nn as nn
import torch.optim as optim

# 训练数据：简单的 XOR 类问题
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 输入
y = torch.tensor([[0.], [1.], [1.], [0.]])  # 目标输出

# 定义一个简单的两层神经网络（2 -> 2 -> 1）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 第一层：2输入 -> 2隐藏层神经元
        self.fc2 = nn.Linear(2, 1)  # 第二层：2隐藏层神经元 -> 1输出

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 隐藏层 + ReLU 激活
        x = self.fc2(x)              # 输出层（线性）
        return x

# 实例化网络和优化器
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 使用 SGD 优化器
criterion = nn.MSELoss()  # 均方误差损失函数

# 训练 1000 次
for epoch in range(1000):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 后向传播：自动计算梯度
    optimizer.zero_grad()  # 清除之前的梯度
    loss.backward()        # 反向传播，计算当前梯度

    # 参数更新：使用优化器更新参数
    optimizer.step()

    # 每 100 次打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
