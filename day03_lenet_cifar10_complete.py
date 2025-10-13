
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os

# 1. 选择设备（GPU / CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('当前设备:', device)

# 2. 数据预处理：归一化到 [-1, 1]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. 下载并加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

classes = trainset.classes  # 10 类标签名

# 4. 定义网络（与官方教程一致）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)        # 输入 3 通道，输出 6 通道，卷积 5×5
        self.pool = nn.MaxPool2d(2, 2)         # 2×2 最大池化
        self.conv2 = nn.Conv2d(6, 16, 5)       # 输入 6 通道，输出 16 通道，卷积 5×5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 展平后全连接
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 最后输出 10 类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 卷积 → ReLU → 池化
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                # 展平成一维向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # 输出 logits（无 softmax）
        return x

# 5. 实例化网络并搬到 GPU/CPU
net = Net().to(device)

# 6. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()                           # 交叉熵损失（内部含 softmax）
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 7. 创建 TensorBoard 记录器
writer = SummaryWriter('runs/cifar10-blitz-1.4')

# 8. 计算整个测试集准确率的辅助函数
def test_accuracy():
    net.eval()                                    # 切换为评估模式（Dropout/BN 行为不同）
    correct = total = 0
    with torch.no_grad():                         # 关闭梯度计算，加速推理
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 取最大值的索引作为预测类别
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 9. 训练循环
EPOCH = 10
start_time = time.time()

for epoch in range(EPOCH):
    net.train()                                   # 切换回训练模式
    running_loss = 0.0                            # 累计 loss
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        #训练标配
        optimizer.zero_grad()                     # 清空过往梯度（PyTorch 默认累加）
        outputs = net(inputs)                     # 前向传播
        loss = criterion(outputs, labels)         # 计算损失
        loss.backward()                           # 反向传播求梯度
        optimizer.step()                          # 更新权重

        running_loss += loss.item()

        # 每 100 个 mini-batch 打印一次并写 TensorBoard
        if i % 100 == 99:
            avg_loss = running_loss / 100
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.3f}')
            # 记录训练 loss（x 轴为全局步数）
            global_step = epoch * len(trainloader) + i
            writer.add_scalar('training loss', avg_loss, global_step)
            running_loss = 0.0

    # 每个 epoch 结束后测试准确率并画曲线
    acc = test_accuracy()
    print(f'>>> Epoch {epoch + 1} test accuracy: {acc:.2f} %')
    writer.add_scalar('test accuracy', acc, epoch + 1)

    # 10. 保存断点（可随时恢复继续训练）
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')

# 11. 保存最终权重（供推理或继续微调）
final_path = './cifar_net.pth'
torch.save(net.state_dict(), final_path)
print('最终权重已保存至', final_path)

total_time = time.time() - start_time
print(f'训练完成！总耗时 {total_time // 60:.0f} 分 {total_time % 60:.0f} 秒')
writer.close()
