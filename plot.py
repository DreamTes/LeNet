from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

train_dataset = FashionMNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# 获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step == 0:  # 只获取第一个batch的数据
        break
batch_x = b_x.squeeze().numpy()  # 将数据转换为numpy数组
batch_y = b_y.numpy()  # 将标签转换为numpy数组

class_label = train_dataset.classes
print(class_label)
print(batch_x.shape)  # 输出数据的形状

# 可视化一个batch
plt.figure(figsize=(12, 5))
for i in np.arange(len(batch_y)):  # 显示64张图像
    plt.subplot(4, 16, i + 1)  # 8行8列
    plt.imshow(batch_x[i,:,:], cmap='gray')  # squeeze去掉单通道的维度
    plt.title(class_label[batch_y[i]], size=10)  # 显示标签
    plt.axis('off')  # 不显示坐标轴
    plt.subplots_adjust(wspace=0.5)  # 调整子图间距
plt.show()


