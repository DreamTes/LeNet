from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data


def train_val_process():
    train_dataset = FashionMNIST(root='./data',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]))

    train_data, val_data = data.random_split(train_dataset, [round(0.8*len(train_dataset)), round(0.2*len(train_dataset))])  # 划分训练集和验证集

    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)
    val_loader = data.DataLoader(dataset=val_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)
    return train_loader, val_loader
