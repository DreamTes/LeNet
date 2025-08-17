from torch.utils.data import DataLoader, Dataset

class DataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)  # 返回数据集的长度

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# 划分训练集和验证集
