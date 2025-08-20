# LeNet-5 (PyTorch)

一个用于在 FashionMNIST 上训练与评估 LeNet-5 的最小可复用模板，适合学习阶段快速上手、改网络和复用训练脚本。

## 特性
- 训练/验证循环分明，验证阶段使用 no_grad/inference_mode
- 自动保存验证集最佳权重到 `checkpoints/best_model.pth`
- 自动将训练曲线图保存到 `results/train_curves_YYYYMMDD_HHMMSS.png`
- 数据标准化、验证集不打乱、Windows 友好（DataLoader 在 `__main__` 内创建）

## 环境准备
- Python 3.8+
- 依赖安装（CPU 版本）：
```bash
pip install torch torchvision pandas matplotlib
```
- 如果使用 CUDA，请到 PyTorch 官网选择与你显卡/CUDA 匹配的安装命令。

可选（用于在 `model.py` 中打印结构摘要）：
```bash
pip install torchsummary
```

## 数据集
首次运行会自动下载 FashionMNIST 到 `./data`。数据预处理包含 `Resize(28)` 与标准化 `Normalize((0.2861,), (0.3530,))`。

## 目录结构
```text
LeNet/
  checkpoints/          # 训练后的最佳模型（自动创建）
  data/                 # 数据集缓存（自动下载）
  results/              # 训练曲线图（自动保存）
  dataset.py            # 数据集与 DataLoader
  model.py              # LeNet-5 定义
  train.py              # 训练与验证主脚本
  evaluate.py           # 测试集评估脚本
  plot.py               # 可扩展的绘图脚本
```

## 训练
```bash
python train.py
```
- 超参数在 `train.py` 顶部：
  - `EPOCHS`：训练轮数（默认 20，示例可能已改为 50）
  - `LEARNING_RATE`：学习率（默认 1e-3）
- 训练结束后：
  - 最佳权重保存到 `checkpoints/best_model.pth`
  - 训练/验证曲线保存到 `results/train_curves_*.png`

## 评估
```bash
python evaluate.py
```
脚本会从 `checkpoints/best_model.pth` 加载权重，并在测试集上计算准确率。权重加载使用 `weights_only=True`（新版本 PyTorch 支持），旧版本会自动回退。

## 常见问题
- 看不到图片或权重被提交到 Git？
  - `.gitignore` 默认忽略 `data/`、`results/`、`checkpoints/` 等产物。如需上传单个文件，可用例外规则或 `git add -f 路径` 强制添加。
- Windows 上 DataLoader 卡住/重复构造？
  - 本项目已将 DataLoader 创建放在 `if __name__ == "__main__":` 中，并将 `num_workers` 默认为 0，通常可避免问题。
- 验证集准确率低于训练集？
  - 正常现象（过拟合）。可考虑添加 Dropout / Weight Decay、数据增强、或学习率调度等。

## 网络结构（示意）

![image-20250816141620710](https://gitee.com/TChangQing/qing_images/raw/master/images/20250816141620802.png)

## 参数详解（示意）

![image-20250816141726046](https://gitee.com/TChangQing/qing_images/raw/master/images/20250816141726152.png)

## 参考
- `https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=1e1bff61f759a9c523c4deb0ae9612fc&p=44`