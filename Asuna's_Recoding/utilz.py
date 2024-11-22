import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics


class Utils:
    """提供数据处理和转换的工具函数"""

    @staticmethod
    def to_numpy(tensor_or_array):
        """将Tensor或ndarray转换为numpy数组"""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.cpu().detach().numpy()
        return tensor_or_array

    @staticmethod
    def ensure_shape(data, target_dim):
        """确保数据具有目标形状"""
        if data.shape[0] == target_dim:
            return data.transpose()
        return data


class PointCloudVisualizer:
    """点云可视化工具"""

    @staticmethod
    def show_pointcloud(pc, size=10):
        pc = Utils.to_numpy(pc)
        pc = Utils.ensure_shape(pc, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=size)
        plt.show()

    @staticmethod
    def show_pointcloud_batch(pc_batch, size=10):
        pc_batch = Utils.to_numpy(pc_batch)
        pc_batch = Utils.ensure_shape(pc_batch, 3)
        B, N, C = pc_batch.shape
        fig = plt.figure()
        for i in range(B):
            ax = fig.add_subplot(2, int(B / 2), i + 1, projection='3d')
            ax.scatter(pc_batch[i, :, 0], pc_batch[i, :, 1], pc_batch[i, :, 2], s=size)
        plt.show()

    @staticmethod
    def show_pointcloud_colored(pc, colors, size=10):
        pc = Utils.to_numpy(pc)
        colors = Utils.to_numpy(colors).squeeze()
        pc = Utils.ensure_shape(pc, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=size, alpha=0.5)
        plt.colorbar(sc, ax=ax)
        plt.show()


class MetricCalculator:
    """性能指标计算工具"""

    @staticmethod
    def calculate_auc(label, pred, pos_label=1):
        label, pred = Utils.to_numpy(label), Utils.to_numpy(pred)
        fpr, tpr, _ = metrics.roc_curve(label, pred, pos_label=pos_label)
        auc_score = metrics.auc(fpr, tpr)
        return auc_score

    @staticmethod
    def calculate_accuracy(label, pred, threshold=0.5):
        label, pred = Utils.to_numpy(label), Utils.to_numpy(pred)
        pred_binary = (pred > threshold).astype(int)
        accuracy = np.mean(pred_binary == label)
        return accuracy


class Logger:
    """记录模型参数的工具"""

    def __init__(self, model):
        self.model = model
        self.gradients = {name: [] for name, param in model.named_parameters() if param.requires_grad}
        self.weights = {name: [param.data.abs().mean().item()] for name, param in model.named_parameters()}

    def log_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.gradients:
                self.gradients[name].append(param.grad.abs().mean().item())

    def log_weights(self):
        for name, param in self.model.named_parameters():
            if name in self.weights:
                self.weights[name].append(param.data.abs().mean().item())

    def plot(self, data_dict, title):
        fig, ax = plt.subplots()
        for name, values in data_dict.items():
            ax.plot(values, label=name)
        ax.set_title(title)
        ax.legend()
        plt.show()


# 示例使用
if __name__ == "__main__":
    # 点云可视化
    pc = torch.rand(1000, 3)
    PointCloudVisualizer.show_pointcloud(pc)

    # 计算AUC
    labels = torch.randint(0, 2, (100,))
    predictions = torch.rand(100)
    auc_score = MetricCalculator.calculate_auc(labels, predictions)
    print(f"AUC: {auc_score:.2f}")