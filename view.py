import matplotlib.pyplot as plt
import random
import numpy as np
def plot_sample_images(dataset, num_samples=20, num_columns=2):
    num_rows = (num_samples + num_columns - 1) // num_columns  # 计算所需的行数
    plt.figure(figsize=(5 * num_columns, 5 * num_rows))
    count = 0
    i = 0
    while count < num_samples and i < len(dataset):
        image, label = dataset[i]
        if label == 0:  # 仅当标签为0时才显示图像
            plt.subplot(num_rows, num_columns, count + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            # plt.title(f'Label: {label}')
            plt.axis('off')
            count += 1
        i += 1
    plt.show()
# 例如，对于MNIST数据集：
from torchvision import datasets, transforms

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
plot_sample_images(mnist_dataset)


