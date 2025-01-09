import time
import torch
import torch.nn as nn
import numpy as np
import math
from metrics import *
import torch.nn.functional as F
from torch.nn.functional import normalize
from dataprocessing import *
import matplotlib.pyplot as plt


class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    #-----特征层面--------#
    def forward_feature(self, h_i, h_j):

        #------------------对角线损失函数---------------------#
        # 计算特征的均值和标准差
        lambda_param = 1e-4 #MNist-usps 1e-4 BDGP 1e-3 Fashion 1e-4 COIL20 5e-3
        z1_mean = torch.mean(h_i,dim=0)
        z2_mean = torch.mean(h_j,dim=0)
        z1_std = torch.std(h_i,dim=0)
        z2_std = torch.std(h_j,dim=0)
        # 中心化和归一化特征
        z1_centered = h_i - z1_mean
        z2_centered = h_j - z2_mean
        z1_normalized = z1_centered / z1_std
        z2_normalized = z2_centered / z2_std
        # 计算相关矩阵
        c = torch.matmul(z1_normalized.T, z2_normalized) / h_i.size(0)

        # 计算损失函数
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c[~torch.eye(c.size(0), dtype=bool)].pow_(2).sum()
        loss = on_diag + lambda_param * off_diag

        return loss

    #------聚类标签层面----------#
    def forward_label(self, q_i, q_j, temperature_l, normalized=False):

        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + entropy

    # ------视图标签层面----------#
    def forward_label2(self, q_i, q_j):
        temperature_l= 0.5  #MNIST-usps 0.5
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i = torch.clamp(p_i, min=1e-8)
        ne_i = (p_i * torch.log(p_i)).sum()

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l
        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = torch.zeros(N, 1).to(positive_clusters.device)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        ne_i /= (N / 2)
        return loss + ne_i



    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
