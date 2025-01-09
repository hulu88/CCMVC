import torch.nn as nn
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import normalize

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())
        # self.residual = nn.Linear(input_dim, feature_dim)
    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


def kmeans(X, num_clusters, num_iterations):
    device = X.device
    indices = torch.randperm(X.shape[0], device=device)[:num_clusters]
    centroids = X[indices]
    centroids = centroids.to(device)
    for _ in range(num_iterations):
        distances = torch.cdist(X, centroids)
        cluster_indices = torch.argmin(distances, dim=1)
        for i in range(num_clusters):
            mask = cluster_indices == i
            if torch.any(mask):
                centroids[i] = X[mask].mean(dim=0)

    return centroids, cluster_indices



class CCMVCNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(CCMVCNetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.classnum =num_clusters
        self.feature_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature,1024), #MNIST_USPS 2048  Fashion BDGP 256 COIL128 hand 1024
        )
        self.label_learning_module = nn.Sequential(  # soft软概率聚类
            nn.Linear(dim_high_feature, num_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, data_views):
        lbps = list()
        dvs = list()
        features = list()

        num_views = len(data_views)
        for idx in range(num_views):
            data_view = data_views[idx]
            high_features = self.encoders[idx](data_view)
            features_low = self.feature_module(high_features)
            features_norm = normalize(features_low,dim=1)
            label_probs = self.label_learning_module(high_features) # soft软聚类
            data_view_recon = self.decoders[idx](high_features)
            features.append(features_norm)
            lbps.append(label_probs)
            dvs.append(data_view_recon)
        return lbps, dvs, features

    def forward_plot1(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        pres = []
        num_views=len(xs)
        for v in range(num_views):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.label_learning_module(z)
            predicted_labels = torch.argmax(h, dim=1)

            unique_labels = torch.unique(predicted_labels)

            k = unique_labels.size(0)
            w = torch.matmul(h, h.T)

            d = torch.sum(w, dim=1)
            d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
            l = torch.diag(d_sqrt_inv.float()) @ w.float() @ torch.diag(d_sqrt_inv.float())
            l[torch.isnan(l)] = 0.0
            l[torch.isinf(l)] = 0.0

            eigenvalues, eigenvectors = torch.linalg.eig(l)
            real_parts = eigenvalues.real

            sorted_indices = torch.argsort(real_parts)
            sorted_indices = sorted_indices
            k_eigenvectors = eigenvectors[:, sorted_indices[0:self.classnum]].real

            centroids, labels = kmeans(k_eigenvectors, k, 50)
            similarity_matrix = torch.zeros((torch.max(predicted_labels) + 1, torch.max(labels) + 1))
            similarity_matrix = similarity_matrix.to('cuda:0')

            for i, j in zip(predicted_labels, labels):
                similarity_matrix[i, j] += 1

            similarity_matrix_cpu = similarity_matrix.cpu()
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix_cpu)  #匈牙利算法
            y_pred_mapped = torch.clone(labels)
            for i, j in zip(col_ind, row_ind):
                y_pred_mapped[labels == i] = j

            num_labels = self.classnum

            one_hot_labels = torch.zeros((labels.size(0), num_labels), device=labels.device)
            one_hot_labels.scatter_(1, y_pred_mapped.unsqueeze(1), 1)

            xr = self.decoders[v](z)

            hs.append(h)
            pres.append(h)
            zs.append(z)
            qs.append(one_hot_labels)

            xrs.append(xr)
        return hs, pres, qs, xrs, zs

