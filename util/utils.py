import os

import random
import json

import numpy as np
import trimesh
import math

import torch
import torch.backends.cudnn as cudnn

from sklearn.cluster import KMeans

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import scipy.sparse.linalg as sla
import potpourri3d as pp3d
import scipy

import time


class RunningAverage:
    """
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def same_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def warm_up_with_cosine_lr(warm_up_epochs, eta_min, base_lr, epochs, T_max=0):
    if T_max == 0:
        T_max = epochs
    warm_up_with_cosine_lr = lambda \
            epoch: (eta_min + (
            base_lr - eta_min) * epoch / warm_up_epochs) / base_lr if epoch <= warm_up_epochs else (eta_min + 0.5 * (
            base_lr - eta_min) * (math.cos(
        (epoch - warm_up_epochs) * math.pi / (T_max - warm_up_epochs)) + 1)) / base_lr
    # warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
    #             math.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    return warm_up_with_cosine_lr


def save_dict_to_json(d, json_path):
    """
    :param d:
    :param json_path:
    :return:
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_logging(args, test_loss, test_acc, epoch=None, net=None, train_loss=0.0, train_acc=0.0, save_name='last'):
    if args.mode == 'train':
        save_dict_to_json(
            {
                'epoch': epoch,
                'Train': {
                    'epoch_loss': train_loss,
                    'epoch_acc': train_acc,
                },
                'Val': {
                    'epoch_loss': test_loss,
                    'epoch_acc': test_acc
                }
            }, os.path.join('checkpoints', args.name, '{}.json'.format(save_name))
        )
        torch.save(net.state_dict(), os.path.join('checkpoints', args.name, '{}.pth'.format(save_name)))
    else:
        save_dict_to_json(
            {
                'Metric': {
                    'loss': test_loss,
                    'face_acc': test_acc
                }
            }, os.path.join('visualization_result', args.name, 'metric.json')
        )


def generate_adjacency_matrix(mesh):
    adjacency_matrix = np.zeros((mesh.vertices.shape[0], mesh.vertices.shape[0]))
    vertex_neighbors = mesh.vertex_neighbors

    for i in range(len(vertex_neighbors)):
        for _, j in enumerate(vertex_neighbors[i]):
            adjacency_matrix[i][j] = 1

    return adjacency_matrix


class LossAdjacency(nn.Module):
    def __init__(self):
        super(LossAdjacency, self).__init__()

    def forward(self, pred, gt, mesh_path, device, bandwidth):

        normal_idx = gt != -1

        mesh = trimesh.load(mesh_path, process=False)
        mesh: trimesh.Trimesh

        pred = F.log_softmax(pred, dim=1)

        pred_in_gt_position = torch.zeros(gt.shape).to(device)
        pred_in_gt_position[normal_idx] = pred[normal_idx, gt[normal_idx]]
        pred_in_gt_position = pred_in_gt_position.unsqueeze(1)

        pred_cdist = torch.cdist(pred_in_gt_position, pred_in_gt_position, p=2,
                                 compute_mode='use_mm_for_euclid_dist_if_necessary')

        # Euclidean_distance
        euclidean_distance = torch.cdist(torch.Tensor(mesh.vertices), torch.Tensor(mesh.vertices), p=2,
                                         compute_mode='use_mm_for_euclid_dist_if_necessary').float().to(device)
        # padding
        if (int(pred_cdist.shape[1] - euclidean_distance.shape[1]) != 0) or (
                int(pred_cdist.shape[0] - euclidean_distance.shape[0]) != 0):
            euclidean_distance = F.pad(euclidean_distance, pad=(
                0, int(pred_cdist.shape[1] - euclidean_distance.shape[1]), 0,
                int(pred_cdist.shape[0] - euclidean_distance.shape[0])), value=0)

        euc_dis_filter = torch.exp(- euclidean_distance / (2 * bandwidth))

        # get the adjacency matrix
        adjacency_matrix = torch.from_numpy(generate_adjacency_matrix(mesh)).float().to(device)
        # padding
        if (int(pred_cdist.shape[1] - adjacency_matrix.shape[1]) != 0) or (
                int(pred_cdist.shape[0] - adjacency_matrix.shape[0]) != 0):
            adjacency_matrix = F.pad(adjacency_matrix, pad=(0, int(pred_cdist.shape[1] - adjacency_matrix.shape[1]), 0,
                                                            int(pred_cdist.shape[0] - adjacency_matrix.shape[0])),
                                     value=0)

        pred_cdist = euc_dis_filter * pred_cdist
        pred_adjacency = (pred_cdist * adjacency_matrix).sum(1)
        adj_num = adjacency_matrix.sum(1)
        adj_num[adj_num[:] == 0] = 1
        loss_neighbor = (pred_adjacency.div(adj_num)).sum() / torch.nonzero(normal_idx).shape[0]

        return loss_neighbor


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, weight, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = weight * (confidence * nll_loss + smoothing * smooth_loss)
        return loss.mean()


def segmentation_loss(pred, gt, mesh_path, weight, device, loss_rate=1.8, bandwidth=1.0):
    class_weight = weight[gt[gt != -1]].to(device)

    criterion = LabelSmoothingCrossEntropy().to(device)

    adjacency = LossAdjacency().to(device)
    loss_neighbor = adjacency(pred, gt, mesh_path, device, bandwidth)

    loss = criterion(pred[gt != -1], gt[gt != -1], class_weight) + loss_rate * loss_neighbor

    return loss


def faces_label(mesh_path, pred_face_label, save_path_name, max_label):
    mesh = trimesh.load(mesh_path, process=False)
    mesh: trimesh.Trimesh

    pred_face_label = pred_face_label.numpy()

    colors = plt.get_cmap("tab20")(pred_face_label[:] / (max_label - 1))  # the color of 'Accent' from HodgeNet

    mesh.visual.face_colors = colors[:, :3]

    if not os.path.exists(os.path.join('visualization_result', save_path_name, 'faces')):
        os.makedirs(os.path.join('visualization_result', save_path_name, 'faces'))

    # exporting meshes are '.ply', the '.obj' files will generate the smooth boundaries.
    mesh.export(os.path.join('visualization_result', save_path_name, 'faces',
                             'face_' + os.path.splitext(mesh_path.split('/')[-1])[0] + '.ply'))
