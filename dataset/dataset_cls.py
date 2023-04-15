import gc
import os
import sys

import numpy as np
import trimesh

import torch
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, args, set_type='train', is_train=True):
        if not os.path.exists(args.data_path):
            print("Error")

        # init
        self.args = args
        self.set_type = set_type
        self.is_train = is_train

        self._load_dataset()

    def _load_dataset(self):
        datas = list()
        gt_list = list()

        class_num = torch.zeros(self.args.num_classes)
        gt_dict = {d: int(i) for i, d in
                   enumerate(sorted(os.listdir(os.path.join(self.args.data_path, self.set_type + '_norm'))))}

        # data path
        mesh_path = os.path.join(self.args.data_path, self.set_type + '_norm')
        vector_path = os.path.join(self.args.data_path, 'eigen_vectors', self.set_type)
        hks_path = os.path.join(self.args.data_path, 'HKS', self.set_type)
        gaussian_curvature_path = os.path.join(self.args.data_path, 'gaussian_curvatures', self.set_type)
        vf_dihedral_angle_path = os.path.join(self.args.data_path, 'VF_3innerProducts', self.set_type)

        dirs_list = [f for f in sorted(os.listdir(mesh_path))]

        for d in dirs_list:
            if not os.path.isdir(os.path.join(mesh_path, d)):
                continue
            file_list = sorted(os.listdir(os.path.join(mesh_path, d)))
            for file in file_list:
                file.strip()
                if self.is_train and 'Manifold40' in self.args.data_path.split('/'):
                    if file.split('_')[-1] not in ['S0.obj', 'S1.obj', 'S2.obj', 'S3.obj']:
                        continue

                print(os.path.join(mesh_path, d, file))

                # load mesh
                mesh = trimesh.load(os.path.join(mesh_path, d, file), process=False, force='mesh')
                mesh: trimesh.Trimesh

                eigen_vector = torch.from_numpy(
                    np.load(os.path.join(vector_path, d, os.path.splitext(file)[0] + '_eigen.npy'))).float()
                if torch.isnan(eigen_vector).sum() > 0 or torch.isinf(eigen_vector).sum() > 0:
                    print('eigen_vector errors, exit')
                    sys.exit()

                hks_cat = torch.from_numpy(
                    np.load(os.path.join(hks_path, d, os.path.splitext(file)[0] + '_hks.npy'))).float()
                if torch.isnan(hks_cat).sum() > 0 or torch.isinf(hks_cat).sum() > 0:
                    print('HKS errors, exit')
                    sys.exit()

                # normalize the gaussian curvature
                gaussian_curvature = torch.exp(-(torch.from_numpy(np.load(os.path.join(gaussian_curvature_path, d,
                                                                                       os.path.splitext(file)[
                                                                                           0] + '_gaussian_curvature.npy'))).float()))
                if torch.isnan(gaussian_curvature).sum() > 0 or torch.isinf(gaussian_curvature).sum() > 0:
                    print('gaussian_curvature errors, exit')
                    sys.exit()

                gaussian_curvature = ((gaussian_curvature - gaussian_curvature.min()) / (
                        gaussian_curvature.max() - gaussian_curvature.min())).unsqueeze(1)

                vf_dihedral_angle = torch.from_numpy(
                    np.load(
                        os.path.join(vf_dihedral_angle_path, d, os.path.splitext(file)[0] + '_vf_3innerProduct.npy')))
                if torch.isnan(vf_dihedral_angle).sum() > 0 or torch.isinf(vf_dihedral_angle).sum() > 0:
                    print('vf_dihedral_angle errors, exit')
                    sys.exit()
                # the input features
                features = torch.cat(
                    (torch.from_numpy(np.array(mesh.vertices)).float(),
                     torch.from_numpy(np.array(mesh.vertex_normals)).float(),
                     eigen_vector[:, 1:21], gaussian_curvature, vf_dihedral_angle,
                     hks_cat), dim=1)

                eigen_list = list()
                levels = list()
                # final_mat = None
                for i, k in enumerate(self.args.num_inputs):
                    level = eigen_vector[:, :k].T.to(self.args.device) @ features.float().to(self.args.device)
                    levels.append(level.cpu())
                    eigen_list.append(eigen_vector[:, :k].cpu())

                # different level
                c_list = list()
                c_list.append((eigen_list[1].T @ eigen_list[0]).float())
                c_list.append((eigen_list[2].T @ eigen_list[1]).float())
                c_list.append((eigen_list[2].T @ eigen_list[0]).float())

                data = dict()
                data['levels'] = levels
                data['c'] = c_list
                # data['final_mat'] = final_mat
                data['mesh_path'] = os.path.join(mesh_path, d, file)
                datas.append(data)
                gt_list.append(gt_dict[d])

                del mesh, eigen_vector, hks_cat, gaussian_curvature, vf_dihedral_angle, features, eigen_list, levels, c_list, data
                gc.collect()

        self.datas = datas
        self.gt = gt_list
        self.class_num = class_num

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item], self.gt[item]

