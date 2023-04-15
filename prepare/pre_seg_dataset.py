import argparse
import os

import numpy as np
import torch
import igl
import trimesh
from types import SimpleNamespace

from mesh import Mesh

import scipy as sp
import open3d as o3d
import json


# mesh normalization, vertex in [-0.5, 0.5]
def normalize_meshes(args):
    print('------normalize meshes------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name)

        norm_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        if not os.path.exists(norm_mesh_path):
            os.makedirs(norm_mesh_path)

        for file in os.listdir(subset_mesh_path):
            file.strip()
            if os.path.splitext(file)[1] not in ['.ply']:
                continue
            print(os.path.join(subset_mesh_path, file))

            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False)

            if subset_name == 'test':
                vertices = mesh.vertices - mesh.vertices.min(axis=0)
                vertices = vertices / vertices.max()
                mesh.vertices = vertices
                mesh.export(os.path.join(norm_mesh_path, os.path.splitext(file)[0] + '.obj'))

                # # add noise
                # noise_rates = [0.00, 0.005, 0.010, 0.050, 0.080, 0.1]
                # for ind, noise_rate in enumerate(noise_rates):
                #     vertices_noise = np.random.rand(vertices.shape[0], vertices.shape[1]) * noise_rate * np.linalg.norm(
                #         vertices.max(0) - vertices.min(0)) + vertices
                #     mesh.vertices = vertices_noise
                #     mesh.export(os.path.join(norm_mesh_path, os.path.splitext(file)[0] + '{}'.format(ind) + '.obj'))
            else:

                if args.augment_orient:
                    rotations_ratio = np.random.choice([0, 1, 2, 3], size=3, replace=False)
                    scales_ratio = np.random.normal(1, 0.1, size=(15, 3))
                else:
                    rotations_ratio = [0]
                    scales_ratio = np.random.normal(1, 0.1, size=(45, 3))

                for i in range(len(rotations_ratio)):
                    # trimesh.copy() is deepcopy. copy(include_cache=False):If True, will shallow copy cached data to new mesh
                    mesh_tans_rotation = mesh.copy()
                    rotation = trimesh.transformations.rotation_matrix((np.pi / 2) * rotations_ratio[i],
                                                                       [0, 1, 0])
                    mesh_tans_rotation.apply_transform(rotation)

                    for j in range(len(scales_ratio)):
                        mesh_trans_scale = mesh_tans_rotation.copy()
                        mesh_trans_scale.vertices = mesh_trans_scale.vertices * scales_ratio[j]

                        vertices = mesh_trans_scale.vertices - mesh_trans_scale.vertices.min(axis=0)
                        vertices = vertices / vertices.max()
                        mesh_trans_scale.vertices = vertices
                        mesh_trans_scale.export(
                            os.path.join(norm_mesh_path, os.path.splitext(file)[0] + '_R{0}_S{1}.obj'.format(i, j)))


def generate_cot_eigen_vectors(args):
    print('------generate cot eigen vectors------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)
        eigen_values_path = os.path.join(args.data_path, 'eigen_values', subset_name)
        if not os.path.exists(eigen_vectors_path):
            os.makedirs(eigen_vectors_path)
        if not os.path.exists(eigen_values_path):
            os.makedirs(eigen_values_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False)
            mesh: trimesh.Trimesh
            cot = -igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
            cot = torch.from_numpy(cot).float().to(args.device)
            eigen_values, eigen_vectors = torch.symeig(cot, eigenvectors=True)
            ind = torch.argsort(eigen_values)[:]

            np.save(os.path.join(eigen_vectors_path, os.path.splitext(file)[0] + '_eigen.npy'),
                    eigen_vectors[:, ind].cpu().numpy())
            np.save(os.path.join(eigen_values_path, os.path.splitext(file)[0] + '_eigenValues.npy'),
                    eigen_values[ind].cpu().numpy())


def generate_gaussian_curvature(args):
    print('------generate gaussian curvature------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        gaussian_curvatures_path = os.path.join(args.data_path, 'gaussian_curvatures', subset_name)
        if not os.path.exists(gaussian_curvatures_path):
            os.makedirs(gaussian_curvatures_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            # mesh_curvature_igl
            mesh_vertices, mesh_faces = igl.read_triangle_mesh(os.path.join(subset_mesh_path, file))
            mesh_gaussian_curvature = igl.gaussian_curvature(mesh_vertices, mesh_faces)

            np.save(os.path.join(gaussian_curvatures_path, os.path.splitext(file)[0] + '_gaussian_curvature.npy'),
                    mesh_gaussian_curvature)


def generate_VF_3innerProducts(args):
    print('------generate Vertex_Face 3innerProducts------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        vf_3innerProducts_path = os.path.join(args.data_path, 'VF_3innerProducts', subset_name)
        if not os.path.exists(vf_3innerProducts_path):
            os.makedirs(vf_3innerProducts_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False)
            mesh: trimesh.Trimesh

            vertex_faces_adjacency_matrix = np.zeros((mesh.vertices.shape[0], mesh.faces.shape[0]))
            for vertex, faces in enumerate(mesh.vertex_faces):
                for i, face in enumerate(faces):
                    if face == -1:
                        break
                    vertex_faces_adjacency_matrix[vertex, face] = 1

            dihedral_angle = list()
            for i in range(mesh.faces.shape[0]):
                dihedral_angle.append(list())

            face_adjacency = mesh.face_adjacency

            for adj_faces in face_adjacency:
                dihedral_angle[adj_faces[0]].append(
                    np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))
                dihedral_angle[adj_faces[1]].append(
                    np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))

            # process the non-watertight mesh which include some faces which dont have three neighbors.
            for i, l in enumerate(dihedral_angle):
                if (len(l)) == 3:
                    continue
                l.append(1)
                if (len(l)) == 3:
                    continue
                l.append(1)
                if (len(l)) == 3:
                    continue
                l.append(1)
                if (len(l)) != 3:
                    print(i, 'Padding Failed')
            face_dihedral_angle = np.array(dihedral_angle).reshape(-1, 3)

            # --------------------------------------------------------------------------------------------
            # sort the face_dihedral_angle, e.g. three dihedral angle [0.5, 0.9, 0.1]--> [0.1, 0.5, 0.9]
            # ind = np.argsort(face_dihedral_angle, axis=-1)
            # ind_axis_0 = (np.argwhere(ind > -1)[:, 0]).reshape(-1, 3)
            #
            # face_dihedral_angle = face_dihedral_angle[ind_axis_0, ind]
            # --------------------------------------------------------------------------------------------

            vf_3innerProducts = np.dot(vertex_faces_adjacency_matrix, face_dihedral_angle)

            np.save(os.path.join(vf_3innerProducts_path, os.path.splitext(file)[0] + '_vf_3innerProduct.npy'),
                    vf_3innerProducts)


def generate_vertices_ground_truth_from_edges(args):
    print('------generate vertices ground truth from edges------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_gt_edge_path = os.path.join(args.data_path, 'seg')
        subset_mesh_path = os.path.join(args.data_path, subset_name)

        ground_truth_path = os.path.join(args.data_path, 'ground_truth', subset_name)
        if not os.path.exists(ground_truth_path):
            os.makedirs(ground_truth_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            gt_list = np.loadtxt(os.path.join(subset_gt_edge_path, os.path.splitext(file)[0] + '.eseg'))
            gt_list = gt_list.astype(np.int32)
            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=True, force='mesh')
            opt = SimpleNamespace()
            opt.num_aug = 1
            meshcnn_mesh = Mesh(os.path.join(subset_mesh_path, file), opt)
            edges_vertices = meshcnn_mesh.edges

            labels = np.zeros((mesh.vertices.shape[0], gt_list.max()), dtype=int)
            v_labels = []
            for e, vertices in enumerate(edges_vertices):
                for v in vertices:
                    labels[v, (gt_list[e] - 1)] += 1
                # get most appear face label as vertex label
                # start from 0
            v_labels.append(np.argmax(labels, axis=1))

            np.save(os.path.join(ground_truth_path, os.path.splitext(file)[0] + '.npy'),
                    v_labels)


def generate_faces_ground_truth_from_vertices(args):
    print('------generate faces ground truth from vertices------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_gt_vertices_path = os.path.join(args.data_path, 'ground_truth', subset_name)
        if subset_name == 'train':
            subset_mesh_path = os.path.join(args.data_path, subset_name)
        else:
            subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        ground_truth_faces_path = os.path.join(args.data_path, 'ground_truth_faces', subset_name)
        if not os.path.exists(ground_truth_faces_path):
            os.makedirs(ground_truth_faces_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            gt_list = np.load(os.path.join(subset_gt_vertices_path, os.path.splitext(file)[0] + '.npy')).squeeze()
            gt_list = gt_list.astype(np.int32)
            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False, force='mesh')

            vertex_faces = mesh.vertex_faces
            labels = np.zeros((mesh.faces.shape[0], gt_list.max() + 1), dtype=int)
            f_labels = []

            for vertex, faces in enumerate(vertex_faces):
                for f in faces:
                    if f == -1:
                        break
                    labels[f, gt_list[vertex]] += 1

            f_labels.append(np.argmax(labels, axis=1))

            np.save(os.path.join(ground_truth_faces_path, os.path.splitext(file)[0] + '_face_gt.npy'),
                    f_labels)


def HKS(args):
    print('------generate HKS------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')
        eigen_vectors_path = os.path.join(args.data_path, 'eigen_vectors', subset_name)
        eigen_values_path = os.path.join(args.data_path, 'eigen_values', subset_name)

        HKS_path = os.path.join(args.data_path, 'HKS', subset_name)
        if not os.path.exists(HKS_path):
            os.makedirs(HKS_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            eigen_vector = torch.from_numpy(
                np.load(os.path.join(eigen_vectors_path, os.path.splitext(file)[0] + '_eigen.npy'))).float()
            eigen_values = torch.from_numpy(
                np.load(os.path.join(eigen_values_path, os.path.splitext(file)[0] + '_eigenValues.npy'))).float()

            t_min = 4 * np.log(10) / eigen_values.max()
            t_max = 4 * np.log(10) / np.sort(eigen_values)[1]
            ts = np.linspace(t_min, t_max, num=100)
            hkss = (eigen_vector[:, :, None] ** 2) * np.exp(
                -eigen_values[None, :, None] * ts.flatten()[None, None, :])
            hks = torch.sum(hkss, dim=1)
            hks_cat = ((hks[:, 1] - hks[:, 1].min()) / (hks[:, 1].max() - hks[:, 1].min())).unsqueeze(1)
            for i, k in enumerate([2, 3, 4, 5, 8, 10, 15, 20]):
                hks_norm = ((hks[:, k] - hks[:, k].min()) / (hks[:, k].max() - hks[:, k].min())).unsqueeze(1)
                hks_cat = torch.cat((hks_cat, hks_norm), dim=1)

            np.save(os.path.join(HKS_path, os.path.splitext(file)[0] + '_hks.npy'), hks_cat)


def generate_VF_3innerProducts_select_min_max(args):
    print('------generate Vertex_Face 3innerProducts------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        vf_3innerProducts_path = os.path.join(args.data_path, 'vertex_dihedral_angles_min_max', subset_name)
        if not os.path.exists(vf_3innerProducts_path):
            os.makedirs(vf_3innerProducts_path)

        for file in sorted(os.listdir(subset_mesh_path)):
            file.strip()
            if os.path.splitext(file)[1] not in ['.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False)
            mesh: trimesh.Trimesh

            faces_for_each_edge = mesh.face_adjacency  # Pairs of faces which share an edge
            edges_of_face_adjacency = mesh.face_adjacency_edges  # Vertex indices which correspond to face_adjacency (edge)
            face_normals = mesh.face_normals
            mesh_vertices = mesh.vertices

            # dihedral_angle_values = np.zeros((faces_for_each_edge.shape[0], 1), dtype=np.float)
            dihedral_angle_values = []
            # faces_cross_values = np.zeros((faces_for_each_edge.shape[0], 1), dtype=np.float)
            faces_cross_values = []
            for i_edge, faces in enumerate(faces_for_each_edge):
                # the value of dihedral angle
                dihedral_angle_value = np.dot(face_normals[faces[0]], face_normals[faces[1]])
                dihedral_angle_values.append(np.exp(-dihedral_angle_value))

                # the cross value of faces
                faces_cross_values.append(np.cross(face_normals[faces[0]], face_normals[faces[1]]))

            dihedral_angle_values = np.array(dihedral_angle_values)
            faces_cross_values = np.array(faces_cross_values)

            # dihedral_angle_values = dihedral_angle_values.squeeze()
            # faces_cross_values = faces_cross_values.squeeze()

            vertex_all_dihedral_angles = [[] for i in range(mesh_vertices.shape[0])]

            for i_edge, vertices in enumerate(edges_of_face_adjacency):
                # the direction of the dihedral angle
                vertex_direction_0to1 = mesh_vertices[vertices[1]] - mesh_vertices[vertices[0]]
                dihedral_angle_direction_0to1 = np.dot(faces_cross_values[i_edge], vertex_direction_0to1)

                if dihedral_angle_direction_0to1 > 0:
                    vertex_all_dihedral_angles[vertices[0]].append(dihedral_angle_values[i_edge])
                    vertex_all_dihedral_angles[vertices[1]].append(-dihedral_angle_values[i_edge])
                else:
                    vertex_all_dihedral_angles[vertices[0]].append(-dihedral_angle_values[i_edge])
                    vertex_all_dihedral_angles[vertices[1]].append(dihedral_angle_values[i_edge])

            min_max_each_vertex_dihedral_angle = [[] for i in range(mesh.vertices.shape[0])]

            for i_vertex in range(len(vertex_all_dihedral_angles)):
                min_max_each_vertex_dihedral_angle[i_vertex].append(min(vertex_all_dihedral_angles[i_vertex]))
                min_max_each_vertex_dihedral_angle[i_vertex].append(max(vertex_all_dihedral_angles[i_vertex]))

            min_max_each_vertex_dihedral_angle = np.array(min_max_each_vertex_dihedral_angle)

            np.save(os.path.join(vf_3innerProducts_path, os.path.splitext(file)[0] + '_vf_3innerProduct.npy'),
                    min_max_each_vertex_dihedral_angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/humanbody')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    args = parser.parse_args()

    normalize_meshes(args)
    generate_cot_eigen_vectors(args)
    generate_gaussian_curvature(args)
    generate_VF_3innerProducts(args)
    generate_vertices_ground_truth_from_edges(args)
    generate_faces_ground_truth_from_vertices(args)
    HKS(args)
