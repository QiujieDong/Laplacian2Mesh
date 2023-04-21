import argparse
import os

import numpy as np
import torch
import igl
import trimesh
from types import SimpleNamespace

from mesh import Mesh


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
            if os.path.splitext(file)[1] not in ['.ply', '.obj']:
                continue
            print(os.path.join(subset_mesh_path, file))

            mesh = trimesh.load(os.path.join(subset_mesh_path, file), process=False)

            if subset_name == 'test':
                vertices = mesh.vertices - mesh.vertices.min(axis=0)
                vertices = vertices / vertices.max()
                mesh.vertices = vertices
                mesh.export(os.path.join(norm_mesh_path, os.path.splitext(file)[0] + '.obj'))
            else:

                if args.augment_orient:
                    if args.is_humanbody:
                        rotations_ratio = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=5, replace=False)
                        scales_ratio = np.random.normal(1, 0.1, size=(9, 3))
                        axis_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    else:
                        rotations_ratio = np.random.choice([0, 1, 2, 3], size=3, replace=False)
                        scales_ratio = np.random.normal(1, 0.1, size=(15, 3))
                else:
                    rotations_ratio = [0]
                    scales_ratio = np.random.normal(1, 0.1, size=(45, 3))

                for i in range(len(rotations_ratio)):
                    # trimesh.copy() is deepcopy. copy(include_cache=False):If True, will shallow copy cached data to new mesh
                    mesh_tans_rotation = mesh.copy()
                    if args.is_humanbody:
                        axis_ind = np.random.choice([0, 1, 2], size=1, replace=False)
                        rotation = trimesh.transformations.rotation_matrix((np.pi / 4) * rotations_ratio[i],
                                                                           axis_list[axis_ind[0]])
                    else:
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
            eigen_values, eigen_vectors = torch.linalg.eigh(cot)
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


def generate_dihedral_angles(args):
    print('------generate Vertex_Face 3innerProducts------')
    for _, subset_name in enumerate(['train', 'test']):

        subset_mesh_path = os.path.join(args.data_path, subset_name + '_norm')

        V_dihedral_angles_path = os.path.join(args.data_path, 'V_dihedral_angles', subset_name)
        if not os.path.exists(V_dihedral_angles_path):
            os.makedirs(V_dihedral_angles_path)

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

            V_dihedral_angles = np.dot(vertex_faces_adjacency_matrix, face_dihedral_angle)

            np.save(os.path.join(V_dihedral_angles_path, os.path.splitext(file)[0] + '_V_dihedralAngles.npy'),
                    V_dihedral_angles)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/humanbody')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--is_humanbody', action='store_true')
    args = parser.parse_args()

    normalize_meshes(args)
    generate_cot_eigen_vectors(args)
    generate_gaussian_curvature(args)
    generate_dihedral_angles(args)
    generate_vertices_ground_truth_from_edges(args)
    generate_faces_ground_truth_from_vertices(args)
    HKS(args)
