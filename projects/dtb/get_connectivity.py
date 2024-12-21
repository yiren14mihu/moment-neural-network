# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:42:27 2024

@author: qiyangku
"""
import h5py
import numpy as np
from scipy import io
import pandas as pd
import networkx as nx
import sparse


def calculate_degree_zcsdata_onlysortical(single_voxel_size=2500, degree=500, n_region=360, _EI_ratio=0.8, use_cluster=True):
    if use_cluster:
        sc_path = r'/public/home/ssct004t/project/wangjiexiang/criticality_test/data/HCPex-structure/connectome_radial4mm_101915.csv'
    else:
        sc_path = r'D:\graduate_life\dtb_generation\DTB_code\criticality_test\HCP_data\HCPex-structure\connectome_radial4mm_101915.csv'

    sc = pd.read_csv(sc_path, header=None, )
    sc = sc.values[:n_region, :n_region]
    sc = sc + np.triu(sc, k=1).T
    sc[np.diag_indices_from(sc)] = 0

    if use_cluster:
        aseg_volume_stats_dir = '/public/home/ssct004t/project/wangjiexiang/criticality_test/HCP_data/aseg_volume_stats.txt'
    else:
        aseg_volume_stats_dir = r'D:\graduate_life\dtb_generation\DTB_code\criticality_test\HCP_data\aseg_volume_stats.txt'
    with open(aseg_volume_stats_dir, 'r') as file:
        content = file.readlines()
    file_row_id = 9
    file_row_id_Left_Hippocampus_idx = content[0].strip().split(',').index('Left_Hippocampus')
    file_row_id_Right_Hippocampus_idx = content[0].strip().split(',').index('Right_Hippocampus')
    # print("file_row_id_Left_Hippocampus_idx", file_row_id_Left_Hippocampus_idx)
    # print("file_row_id_Right_Hippocampus_idx", file_row_id_Right_Hippocampus_idx)
    file_subject_list = content[file_row_id].strip().split(',')
    replace_grayvolume_left_value = float(file_subject_list[file_row_id_Left_Hippocampus_idx])
    replace_grayvolume_Right_value = float(file_subject_list[file_row_id_Right_Hippocampus_idx])

    _id = int(file_subject_list[0])
    # print("_id", _id)
    # print("replace_grayvolume_left_value", replace_grayvolume_left_value)
    # print("replace_grayvolume_Right_value", replace_grayvolume_Right_value)

    if use_cluster:
        GrayVol_path = '/public/home/ssct004t/project/wangjiexiang/criticality_test/HCP_data/HCP1200_HCPatlas_GrayVol.csv'
    else:
        GrayVol_path = r'D:\graduate_life\dtb_generation\DTB_code\criticality_test\HCP_data\HCP1200_HCPatlas_GrayVol.csv'
    GrayVol_df = pd.read_csv(GrayVol_path)
    subject_idx = GrayVol_df.index[GrayVol_df['SubID'] == _id].tolist()[0]
    # print("subject_idx", subject_idx)
    regionwize_NSR_dti_grey_matter = np.array(GrayVol_df.iloc[subject_idx, 1:])
    regionwize_NSR_dti_grey_matter[119] = replace_grayvolume_left_value
    regionwize_NSR_dti_grey_matter[299] = replace_grayvolume_Right_value
    if use_cluster:
        HCPregion_name_label = r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/HCPex_3mm_modified_label_mayraw.csv'
    else:
        HCPregion_name_label = r'D:\graduate_life\dtb_generation\DTB_code\criticality_test\JFFengdata\HCPex_3mm_modified_label_mayraw.csv'
    HCPex_3mm_modified_label = pd.read_csv(HCPregion_name_label)
    reorder_region_label_raw = np.array(HCPex_3mm_modified_label['Label'])[:n_region]
    regionwize_NSR_dti_grey_matter = regionwize_NSR_dti_grey_matter[reorder_region_label_raw-1]

    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region  # number of neurons in each region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO

    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges

    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO

    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    K_EE_long = K_EE_long * sc / sc.sum(axis=1, keepdims=True)

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


def calculate_degree_eigenmode(single_voxel_size=2500, degree=500, n_region=378, _EI_ratio=0.8, use_cluster=True,
                     kunshan_cluster=True, reconstruct_num=1):
    if use_cluster:
        if kunshan_cluster:
            brain_file = h5py.File(
                r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
        else:
            brain_file = h5py.File(
                r"/public/home/dtbrain/project/wjx/moment-neural-network-develop/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
    else:
        brain_file = h5py.File(
            r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
            "r")

    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)

    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Laplace_matrix = np.diag(sc.sum(axis=1)) - sc
    eigenvalues, eigenvectors = np.linalg.eig(Laplace_matrix)
    _eigenvalues_argsort = np.argsort(eigenvalues)
    _eigenvectors_argsort_total = eigenvectors[:, _eigenvalues_argsort]
    reconstruct_Laplace_matrix = _eigenvectors_argsort_total[:, :reconstruct_num] @ np.diag(eigenvalues)[:reconstruct_num, :reconstruct_num] @ _eigenvectors_argsort_total[:, :reconstruct_num].T
    reconstruct_sc = np.diag(sc.sum(axis=1)) - reconstruct_Laplace_matrix

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO

    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges

    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO

    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    K_EE_long = K_EE_long * sc / sc.sum(axis=1, keepdims=True)
    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size

def calculate_degree_DTIdestruct_random_rewiring_new(single_voxel_size=2500, degree=500, n_region=378, _EI_ratio=0.8, use_cluster=True,
                     kunshan_cluster=True, ws_rewiring_p=0.1):
    if use_cluster:
        if kunshan_cluster:
            brain_file = h5py.File(
                r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
        else:
            brain_file = h5py.File(
                r"/public/home/dtbrain/project/wjx/moment-neural-network-develop/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
    else:
        brain_file = h5py.File(
            r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
            "r")
    # brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)

    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([_EI_ratio, 1 - _EI_ratio])[None, :]).reshape(
        [-1]).astype(np.int64)

    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges

    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO

    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    conn = sparse.COO(sc)
    new_coords = []
    diff = 2
    last = None
    for coord in conn.coords.T:
        value = np.random.rand(1)
        if value < ws_rewiring_p:
            if last is None:
                diff = 2
                last = coord[0]
            else:
                if last == coord[0]:
                    diff += 1
                else:
                    last = coord[0]
                    diff = 2
            y = ((-1) ** (diff % 2)) * (diff // 2) + coord[0]
            y = y % n_region
            coord[1] = y
            new_coords.append(coord)
        else:
            new_coords.append(coord)
    new_coords = np.stack(new_coords, axis=1)
    shape = conn.shape
    data = conn.data
    new_conn = sparse.COO(coords=new_coords, data=data, shape=shape)
    new_conn = new_conn / new_conn.sum(axis=1, keepdims=True)
    network = new_conn.todense()

    # conn_bin = sc.astype(bool).astype(int)
    # deg = int(np.mean(np.sum(conn_bin, axis=1)))
    # G = nx.watts_strogatz_graph(n_region, deg, ws_rewiring_p)
    # network = nx.to_numpy_array(G)
    #
    # # coincide with distribution of dti
    # # mask = np.nonzero(network)
    # # actual_conns = network[mask]
    # # network[mask] = data_value[np.argsort(actual_conns)]
    #
    # network = network.T
    # np.fill_diagonal(network, 0)
    # network /= network.sum(axis=1, keepdims=True)

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    K_EE_long = K_EE_long * network / network.sum(axis=1, keepdims=True)

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


def calculate_degree_DTIdestruct_local_ws(single_voxel_size=2500, degree=500, n_region=378, _EI_ratio=0.8, use_cluster=True,
                     kunshan_cluster=True, ws_rewiring_p=0.1):
    if use_cluster:
        if kunshan_cluster:
            brain_file = h5py.File(
                r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
        else:
            brain_file = h5py.File(
                r"/public/home/dtbrain/project/wjx/moment-neural-network-develop/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
    else:
        brain_file = h5py.File(
            r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
            "r")
    # brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)

    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([_EI_ratio, 1 - _EI_ratio])[None, :]).reshape(
        [-1]).astype(np.int64)

    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges

    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO

    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    # p, *other = args
    # N = conn.shape[0]

    conn_bin = sc.astype(bool).astype(int)
    deg = int(np.mean(np.sum(conn_bin, axis=1)))
    G = nx.watts_strogatz_graph(n_region, deg, ws_rewiring_p)
    network = nx.to_numpy_array(G)

    # coincide with distribution of dti
    # mask = np.nonzero(network)
    # actual_conns = network[mask]
    # network[mask] = data_value[np.argsort(actual_conns)]

    network = network.T
    np.fill_diagonal(network, 0)
    network /= network.sum(axis=1, keepdims=True)

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    K_EE_long = K_EE_long * network / network.sum(axis=1, keepdims=True)

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


def calculate_degree_DTIdestruct(single_voxel_size=2500, degree=500, n_region=378, _EI_ratio=0.8, use_cluster=True,
                     kunshan_cluster=True):
    if use_cluster:
        if kunshan_cluster:
            brain_file = h5py.File(
                r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
        else:
            brain_file = h5py.File(
                r"/public/home/dtbrain/project/wjx/moment-neural-network-develop/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
    else:
        brain_file = h5py.File(
            r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
            "r")
    # brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)

    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([_EI_ratio, 1 - _EI_ratio])[None, :]).reshape(
        [-1]).astype(np.int64)

    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges

    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO

    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    density = np.sum(sc.astype(bool).astype(int)) / (len(sc) ** 2)
    new_conn = nx.to_numpy_array(nx.fast_gnp_random_graph(n=len(sc), p=density, directed=False))
    new_conn = new_conn * np.random.uniform(0, 1, new_conn.shape)
    # new_conn[np.where(abs(new_conn) <= 0.00001)] = 0

    upper_diag = new_conn.copy()[np.triu_indices_from(new_conn, 1)]
    new_conn = new_conn.T
    new_conn[np.triu_indices_from(new_conn, 1)] = upper_diag
    np.fill_diagonal(new_conn, 0)

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    K_EE_long = K_EE_long * new_conn / new_conn.sum(axis=1, keepdims=True)

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


def calculate_degree(single_voxel_size=2500, degree=500, n_region=378, _EI_ratio=0.8, use_cluster=True, kunshan_cluster=True):
    if use_cluster:
        if kunshan_cluster:
            brain_file = h5py.File(
                r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
        else:
            brain_file = h5py.File(
                r"/public/home/dtbrain/project/wjx/moment-neural-network-develop/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
                "r")
    else:
        brain_file = h5py.File(r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)
    
    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([_EI_ratio, 1-_EI_ratio])[None, :]).reshape(
        [-1]).astype(np.int64)
    
    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges
    
    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO
    
    # below are custom edits by Qi 
    # local intra-regional connections
    K_EE = degree_*0.6
    K_EI = degree_*0.15
    K_IE = degree_*0.6
    K_II = degree_*0.15
    
    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1))*0.25
    # K_EE_long = degree_*0.25
    K_EE_long = K_EE_long*sc/sc.sum(axis=1, keepdims=True)
    # np.fill_diagonal(K_EE_long, 0.0) # make sure self connection excluded

    # # print("degree_", degree_*0.25)
    # print(degree_.reshape((-1, 1))*0.25 *sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')
    #
    # print(degree_[0] * 0.25 * sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')
    # print(K_EE_long[0], '\n\n\n\n')
    # print(degree_.reshape((1, -1))*0.25 *sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


def calculate_degree_nograyvol_rownormalized(single_voxel_size=2500, degree=500, n_region=378, sc_intra_cluster=int(4e3), sc_inter_cluster=int(1e2), _EI_ratio=0.8, use_cluster=True):
    if use_cluster:
        brain_file = h5py.File(
            r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
            "r")
    else:
        brain_file = h5py.File(r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r"/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
    #     "r")
    # brain_file = h5py.File(
    #     r"D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\projects\dtb\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat",
    #     "r")
    # brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)

    sc = sc[:n_region, :n_region]
    sc[:round(n_region/2), :round(n_region/2)] = sc_intra_cluster
    sc[round(n_region/2):, round(n_region/2):] = sc_intra_cluster
    sc[:round(n_region/2), round(n_region/2):] = sc_inter_cluster
    sc[round(n_region/2):, :round(n_region/2)] = sc_inter_cluster

    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    sc[np.diag_indices_from(sc)] = 0

    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()

    regionwize_NSR_dti_grey_matter = np.ones_like(Region_GM)
    # regionwize_NSR_dti_grey_matter = Region_GM

    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([_EI_ratio, 1-_EI_ratio])[None, :]).reshape(
        [-1]).astype(np.int64)


    degree_ = np.ones(sc.shape[0]) * degree  # TODO ！！！要不要取整？.astype(np.int32)


    # below are custom edits by Qi
    # local intra-regional connections
    K_EE = degree_ * 0.6
    K_EI = degree_ * 0.15
    K_IE = degree_ * 0.6
    K_II = degree_ * 0.15

    # long range inter-regional excitatory connections
    K_EE_long = degree_.reshape((-1, 1)) * 0.25
    # K_EE_long = degree_*0.25
    K_EE_long = K_EE_long * sc / sc.sum(axis=1, keepdims=True)
    # np.fill_diagonal(K_EE_long, 0.0) # make sure self connection excluded

    # # print("degree_", degree_*0.25)
    # print(degree_.reshape((-1, 1))*0.25 *sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')
    #
    # print(degree_[0] * 0.25 * sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')
    # print(K_EE_long[0], '\n\n\n\n')
    # print(degree_.reshape((1, -1))*0.25 *sc/sc.sum(axis=1, keepdims=True), '\n\n\n\n')

    return K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size


if __name__=='__main__':    
    K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size = calculate_degree_zcsdata_onlysortical(use_cluster=False)
    print("K_EE", K_EE.shape)
    print("K_EI", K_EI.shape)
    print("K_IE", K_IE.shape)
    print("K_II", K_II.shape)
    print("K_EE_long", K_EE_long.shape)
    print("degree_", degree_.shape)
    print("block_size", block_size.shape)

