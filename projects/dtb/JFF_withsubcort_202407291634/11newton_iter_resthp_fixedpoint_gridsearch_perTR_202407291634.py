# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: newton_iter_resthp_fixedpoint_gridsearch_perTR_0717.py
# @time: 2024/7/17 20:38

# -*- coding: utf-8 -*-
# @author: yiren14mihu
# @file: newton_iter_resthp_fixedpoint_gridsearch.py
# @time: 2024/7/10 16:51

import os

from matplotlib import pyplot as plt
import torch
import numpy as np
from mnn_dtb_withhp_float64_202407291634 import Cond_MNN_DTB
import argparse
from mpi4py import MPI
from scipy import io
from torch.autograd.functional import jacobian
from utils.helpers import numpy2torch, torch_2_numpy
# import nibabel as nib
import h5py
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import pandas as pd
from plot.dynamics_project_Encapsulation import dynamics_project_Encapsulation_ode_aspect_variancemapping


def get_args():
    parser = argparse.ArgumentParser(description="MNN")
    parser.add_argument("--multiply_value", type=str, default="1")
    args = parser.parse_args()
    return args



# load_dir_list = list(np.arange(200, 261))
# load_dir_list = list(np.arange(200, 201))
# load_dir_list = list(np.arange(209, 212))
# load_dir_list = list(np.arange(209, 210))
# load_dir_list = list(np.arange(211, 212))
# load_dir_list = list(np.arange(240, 241))
load_dir_list = [228, 237, 250, 255]
# load_dir_list = list(np.arange(201, 209, 1))
# load_dir_list = list(np.arange(0, 1, 10))
# load_dir_list = list(np.arange(200, 201, 10))
# load_dir_list = list(np.arange(0, 191, 10))
# load_dir_list = [200, 209, 240]
# load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
# load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
# load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))

_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in load_dir_list]
_gui_ampa_scale_values = np.array(_gui_ampa_scale_values)
total_paras = _gui_ampa_scale_values

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_per_proc = int(np.ceil(len(total_paras) / size))
start_idx = rank * n_per_proc
end_idx = min(start_idx + n_per_proc, len(total_paras))
params = total_paras[start_idx:end_idx]
selected_load_dir_list = np.array(load_dir_list)[start_idx:end_idx]
# selected_total___load_dir_list = np.array(total___load_dir_list)[start_idx:end_idx]

_righteigenvalues_record = []
_lefteigenvalues_record = []
_fixed_point_error_record = []
_righteigenvector_record1 = []
_lefteigenvector_record1 = []

_fixedpoint_record = []

_reconstruct_error_record = []

for rela_param_idx in range(len(params)):
    _gui_ampa_scalevalue = params[rela_param_idx]
    load_dir_list_idx = selected_load_dir_list[rela_param_idx]

    # _gui_ampa_scalevalue = params[rela_param_idx]
    # load_dir_list_idx = selected_load_dir_list[rela_param_idx]

    scale_factor = 2500
    modified_parameter_tao_ui = np.array([4., 4., 10., 50.])
    # all_scale = 0
    all_scale = 0.25
    _J_EE = all_scale * 2
    _J_IE = all_scale * 4
    _J_EI = all_scale * 27
    _J_II = all_scale * 48
    _J_EE_scaled = _J_EE / np.sqrt(scale_factor)
    _J_IE_scaled = _J_IE / np.sqrt(scale_factor)
    _J_EI_scaled = _J_EI / np.sqrt(scale_factor)
    _J_II_scaled = _J_II / np.sqrt(scale_factor)

    _modified_J_EE = 0.05
    _modified_J_IE = 0.55
    _modified_J_EI = 0.4
    _modified_J_II = 0.1

    _gui_voxel = np.array([[_gui_ampa_scalevalue * _J_EE_scaled / modified_parameter_tao_ui[0],
                            _modified_J_EE * _J_EE_scaled / modified_parameter_tao_ui[0],
                            _modified_J_EI * _J_EI_scaled / modified_parameter_tao_ui[2], 0],
                           [1.1 * 0 * _J_IE_scaled / modified_parameter_tao_ui[0],
                            _modified_J_IE * _J_IE_scaled / modified_parameter_tao_ui[0],
                            _modified_J_II * _J_II_scaled / modified_parameter_tao_ui[2], 0]], dtype=np.float32)

    initial_parameter_g_Li = np.array([1 / 20, 1 / 10])

    _hp_label = 'restintero_hptimeaverage'
    # _hp_label = 'gridsearch_resthp_intero'
    # _hp_label = 'nohp'
    if _hp_label == 'sin_toytest':
        # generate hyperparameter
        _assi_region = np.array([1])

        _hp_timepoint = 349  # The total time is _hp_timepoint*800 ms
        x_iter = np.arange(349) / 1.25
        X = np.sin(2 * np.pi / 40 * (x_iter))
        X = X * 0.3 + 0.4
        hp_total = X.reshape((-1, _assi_region.shape[0]))

        single_voxel_size = 500
        degree = 100
        n_region = 20
    elif _hp_label == 'rest_intero':
        _assi_region = np.sort(np.array(
            [120, 300, 109, 111, 112, 289, 291, 292, 57, 59, 61, 62, 179, 180, 237, 239, 241, 242, 359, 360, 165, 166,
             345, 346,
             361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378]))

        hp_path = '/public/home/ssct004t/project/wangjiexiang/criticality_test/OU_ZCS_cudasearch/cuda_test_code_debug/voxel_between_ampa/JFF_data_withsubcortical/DA/Epopu_ext/d7_7/intero/d_inv/da_rest_intero_1.5outer_regionwise_202402262208_12card/12card_newdebug_1.5_50ensemble_dt0.1/hp.npy'
        hp_total = np.load(hp_path).reshape((-1, _assi_region.shape[0]))

        # TODO 小心get_connectivity里调用的函数是什么!!!
        single_voxel_size = 25000
        degree = 500
        n_region = 378
    elif _hp_label == 'restintero_hptimeaverage':
        _assi_region = np.sort(np.array(
            [120, 300, 109, 111, 112, 289, 291, 292, 57, 59, 61, 62, 179, 180, 237, 239, 241, 242, 359, 360, 165, 166,
             345, 346,
             361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378]))

        hp_path = '/public/home/ssct004t/project/wangjiexiang/criticality_test/OU_ZCS_cudasearch/cuda_test_code_debug/voxel_between_ampa/JFF_data_withsubcortical/DA/Epopu_ext/d7_7/intero/d_inv/da_rest_intero_1.5outer_regionwise_202402262208_12card/12card_newdebug_1.5_50ensemble_dt0.1/hp.npy'
        raw_hp_total = np.load(hp_path).reshape((-1, _assi_region.shape[0]))
        hp_total = raw_hp_total.mean(axis=0).reshape((1, -1)).repeat(raw_hp_total.shape[0], axis=0)

        # TODO 小心get_connectivity里调用的函数是什么!!!
        single_voxel_size = 25000
        degree = 500
        n_region = 378
    elif _hp_label == 'task_visual':
        _assi_region = np.array([1, 181])

        hp_path = '/public/home/ssct004t/project/wangjiexiang/criticality_test/OU_ZCS_cudasearch/cuda_test_code_debug/voxel_between_ampa/JFF_data_withsubcortical/V1_task/DA/Epopu_ext/d7_7/d_inv/da_task_1.5outer_regionwise_202403061545_12card/12card_newdebug_1.5_50ensemble_dt0.1/hp.npy'
        hp_total = np.load(hp_path).reshape((-1, _assi_region.shape[0]))

        # TODO 小心get_connectivity里调用的函数是什么!!!
        single_voxel_size = 25000
        degree = 500
        n_region = 378
    elif _hp_label == 'nohp':
        _assi_region = np.array([1])

        _hp_timepoint = 349  # The total time is _hp_timepoint*800 ms
        hp_total = np.zeros((_hp_timepoint, _assi_region.shape[0]))

        single_voxel_size = 25000
        degree = 500
        n_region = 378
    else:
        raise 'Wrong _hp_label!'

    dt_mnn = 0.5
    # dt_mnn = 0.05

    _seed = 20
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    config_for_Cond_MNN_DTB = {
        '_EI_ratio': 0.8,
        'is_cuda': False,
        'single_voxel_size': single_voxel_size,
        'degree': degree,
        'n_region': n_region,
        'xregion_gain_for_E_popu': _gui_voxel[0, 0] / initial_parameter_g_Li[0],  # shape is batchsize x 1
        'local_w_EE': _gui_voxel[0, 1] / initial_parameter_g_Li[0],  # diff between exh and inh
        'local_w_EI': _gui_voxel[0, 2] / initial_parameter_g_Li[0],  # diff between exh and inh
        'local_w_IE': _gui_voxel[1, 1] / initial_parameter_g_Li[1],  # diff between exh and inh
        'local_w_II': _gui_voxel[1, 2] / initial_parameter_g_Li[1],  # diff between exh and inh
        'hp': hp_total / initial_parameter_g_Li[0],
        '_assi_region': _assi_region,
        'hp_update_time': 800,
        'dt_mnn': dt_mnn,
        # 'dt_mnn': 0.5,
        'mnn_membrane_constant': 15,
    }

    config_for_Moment_exc_activation = {
        'tau_L': 1 / initial_parameter_g_Li[0],  # diff between exh and inh
        'tau_E': 4,
        'tau_I': 10,
        'VL': -70,
        'VE': 0,
        'VI': -70,
        'Vth': -50,
        'Vres': -60,
        'Tref': 2,  # diff between exh and inh
        'bgOU_mean': 0.36 / initial_parameter_g_Li[0],
        'bgOU_std': 1 / initial_parameter_g_Li[0],
        'bgOU_tau': 4,
    }

    config_for_Moment_inh_activation = {
        'tau_L': 1 / initial_parameter_g_Li[1],  # diff between exh and inh
        'tau_E': 4,
        'tau_I': 10,
        'VL': -70,
        'VE': 0,
        'VI': -70,
        'Vth': -50,
        'Vres': -60,
        'Tref': 1,  # diff between exh and inh
        'bgOU_mean': 0.54 / initial_parameter_g_Li[1],
        'bgOU_std': 1.2 / initial_parameter_g_Li[1],
        'bgOU_tau': 4,
    }

    T = round(0.8 * 1000 * 5)
    init_time = round(0.8 * 1000 * 4)

    _base_save_dir = os.path.join('202407101655_fixedpoint_gridsearch_perTR', 'NEWsave_data', )
    # _base_save_dir = os.path.join('202406302134', 'NEWsave_data', )
    # _base_save_dir = os.path.join('202406302134', 'save_data', f'{_hp_label}_{single_voxel_size}_{degree}_{n_region}_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{init_time}_hp{config_for_Cond_MNN_DTB["_assi_region"][0]}_{config_for_Cond_MNN_DTB["_assi_region"][-1]}_{config_for_Cond_MNN_DTB["degree"]}_{config_for_Cond_MNN_DTB["n_region"]}_{all_scale}_{_modified_J_EE}_{_modified_J_IE}_{_modified_J_EI}_{_modified_J_II}_{load_dir_list[0]}_{load_dir_list[-1]}_{_seed}_{config_for_Cond_MNN_DTB["is_cuda"]}')
    os.makedirs(_base_save_dir, exist_ok=True)

    newton_method_save_path = os.path.join(_base_save_dir, 'NEW_newton_method_save_path')
    os.makedirs(newton_method_save_path, exist_ok=True)
    # eigenvectors_save_path = os.path.join(_base_save_dir, '_eigenvectors_save_path_0803')
    # os.makedirs(eigenvectors_save_path, exist_ok=True)

    mpi_result_save_path = os.path.join(_base_save_dir, '_mpi_result_save_path_0803')
    os.makedirs(mpi_result_save_path, exist_ok=True)


    mnn = Cond_MNN_DTB(config_for_Cond_MNN_DTB=config_for_Cond_MNN_DTB,
                       config_for_Moment_exc_activation=config_for_Moment_exc_activation,
                       config_for_Moment_inh_activation=config_for_Moment_inh_activation)
    if rank == 0:
        print('Parameter initialization done')

    _external_current_forEpopu = torch.zeros(1, mnn.N, dtype=torch.float64, device=mnn.device)
    _external_current_forEpopu[:, _assi_region - 1] = torch.tensor(config_for_Cond_MNN_DTB['hp'][0, :].reshape((1, -1)),
                                                                   dtype=torch.float64, device=mnn.device)

    # save_data_done = True
    save_data_done = False
    # if not save_data_done and load_dir_list_idx < 200:  # TODO
    if not save_data_done:
        load_dir_list_idx_init = 200
        # Euler, no hp
        ue_record = np.load(os.path.join(r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202406302134', 'save_data', f'ue_record_{load_dir_list_idx_init}.npy'))
        se_record = np.load(os.path.join(r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202406302134', 'save_data', f'se_record_{load_dir_list_idx_init}.npy'))
        ui_record = np.load(os.path.join(r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202406302134', 'save_data', f'ui_record_{load_dir_list_idx_init}.npy'))
        si_record = np.load(os.path.join(r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202406302134', 'save_data', f'si_record_{load_dir_list_idx_init}.npy'))

        ue_oldinit = ue_record[-1, :]
        se_oldinit = se_record[-1, :]
        ui_oldinit = ui_record[-1, :]
        si_oldinit = si_record[-1, :]

        x0 = np.concatenate((ue_oldinit, se_oldinit, ui_oldinit, si_oldinit)).reshape((-1, 1))
        point_record = mnn.run_newton_fixedpoint_iter(init_point=x0, nsteps=20,
                                                      _external_current_forEpopu=_external_current_forEpopu)
        # point_record = mnn.run_newton_fixedpoint_iter(init_point=x0, nsteps=30)
        np.save(os.path.join(newton_method_save_path, f'point_record_{load_dir_list_idx}.npy'),
                point_record)
    else:
        point_record = np.load(
            os.path.join(newton_method_save_path, f'point_record_{load_dir_list_idx}.npy'))

    _fixedpoint_record.append(point_record[-1, :])

    ue_new = numpy2torch(point_record[-1, :config_for_Cond_MNN_DTB['n_region']].reshape((1, -1)),
                         is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    se_new = numpy2torch(
        point_record[-1, config_for_Cond_MNN_DTB['n_region']:config_for_Cond_MNN_DTB['n_region'] * 2].reshape((1, -1)),
        is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    ui_new = numpy2torch(
        point_record[-1, config_for_Cond_MNN_DTB['n_region'] * 2:config_for_Cond_MNN_DTB['n_region'] * 3].reshape(
            (1, -1)), is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    si_new = numpy2torch(
        point_record[-1, config_for_Cond_MNN_DTB['n_region'] * 3:config_for_Cond_MNN_DTB['n_region'] * 4].reshape(
            (1, -1)), is_cuda=config_for_Cond_MNN_DTB['is_cuda'])

    _test_ue, _test_se, _test_ui, _test_si = mnn.forward(ue_new, se_new, ui_new, si_new,
                                                         external_current_forEpopu=_external_current_forEpopu)
    ue_error = (ue_new - _test_ue).abs().max().item()
    se_error = (se_new - _test_se).abs().max().item()
    ui_error = (ui_new - _test_ui).abs().max().item()
    si_error = (si_new - _test_si).abs().max().item()
    _fixed_point_error = max(ue_error, se_error, ui_error, si_error)

    _jacobian_for_MA, _jacobian_summation_and_effective_current = mnn.backward_for_MA_and_summation_and_effective_current(ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1),
                                                 si_new.reshape(-1),
                                                 _external_current_forEpopu=_external_current_forEpopu)
    np.save(os.path.join(_base_save_dir, f'_jacobian_for_MA_{load_dir_list_idx}.npy'), _jacobian_for_MA)
    np.save(os.path.join(_base_save_dir, f'_jacobian_summation_and_effective_current_{load_dir_list_idx}.npy'), _jacobian_summation_and_effective_current)

    # _jacobian_onestep = mnn.backward_for_onestep(ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1),
    #                                              si_new.reshape(-1),
    #                                              _external_current_forEpopu=_external_current_forEpopu)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_{load_dir_list_idx}.npy'), _jacobian_onestep)
    #
    # _jacobian_onestep_forexternalcurrent = mnn.backward_for_onestep_forexternalcurrent(ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1),
    #                                              si_new.reshape(-1),
    #                                              _external_current_forEpopu=_external_current_forEpopu.reshape(-1))
    # print("_jacobian_onestep_forexternalcurrent", _jacobian_onestep_forexternalcurrent.shape)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_forexternalcurrent_{load_dir_list_idx}.npy'), _jacobian_onestep_forexternalcurrent)

    var_e_new = se_new ** 2
    var_i_new = si_new ** 2
    _jacobian_onestep_mapping_variance = mnn.backward_for_onestep_mapping_variance(ue_new.reshape(-1), var_e_new.reshape(-1), ui_new.reshape(-1),
                                                                  var_i_new.reshape(-1),
                                                                  _external_current_forEpopu=_external_current_forEpopu)
    np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_mapping_variance_{load_dir_list_idx}.npy'), _jacobian_onestep_mapping_variance)

    _jacobian_onestep_forexternalcurrent_mapping_variance = mnn.backward_for_onestep_forexternalcurrent_mapping_variance(
        ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1), si_new.reshape(-1),
        _external_current_forEpopu=_external_current_forEpopu.reshape(-1))
    np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_forexternalcurrent_mapping_variance_{load_dir_list_idx}.npy'), _jacobian_onestep_forexternalcurrent_mapping_variance)


    eigenvalues, eigenvectors = np.linalg.eig(_jacobian_onestep_mapping_variance - np.eye(_jacobian_onestep_mapping_variance.shape[0]))

    # 提取特征值的实部
    real_parts = np.real(eigenvalues)
    # 检查实部的最大值是否小于0
    max_real_part = np.max(real_parts)
    min_real_part = np.min(real_parts)

    print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx
          , "_fixed_point_error", _fixed_point_error, "max_real_part", max_real_part, 'done')

    _real_parts_argsort = np.argsort(real_parts)[::-1]
    _eigenvectors_argsort_total = eigenvectors[:, _real_parts_argsort]
    eigenvalues = eigenvalues[_real_parts_argsort]

    _lefteigenvalues, _lefteigenvectors = np.linalg.eig(_jacobian_onestep_mapping_variance.T - np.eye(_jacobian_onestep_mapping_variance.shape[0]))
    _leftreal_parts = np.real(_lefteigenvalues)
    _leftreal_parts_argsort = np.argsort(_leftreal_parts)[::-1]
    _lefteigenvalues = _lefteigenvalues[_leftreal_parts_argsort]
    _lefteigenvectors_argsort_total = _lefteigenvectors[:, _leftreal_parts_argsort]

    assert np.max(np.abs(np.real(eigenvalues - _lefteigenvalues))) < 1e-6
    eigenvalue_diff_abs = (np.abs(eigenvalues - _lefteigenvalues))
    eigenvalue_diff_abs_indices = eigenvalue_diff_abs > 1e-7
    eigenvalues[eigenvalue_diff_abs_indices] = np.conj(eigenvalues[eigenvalue_diff_abs_indices])
    _eigenvectors_argsort_total[:, eigenvalue_diff_abs_indices] = np.conj(_eigenvectors_argsort_total[:, eigenvalue_diff_abs_indices])

    _righteigenvalues_record.append(eigenvalues)
    _fixed_point_error_record.append(_fixed_point_error)
    _lefteigenvalues_record.append(_lefteigenvalues)

    _MNN_simu_load_path = os.path.join('/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739', 'save_data_0.05_360000')

    # _base_project_save_path = 'dynamics_reconstruct_odeproject_variance_online_new'
    # os.makedirs(_base_project_save_path, exist_ok=True)

    save_eigenmode = False
    if save_eigenmode:
        TB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed = io.loadmat(
            '/public/home/ssct004t/project/wangjiexiang/moment-neural-network/DTB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed.mat')
        np_DTB_3_res_2mm_Region_Label = TB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed[
            'DTB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed'].astype(np.float64)

        np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 379] = 0
        np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 380] = 0
        # for _ttt in range(361, 379):
        #     np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == _ttt] = 0

        x = np.arange(np_DTB_3_res_2mm_Region_Label.shape[0])
        y = np.arange(np_DTB_3_res_2mm_Region_Label.shape[1])
        z = np.arange(np_DTB_3_res_2mm_Region_Label.shape[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        xx_selected = xx[np_DTB_3_res_2mm_Region_Label != 0]
        yy_selected = yy[np_DTB_3_res_2mm_Region_Label != 0]
        zz_selected = zz[np_DTB_3_res_2mm_Region_Label != 0]

        _nii_eigenvectors_basesavepath = os.path.join(_base_save_dir, 'nii_eigenvectors')
        os.makedirs(_nii_eigenvectors_basesavepath, exist_ok=True)
        _nii_eigenvectors_savepath = os.path.join(_nii_eigenvectors_basesavepath, f'{load_dir_list_idx}')
        os.makedirs(_nii_eigenvectors_savepath, exist_ok=True)
        _pic_eigenvectors_savepath = os.path.join(_nii_eigenvectors_basesavepath, 'picture', f'{load_dir_list_idx}')
        os.makedirs(_pic_eigenvectors_savepath, exist_ok=True)

        _eigenvectors_argsort_selected_idx = np.arange(10)
        _eigenvectors_argsort_total_reshape = _eigenvectors_argsort_total.reshape((4, n_region, _eigenvectors_argsort_total.shape[1]))
        _eigenvectors_argsort_selected = _eigenvectors_argsort_total_reshape[:, :, _eigenvectors_argsort_selected_idx]

        HCP_region = np.arange(1, n_region+1)
        _nii_eigenvectors_angle_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
        _nii_eigenvectors_abs_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
        _nii_eigenvectors_real_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
        _nii_eigenvectors_imag_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))

        assert ((np.sqrt((_eigenvectors_argsort_selected ** 2).reshape((n_region*4, _eigenvectors_argsort_selected_idx.shape[0])).sum(axis=0)) - _eigenvectors_argsort_selected_idx.shape[0]) < 1e-6).all()

        for ii in range(_eigenvectors_argsort_selected.shape[2]):
            for _variable_idx in range(4):
                _eigenvectors_single = _eigenvectors_argsort_selected[_variable_idx, :, ii]
                _eigenvectors_single_angle = np.angle(_eigenvectors_single)
                _eigenvectors_single_abs = np.abs(_eigenvectors_single)
                _eigenvectors_single_real = np.real(_eigenvectors_single)
                _eigenvectors_single_imag = np.imag(_eigenvectors_single)

                _nii_eigenvectors_angle = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
                _nii_eigenvectors_abs = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
                _nii_eigenvectors_real = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
                _nii_eigenvectors_imag = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
                for iii in range(len(HCP_region)):
                    _nii_eigenvectors_angle[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_angle[iii]
                    _nii_eigenvectors_abs[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_abs[iii]
                    _nii_eigenvectors_real[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_real[iii]
                    _nii_eigenvectors_imag[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_imag[iii]

                _nii_eigenvectors_angle_selected = _nii_eigenvectors_angle[np_DTB_3_res_2mm_Region_Label != 0]
                _nii_eigenvectors_abs_selected = _nii_eigenvectors_abs[np_DTB_3_res_2mm_Region_Label != 0]
                _nii_eigenvectors_real_selected = _nii_eigenvectors_real[np_DTB_3_res_2mm_Region_Label != 0]
                _nii_eigenvectors_imag_selected = _nii_eigenvectors_imag[np_DTB_3_res_2mm_Region_Label != 0]

                _nii_eigenvectors_angle_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_angle_selected
                _nii_eigenvectors_abs_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_abs_selected
                _nii_eigenvectors_real_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_real_selected
                _nii_eigenvectors_imag_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_imag_selected

        np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_angle_{load_dir_list_idx}'), _nii_eigenvectors_angle_iterrecords)
        np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_abs_{load_dir_list_idx}'), _nii_eigenvectors_abs_iterrecords)
        np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_real_{load_dir_list_idx}'), _nii_eigenvectors_real_iterrecords)
        np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_imag_{load_dir_list_idx}'), _nii_eigenvectors_imag_iterrecords)

    # fixed_point = point_record[-1, :]
    # sigma_idxes = np.concatenate((np.arange(n_region, n_region*2), np.arange(n_region*3, n_region*4)), axis=0)
    # fixed_point[sigma_idxes] = fixed_point[sigma_idxes] ** 2
    #
    # _hp_TR_idx_scalevalue = 0
    # dynamics_project_Encapsulation_ode_aspect_variancemapping(nregion=n_region,
    #                                fixed_point=fixed_point,
    #                                np_all_righteigenvector_results=_eigenvectors_argsort_total,
    #                                np_all_lefteigenvector_results=_lefteigenvectors_argsort_total,
    #                                _MNN_simu_load_path=_MNN_simu_load_path,
    #                                load_dir_list_idx=load_dir_list_idx,
    #                                _hp_TR_idx_scalevalue=_hp_TR_idx_scalevalue,
    #                                           _base_save_path=_base_project_save_path)


# all_fixedpoint_results = comm.gather(_fixedpoint_record, root=0)
# all_lefteigenvalues_results = comm.gather(_lefteigenvalues_record, root=0)
# all_righteigenvalues_results = comm.gather(_righteigenvalues_record, root=0)
# all_error_results = comm.gather(_fixed_point_error_record, root=0)
#
#
# if rank == 0:
#     filtered_all_fixedpoint_results = [item for item in all_fixedpoint_results if item]
#     filtered_all_righteigenvalues_results = [item for item in all_righteigenvalues_results if item]
#     filtered_all_lefteigenvalues_results = [item for item in all_lefteigenvalues_results if item]
#     filtered_all_error_results = [item for item in all_error_results if item]
#
#     np_all_fixedpoint_results = np.concatenate(filtered_all_fixedpoint_results, axis=0)
#     np_all_lefteigenvalues_results = np.concatenate(filtered_all_lefteigenvalues_results, axis=0)
#     np_all_righteigenvalues_results = np.concatenate(filtered_all_righteigenvalues_results, axis=0)
#     np_all_error_results = np.concatenate(filtered_all_error_results, axis=0)
#
#     print("done", "np_all_righteigenvalues_results.shape", np_all_righteigenvalues_results.shape,
#           "np_all_error_results.shape", np_all_error_results.shape,
#           "np_all_fixedpoint_results.shape", np_all_fixedpoint_results.shape,
#           )
#
#     np.save(os.path.join(mpi_result_save_path, f"np_all_fixedpoint_results_{load_dir_list[0]}_{load_dir_list[-1]}.npy"), np_all_fixedpoint_results)
#     np.save(os.path.join(mpi_result_save_path, f"np_all_righteigenvalues_results_{load_dir_list[0]}_{load_dir_list[-1]}.npy"), np_all_righteigenvalues_results)
#     np.save(os.path.join(mpi_result_save_path, f"np_all_lefteigenvalues_results_{load_dir_list[0]}_{load_dir_list[-1]}.npy"), np_all_lefteigenvalues_results)
#     np.save(os.path.join(mpi_result_save_path, f"np_all_error_results_{load_dir_list[0]}_{load_dir_list[-1]}.npy"), np_all_error_results)
