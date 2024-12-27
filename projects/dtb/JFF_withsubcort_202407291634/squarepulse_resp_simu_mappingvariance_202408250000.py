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
    parser.add_argument("--para_index", type=str, default="1")
    args = parser.parse_args()
    return args



# load_dir_list = list(np.arange(200, 261))
# load_dir_list = list(np.arange(200, 201))
# load_dir_list = list(np.arange(209, 212))
# load_dir_list = list(np.arange(209, 210))
# load_dir_list = list(np.arange(211, 212))
# load_dir_list = list(np.arange(240, 241))
# load_dir_list = list(np.arange(209, 211, 1))
# load_dir_list = list(np.arange(210, 230, 1))
# load_dir_list = list(np.arange(201, 209, 1))
# load_dir_list = list(np.arange(0, 1, 10))
# load_dir_list = list(np.arange(200, 201, 10))
# load_dir_list = list(np.arange(0, 191, 10))
# load_dir_list = [200, 209, 240]
load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
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

    # dt_mnn = 0.5
    dt_mnn = 0.01
    # dt_mnn = 0.0005
    # dt_mnn = 0.001
    # dt_mnn = 0.002
    # dt_mnn = 0.005
    save_dt = 0.01

    _seed = 20
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    # stimulus_strength_list = [0, 5, 10, 20, 40]
    stimulus_strength_list = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # stimulus_strength_list = [0.5, 1, 2, 3, 4]
    # stimulus_strength_list = [0, 0.5, 1, 2, 3, 4, 5, 10, 12, 14, 16, 18, 20]
    # stimulus_strength_list = [6, 7, 8, 9, 12, 14, 16, 18, 20]
    # stimulus_strength_list = [0, 0.5, 1, 2, 3, 4, 5, 10]
    # stimulus_strength_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10]

    # stimulus_strength = 1000
    # stimulus_strength = 0
    # stimulus_strength = 100
    # stimulus_strength = 0.5
    args = get_args()
    rela_stimulus_strength_idx = args.para_index
    stimulus_strength = stimulus_strength_list[int(rela_stimulus_strength_idx)]
    # stimulus_strength = 2
    # stimulus_strength = 1
    # stimulus_strength = 10

    stimulus_region = np.array([1, 181])
    stimulus_region__dir = '_'.join(map(str, stimulus_region))

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
        'stimulus_strength': stimulus_strength,
        'stimulus_region': stimulus_region,
        # 'dt_mnn': 0.5,
        'mnn_membrane_constant': 11,
        'save_dt': save_dt,
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
    # T = round(0.8 * 1000 * 5)
    # T = round(0.9 * 1000 * 1)
    # T = round(0.8 * 1000 * 10)
    # T = round(0.8 * 1000 * 1)
    # T = round(0.4 * 1000 * 1)
    # init_time = round(0.805 * 1000)
    # stimulus_strength_step = round(round(0.8 * 1000) / dt_mnn)

    # init_time = round(0.55 * 1000)
    # stimulus_strength_time = round(0.05 * 1000)

    # init_time = round(1.5 * 1000)
    # stimulus_strength_time = round(1 * 1000)

    # init_time = round(0.5 * 1000 + 1)
    # stimulus_strength_time = round(1)
    # init_time = round(1 * 1000)
    # stimulus_strength_time = round(0.5 * 1000)

    # init_time = round(0.7 * 1000)
    # stimulus_strength_time = round(0.2 * 1000)

    init_time = round(0.6 * 1000)
    stimulus_strength_time = round(0.1 * 1000)
    stimulus_strength_step = round(stimulus_strength_time / dt_mnn)

    _base_save_dir = os.path.join('squpul_resp_simu_mappingvariance_202408250000_moreprecise', 'diff_dtmnn', f'{_hp_label}_{stimulus_strength}_{stimulus_region__dir}_{stimulus_strength_time}_{single_voxel_size}_{degree}_{n_region}_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{init_time}_hp{config_for_Cond_MNN_DTB["_assi_region"][0]}_{config_for_Cond_MNN_DTB["_assi_region"][-1]}_{config_for_Cond_MNN_DTB["degree"]}_{config_for_Cond_MNN_DTB["n_region"]}_{all_scale}_{_modified_J_EE}_{_modified_J_IE}_{_modified_J_EI}_{_modified_J_II}_{load_dir_list[0]}_{load_dir_list[-1]}_{_seed}_{config_for_Cond_MNN_DTB["is_cuda"]}')
    # _base_save_dir = os.path.join('squpul_resp_simu_mappingvariance_202408250000_moreprecise', 'save_data', f'{_hp_label}_{stimulus_strength}_{stimulus_region__dir}_{stimulus_strength_time}_{single_voxel_size}_{degree}_{n_region}_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{init_time}_hp{config_for_Cond_MNN_DTB["_assi_region"][0]}_{config_for_Cond_MNN_DTB["_assi_region"][-1]}_{config_for_Cond_MNN_DTB["degree"]}_{config_for_Cond_MNN_DTB["n_region"]}_{all_scale}_{_modified_J_EE}_{_modified_J_IE}_{_modified_J_EI}_{_modified_J_II}_{load_dir_list[0]}_{load_dir_list[-1]}_{_seed}_{config_for_Cond_MNN_DTB["is_cuda"]}')
    # _base_save_dir = os.path.join('squpul_resp_simu_mappingvariance_202408250000_moreprecise', 'diffstimutime_save_data', f'{_hp_label}_{stimulus_strength}_{stimulus_region__dir}_{stimulus_strength_time}_{single_voxel_size}_{degree}_{n_region}_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{init_time}_hp{config_for_Cond_MNN_DTB["_assi_region"][0]}_{config_for_Cond_MNN_DTB["_assi_region"][-1]}_{config_for_Cond_MNN_DTB["degree"]}_{config_for_Cond_MNN_DTB["n_region"]}_{all_scale}_{_modified_J_EE}_{_modified_J_IE}_{_modified_J_EI}_{_modified_J_II}_{load_dir_list[0]}_{load_dir_list[-1]}_{_seed}_{config_for_Cond_MNN_DTB["is_cuda"]}')
    os.makedirs(_base_save_dir, exist_ok=True)

    newton_method_save_path = os.path.join(_base_save_dir, 'NEW_newton_method_save_path')
    os.makedirs(newton_method_save_path, exist_ok=True)
    # eigenvectors_save_path = os.path.join(_base_save_dir, '_eigenvectors_save_path_0803')
    # os.makedirs(eigenvectors_save_path, exist_ok=True)

    mpi_result_save_path = os.path.join(_base_save_dir, '_mpi_result_save_path_0824')
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
    if not save_data_done:
        io.savemat(os.path.join(_base_save_dir, f'config_for_Cond_MNN_DTB.mat'), config_for_Cond_MNN_DTB)
        io.savemat(os.path.join(_base_save_dir, f'config_for_Moment_exc_activation.mat'),
                   config_for_Moment_exc_activation)
        io.savemat(os.path.join(_base_save_dir, f'config_for_Moment_inh_activation.mat'),
                   config_for_Moment_inh_activation)

        if load_dir_list_idx <= 209:
            _load_init_path = r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407291634/202407101655_fixedpoint_gridsearch_perTR/NEWsave_data/NEW_newton_method_save_path'
            x0 = np.load(os.path.join(_load_init_path, f'point_record_{load_dir_list_idx}.npy'))[-1, :]
            x0 = numpy2torch(x0, is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))
        else:
            _load_init_path = r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407291634/squpul_resp_simu_mappingvariance_202408241520/save_data/restintero_hptimeaverage_0_1_181_100_25000_500_378_0.001_4000_600_hp57_378_500_378_0.25_0.05_0.55_0.4_0.1_0_260_20_False'
            ue_oldinit = np.load(os.path.join(_load_init_path, f'ue_record_{load_dir_list_idx}.npy'))[-1, :]
            se_oldinit = np.load(os.path.join(_load_init_path, f'se_record_{load_dir_list_idx}.npy'))[-1, :]
            ui_oldinit = np.load(os.path.join(_load_init_path, f'ui_record_{load_dir_list_idx}.npy'))[-1, :]
            si_oldinit = np.load(os.path.join(_load_init_path, f'si_record_{load_dir_list_idx}.npy'))[-1, :]

            x0 = np.concatenate((ue_oldinit, se_oldinit, ui_oldinit, si_oldinit)).reshape((1, -1))
            x0 = numpy2torch(x0, is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))

        ue_record, se_record, ui_record, si_record = mnn.run_with_impulse_stimulus_mapping_variance(
            T=T, init_time=init_time, init_point=x0,
            stimulus_strength=stimulus_strength, stimulus_region=stimulus_region,
            stimulus_strength_step=stimulus_strength_step, savebold=False)

        np.save(os.path.join(_base_save_dir, f'ue_record_{load_dir_list_idx}.npy'), ue_record)
        np.save(os.path.join(_base_save_dir, f'se_record_{load_dir_list_idx}.npy'), se_record)
        np.save(os.path.join(_base_save_dir, f'ui_record_{load_dir_list_idx}.npy'), ui_record)
        np.save(os.path.join(_base_save_dir, f'si_record_{load_dir_list_idx}.npy'), si_record)
    else:
        ue_record = np.load(os.path.join(_base_save_dir, f'ue_record_{load_dir_list_idx}.npy'))
        se_record = np.load(os.path.join(_base_save_dir, f'se_record_{load_dir_list_idx}.npy'))
        ui_record = np.load(os.path.join(_base_save_dir, f'ui_record_{load_dir_list_idx}.npy'))
        si_record = np.load(os.path.join(_base_save_dir, f'si_record_{load_dir_list_idx}.npy'))

    ue_new = numpy2torch(ue_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))
    se_new = numpy2torch(se_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))
    ui_new = numpy2torch(ui_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))
    si_new = numpy2torch(si_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda']).reshape((1, -1))


    _test_ue, _test_se, _test_ui, _test_si = mnn.forward(ue_new, se_new, ui_new, si_new,
                                                         external_current_forEpopu=_external_current_forEpopu)
    ue_error = (ue_new - _test_ue).abs().max().item()
    var_e_error = (se_new ** 2 - _test_se ** 2).abs().max().item()
    ui_error = (ui_new - _test_ui).abs().max().item()
    var_i_error = (si_new ** 2 - _test_si ** 2).abs().max().item()
    _fixed_point_error = max(ue_error, var_e_error, ui_error, var_i_error)

    # _jacobian_for_MA, _jacobian_summation_and_effective_current = mnn.backward_for_MA_and_summation_and_effective_current(ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1),
    #                                              si_new.reshape(-1),
    #                                              _external_current_forEpopu=_external_current_forEpopu)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_for_MA_{load_dir_list_idx}.npy'), _jacobian_for_MA)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_summation_and_effective_current_{load_dir_list_idx}.npy'), _jacobian_summation_and_effective_current)

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

    # var_e_new = se_new ** 2
    # var_i_new = si_new ** 2
    # _jacobian_onestep_mapping_variance = mnn.backward_for_onestep_mapping_variance(ue_new.reshape(-1), var_e_new.reshape(-1), ui_new.reshape(-1),
    #                                                               var_i_new.reshape(-1),
    #                                                               _external_current_forEpopu=_external_current_forEpopu)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_mapping_variance_{load_dir_list_idx}.npy'), _jacobian_onestep_mapping_variance)
    #
    # _jacobian_onestep_forexternalcurrent_mapping_variance = mnn.backward_for_onestep_forexternalcurrent_mapping_variance(
    #     ue_new.reshape(-1), se_new.reshape(-1), ui_new.reshape(-1), si_new.reshape(-1),
    #     _external_current_forEpopu=_external_current_forEpopu.reshape(-1))
    # np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_forexternalcurrent_mapping_variance_{load_dir_list_idx}.npy'), _jacobian_onestep_forexternalcurrent_mapping_variance)
    #
    #
    # eigenvalues, eigenvectors = np.linalg.eig(_jacobian_onestep_mapping_variance - np.eye(_jacobian_onestep_mapping_variance.shape[0]))

    # # 提取特征值的实部
    # real_parts = np.real(eigenvalues)
    # # 检查实部的最大值是否小于0
    # max_real_part = np.max(real_parts)
    # min_real_part = np.min(real_parts)

    print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx
          , "_fixed_point_error", _fixed_point_error, 'done')

    # _real_parts_argsort = np.argsort(real_parts)[::-1]
    # _eigenvectors_argsort_total = eigenvectors[:, _real_parts_argsort]
    # eigenvalues = eigenvalues[_real_parts_argsort]
    #
    # _lefteigenvalues, _lefteigenvectors = np.linalg.eig(_jacobian_onestep_mapping_variance.T - np.eye(_jacobian_onestep_mapping_variance.shape[0]))
    # _leftreal_parts = np.real(_lefteigenvalues)
    # _leftreal_parts_argsort = np.argsort(_leftreal_parts)[::-1]
    # _lefteigenvalues = _lefteigenvalues[_leftreal_parts_argsort]
    # _lefteigenvectors_argsort_total = _lefteigenvectors[:, _leftreal_parts_argsort]
    #
    # assert np.max(np.abs(np.real(eigenvalues - _lefteigenvalues))) < 1e-6
    # eigenvalue_diff_abs = (np.abs(eigenvalues - _lefteigenvalues))
    # eigenvalue_diff_abs_indices = eigenvalue_diff_abs > 1e-7
    # eigenvalues[eigenvalue_diff_abs_indices] = np.conj(eigenvalues[eigenvalue_diff_abs_indices])
    # _eigenvectors_argsort_total[:, eigenvalue_diff_abs_indices] = np.conj(_eigenvectors_argsort_total[:, eigenvalue_diff_abs_indices])
    #
    # _righteigenvalues_record.append(eigenvalues)
    # _fixed_point_error_record.append(_fixed_point_error)
    # _lefteigenvalues_record.append(_lefteigenvalues)
    #
    # _MNN_simu_load_path = os.path.join('/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739', 'save_data_0.05_360000')
    #
    # _base_project_save_path = 'dynamics_reconstruct_odeproject_variance_online_new'
    # os.makedirs(_base_project_save_path, exist_ok=True)


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
