# -*- coding: utf-8 -*-
# @author: yiren14mihu
# @file: wjx_criticality_revise.py
# @time: 2024/5/10 23:43
import os.path

from matplotlib import pyplot as plt
import torch
import numpy as np
from mnn_dtb_withhp_float64_202407101004 import Cond_MNN_DTB
import argparse
from mpi4py import MPI
from scipy import io
from torch.autograd.functional import jacobian
from utils.helpers import numpy2torch, torch_2_numpy
import nibabel as nib
import h5py
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="MNN")
    parser.add_argument("--multiply_value", type=str, default="1")
    args = parser.parse_args()
    return args


# load_dir_list = list(np.arange(200, 261, 1))
load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))

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


_error_amplitude_list = []

# _nii_eigenvectors_angle_iterrecords_totalparams = []
# _nii_eigenvectors_abs_iterrecords_totalparams = []
# _nii_eigenvectors_real_iterrecords_totalparams = []
# _nii_eigenvectors_imag_iterrecords_totalparams = []

for rela_param_idx in range(len(params)):
    _gui_ampa_scalevalue = params[rela_param_idx]
    load_dir_list_idx = selected_load_dir_list[rela_param_idx]

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

    _gui_voxel = np.array([[_gui_ampa_scalevalue * _J_EE_scaled / modified_parameter_tao_ui[0], _modified_J_EE * _J_EE_scaled / modified_parameter_tao_ui[0], _modified_J_EI * _J_EI_scaled / modified_parameter_tao_ui[2], 0],
                           [1.1 * 0 * _J_IE_scaled / modified_parameter_tao_ui[0], _modified_J_IE * _J_IE_scaled / modified_parameter_tao_ui[0], _modified_J_II * _J_II_scaled / modified_parameter_tao_ui[2], 0]], dtype=np.float32)

    initial_parameter_g_Li = np.array([1 / 20, 1 / 10])

    _hp_label = 'rest_intero'
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
    # dt_mnn = 0.1
    # dt_mnn = 0.05
    # dt_mnn = 0.025
    dt_mnn = 0.01
    save_dt = 1

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
        'save_dt': save_dt,
        # 'dt_mnn': 0.5,
        'mnn_membrane_constant': 11,
    }


    config_for_Moment_exc_activation = {
        'tau_L': 1 / initial_parameter_g_Li[0], # diff between exh and inh
        'tau_E': 4,
        'tau_I': 10,
        'VL': -70,
        'VE': 0,
        'VI': -70,
        'Vth': -50,
        'Vres': -60,
        'Tref': 2, # diff between exh and inh
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

    # T = round(0.8 * 1000 * 150)
    # T = round(0.8 * 1000 * 5)
    # T = round(0.8 * 1000 * 10)
    T = round(0.8 * 1000 * hp_total.shape[0])
    # T = 20
    # T = round(0.8 * 1000 * 349)
    # T = round(0.8 * 1000 * 349)
    # T = round(0.8 * 1000 * 19.5)
    # T = round(0.8 * 1000 * 20)
    init_time = round(0.8 * 1000 * 4)

    # _base_save_dir = os.path.join('202407131739', f'save_data_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}', )
    # _base_save_dir = os.path.join('202407131449', f'save_data_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}', )
    # _base_save_dir = os.path.join('202406302134', f'save_data_{config_for_Cond_MNN_DTB["dt_mnn"]}', )
    # _base_save_dir = os.path.join('202406302134', 'save_data', )
    _base_save_dir = os.path.join('202407131739', f'save_data_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{config_for_Cond_MNN_DTB["mnn_membrane_constant"]}', f'{_hp_label}_{single_voxel_size}_{degree}_{n_region}_{config_for_Cond_MNN_DTB["dt_mnn"]}_{T}_{init_time}_hp{config_for_Cond_MNN_DTB["_assi_region"][0]}_{config_for_Cond_MNN_DTB["_assi_region"][-1]}_{config_for_Cond_MNN_DTB["degree"]}_{config_for_Cond_MNN_DTB["n_region"]}_{all_scale}_{_modified_J_EE}_{_modified_J_IE}_{_modified_J_EI}_{_modified_J_II}_{load_dir_list[0]}_{load_dir_list[-1]}_{_seed}_{config_for_Cond_MNN_DTB["is_cuda"]}')
    os.makedirs(_base_save_dir, exist_ok=True)

    mnn = Cond_MNN_DTB(config_for_Cond_MNN_DTB=config_for_Cond_MNN_DTB, config_for_Moment_exc_activation=config_for_Moment_exc_activation,
                       config_for_Moment_inh_activation=config_for_Moment_inh_activation)
    if rank == 0:
        print('Parameter initialization done')

    # save_data_done = True
    save_data_done = False
    if not save_data_done:
        io.savemat(os.path.join(_base_save_dir, f'config_for_Cond_MNN_DTB.mat'), config_for_Cond_MNN_DTB)
        io.savemat(os.path.join(_base_save_dir, f'config_for_Moment_exc_activation.mat'),
                   config_for_Moment_exc_activation)
        io.savemat(os.path.join(_base_save_dir, f'config_for_Moment_inh_activation.mat'),
                   config_for_Moment_inh_activation)

        ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record = mnn.run(T=T, init_time=init_time)

        np.save(os.path.join(_base_save_dir, f'ue_record_{load_dir_list_idx}.npy'), ue_record)
        np.save(os.path.join(_base_save_dir, f'se_record_{load_dir_list_idx}.npy'), se_record)
        np.save(os.path.join(_base_save_dir, f'ui_record_{load_dir_list_idx}.npy'), ui_record)
        np.save(os.path.join(_base_save_dir, f'si_record_{load_dir_list_idx}.npy'), si_record)
        np.save(os.path.join(_base_save_dir, f'voxelwise_firingrate_record_{load_dir_list_idx}.npy'), voxelwise_firingrate_record)
        np.save(os.path.join(_base_save_dir, f'BOLD_record_{load_dir_list_idx}.npy'), BOLD_record)
    else:
        ue_record = np.load(os.path.join(_base_save_dir, f'ue_record_{load_dir_list_idx}.npy'))
        se_record = np.load(os.path.join(_base_save_dir, f'se_record_{load_dir_list_idx}.npy'))
        ui_record = np.load(os.path.join(_base_save_dir, f'ui_record_{load_dir_list_idx}.npy'))
        si_record = np.load(os.path.join(_base_save_dir, f'si_record_{load_dir_list_idx}.npy'))
        voxelwise_firingrate_record = np.load(os.path.join(_base_save_dir, f'voxelwise_firingrate_record_{load_dir_list_idx}.npy'))
        BOLD_record = np.load(os.path.join(_base_save_dir, f'BOLD_record_{load_dir_list_idx}.npy'))

    ue = numpy2torch(ue_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    se = numpy2torch(se_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    ui = numpy2torch(ui_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda'])
    si = numpy2torch(si_record[-1, :], is_cuda=config_for_Cond_MNN_DTB['is_cuda'])

    _residual_time = 400
    init_steps = round(init_time / save_dt)
    ue_record_discardtransient = ue_record[init_steps:].reshape((-1, round(config_for_Cond_MNN_DTB['hp_update_time'] / save_dt), n_region))[:, -round(_residual_time / save_dt):]
    ui_record_discardtransient = ui_record[init_steps:].reshape((-1, round(config_for_Cond_MNN_DTB['hp_update_time'] / save_dt), n_region))[:, -round(_residual_time / save_dt):]
    ue_record_discardtransient_amplitude = (ue_record_discardtransient.max(axis=1) - ue_record_discardtransient.min(axis=1)).mean()
    ui_record_discardtransient_amplitude = (ui_record_discardtransient.max(axis=1) - ui_record_discardtransient.min(axis=1)).mean()

    _jacobian_onestep = mnn.backward_for_onestep(ue, se, ui, si)
    # np.save(os.path.join(_base_save_dir, f'_jacobian_onestep_{load_dir_list_idx}.npy'), _jacobian_onestep)

    eigenvalues, eigenvectors = np.linalg.eig(_jacobian_onestep - np.eye(_jacobian_onestep.shape[0]))

    # 提取特征值的实部
    real_parts = np.real(eigenvalues)
    # 检查实部的最大值是否小于0
    max_real_part = np.max(real_parts)
    min_real_part = np.min(real_parts)

    # _real_parts_argsort = np.argsort(real_parts)[::-1]
    # _eigenvectors_argsort_total = eigenvectors[:, _real_parts_argsort]
    # _eigenvectors_argsort_total_reshape = _eigenvectors_argsort_total.reshape((4, n_region, _eigenvectors_argsort_total.shape[1]))
    #
    # _eigenvectors_argsort_selected_idx = np.concatenate((np.arange(10), np.arange(_eigenvectors_argsort_total.shape[1]-10, _eigenvectors_argsort_total.shape[1])))
    # _eigenvectors_argsort_selected = _eigenvectors_argsort_total_reshape[:, :, _eigenvectors_argsort_selected_idx]
    #
    # DTB_3_res_2mm_Region_Label = nib.load(
    #     r'/public/home/ssct004t/project/wangjiexiang/moment-neural-network/DTB_3_res_2mm_Region_Label.nii')
    # # DTB_3_res_2mm_Region_Label = nib.load(
    # #     r'E:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\DTB_3_res_2mm_Region_Label.nii')
    # np_DTB_3_res_2mm_Region_Label = np.array(DTB_3_res_2mm_Region_Label.dataobj)
    # np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 379] = 0
    # np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 380] = 0
    # for _ttt in range(361, 379):
    #     np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == _ttt] = 0
    #
    # x = np.arange(np_DTB_3_res_2mm_Region_Label.shape[0])
    # y = np.arange(np_DTB_3_res_2mm_Region_Label.shape[1])
    # z = np.arange(np_DTB_3_res_2mm_Region_Label.shape[2])
    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # xx_selected = xx[np_DTB_3_res_2mm_Region_Label != 0]
    # yy_selected = yy[np_DTB_3_res_2mm_Region_Label != 0]
    # zz_selected = zz[np_DTB_3_res_2mm_Region_Label != 0]
    #
    # _nii_eigenvectors_basesavepath = os.path.join(_base_save_dir, 'nii_eigenvectors')
    # os.makedirs(_nii_eigenvectors_basesavepath, exist_ok=True)
    # _nii_eigenvectors_savepath = os.path.join(_nii_eigenvectors_basesavepath, f'{load_dir_list_idx}')
    # os.makedirs(_nii_eigenvectors_savepath, exist_ok=True)
    # _pic_eigenvectors_savepath = os.path.join(_nii_eigenvectors_basesavepath, 'picture', f'{load_dir_list_idx}')
    # os.makedirs(_pic_eigenvectors_savepath, exist_ok=True)
    #
    # _variable_name = ['ue', 'se', 'ui', 'si']
    #
    # # HCPregion_name_label = r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/HCPex_3mm_modified_label_mayraw.csv'
    # # HCPex_3mm_modified_label = pd.read_csv(HCPregion_name_label)
    # # reorder_region_label_raw = np.array(HCPex_3mm_modified_label['Label'])
    # # HCP_region = reorder_region_label_raw[:n_region]
    # HCP_region = np.arange(1, n_region+1)
    #
    # _nii_eigenvectors_angle_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
    # _nii_eigenvectors_abs_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
    # _nii_eigenvectors_real_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
    # _nii_eigenvectors_imag_iterrecords = np.zeros((_eigenvectors_argsort_selected.shape[2], 4, xx_selected.shape[0]))
    #
    # assert ((np.sqrt((_eigenvectors_argsort_selected ** 2).reshape((n_region*4, _eigenvectors_argsort_selected_idx.shape[0])).sum(axis=0)) - _eigenvectors_argsort_selected_idx.shape[0]) < 1e-6).all()
    #
    # for ii in range(_eigenvectors_argsort_selected.shape[2]):
    #     for _variable_idx in range(4):
    #         _eigenvectors_single = _eigenvectors_argsort_selected[_variable_idx, :, ii]
    #         _eigenvectors_single_angle = np.angle(_eigenvectors_single)
    #         _eigenvectors_single_abs = np.abs(_eigenvectors_single)
    #         _eigenvectors_single_real = np.real(_eigenvectors_single)
    #         _eigenvectors_single_imag = np.imag(_eigenvectors_single)
    #
    #         _nii_eigenvectors_angle = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
    #         _nii_eigenvectors_abs = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
    #         _nii_eigenvectors_real = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
    #         _nii_eigenvectors_imag = np.zeros_like(np_DTB_3_res_2mm_Region_Label)
    #         for iii in range(len(HCP_region)):
    #             _nii_eigenvectors_angle[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_angle[iii]
    #             _nii_eigenvectors_abs[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_abs[iii]
    #             _nii_eigenvectors_real[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_real[iii]
    #             _nii_eigenvectors_imag[np_DTB_3_res_2mm_Region_Label == HCP_region[iii]] = _eigenvectors_single_imag[iii]
    #
    #         nii_image_angle = nib.Nifti1Image(_nii_eigenvectors_angle, affine=DTB_3_res_2mm_Region_Label.affine, header=DTB_3_res_2mm_Region_Label.header)
    #         nii_image_abs = nib.Nifti1Image(_nii_eigenvectors_abs, affine=DTB_3_res_2mm_Region_Label.affine, header=DTB_3_res_2mm_Region_Label.header)
    #         nii_image_real = nib.Nifti1Image(_nii_eigenvectors_real, affine=DTB_3_res_2mm_Region_Label.affine, header=DTB_3_res_2mm_Region_Label.header)
    #         nii_image_imag = nib.Nifti1Image(_nii_eigenvectors_imag, affine=DTB_3_res_2mm_Region_Label.affine, header=DTB_3_res_2mm_Region_Label.header)
    #
    #         nib.save(nii_image_angle, os.path.join(_nii_eigenvectors_savepath, f'eigenvectors_angle_{load_dir_list_idx}_{_variable_name[_variable_idx]}_{_eigenvectors_argsort_selected_idx[ii]}.nii'))
    #         nib.save(nii_image_abs, os.path.join(_nii_eigenvectors_savepath, f'eigenvectors_abs_{load_dir_list_idx}_{_variable_name[_variable_idx]}_{_eigenvectors_argsort_selected_idx[ii]}.nii'))
    #         nib.save(nii_image_real, os.path.join(_nii_eigenvectors_savepath, f'eigenvectors_real_{load_dir_list_idx}_{_variable_name[_variable_idx]}_{_eigenvectors_argsort_selected_idx[ii]}.nii'))
    #         nib.save(nii_image_imag, os.path.join(_nii_eigenvectors_savepath, f'eigenvectors_imag_{load_dir_list_idx}_{_variable_name[_variable_idx]}_{_eigenvectors_argsort_selected_idx[ii]}.nii'))
    #
    #         _nii_eigenvectors_angle_selected = _nii_eigenvectors_angle[np_DTB_3_res_2mm_Region_Label != 0]
    #         _nii_eigenvectors_abs_selected = _nii_eigenvectors_abs[np_DTB_3_res_2mm_Region_Label != 0]
    #         _nii_eigenvectors_real_selected = _nii_eigenvectors_real[np_DTB_3_res_2mm_Region_Label != 0]
    #         _nii_eigenvectors_imag_selected = _nii_eigenvectors_imag[np_DTB_3_res_2mm_Region_Label != 0]
    #
    #         _nii_eigenvectors_angle_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_angle_selected
    #         _nii_eigenvectors_abs_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_abs_selected
    #         _nii_eigenvectors_real_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_real_selected
    #         _nii_eigenvectors_imag_iterrecords[ii, _variable_idx, :] = _nii_eigenvectors_imag_selected
    #
    # np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_angle_{load_dir_list_idx}'), _nii_eigenvectors_angle_iterrecords)
    # np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_abs_{load_dir_list_idx}'), _nii_eigenvectors_abs_iterrecords)
    # np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_real_{load_dir_list_idx}'), _nii_eigenvectors_real_iterrecords)
    # np.save(os.path.join(_nii_eigenvectors_savepath, f'np_eigenvectors_imag_{load_dir_list_idx}'), _nii_eigenvectors_imag_iterrecords)
    #
    # _nii_eigenvectors_angle_iterrecords_totalparams.append(_nii_eigenvectors_angle_iterrecords)
    # _nii_eigenvectors_abs_iterrecords_totalparams.append(_nii_eigenvectors_abs_iterrecords)
    # _nii_eigenvectors_real_iterrecords_totalparams.append(_nii_eigenvectors_real_iterrecords)
    # _nii_eigenvectors_imag_iterrecords_totalparams.append(_nii_eigenvectors_imag_iterrecords)


    ue = ue.reshape((1, -1))
    se = se.reshape((1, -1))
    ui = ui.reshape((1, -1))
    si = si.reshape((1, -1))
    _test_ue, _test_se, _test_ui, _test_si = mnn.forward(ue, se, ui, si, external_current_forEpopu=torch.zeros_like(ue).type_as(ue))
    ue_error = (ue - _test_ue).abs().max().item()
    se_error = (se - _test_se).abs().max().item()
    ui_error =(ui - _test_ui).abs().max().item()
    si_error = (si - _test_si).abs().max().item()
    error = max(ue_error, se_error, ui_error, si_error)

    print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx, "error",
          error, ue_record_discardtransient_amplitude, ui_record_discardtransient_amplitude, "real_part", max_real_part, 'done')
    _error_amplitude_list.append([error, ue_record_discardtransient_amplitude, ui_record_discardtransient_amplitude])

    # if load_dir_list_idx > 200 and load_dir_list_idx < 210:
    #     print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx, "error", error, "real_part", max_real_part, min_real_part, 'done')

    # print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx, 'done')

    # ue = ue.reshape((1, -1))
    # se = se.reshape((1, -1))
    # ui = ui.reshape((1, -1))
    # si = si.reshape((1, -1))
    # _test_ue, _test_se, _test_ui, _test_si = mnn.forward(ue, se, ui, si, external_current_forEpopu=torch.zeros_like(ue).type_as(ue))
    # # print()
    # ue_error = (ue - _test_ue).abs().max().item()
    # se_error = (se - _test_se).abs().max().item()
    # ui_error =(ui - _test_ui).abs().max().item()
    # si_error = (si - _test_si).abs().max().item()
    # error = max(ue_error, se_error, ui_error, si_error)
    #
    # exc_channel_current_mean_for_Epopu, exc_channel_current_std_for_Epopu, \
    # inh_channel_current_mean_for_Epopu, inh_channel_current_std_for_Epopu, \
    # exc_channel_current_mean_for_Ipopu, exc_channel_current_std_for_Ipopu, \
    # inh_channel_current_mean_for_Ipopu, inh_channel_current_std_for_Ipopu = mnn.synaptic_current_stat(ue, se, ui, si)
    #
    # current_mean_for_Epopu = exc_channel_current_mean_for_Epopu+inh_channel_current_mean_for_Epopu
    # current_std_for_Epopu = (exc_channel_current_std_for_Epopu**2+inh_channel_current_std_for_Epopu**2).sqrt()
    #
    # current_mean_for_Ipopu = exc_channel_current_mean_for_Ipopu+inh_channel_current_mean_for_Ipopu
    # current_std_for_Ipopu = (exc_channel_current_std_for_Ipopu**2+inh_channel_current_std_for_Ipopu**2).sqrt()
    #
    # print(current_mean_for_Epopu.min(), current_mean_for_Epopu.mean(), current_mean_for_Epopu.max())
    # print(current_std_for_Epopu.min(), current_std_for_Epopu.mean(), current_std_for_Epopu.max())
    # print(current_mean_for_Ipopu.min(), current_mean_for_Ipopu.mean(), current_mean_for_Ipopu.max())
    # print(current_std_for_Ipopu.min(), current_std_for_Ipopu.mean(), current_std_for_Ipopu.max())
    #
    # print(f"rank", rank, '_gui_ampa_scalevalue', _gui_ampa_scalevalue, "load_dir_list_idx", load_dir_list_idx, "error", error, "real_part", max_real_part, min_real_part, 'done')

all_error_amplitude_results = comm.gather(_error_amplitude_list, root=0)

if rank == 0:
    filtered_all_error_amplitude_results = [item for item in all_error_amplitude_results if item]

    if filtered_all_error_amplitude_results != []:
        np_filtered_all_error_amplitude_results = np.concatenate(filtered_all_error_amplitude_results, axis=0)

        np.save(os.path.join(_base_save_dir, "np_filtered_all_error_amplitude_results.npy"), np_filtered_all_error_amplitude_results)


