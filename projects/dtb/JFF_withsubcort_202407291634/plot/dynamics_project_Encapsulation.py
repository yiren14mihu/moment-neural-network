# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: dynamics_project.py
# @time: 2024/7/13 19:41

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch


def dynamics_project_Encapsulation_stochastic_aspect(nregion, fixed_point,
                                   np_all_righteigenvector_results, np_all_lefteigenvector_results,
                                   _MNN_simu_load_path, load_dir_list_idx, _hp_TR_idx_scalevalue):

    assert fixed_point.shape == (nregion*4, )
    assert np_all_righteigenvector_results.shape[0] == nregion*2
    assert np_all_lefteigenvector_results.shape[0] == nregion*2

    _record_mode_num = np_all_righteigenvector_results.shape[1]
    # eigenmodes_range = np.arange(_record_mode_num, _record_mode_num + 1)
    eigenmodes_range = np.arange(1, _record_mode_num + 1)

    save_dt = 1
    time_unit_converted = 1000
    hp_update_time = 800
    hp_update_step = round(200 / save_dt)
    # hp_update_step = round(hp_update_time / save_dt)
    init_time = round(0.8 * time_unit_converted * 4)
    init_step = round(init_time / save_dt)

    _base_load_path = '/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004'

    ue_record = np.load(os.path.join(_MNN_simu_load_path, f'ue_record_{load_dir_list_idx}.npy'))
    se_record = np.load(os.path.join(_MNN_simu_load_path, f'se_record_{load_dir_list_idx}.npy'))
    ui_record = np.load(os.path.join(_MNN_simu_load_path, f'ui_record_{load_dir_list_idx}.npy'))
    si_record = np.load(os.path.join(_MNN_simu_load_path, f'si_record_{load_dir_list_idx}.npy'))
    dynamics_record = np.concatenate((ue_record, se_record, ui_record, si_record), axis=1)
    dynamics_record = dynamics_record[init_step:, :]
    dynamics_record = dynamics_record.reshape((-1, round(hp_update_time / save_dt), nregion*4))[:, :hp_update_step, :]
    # dynamics_record = dynamics_record.reshape((-1, hp_update_step, nregion*4))
    dynamics_record = dynamics_record[:-1, :, :]
    dynamics_record = dynamics_record[_hp_TR_idx_scalevalue, :, :]  # important

    del ue_record, se_record, ui_record, si_record
    mu_idxes = np.concatenate((np.arange(0, nregion), np.arange(nregion*2, nregion*3)), axis=0)
    sigma_idxes = np.concatenate((np.arange(nregion, nregion*2), np.arange(nregion*3, nregion*4)), axis=0)

    dynamics_record_demean = dynamics_record - fixed_point.reshape((1, nregion*4))
    mu_dynamics_record_demean = dynamics_record_demean[:, mu_idxes]
    del dynamics_record_demean

    _mu_error_normalized_factor = np.linalg.norm(dynamics_record)
    _sigma_error_normalized_factor = _mu_error_normalized_factor
    # _mu_error_normalized_factor = np.linalg.norm(dynamics_record[:, mu_idxes])
    # _sigma_error_normalized_factor = np.linalg.norm(dynamics_record[:, sigma_idxes])
    
    sigma_dynamics_record = dynamics_record[:, sigma_idxes]
    sigma_dynamics_record_reshape = np.zeros((hp_update_step, nregion*2, nregion*2))
    sigma_dynamics_record_reshape[:, np.arange(nregion*2), np.arange(nregion*2)] = sigma_dynamics_record

    del dynamics_record

    assert sigma_dynamics_record.shape == (hp_update_step, nregion*2)
    assert sigma_dynamics_record_reshape.shape == (hp_update_step, nregion*2, nregion*2)

    torch_all_righteigenvector_results = torch.from_numpy(np_all_righteigenvector_results).cuda()[:, :_record_mode_num]
    torch_variance_dynamics_record_reshape = torch.from_numpy(sigma_dynamics_record_reshape).cuda().type_as(torch_all_righteigenvector_results) ** 2
    torch_all_lefteigenvector_results_normalized = torch.from_numpy(np_all_lefteigenvector_results).cuda()[:, :_record_mode_num] / \
                                                   torch.einsum('ij,ij->j',
                                                                torch.from_numpy(np_all_lefteigenvector_results).cuda()[:, :_record_mode_num],
                                                                torch_all_righteigenvector_results).reshape(1, _record_mode_num)
    del sigma_dynamics_record_reshape
    torch.cuda.empty_cache()

    P = torch.einsum('ji,tjk,kl->til', torch_all_lefteigenvector_results_normalized, torch_variance_dynamics_record_reshape, torch.conj(torch_all_lefteigenvector_results_normalized), )
    del torch_all_lefteigenvector_results_normalized, torch_variance_dynamics_record_reshape
    torch.cuda.empty_cache()

    _base_save_path = os.path.join(_base_load_path, 'dynamics_reconstruct_stochastic_aspect_online_newNEW')
    os.makedirs(_base_save_path, exist_ok=True)
    _reconstruct_data_save_path = os.path.join(_base_save_path, 'reconstruct_data')
    os.makedirs(_reconstruct_data_save_path, exist_ok=True)
    _mode_coefficient_save_path = os.path.join(_base_save_path, '_mode_coefficient_save')
    os.makedirs(_mode_coefficient_save_path, exist_ok=True)

    sigma_dynamics_reconstruct_local_all = np.zeros((_record_mode_num, hp_update_step, nregion*2))
    torch.cuda.empty_cache()
    for _eigenmode_num in eigenmodes_range:
        if _eigenmode_num % 100 == 1:
            print("load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue, "_eigenmode_num", _eigenmode_num)
        sigma_dynamics_reconstruct_local_all[_eigenmode_num-1, :, :] = np.abs((torch.einsum('ij,tjk,lk->til', torch_all_righteigenvector_results[:, :_eigenmode_num], P[:, :_eigenmode_num, :_eigenmode_num], torch.conj(torch_all_righteigenvector_results[:, :_eigenmode_num]))[:, np.arange(nregion*2), np.arange(nregion*2)]).cpu().numpy())
        torch.cuda.empty_cache()
    sigma_dynamics_reconstruct_local_all = np.sqrt(sigma_dynamics_reconstruct_local_all)

    del torch_all_righteigenvector_results, P
    torch.cuda.empty_cache()

    assert sigma_dynamics_reconstruct_local_all.shape == (_record_mode_num, hp_update_step, nregion*2)

    sigma_error_record = np.linalg.norm(sigma_dynamics_reconstruct_local_all - sigma_dynamics_record.reshape((1, hp_update_step, nregion*2)), axis=1) / _sigma_error_normalized_factor
    np.save(os.path.join(_mode_coefficient_save_path, f'sigma_error_record_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), sigma_error_record)
    # if np.max(np.abs(sigma_dynamics_reconstruct_local_all[-1, :, :] - sigma_dynamics_record)) > 1e-5:
    print("load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue,
          "np.linalg.norm(sigma_dynamics_reconstruct_local_all[-1, :, :] - sigma_dynamics_record)",
          np.linalg.norm(sigma_dynamics_reconstruct_local_all[-1, :, :] - sigma_dynamics_record),
          "np.linalg.norm(sigma_error_record, axis=1)[-1]", np.linalg.norm(sigma_error_record, axis=1)[-1])
    if _hp_TR_idx_scalevalue == 0:
        for _eigenmode_num in eigenmodes_range:
            np.save(os.path.join(_reconstruct_data_save_path, f'sigma_dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{_eigenmode_num}mode.npy'),
                    sigma_dynamics_reconstruct_local_all[_eigenmode_num - 1])
    del sigma_dynamics_reconstruct_local_all

    # Optimized computation
    mu_not_normalized_coefficients = np.einsum('ij,jk->ki', mu_dynamics_record_demean, np_all_lefteigenvector_results)
    mu_normalized_factors = np.einsum('ij,ij->j', np_all_lefteigenvector_results, np_all_righteigenvector_results)
    mu_mode_coefficient_totallist = mu_not_normalized_coefficients / mu_normalized_factors.reshape((nregion*2, 1))

    del mu_not_normalized_coefficients, mu_normalized_factors, np_all_lefteigenvector_results

    assert mu_mode_coefficient_totallist.shape == (nregion*2, hp_update_step)

    np.save(os.path.join(_mode_coefficient_save_path, f'mu_mode_coefficient_totallist_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), mu_mode_coefficient_totallist)
    if _hp_TR_idx_scalevalue == 0:
        np.save(os.path.join(_reconstruct_data_save_path, f'mu_dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{0}mode.npy'),
                fixed_point.reshape((1, nregion*4)).repeat(hp_update_step, axis=0)[:, mu_idxes])

    fixed_point_reshaped = fixed_point.reshape((1, nregion*4))
    fixed_point_repeated = fixed_point_reshaped.repeat(hp_update_step, axis=0)

    mu_fixed_point_repeated = fixed_point_repeated[:, mu_idxes]
    del fixed_point, fixed_point_reshaped, fixed_point_repeated

    mu_mode_coefficients_cumsum = np.real(np.cumsum(mu_mode_coefficient_totallist[:, np.newaxis] * np_all_righteigenvector_results.T[:, :, np.newaxis], axis=0))

    mu_dynamics_reconstruct_local_all = mu_fixed_point_repeated.reshape((1, hp_update_step, nregion*2)) + mu_mode_coefficients_cumsum.transpose(0, 2, 1)

    mu_error_record = np.linalg.norm(mu_mode_coefficients_cumsum.transpose(0, 2, 1) - mu_dynamics_record_demean.reshape((1, hp_update_step, nregion*2)), axis=1) / _mu_error_normalized_factor
    np.save(os.path.join(_mode_coefficient_save_path, f'mu_error_record_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), mu_error_record)
    if np.max(np.abs(mu_mode_coefficients_cumsum[-1, :, :].T - mu_dynamics_record_demean)) > 1e-5:
        print("load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue,
              "np.linalg.norm(mode_coefficients_cumsum[-1, :, :].T - mu_dynamics_record_demean)",
              np.linalg.norm(mu_mode_coefficients_cumsum[-1, :, :].T - mu_dynamics_record_demean),
              "np.linalg.norm(mu_error_record, axis=1)[-1]", np.linalg.norm(mu_error_record, axis=1)[-1])
    del mu_error_record

    del mu_dynamics_record_demean, mu_mode_coefficients_cumsum
    if _hp_TR_idx_scalevalue == 0:
        for _eigenmode_num in eigenmodes_range:
            np.save(os.path.join(_reconstruct_data_save_path, f'mu_dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{_eigenmode_num}mode.npy'),
                    mu_dynamics_reconstruct_local_all[_eigenmode_num - 1])
    del mu_dynamics_reconstruct_local_all

    print("done", "load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue)



def dynamics_project_Encapsulation_ode_aspect(nregion, fixed_point,
                                   np_all_righteigenvector_results, np_all_lefteigenvector_results,
                                   _MNN_simu_load_path, load_dir_list_idx, _hp_TR_idx_scalevalue,
                                    _base_save_path, ):

    assert fixed_point.shape == (nregion*4, )
    assert np_all_righteigenvector_results.shape[0] == nregion*4
    assert np_all_lefteigenvector_results.shape[0] == nregion*4

    save_dt = 1
    time_unit_converted = 1000
    hp_update_time = 800
    hp_update_step = round(hp_update_time / save_dt)
    init_time = round(0.8 * time_unit_converted * 4)
    init_step = round(init_time / save_dt)

    ue_record = np.load(os.path.join(_MNN_simu_load_path, f'ue_record_{load_dir_list_idx}.npy'))
    se_record = np.load(os.path.join(_MNN_simu_load_path, f'se_record_{load_dir_list_idx}.npy'))
    ui_record = np.load(os.path.join(_MNN_simu_load_path, f'ui_record_{load_dir_list_idx}.npy'))
    si_record = np.load(os.path.join(_MNN_simu_load_path, f'si_record_{load_dir_list_idx}.npy'))
    dynamics_record = np.concatenate((ue_record, se_record, ui_record, si_record), axis=1)
    dynamics_record = dynamics_record[init_step:, :]
    dynamics_record = dynamics_record.reshape((-1, hp_update_step, nregion*4))
    dynamics_record = dynamics_record[:-1, :, :]

    del ue_record, se_record, ui_record, si_record

    dynamics_record = dynamics_record[_hp_TR_idx_scalevalue, :, :]  # important
    dynamics_record_demean = dynamics_record - fixed_point.reshape((1, nregion*4))

    _error_normalized_factor = np.linalg.norm(dynamics_record)
    del dynamics_record

    _record_mode_num = np_all_righteigenvector_results.shape[1]

    # Optimized computation
    not_normalized_coefficients = np.einsum('ij,jk->ki', dynamics_record_demean, np_all_lefteigenvector_results)
    normalized_factors = np.einsum('ij,ij->j', np_all_lefteigenvector_results, np_all_righteigenvector_results)
    mode_coefficient_totallist = not_normalized_coefficients / normalized_factors.reshape((_record_mode_num, 1))

    del not_normalized_coefficients, normalized_factors

    assert mode_coefficient_totallist.shape == (_record_mode_num, hp_update_step)

    os.makedirs(_base_save_path, exist_ok=True)
    _reconstruct_data_save_path = os.path.join(_base_save_path, 'reconstruct_data')
    os.makedirs(_reconstruct_data_save_path, exist_ok=True)
    _mode_coefficient_save_path = os.path.join(_base_save_path, '_mode_coefficient_save')
    os.makedirs(_mode_coefficient_save_path, exist_ok=True)

    np.save(os.path.join(_mode_coefficient_save_path, f'mode_coefficient_totallist_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), mode_coefficient_totallist)
    if _hp_TR_idx_scalevalue == 0:
        np.save(os.path.join(_reconstruct_data_save_path, f'dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{0}mode.npy'),
                fixed_point.reshape((1, nregion*4)).repeat(hp_update_step, axis=0))

    fixed_point_reshaped = fixed_point.reshape((1, nregion*4))
    fixed_point_repeated = fixed_point_reshaped.repeat(hp_update_step, axis=0)

    del fixed_point

    eigenmodes_range = np.arange(1, _record_mode_num + 1)
    mode_coefficients_cumsum = np.real(np.cumsum(mode_coefficient_totallist[:, np.newaxis] * np_all_righteigenvector_results.T[:, :, np.newaxis], axis=0))

    dynamics_reconstruct_local_all = fixed_point_repeated.reshape((1, hp_update_step, nregion*4)) + mode_coefficients_cumsum.transpose(0, 2, 1)
    del fixed_point_repeated

    error_record = np.linalg.norm(mode_coefficients_cumsum.transpose(0, 2, 1) - dynamics_record_demean.reshape((1, hp_update_step, nregion*4)), axis=1) / _error_normalized_factor
    np.save(os.path.join(_mode_coefficient_save_path, f'error_record_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), error_record)
    if np.max(np.abs(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean)) > 1e-5:
        print("load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue,
              "np.linalg.norm(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean)",
              np.linalg.norm(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean),
              "np.linalg.norm(error_record, axis=1)[-1]", np.linalg.norm(error_record, axis=1)[-1])
    del error_record

    del dynamics_record_demean, mode_coefficients_cumsum
    if _hp_TR_idx_scalevalue == 0:
        for _eigenmode_num in eigenmodes_range:
            np.save(os.path.join(_reconstruct_data_save_path, f'dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{_eigenmode_num}mode.npy'),
                    dynamics_reconstruct_local_all[_eigenmode_num - 1])
    del dynamics_reconstruct_local_all

    print("done", "load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue)


def dynamics_project_Encapsulation_ode_aspect_variancemapping(nregion, fixed_point,
                                                              np_all_righteigenvector_results, np_all_lefteigenvector_results,
                                                              _MNN_simu_load_path, load_dir_list_idx, _hp_TR_idx_scalevalue,
                                                              _base_save_path, ):

    assert fixed_point.shape == (nregion*4, )
    assert np_all_righteigenvector_results.shape[0] == nregion*4
    assert np_all_lefteigenvector_results.shape[0] == nregion*4

    save_dt = 1
    time_unit_converted = 1000
    hp_update_time = 800
    hp_update_step = round(hp_update_time / save_dt)
    init_time = round(0.8 * time_unit_converted * 4)
    init_step = round(init_time / save_dt)

    ue_record = np.load(os.path.join(_MNN_simu_load_path, f'ue_record_{load_dir_list_idx}.npy'))
    se_record = np.load(os.path.join(_MNN_simu_load_path, f'se_record_{load_dir_list_idx}.npy'))
    ui_record = np.load(os.path.join(_MNN_simu_load_path, f'ui_record_{load_dir_list_idx}.npy'))
    si_record = np.load(os.path.join(_MNN_simu_load_path, f'si_record_{load_dir_list_idx}.npy'))
    dynamics_record = np.concatenate((ue_record, se_record ** 2, ui_record, si_record ** 2), axis=1)
    dynamics_record = dynamics_record[init_step:, :]
    dynamics_record = dynamics_record.reshape((-1, hp_update_step, nregion*4))
    dynamics_record = dynamics_record[:-1, :, :]

    del ue_record, se_record, ui_record, si_record

    dynamics_record = dynamics_record[_hp_TR_idx_scalevalue, :, :]  # important
    dynamics_record_demean = dynamics_record - fixed_point.reshape((1, nregion*4))

    _error_normalized_factor = np.linalg.norm(dynamics_record)
    del dynamics_record

    _record_mode_num = np_all_righteigenvector_results.shape[1]

    # Optimized computation
    not_normalized_coefficients = np.einsum('ij,jk->ki', dynamics_record_demean, np_all_lefteigenvector_results)
    normalized_factors = np.einsum('ij,ij->j', np_all_lefteigenvector_results, np_all_righteigenvector_results)
    mode_coefficient_totallist = not_normalized_coefficients / normalized_factors.reshape((_record_mode_num, 1))

    del not_normalized_coefficients, normalized_factors

    assert mode_coefficient_totallist.shape == (_record_mode_num, hp_update_step)

    os.makedirs(_base_save_path, exist_ok=True)
    _reconstruct_data_save_path = os.path.join(_base_save_path, 'reconstruct_data')
    os.makedirs(_reconstruct_data_save_path, exist_ok=True)
    _mode_coefficient_save_path = os.path.join(_base_save_path, '_mode_coefficient_save')
    os.makedirs(_mode_coefficient_save_path, exist_ok=True)

    np.save(os.path.join(_mode_coefficient_save_path, f'mode_coefficient_totallist_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), mode_coefficient_totallist)
    if _hp_TR_idx_scalevalue == 0:
        np.save(os.path.join(_reconstruct_data_save_path, f'dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{0}mode.npy'),
                fixed_point.reshape((1, nregion*4)).repeat(hp_update_step, axis=0))

    fixed_point_reshaped = fixed_point.reshape((1, nregion*4))
    fixed_point_repeated = fixed_point_reshaped.repeat(hp_update_step, axis=0)

    del fixed_point

    eigenmodes_range = np.arange(1, _record_mode_num + 1)
    mode_coefficients_cumsum = np.real(np.cumsum(mode_coefficient_totallist[:, np.newaxis] * np_all_righteigenvector_results.T[:, :, np.newaxis], axis=0))

    dynamics_reconstruct_local_all = fixed_point_repeated.reshape((1, hp_update_step, nregion*4)) + mode_coefficients_cumsum.transpose(0, 2, 1)
    del fixed_point_repeated

    error_record = np.linalg.norm(mode_coefficients_cumsum.transpose(0, 2, 1) - dynamics_record_demean.reshape((1, hp_update_step, nregion*4)), axis=1) / _error_normalized_factor
    np.save(os.path.join(_mode_coefficient_save_path, f'error_record_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}.npy'), error_record)
    if np.max(np.abs(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean)) > 1e-5:
        print("load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue,
              "np.linalg.norm(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean)",
              np.linalg.norm(mode_coefficients_cumsum[-1, :, :].T - dynamics_record_demean),
              "np.linalg.norm(error_record, axis=1)[-1]", np.linalg.norm(error_record, axis=1)[-1])
    del error_record

    del dynamics_record_demean, mode_coefficients_cumsum
    if _hp_TR_idx_scalevalue == 0:
        for _eigenmode_num in eigenmodes_range:
            np.save(os.path.join(_reconstruct_data_save_path, f'dynamics_reconstruct_{load_dir_list_idx}_{_hp_TR_idx_scalevalue}_{_eigenmode_num}mode.npy'),
                    dynamics_reconstruct_local_all[_eigenmode_num - 1])
    del dynamics_reconstruct_local_all

    print("done", "load_dir_list_idx", load_dir_list_idx, "_hp_TR_idx_scalevalue", _hp_TR_idx_scalevalue)

