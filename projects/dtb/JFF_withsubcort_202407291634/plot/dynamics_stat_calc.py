# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: bold_generate.py
# @time: 2024/5/12 23:19

import os

from matplotlib import pyplot as plt
import torch
import numpy as np
import argparse
from mpi4py import MPI
from scipy.signal import welch, get_window


def get_psd(data, fs, resolution, axis=0):
    '''
    data: average of postsynaptic current
    fs: sampling frequency
    resolution: mininum frequency in psd. If resolution=0.5 Hz, length of data should be no less than 2s.
    '''
    win = int(fs / resolution)
    the_window = get_window('boxcar', win)
    freqs, power = welch(data,
                         fs=fs,
                         window=the_window,
                         noverlap=int(0.75 * win),
                         nfft=win,
                         axis=axis,
                         scaling='density',
                         detrend=False)
    return freqs, np.array(power)


def get_args():
    parser = argparse.ArgumentParser(description="MNN")
    parser.add_argument("--para_index", type=str, default="1")
    args = parser.parse_args()
    return args


delta_t = 1
time_unit_converted = 1000
hp_update_time = 800
n_region = 378

args = get_args()
rela_param_idx = args.para_index
tau_constant_list = np.arange(1, 16)
tau = tau_constant_list[int(rela_param_idx)]

fs = time_unit_converted / (delta_t * (tau/11))

_residual_time = 400

load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in load_dir_list]
_gui_ampa_scale_values = np.array(_gui_ampa_scale_values)
total_paras = _gui_ampa_scale_values  # different from simu code

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_per_proc = int(np.ceil(len(total_paras) / size))
start_idx = rank * n_per_proc
end_idx = min(start_idx + n_per_proc, len(total_paras))
params = total_paras[start_idx:end_idx]
selected_load_dir_list = np.array(load_dir_list)[start_idx:end_idx]

data_save_done = False

_base_load_dir = '/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739/save_data_0.01_360000_11/rest_intero_25000_500_378_0.01_360000_3200_hp57_378_500_378_0.25_0.05_0.55_0.4_0.1_0_260_20_False'
# _base_load_dir = '/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739/save_data_0.05_360000'
_base_save_dir = os.path.join(_base_load_dir, f'modified_refine_dynamics_plot_debugtau_tau{tau}')
os.makedirs(_base_save_dir, exist_ok=True)

columns_name = ['mean_E_popu_firing_rate', 'mean_I_popu_firing_rate', 'mean_voxelwise_firing_rate',
                'mean_E_popu_oscillation_amplitude', 'mean_I_popu_oscillation_amplitude', 'mean_voxelwise_oscillation_amplitude',
                'mean_E_popu_oscillation_amplitude_std', 'mean_I_popu_oscillation_amplitude_std', 'mean_voxelwise_oscillation_amplitude_std',
                'mean_E_popu_oscillation_freq', 'mean_I_popu_oscillation_freq', 'mean_voxelwise_oscillation_freq',
                'mean_E_popu_oscillation_peakpower', 'mean_I_popu_oscillation_peakpower', 'mean_voxelwise_oscillation_peakpower',
                'std_E_popu_firing_rate', 'std_I_popu_firing_rate', 'std_voxelwise_firing_rate',
                'FanoFactor_E_popu_firing_rate', 'FanoFactor_I_popu_firing_rate', 'FanoFactor_voxelwise_firing_rate',
                'signalFanoFactor_E_popu_firing_rate', 'signalFanoFactor_I_popu_firing_rate', 'singalFanoFactor_voxelwise_firing_rate',
                ]
# stat_whole_brain = np.zeros((len(_gui_ampa_scale_values), n_region, len(columns_name)))
stat_whole_brain_record = []

T = round(0.8 * 1000 * 449)
# T = round(0.8 * 1000 * 349)
init_time = round(0.8 * 1000 * 4)
record_steps = int(T / delta_t)
init_steps = int(init_time / delta_t)

EI_ratio = 0.8

for rela_param_idx in range(len(params)):
    _singlepara_stat_whole_brain = np.zeros((n_region, len(columns_name)))

    # _gui_ampa_scalevalue = params[rela_param_idx]
    load_dir_list_idx = selected_load_dir_list[rela_param_idx]

    ue_record = np.load(os.path.join(_base_load_dir, f'ue_record_{load_dir_list_idx}.npy'))[init_steps:].astype(np.float32)
    ui_record = np.load(os.path.join(_base_load_dir, f'ui_record_{load_dir_list_idx}.npy'))[init_steps:].astype(np.float32)
    se_record = np.load(os.path.join(_base_load_dir, f'se_record_{load_dir_list_idx}.npy'))[init_steps:].astype(np.float32)
    si_record = np.load(os.path.join(_base_load_dir, f'si_record_{load_dir_list_idx}.npy'))[init_steps:].astype(np.float32)
    
    mean_voxelwise_firingrate = ue_record * 0.8 + ui_record * 0.2

    std_voxelwise_firingrate = np.sqrt(EI_ratio * se_record ** 2 + (1 - EI_ratio) * si_record ** 2)
    FanoFactor_E_popu = se_record ** 2 / ue_record
    FanoFactor_I_popu = si_record ** 2 / ui_record
    FanoFactor_voxelwise_firingrate = std_voxelwise_firingrate ** 2 / mean_voxelwise_firingrate

    signalFanoFactor_timeaveraged_E_popu_firingrate = ((se_record ** 2).mean(axis=0) + ue_record.var(axis=0)) / ue_record.mean(axis=0)
    signalFanoFactor_timeaveraged_I_popu_firingrate = ((si_record ** 2).mean(axis=0) + ui_record.var(axis=0)) / ui_record.mean(axis=0)
    signalFanoFactor_timeaveraged_voxelwise_firingrate = ((std_voxelwise_firingrate ** 2).mean(axis=0) + mean_voxelwise_firingrate.var(axis=0)) / mean_voxelwise_firingrate.mean(axis=0)

    _singlepara_stat_whole_brain[:, 0] = ue_record.mean(axis=0)
    _singlepara_stat_whole_brain[:, 1] = ui_record.mean(axis=0)
    _singlepara_stat_whole_brain[:, 2] = mean_voxelwise_firingrate.mean(axis=0)

    ue_record_discardtransient = ue_record.reshape((-1, round(hp_update_time/delta_t), n_region))[:, -round(_residual_time/delta_t):]
    ui_record_discardtransient = ui_record.reshape((-1, round(hp_update_time/delta_t), n_region))[:, -round(_residual_time/delta_t):]
    mean_voxelwise_firingrate_discardtransient = mean_voxelwise_firingrate.reshape((-1, round(hp_update_time/delta_t), n_region))[:, -round(_residual_time/delta_t):]

    ue_record_discardtransient_amplitude = ue_record_discardtransient.max(axis=1) - ue_record_discardtransient.min(axis=1)
    ui_record_discardtransient_amplitude = ui_record_discardtransient.max(axis=1) - ui_record_discardtransient.min(axis=1)
    mean_voxelwise_firingrate_discardtransient_amplitude = mean_voxelwise_firingrate_discardtransient.max(axis=1) - mean_voxelwise_firingrate_discardtransient.min(axis=1)

    _singlepara_stat_whole_brain[:, 3] = ue_record_discardtransient_amplitude.mean(axis=0)
    _singlepara_stat_whole_brain[:, 4] = ui_record_discardtransient_amplitude.mean(axis=0)
    _singlepara_stat_whole_brain[:, 5] = mean_voxelwise_firingrate_discardtransient_amplitude.mean(axis=0)

    _singlepara_stat_whole_brain[:, 6] = ue_record_discardtransient_amplitude.std(axis=0)
    _singlepara_stat_whole_brain[:, 7] = ui_record_discardtransient_amplitude.std(axis=0)
    _singlepara_stat_whole_brain[:, 8] = mean_voxelwise_firingrate_discardtransient_amplitude.std(axis=0)

    freqs_ue_record, power_ue_record = get_psd(ue_record, fs=fs, resolution=0.5, axis=0)
    freqs_ue_record_selected = freqs_ue_record[1:201]  # TODO 这里201的含义？
    power_ue_record_selected = power_ue_record[1:201, :]

    freqs_ui_record, power_ui_record = get_psd(ui_record, fs=fs, resolution=0.5, axis=0)
    freqs_ui_record_selected = freqs_ui_record[1:201]
    power_ui_record_selected = power_ui_record[1:201, :]

    if rank == 0:
        print(freqs_ui_record_selected)

    freqs_mean_voxelwise_firingrate, power_mean_voxelwise_firingrate = get_psd(mean_voxelwise_firingrate, fs=fs, resolution=0.5, axis=0)
    freqs_mean_voxelwise_firingrate_selected = freqs_mean_voxelwise_firingrate[1:201]
    power_mean_voxelwise_firingrate_selected = power_mean_voxelwise_firingrate[1:201, :]

    _singlepara_stat_whole_brain[:, 9] = freqs_ue_record_selected[power_ue_record_selected.argmax(axis=0)]
    _singlepara_stat_whole_brain[:, 10] = freqs_ui_record_selected[power_ui_record_selected.argmax(axis=0)]
    _singlepara_stat_whole_brain[:, 11] = freqs_mean_voxelwise_firingrate_selected[power_mean_voxelwise_firingrate_selected.argmax(axis=0)]

    _singlepara_stat_whole_brain[:, 12] = power_ue_record_selected.max(axis=0)
    _singlepara_stat_whole_brain[:, 13] = power_ui_record_selected.max(axis=0)
    _singlepara_stat_whole_brain[:, 14] = power_mean_voxelwise_firingrate_selected.max(axis=0)

    _singlepara_stat_whole_brain[:, 15] = (se_record ** 2).mean(axis=0)
    _singlepara_stat_whole_brain[:, 16] = (si_record ** 2).mean(axis=0)
    _singlepara_stat_whole_brain[:, 17] = (std_voxelwise_firingrate ** 2).mean(axis=0)

    _singlepara_stat_whole_brain[:, 18] = FanoFactor_E_popu.mean(axis=0)
    _singlepara_stat_whole_brain[:, 19] = FanoFactor_I_popu.mean(axis=0)
    _singlepara_stat_whole_brain[:, 20] = FanoFactor_voxelwise_firingrate.mean(axis=0)

    _singlepara_stat_whole_brain[:, 21] = signalFanoFactor_timeaveraged_E_popu_firingrate
    _singlepara_stat_whole_brain[:, 22] = signalFanoFactor_timeaveraged_I_popu_firingrate
    _singlepara_stat_whole_brain[:, 23] = signalFanoFactor_timeaveraged_voxelwise_firingrate

    _singlepara_stat_whole_brain[:, 3:6] = _singlepara_stat_whole_brain[:, 3:6] / 2
    _singlepara_stat_whole_brain[:, :9] = _singlepara_stat_whole_brain[:, :9] * time_unit_converted
    _singlepara_stat_whole_brain[:, 12:15] = np.log10(_singlepara_stat_whole_brain[:, 12:15])
    _singlepara_stat_whole_brain[:, 15:18] = _singlepara_stat_whole_brain[:, 15:18]

    stat_whole_brain_record.append(_singlepara_stat_whole_brain)


all_stat_whole_brain_results = comm.gather(stat_whole_brain_record, root=0)
print(f"rank", rank, "size", size, 'done')

if rank == 0:
    filtered_all_stat_whole_brain_results = [item for item in all_stat_whole_brain_results if item]
    if filtered_all_stat_whole_brain_results != []:
        np_all_stat_whole_brain_results = np.concatenate(filtered_all_stat_whole_brain_results, axis=0)
        np.save(os.path.join(_base_save_dir, "np_all_stat_whole_brain_results.npy"), np_all_stat_whole_brain_results)

        print(f"rank", rank, "size", size, 'done')
