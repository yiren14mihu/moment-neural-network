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
from matplotlib import gridspec
import matplotlib.ticker as ticker


def get_args():
    parser = argparse.ArgumentParser(description="MNN")
    parser.add_argument("--para_index", type=str, default="1")
    args = parser.parse_args()
    return args


args = get_args()
rela_param_idx = args.para_index
tau_constant_list = np.arange(1, 16)
tau = tau_constant_list[int(rela_param_idx)]

delta_t = 1
time_unit_converted = 1000
hp_update_time = 800
n_region = 378

_residual_time = 400

_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_modified_J_EE = 0.05
_gui_ampa_scale_values = _gui_ampa_scale_values / _modified_J_EE

_SNN_load_dir_list = list(np.arange(0, 191, 10))
_SNN_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in _SNN_load_dir_list]
_SNN_gui_ampa_scale_values = np.array(_SNN_gui_ampa_scale_values)

_MNN_load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
_MNN_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in _MNN_load_dir_list]
_MNN_gui_ampa_scale_values = np.array(_MNN_gui_ampa_scale_values)

_plot_start = 0
_plot_end = len(_MNN_load_dir_list)

_MNN_gui_ampa_scale_values = _MNN_gui_ampa_scale_values[_plot_start:_plot_end]

_discard_time = 60
delay = 3
hpc_label = np.arange(1, 379)

_base_load_dir_MNN = '/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739/save_data_0.01_360000_11/rest_intero_25000_500_378_0.01_360000_3200_hp57_378_500_378_0.25_0.05_0.55_0.4_0.1_0_260_20_False'
_base_save_dir_MNN_BOLD = os.path.join(_base_load_dir_MNN, f'NEWrefine_BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')
C_wholebrain_max_record_MNN = np.load(os.path.join(_base_save_dir_MNN_BOLD, 'C_total_record.npy')).T
# C_wholebrain_max_record_MNN = np.load(os.path.join(_base_save_dir_MNN_BOLD, 'C_wholebrain_max_record.npy'))
fc_corr_record_MNN = np.load(os.path.join(_base_save_dir_MNN_BOLD, 'fc_corr_record.npy'))
_base_save_dir_MNN_dynamics = os.path.join(_base_load_dir_MNN, f'modified_refine_dynamics_plot_debugtau_tau{tau}')
np_all_stat_whole_brain_results_MNN = np.load(os.path.join(_base_save_dir_MNN_dynamics, "np_all_stat_whole_brain_results.npy"))  # shape: len(_gui_ampa_scale_values), n_region, len(columns_name)
C_wholebrain_max_record_MNN = C_wholebrain_max_record_MNN[_plot_start:_plot_end, :]
fc_corr_record_MNN = fc_corr_record_MNN[_plot_start:_plot_end, :]
np_all_stat_whole_brain_results_MNN = np_all_stat_whole_brain_results_MNN[_plot_start:_plot_end, :, :]

_discard_time = 20
_base_load_dir_SNN = '/public/home/ssct004t/project/wangjiexiang/criticality_test/OU_ZCS_cudasearch/cuda_test_code_debug/voxel_between_ampa/JFF_data_withsubcortical/simu_af_DA/Epopu_ext/d7_7/intero/minor_local/onlyEE_region/d_inv/202404202110_intero_9e6_d500_subc_12c_JF_trc/alltimes0.25/0_10_1001/J_EO1.2_J_IO1.8_std1_1.2_dt0.1_OUmean0.25_Ipopuampa1.1_steptime0.8_1times349step_nmda0.05_0.55_gaba0.4_0.1_extscale3_0_init4s/imean_nosample/trial_0/inter_region_synap_strenth_search/'
_base_save_dir_SNN_BOLD = os.path.join(_base_load_dir_SNN, f'renew_new_____NEWnew_syn_samelag{delay}_FC_pic{_discard_time}discard_binnum300')
fc_corr_record_SNN = np.load(os.path.join(_base_save_dir_SNN_BOLD, 'fc_corr_record.npy'))
C_wholebrain_max_record_SNN = np.load(os.path.join(_base_save_dir_SNN_BOLD, 'C_total_record.npy')).T
_base_save_dir_SNN_dynamics = os.path.join(_base_load_dir_SNN, f'refine_dynamics_plot_more_para_point')
np_all_stat_whole_brain_results_SNN = np.load(os.path.join(_base_save_dir_SNN_dynamics, "np_all_stat_whole_brain_results.npy"))  # shape: len(_gui_ampa_scale_values), n_region, len(columns_name)
np_all_stat_whole_brain_results_SNN[:, :, 1] = np_all_stat_whole_brain_results_SNN[:, :, 1] / 2

# _base_save_dir_compareSNNMNN = os.path.join(_base_load_dir_MNN, f'compareSNNMNN')
# os.makedirs(_base_save_dir_compareSNNMNN, exist_ok=True)

desired_number_of_ticks = 3
plt_color = ['royalblue', 'tomato', 'violet', 'mediumseagreen']
xticks_index = list(np.sort(np.concatenate((np.arange(0, 21, 4), np.arange(60, _plot_end, 40)))))

_text_list = ['A', 'B', 'C', 'D']
_MNN_plotdata_list = [fc_corr_record_MNN, C_wholebrain_max_record_MNN, np_all_stat_whole_brain_results_MNN[:, :, 2], np_all_stat_whole_brain_results_MNN[:, :, 5]]
_SNN_plotdata_list = [fc_corr_record_SNN, C_wholebrain_max_record_SNN, np_all_stat_whole_brain_results_SNN[:, :, 0], np_all_stat_whole_brain_results_SNN[:, :, 1]]

_save_dir = os.path.join(_base_load_dir_MNN, f'compare_SNN_MNN_rest_test_debugMNNtau_tau{tau}')
os.makedirs(_save_dir, exist_ok=True)
print('a')
np.save(os.path.join(_save_dir, "fc_corr_record_MNN.npy"), fc_corr_record_MNN)
np.save(os.path.join(_save_dir, "C_wholebrain_max_record_MNN.npy"), C_wholebrain_max_record_MNN)
np.save(os.path.join(_save_dir, "np_all_stat_whole_brain_results_MNN.npy"), np_all_stat_whole_brain_results_MNN)
np.save(os.path.join(_save_dir, "fc_corr_record_SNN.npy"), fc_corr_record_SNN)
np.save(os.path.join(_save_dir, "C_wholebrain_max_record_SNN.npy"), C_wholebrain_max_record_SNN)
np.save(os.path.join(_save_dir, "np_all_stat_whole_brain_results_SNN.npy"), np_all_stat_whole_brain_results_SNN)

# _ylabel_list = ['PCC between FC matrices', 'PCC of BOLD in the whole brain', 'Region-wise firing rate (sp/s)', 'Oscillation amplitude of\nregion-wise firing rate (sp/s)']
# # _plot_scatter_title = ['Average firing rate in E population (sp/s)', 'Average firing rate in I population (sp/s)', 'Average region-wise firing rate (sp/s)',
# #                 'Average oscillation amplitude of firing rate in E population (sp/s)', 'Average oscillation amplitude of firing rate in I population (sp/s)', 'Average oscillation amplitude of\nregion-wise firing rate (sp/s)']
#
# fig = plt.figure(figsize=(20, 5), dpi=300)
# # fig.suptitle(r'Performance of the DTB for different inter-region coupling strength',
# #              fontsize=16)
# ax = {}
# figure_num = 4
# gs = gridspec.GridSpec(1, figure_num)
# gs.update(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
# for i in range(figure_num):
#     ax[i] = fig.add_subplot(gs[0, i], frameon=True)
#
#     ax[i].plot(_MNN_gui_ampa_scale_values, _MNN_plotdata_list[i].mean(axis=1), color=plt_color[0])
#     ax[i].plot(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[i].mean(axis=1), color=plt_color[1])
#     legend = ax[i].legend([f"MNN", f'SNN'],
#                           )
#     legend.legendHandles[0]._sizes = [12]
#     legend.legendHandles[1]._sizes = [12]
#     ax[i].scatter(_MNN_gui_ampa_scale_values, _MNN_plotdata_list[i].mean(axis=1), color=plt_color[0], alpha=0.5)
#     ax[i].scatter(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[i].mean(axis=1), color=plt_color[1], alpha=0.5)
#
#     locator = ax[i].yaxis.get_major_locator()
#     ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
#     # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
#     ax[i].tick_params(axis='y', labelsize=9)
#     ax[i].set_xticks(np.round(_MNN_gui_ampa_scale_values[xticks_index], 3))
#     ax[i].set_xticklabels(np.round(_MNN_gui_ampa_scale_values[xticks_index], 3))
#
#     ax[i].text(-0.1, 1.05, _text_list[i],
#                fontdict={'fontsize': 11, 'weight': 'bold',
#                          'horizontalalignment': 'left', 'verticalalignment':
#                              'bottom'}, transform=ax[i].transAxes)
#     ax[i].set_ylabel(_ylabel_list[i], fontsize=12)
#     ax[i].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
# fig.savefig(os.path.join(_base_save_dir_compareSNNMNN,
#                          f"total_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"),
#             dpi=300)
# fig.savefig(os.path.join(_base_save_dir_compareSNNMNN,
#                          f"total_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.eps"),
#             dpi=300)
