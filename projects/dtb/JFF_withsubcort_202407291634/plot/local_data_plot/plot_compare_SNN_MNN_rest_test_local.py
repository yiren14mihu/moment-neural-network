# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: bold_generate.py
# @time: 2024/5/12 23:19

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from matplotlib import pyplot as plt
import torch
import numpy as np
import argparse
# from mpi4py import MPI
from scipy.signal import welch, get_window
from matplotlib import gridspec
import matplotlib.ticker as ticker



time_unit_converted = 1000
hp_update_time = 800
n_region = 378

_residual_time = 400

_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_modified_J_EE = 0.05
_gui_ampa_scale_values = _gui_ampa_scale_values / _modified_J_EE

_SNN_load_dir_list = list(np.sort(np.concatenate((np.arange(0, 191, 10), np.arange(152, 160, 2)))))
# _SNN_load_dir_list = list(np.arange(0, 191, 10))
_SNN_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in _SNN_load_dir_list]
_SNN_gui_ampa_scale_values = np.array(_SNN_gui_ampa_scale_values)

_MNN_load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))
_MNN_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in _MNN_load_dir_list]
_MNN_gui_ampa_scale_values = np.array(_MNN_gui_ampa_scale_values)

print(_MNN_gui_ampa_scale_values[29])

# _MNN_gui_ampa_scale_values = _MNN_gui_ampa_scale_values / _MNN_gui_ampa_scale_values[29]
# _SNN_gui_ampa_scale_values = _SNN_gui_ampa_scale_values / _gui_ampa_scale_values[158]
# # _SNN_gui_ampa_scale_values = _SNN_gui_ampa_scale_values / _SNN_gui_ampa_scale_values[16]
# # _MNN_gui_ampa_scale_values = _MNN_gui_ampa_scale_values - _MNN_gui_ampa_scale_values[29]
# # _SNN_gui_ampa_scale_values = _SNN_gui_ampa_scale_values - _SNN_gui_ampa_scale_values[16]

_plot_start = 0
_plot_end = len(_MNN_load_dir_list)

_MNN_gui_ampa_scale_values = _MNN_gui_ampa_scale_values[_plot_start:_plot_end]

_discard_time = 60
delay = 3
hpc_label = np.arange(1, 379)

desired_number_of_ticks = 3
plt_color = ['royalblue', 'tomato', 'violet', 'mediumseagreen']
xticks_index = list(np.sort(np.concatenate((np.arange(0, 21, 5), np.arange(70, _plot_end, 40)))))
# xticks_index = list(np.sort(np.concatenate((np.arange(0, 21, 4), np.arange(60, _plot_end, 40)))))

_text_list = ['A', 'B', 'C', 'D']

# tau = 1
# tau = 15
# tau = 12
tau = 11
# tau = 10
# tau = 5
_save_dir = f'NEW_SNNMNNcompare_data/compare_SNN_MNN_rest_test_debugMNNtau_tau{tau}'

fc_corr_record_MNN = np.load(os.path.join(_save_dir, "fc_corr_record_MNN.npy"))
C_wholebrain_max_record_MNN = np.load(os.path.join(_save_dir, "C_wholebrain_max_record_MNN.npy"))
np_all_stat_whole_brain_results_MNN = np.load(os.path.join(_save_dir, "np_all_stat_whole_brain_results_MNN.npy"))
fc_corr_record_SNN = np.load(os.path.join(_save_dir, "fc_corr_record_SNN.npy"))
C_wholebrain_max_record_SNN = np.load(os.path.join(_save_dir, "C_wholebrain_max_record_SNN.npy"))
np_all_stat_whole_brain_results_SNN = np.load(os.path.join(_save_dir, "np_all_stat_whole_brain_results_SNN.npy"))

# np_all_stat_whole_brain_results_SNN
# k, q, _, _ = np.linalg.lstsq(np.stack([unique_min_disttoV1_HCP_region_exceptV1[plt_start_idx:plt_end_idx], np.ones(len(power_averagedist_exceptV1[iiiii, plt_start_idx:plt_end_idx]))], axis=1),
#                              np.log10(power_averagedist_exceptV1[iiiii, plt_start_idx:plt_end_idx][:, np.newaxis]), rcond=None)

# print(np_all_stat_whole_brain_results_SNN.shape)
# print(np_all_stat_whole_brain_results_SNN[:, :, 0])
# print(np_all_stat_whole_brain_results_SNN[:, :, 1])
# print(np_all_stat_whole_brain_results_SNN[:, :, 2])
# print(np_all_stat_whole_brain_results_SNN[:, :, 3])
# print(np_all_stat_whole_brain_results_SNN[:, :, 4])
# print(np_all_stat_whole_brain_results_MNN.shape)
# a
# print(fc_corr_record_SNN.argmax(axis=0))

# print(C_wholebrain_max_record_SNN.mean(axis=1).argmax())
# print(C_wholebrain_max_record_SNN.mean(axis=1).max())
# print(C_wholebrain_max_record_MNN.mean(axis=1).argmax())
# print(C_wholebrain_max_record_MNN.mean(axis=1).max())
# print(fc_corr_record_SNN.argmax(axis=0))
# print(fc_corr_record_SNN.argmax(axis=0))

# # print(np_all_stat_whole_brain_results_SNN[:, :, 1].mean(axis=1)[:17])
# a = np_all_stat_whole_brain_results_MNN[:, :, 5].mean(axis=1)
# print(np_all_stat_whole_brain_results_MNN[:, :, 5].shape)
# print(a[20:35])
# plt.scatter(np.arange(len(a[20:35])), a[20:35])
# plt.show()
#
# b = fc_corr_record_MNN.mean(axis=1)
# plt.scatter(np.arange(len(b[20:35])), b[20:35])
# plt.show()

# print(np_all_stat_whole_brain_results_MNN[:, :, 5].mean(axis=1)[:31])
# plt.scatter(range(31), np_all_stat_whole_brain_results_MNN[:, :, 5].mean(axis=1)[:31])
# plt.show()

# print(np_all_stat_whole_brain_results_SNN[:, :, 1].mean(axis=1)[:19])
# plt.scatter(range(19), np_all_stat_whole_brain_results_SNN[:, :, 1].mean(axis=1)[:19])
# plt.show()

np_all_stat_whole_brain_results_MNN[:30, :, 11] = 0
# np_all_stat_whole_brain_results_MNN[:29, :, 11] = 0
np_all_stat_whole_brain_results_SNN[:20, :, 3] = 0
print(np_all_stat_whole_brain_results_SNN[20:, :, 3])
a

_MNN_plotdata_list = [fc_corr_record_MNN, C_wholebrain_max_record_MNN, np_all_stat_whole_brain_results_MNN[:, :, 2], np_all_stat_whole_brain_results_MNN[:, :, 5], np_all_stat_whole_brain_results_MNN[:, :, 11]]
_SNN_plotdata_list = [fc_corr_record_SNN, C_wholebrain_max_record_SNN, np_all_stat_whole_brain_results_SNN[:, :, 0], np_all_stat_whole_brain_results_SNN[:, :, 1], np_all_stat_whole_brain_results_SNN[:, :, 3]]

print(_SNN_plotdata_list[2].mean(axis=1)[:20])
print(_SNN_plotdata_list[3].mean(axis=1)[:20])
print(_SNN_plotdata_list[2].mean(axis=1)[21:22])
print(_SNN_plotdata_list[3].mean(axis=1)[21:22])
# a
# from scipy import io
# save_dict = {}
# save_dict['MNN_gamma_divided_gammaC'] = _MNN_gui_ampa_scale_values[30:]
# save_dict['SNN_gamma_divided_gammaC'] = _SNN_gui_ampa_scale_values[16:]
# save_dict['MNN_freq'] = _MNN_plotdata_list[4].mean(axis=1)[30:]
# save_dict['SNN_freq'] = _SNN_plotdata_list[4].mean(axis=1)[16:]
# # io.savemat('fit_MNN_SNN_tau_data.mat', save_dict)
#
# save_dict['MNN_gamma'] = _MNN_gui_ampa_scale_values[30:]
# save_dict['SNN_gamma'] = _SNN_gui_ampa_scale_values[16:]
# save_dict['MNN_freq'] = _MNN_plotdata_list[4].mean(axis=1)[30:]
# save_dict['SNN_freq'] = _SNN_plotdata_list[4].mean(axis=1)[16:]
# io.savemat('fit_MNN_SNN_tau_data_unscaled.mat', save_dict)


_ylabel_list = ['Correlation of\nFC matrices', 'Correlation of\nBOLD signals', 'Region-wise firing rate (sp/s)', 'Variation amplitude of\nregion-wise firing rate (sp/s)', 'Oscillation frequency (Hz)']
# _plot_scatter_title = ['Average firing rate in E population (sp/s)', 'Average firing rate in I population (sp/s)', 'Average region-wise firing rate (sp/s)',
#                 'Average oscillation amplitude of firing rate in E population (sp/s)', 'Average oscillation amplitude of firing rate in I population (sp/s)', 'Average oscillation amplitude of\nregion-wise firing rate (sp/s)']

_save_name = ['Correlation_of_FC_matrices', 'Correlation_of_BOLD_signals', 'Region-wise_firing_rate', 'Oscillation_amplitude_of_region-wise_firing_rate', 'Oscillation_freq']

unique_elements, first_occurrence_indices = np.unique(_MNN_plotdata_list[4].mean(axis=1)[30:], return_index=True)
first_occurrence_indices = first_occurrence_indices[::-1]
# print(_MNN_plotdata_list[4].mean(axis=1)[30:])
# plt.scatter(range(len(_MNN_plotdata_list[4].mean(axis=1)[30:])), _MNN_plotdata_list[4].mean(axis=1)[30:])
# # plt.show()
#
# print(first_occurrence_indices)
# plt.plot(np.arange(len(_MNN_plotdata_list[4].mean(axis=1)[30:]))[first_occurrence_indices], _MNN_plotdata_list[4].mean(axis=1)[30:][first_occurrence_indices])
# plt.show()
# a

print(_SNN_plotdata_list[2].mean(axis=1)[15])
print(_SNN_plotdata_list[2].mean(axis=1)[16])

print(_MNN_gui_ampa_scale_values[_MNN_plotdata_list[0].mean(axis=1).argmax()])
print(_MNN_gui_ampa_scale_values[_MNN_plotdata_list[1].mean(axis=1).argmax()])
print(_MNN_plotdata_list[0].mean(axis=1).max())
print(_MNN_plotdata_list[1].mean(axis=1).max())

print(_SNN_gui_ampa_scale_values[_SNN_plotdata_list[1].mean(axis=1).argmax()])
print(_SNN_plotdata_list[1].mean(axis=1).max())
a
# plt.scatter(_MNN_gui_ampa_scale_values[:], _MNN_plotdata_list[1].mean(axis=1)[:])
# # plt.scatter(_MNN_gui_ampa_scale_values[:], _MNN_plotdata_list[0].mean(axis=1)[:])
# # plt.scatter(_MNN_gui_ampa_scale_values[:33], _MNN_plotdata_list[0].mean(axis=1)[:33])
# # plt.scatter(range(len(_MNN_plotdata_list[0].mean(axis=1)[:33])), _MNN_plotdata_list[0].mean(axis=1)[:33])
# # plt.scatter(range(len(_MNN_plotdata_list[1].mean(axis=1)[:])), _MNN_plotdata_list[1].mean(axis=1)[:])
#
# # plt.scatter(range(len(_SNN_plotdata_list[1].mean(axis=1))), _SNN_plotdata_list[1].mean(axis=1))
# # plt.scatter(range(len(_SNN_plotdata_list[3].mean(axis=1))), _SNN_plotdata_list[3].mean(axis=1))
# # plt.scatter(range(len(_SNN_plotdata_list[2].mean(axis=1))), _SNN_plotdata_list[2].mean(axis=1))
# plt.show()
# a

# snn_x = _SNN_gui_ampa_scale_values[-4:]
# snn_y = np_all_stat_whole_brain_results_SNN[-4:, :, 3].mean(axis=1)
# k, q, _, _ = np.linalg.lstsq(np.stack([snn_x, np.ones(len(snn_x))], axis=1),
#                              snn_y[:, np.newaxis], rcond=None)
# plt.scatter(snn_x, snn_y)
# plt.plot(snn_x, k[1] + snn_x * k[0], color='red')
# plt.title('SNN')
# plt.xlabel('$\gamma$')
# plt.ylabel('frequency (hz)')
# plt.show()
#
# mnn_x = _MNN_gui_ampa_scale_values[29:]
# mnn_y = np_all_stat_whole_brain_results_MNN[29:, :, 11].mean(axis=1)
# k, q, _, _ = np.linalg.lstsq(np.stack([mnn_x, np.ones(len(mnn_x))], axis=1),
#                              mnn_y[:, np.newaxis], rcond=None)
# plt.scatter(mnn_x, mnn_y)
# plt.plot(mnn_x, k[1] + mnn_x * k[0], color='red')
# plt.title('MNN')
# plt.ylabel('frequency (1/a.u.)')
# plt.xlabel('$\gamma$')
# plt.show()

for iii in range(0, len(_ylabel_list)):
# for iii in range(len(_ylabel_list)-1, len(_ylabel_list)):
    fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
    fig.suptitle(r'$\tau$='f'{tau}',
                 fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.32, right=0.98, top=0.82, bottom=0.16, hspace=0.15)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)
        if iii == 4:
            ax[i].plot(_MNN_gui_ampa_scale_values[:31], _MNN_plotdata_list[iii].mean(axis=1)[:31], color=plt_color[0], linewidth=1, alpha=0.7)
            ax[i].plot(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[iii].mean(axis=1), color=plt_color[1], linewidth=1, alpha=0.7)
            ax[i].plot(_MNN_gui_ampa_scale_values[30:][first_occurrence_indices], _MNN_plotdata_list[iii].mean(axis=1)[30:][first_occurrence_indices], color=plt_color[0], linewidth=1, alpha=0.7)
            # ax[i].plot(_SNN_gui_ampa_scale_values[20:], _SNN_plotdata_list[iii].mean(axis=1)[20:], color=plt_color[1], linewidth=1, alpha=0.7)
        else:
            ax[i].plot(_MNN_gui_ampa_scale_values, _MNN_plotdata_list[iii].mean(axis=1), color=plt_color[0], linewidth=1, alpha=0.7)
            ax[i].plot(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[iii].mean(axis=1), color=plt_color[1], linewidth=1, alpha=0.7)

        legend = ax[i].legend([f"MNN", f'SNN'],
                              )
        legend.legendHandles[0]._sizes = [12]
        legend.legendHandles[1]._sizes = [12]

        if iii >= 1:
            if iii == 4:
                ax[i].fill_between(_MNN_gui_ampa_scale_values[30:][first_occurrence_indices], (_MNN_plotdata_list[iii].mean(axis=1) - _MNN_plotdata_list[iii].std(axis=1))[30:][first_occurrence_indices], (_MNN_plotdata_list[iii].mean(axis=1) + _MNN_plotdata_list[iii].std(axis=1))[30:][first_occurrence_indices], color=plt_color[0], alpha=0.2)
                ax[i].fill_between(_MNN_gui_ampa_scale_values[:31], (_MNN_plotdata_list[iii].mean(axis=1) - _MNN_plotdata_list[iii].std(axis=1))[:31], (_MNN_plotdata_list[iii].mean(axis=1) + _MNN_plotdata_list[iii].std(axis=1))[:31], color=plt_color[0], alpha=0.2)
                ax[i].fill_between(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[iii].mean(axis=1) - _SNN_plotdata_list[iii].std(axis=1), _SNN_plotdata_list[iii].mean(axis=1) + _SNN_plotdata_list[iii].std(axis=1), color=plt_color[1], alpha=0.2)
            else:
                ax[i].fill_between(_MNN_gui_ampa_scale_values, _MNN_plotdata_list[iii].mean(axis=1) - _MNN_plotdata_list[iii].std(axis=1), _MNN_plotdata_list[iii].mean(axis=1) + _MNN_plotdata_list[iii].std(axis=1), color=plt_color[0], alpha=0.2)
                ax[i].fill_between(_SNN_gui_ampa_scale_values, _SNN_plotdata_list[iii].mean(axis=1) - _SNN_plotdata_list[iii].std(axis=1), _SNN_plotdata_list[iii].mean(axis=1) + _SNN_plotdata_list[iii].std(axis=1), color=plt_color[1], alpha=0.2)

        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))

        if iii == 2:
            # ax[i].set_ylim([0, max(_MNN_plotdata_list[iii].mean(axis=1) + _MNN_plotdata_list[iii].std(axis=1))+2])
            ax[i].set_yticks([0, 10, 20, 30])

        ax[i].tick_params(axis='y', labelsize=13)
        ax[i].tick_params(axis='x', labelsize=13)
        ax[i].set_xticks(np.round(_MNN_gui_ampa_scale_values[xticks_index], 0))

        ax[i].set_ylabel(_ylabel_list[iii], fontsize=16)
        ax[i].set_xlabel(r'$\gamma$', fontsize=16)
    fig.savefig(os.path.join(_save_dir,
                             f"{_save_name[iii]}_{_plot_start}_{_plot_end}.pdf"),
                dpi=200)
    # fig.savefig(os.path.join(_save_dir,
    #                          f"{_save_name[iii]}_{_plot_start}_{_plot_end}.png"),
    #             dpi=200)
    # fig.savefig(os.path.join(_save_dir,
    #                          f"{_save_name[iii]}_{_plot_start}_{_plot_end}.eps"),
    #             dpi=200)
    plt.show()
