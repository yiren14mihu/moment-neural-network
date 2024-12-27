# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: b.py
# @time: 2024/7/8 21:31
import os.path

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import gridspec, rc_file, rc, rcParams

_base_dir = r'F:\MNN_transientdata\JFF_withsubcort_202407291634\202407101655_fixedpoint_gridsearch_perTR\_mpi_result_save_path_0803'

mnn_membrane_constant = 11

np_all_eigenvalues_results = np.load(os.path.join(_base_dir, 'np_all_righteigenvalues_results.npy'))
print("np_all_eigenvalues_results", np_all_eigenvalues_results.shape)

np_all_eigenvalues_results = np_all_eigenvalues_results[17:, :]
# np_all_eigenvalues_results = np_all_eigenvalues_results[:, :]

np_all_eigenvalues_results = np_all_eigenvalues_results / mnn_membrane_constant
RE_np_all_eigenvalues_results = np.real(np_all_eigenvalues_results)
IM_np_all_eigenvalues_results = np.imag(np_all_eigenvalues_results)


load_dir_list = list(np.sort(np.concatenate((np.arange(170, 201, 10), np.arange(201, 261)))))
# load_dir_list = list(range(200, 261, 1))
_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in load_dir_list]
_gui_ampa_scale_values = np.array(_gui_ampa_scale_values)

_modified_J_EE = 0.05
_gui_ampa_scale_values = _gui_ampa_scale_values / _modified_J_EE

plt_colors = ['lightcoral', 'royalblue',  'navajowhite', 'limegreen', 'darkorange', 'violet', 'cyan',
              'deepskyblue', 'sienna', 'mediumpurple', 'darkgreen', 'slategray', 'darkblue', 'yellow',
              'r', 'lightseagreen']
eigenvalue_legend_list = [r'1st', r'2nd', r'3rd', r'4th', r'5th']
# eigenvalue_legend_list = [r'1st', r'2nd', r'3rd', r'4th', r'5th', r'6th', r'7th']
_selected_eigenvalue_idx = np.arange(0, 10, 2)

symbol_RE_np_all_eigenvalues_results = np.where(RE_np_all_eigenvalues_results>0, 1, 0)
symbol_RE_np_all_eigenvalues_results = symbol_RE_np_all_eigenvalues_results[::-1, :]
n, _ = symbol_RE_np_all_eigenvalues_results.shape
# 查找从下往上第一个为1的索引
indices = np.argmax(symbol_RE_np_all_eigenvalues_results[::-1, :], axis=0)
indices = indices.astype(np.float64)
print(indices[:14])
print(symbol_RE_np_all_eigenvalues_results)
# 检查列是否全为0，如果全为0则将索引设为0
all_zero_columns = np.all(symbol_RE_np_all_eigenvalues_results == 0, axis=0)
indices[all_zero_columns] = np.NAN
# 将反转的索引转换为实际索引
# indices = n - 1 - indices
# print(indices)

symbol_RE_np_all_eigenvalues_results = symbol_RE_np_all_eigenvalues_results[::-1, :]
print(symbol_RE_np_all_eigenvalues_results.shape)
print(symbol_RE_np_all_eigenvalues_results[9, 0], RE_np_all_eigenvalues_results[9, 0], load_dir_list[9])
print(symbol_RE_np_all_eigenvalues_results[10, 0], RE_np_all_eigenvalues_results[10, 0], load_dir_list[10])
print(symbol_RE_np_all_eigenvalues_results[11, 0], RE_np_all_eigenvalues_results[11, 0], load_dir_list[11])
print(symbol_RE_np_all_eigenvalues_results[12, 0], RE_np_all_eigenvalues_results[12, 0], load_dir_list[12])
print(symbol_RE_np_all_eigenvalues_results[13, 0], RE_np_all_eigenvalues_results[13, 0], load_dir_list[13])

fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
fig.suptitle('Real part of eigenvalues\nof jacobian at fixed point', fontsize=20)
# yticks_index = [0, int(theta_diff_density_histnum / 2), theta_diff_density_histnum]
figure_num = 1
ax = {}
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.3, right=0.98, top=0.8, bottom=0.16, hspace=0.15)
ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
for i in range(len(_selected_eigenvalue_idx)):
    ax[0].plot((_gui_ampa_scale_values), RE_np_all_eigenvalues_results[:, _selected_eigenvalue_idx[i]], c=plt_colors[i], linewidth=1.5, alpha=0.7)
legend = ax[0].legend([f'{i}' for i in eigenvalue_legend_list],
                        loc='best')
ax[0].axhline(0, linestyle=':', c='k', linewidth=0.8, alpha=0.7)
ax[0].set_xlabel(r"$\gamma$", fontsize=16)
for i in range(len(eigenvalue_legend_list)):
    legend.legendHandles[i]._sizes = [12]
ax[0].tick_params(axis='y', labelsize=13)
ax[0].tick_params(axis='x', labelsize=13)
ax[0].yaxis.set_major_locator(ticker.MaxNLocator(3))
fig.savefig(os.path.join(_base_dir, f"Real_Part_of_Eigenvalues_of_Jacobian_at_MNN_Fixed_Point_JFF_withsubcortical_{mnn_membrane_constant}.eps"),
            dpi=200)
plt.show()

fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
fig.suptitle('Oscillation frequencies of\ndynamical eigenmodes', fontsize=20)
# fig.suptitle('Oscillation frequencies of dynamical\neigenmodes from MNN', fontsize=20)
figure_num = 1
ax = {}
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.3, right=0.98, top=0.8, bottom=0.16, hspace=0.15)
ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
for i in range(len(_selected_eigenvalue_idx)):
    ax[0].plot((_gui_ampa_scale_values[:round(indices[_selected_eigenvalue_idx[i]])]), np.abs(IM_np_all_eigenvalues_results[:round(indices[_selected_eigenvalue_idx[i]]), _selected_eigenvalue_idx[i]])/np.pi/2*1000, c=plt_colors[i], linewidth=1.5, alpha=0.7)
legend = ax[0].legend([f'{i}' for i in eigenvalue_legend_list],
                        loc='best')
for i in range(len(eigenvalue_legend_list)):
    legend.legendHandles[i]._sizes = [12]
for i in range(len(_selected_eigenvalue_idx)):
    ax[0].plot((_gui_ampa_scale_values[round(indices[_selected_eigenvalue_idx[i]]):]), np.abs(IM_np_all_eigenvalues_results[round(indices[_selected_eigenvalue_idx[i]]):, _selected_eigenvalue_idx[i]])/np.pi/2*1000, c=plt_colors[i], linewidth=1.5, alpha=0.7, linestyle='dotted')
ax[0].set_ylabel("Oscillation frequency\n(Hz)", fontsize=16)
ax[0].set_xlabel(r"$\gamma$", fontsize=16)
ax[0].tick_params(axis='y', labelsize=13)
ax[0].tick_params(axis='x', labelsize=13)
ax[0].yaxis.set_major_locator(ticker.MaxNLocator(4))
fig.savefig(os.path.join(_base_dir, f"Oscillation_frequencies_of_dynamical_eigenmodes_from_MNN_JFF_withsubcortical_{mnn_membrane_constant}.eps"),
            dpi=200)
plt.show()

# a
# local_plot_colors = ['navajowhite', 'lightcoral', 'deepskyblue']
local_plot_colors = ['#FFDEAD', '#F08080', 'cornflowerblue']
plt_sizes = [40, 30, 20]
_selected_coupling_idx = [1, 12, 17]
# _selected_coupling_idx = [0, 9, 14]
figure_num = 1
fig = plt.figure(figsize=(figure_num*3.5, 3.5), dpi=200)
fig.suptitle('Eigenvalues of Jacobian\nat Fixed Point', fontsize=20)
# yticks_index = [0, int(theta_diff_density_histnum / 2), theta_diff_density_histnum]
ax = {}
gs = gridspec.GridSpec(1, figure_num)
gs.update(left=0.3, right=0.98, top=0.8, bottom=0.16, hspace=0.15)
for i in range(figure_num):
    ax[i] = fig.add_subplot(gs[0, i], frameon=True)
    # for jj in range(1):
    for jj in range(len(_selected_coupling_idx)):
        ax[i].scatter(RE_np_all_eigenvalues_results[_selected_coupling_idx[jj], :], IM_np_all_eigenvalues_results[_selected_coupling_idx[jj], :], c=local_plot_colors[jj], s=plt_sizes[jj], alpha=0.7, marker='.', edgecolors='none')
    ax[i].set_xlabel("Real part", fontsize=16)
    ax[i].set_ylabel("Imaginary part", fontsize=16)
    ax[i].yaxis.set_label_coords(-0.05, 0.5)

    ax[i].spines['left'].set_position('zero')
    ax[i].spines['bottom'].set_position('zero')
    ax[i].xaxis.set_ticks_position('bottom')
    ax[i].yaxis.set_ticks_position('left')
    ax[i].set_xticks([-0.1, 0])
    ax[i].set_yticks([-0.1, 0.1])

    # ax[i].set_xticks([-2, -1, 0])
    # ax[i].set_yticks([-3, -1, 1, 3])

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    legend = ax[0].legend([r"$\gamma=$"f'{_gui_ampa_scale_values[_i]:.4}' for _i in _selected_coupling_idx],
                          loc='best')
    ax[i].tick_params(axis='y', labelsize=13)
    ax[i].tick_params(axis='x', labelsize=13)
    for _i in range(len(_selected_coupling_idx)):
        # print("_gui_ampa_scale_values[_i]", f'{_gui_ampa_scale_values[_selected_coupling_idx[_i]]:.3}')
        print("_gui_ampa_scale_values[_i]", _gui_ampa_scale_values[_selected_coupling_idx[_i]])
        legend.legendHandles[_i]._sizes = [12]
fig.savefig(os.path.join(_base_dir, f"Eigenvalues_of_Jacobian_at_MNN_Fixed_Point_JFF_withsubcortical_{mnn_membrane_constant}.eps"),
            dpi=200)
plt.show()
