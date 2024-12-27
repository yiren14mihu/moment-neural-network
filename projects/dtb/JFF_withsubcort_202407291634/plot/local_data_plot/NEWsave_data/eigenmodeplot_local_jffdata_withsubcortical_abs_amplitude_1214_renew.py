# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: b.py
# @time: 2024/7/8 21:31


from scipy import io
# from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, rc_file, rc, rcParams
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

import torch
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
import nibabel as nib
import argparse
import pandas as pd
# from colorspacious import cspace_convert
import matplotlib.colors as col
import seaborn as sns


def make_segmented_cmap():
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap



mnn_membrane_constant = 11

_base_dir = '_mpi_result_save_path_0803'
np_all_eigenvalues_results = np.load(os.path.join(_base_dir, 'np_all_righteigenvalues_results.npy'))[20:, :]
print("np_all_eigenvalues_results", np_all_eigenvalues_results.shape)

np_all_eigenvalues_results = np_all_eigenvalues_results / mnn_membrane_constant
RE_np_all_eigenvalues_results = np.real(np_all_eigenvalues_results)
IM_np_all_eigenvalues_results = np.imag(np_all_eigenvalues_results)

load_dir_list = list(range(200, 261, 1))
_gui_ampa_scale_values = np.linspace(0, 10, 1001)
_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in load_dir_list]
_gui_ampa_scale_values = np.array(_gui_ampa_scale_values)

_modified_J_EE = 0.05
_gui_ampa_scale_values = _gui_ampa_scale_values / _modified_J_EE

plt_colors = ['lightcoral', 'royalblue',  'navajowhite', 'limegreen', 'darkorange', 'violet', 'cyan',
              'deepskyblue', 'sienna', 'mediumpurple', 'darkgreen', 'slategray', 'darkblue', 'yellow',
              'r', 'lightseagreen']
eigenvalue_legend_list = [r'1st', r'2nd', r'3rd', r'4th', r'5th']
_selected_eigenvalue_idx = np.arange(0, 10, 2)

symbol_RE_np_all_eigenvalues_results = np.where(RE_np_all_eigenvalues_results>0, 1, 0)
symbol_RE_np_all_eigenvalues_results = symbol_RE_np_all_eigenvalues_results[::-1, :]
n, _ = symbol_RE_np_all_eigenvalues_results.shape
# 查找从下往上第一个为1的索引
indices = np.argmax(symbol_RE_np_all_eigenvalues_results[::-1, :], axis=0)
indices = indices.astype(np.float64)
print(indices[:14])
# print(indices[:14] - 20)
print(symbol_RE_np_all_eigenvalues_results)

TB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed = io.loadmat(
    'D:\graduate_life\dtb_generation\DTB_code\moment-neural-network-develop\DTB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed.mat')
np_DTB_3_res_2mm_Region_Label = TB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed[
    'DTB_3_res_2mm_Region_Label_wjxmodified_matlabprocessed'].astype(np.float64)

np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 379] = 0
np_DTB_3_res_2mm_Region_Label[np_DTB_3_res_2mm_Region_Label == 380] = 0

x = np.arange(np_DTB_3_res_2mm_Region_Label.shape[0])
y = np.arange(np_DTB_3_res_2mm_Region_Label.shape[1])
z = np.arange(np_DTB_3_res_2mm_Region_Label.shape[2])
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
xx_selected = xx[np_DTB_3_res_2mm_Region_Label != 0]
yy_selected = yy[np_DTB_3_res_2mm_Region_Label != 0]
zz_selected = zz[np_DTB_3_res_2mm_Region_Label != 0]
full_coordinate_selected = np.concatenate(
    (xx_selected.reshape((-1, 1)), yy_selected.reshape((-1, 1)), zz_selected.reshape((-1, 1))), axis=1)

# 使用pandas DataFrame来简化操作
df = pd.DataFrame(full_coordinate_selected, columns=['col1', 'col2', 'col3'])
# 使用groupby和idxmax来找到每组第三列元素最大的行
idx_full_coordinate_selected = df.groupby(['col1', 'col2'])['col3'].idxmax()
# 筛选出这些行
full_coordinate_selected_slice = df.loc[idx_full_coordinate_selected].values

_base_save_dir = os.path.join(f'eigenmode_pic')
os.makedirs(_base_save_dir, exist_ok=True)
# total_real_iterrecords_totalparams_results_list = []
# total_imag_iterrecords_totalparams_results_list = []
total_abs_iterrecords_totalparams_results_list = []
total_angle_iterrecords_totalparams_results_list = []

relative_idxed = indices[1:10:2].astype(np.int32) - 1
print(relative_idxed)
load_idx_list = np.array(load_dir_list)[relative_idxed]
# load_idx_list = [211, 229, 239, 252, 257]
_eigenvectors_plot_idx = np.arange(1, 10, 2)
for iii in range(len(load_idx_list)):
    load_idx = load_idx_list[iii]
    _base_load_dir = os.path.join(f'{load_idx}')
    np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results = np.load(os.path.join(_base_load_dir, f'np_eigenvectors_abs_{load_idx}.npy'))
    np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results = np.load(os.path.join(_base_load_dir, f'np_eigenvectors_angle_{load_idx}.npy'))
    # np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results = np.load(os.path.join(_base_load_dir, f'np_eigenvectors_real_{load_idx}.npy'))
    # np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results = np.load(os.path.join(_base_load_dir, f'np_eigenvectors_imag_{load_idx}.npy'))

    # for ttt in range(5):
    #     print((np.abs(np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results[2*ttt, :, :]+np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results[2*ttt+1, :, :])).sum())

    np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results[_eigenvectors_plot_idx, :, :]
    np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results[_eigenvectors_plot_idx, :, :]
    # np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results[_eigenvectors_plot_idx, :, :]
    # np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results[_eigenvectors_plot_idx, :, :]

    np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results[:, :, idx_full_coordinate_selected.values]
    np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results[:, :, idx_full_coordinate_selected.values]
    # np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results[:, :, idx_full_coordinate_selected.values]
    # np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results = np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results[:, :, idx_full_coordinate_selected.values]

    total_abs_iterrecords_totalparams_results_list.append(np_filtered_all_nii_eigenvectors_abs_iterrecords_totalparams_results)
    total_angle_iterrecords_totalparams_results_list.append(np_filtered_all_nii_eigenvectors_angle_iterrecords_totalparams_results)
    # total_real_iterrecords_totalparams_results_list.append(np_filtered_all_nii_eigenvectors_real_iterrecords_totalparams_results)
    # total_imag_iterrecords_totalparams_results_list.append(np_filtered_all_nii_eigenvectors_imag_iterrecords_totalparams_results)

# a = total_angle_iterrecords_totalparams_results_list[1][1, 1, :] - total_angle_iterrecords_totalparams_results_list[1][1, 3, :]
# plt.hist(a, bins=30, edgecolor='black', alpha=0.5,)
# print(a)
# plt.show()

segmented_cmap = make_segmented_cmap()

# fig = plt.figure(figsize=(4, 4), dpi=200)
# fig.suptitle(f'Phase of eigenmodes\nof jacobian', fontsize=20)
# ax = {}
# gs = gridspec.GridSpec(1, 1)
# ax[0] = fig.add_subplot(gs[0, 0])
# _nii_eigenvectors_imag_selected = total_angle_iterrecords_totalparams_results_list[1][1, 2, :]
# # _nii_eigenvectors_imag_selected = total_angle_iterrecords_totalparams_results_list[1][1, 0, :]
# # _nii_eigenvectors_imag_selected = total_angle_iterrecords_totalparams_results_list[1][1, 0, :] - total_angle_iterrecords_totalparams_results_list[1][1, 2, :]
# norm = Normalize(vmin=-np.pi, vmax=np.pi)
# _scatter_ax = ax[0].scatter(full_coordinate_selected_slice[:, 0],
#                                                full_coordinate_selected_slice[:, 1], s=1,
#                                                c=_nii_eigenvectors_imag_selected, cmap=segmented_cmap,
#                                                norm=norm)
# colorbar = fig.colorbar(_scatter_ax, ax=ax[0], shrink=0.9)
# colorbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
# colorbar.update_ticks()
# ax[0].set_axis_off()
# plt.show()

figure_num_h = 4
figure_num_v = _eigenvectors_plot_idx.shape[0]

ylabels_abs = [r'$|\mu_E|$', r'$|\sigma_E^2|$', r'$|\mu_I|$', r'$|\sigma_I^2|$']
ylabels_angle = [r'$\arg{\mu_E}$', r'$\arg{\sigma_E^2}$', r'$\arg{\mu_I}$', r'$\arg{\sigma_I^2}$']

title = [r'0th', r'1st', r'2nd', r'3rd', r'4th']

_save_dir = os.path.join(_base_save_dir, 'single_brainmap')
os.makedirs(_save_dir, exist_ok=True)

_angle_save_dir = os.path.join(_save_dir, 'angle')
os.makedirs(_angle_save_dir, exist_ok=True)
_abs_save_dir = os.path.join(_save_dir, 'abs')
os.makedirs(_abs_save_dir, exist_ok=True)
# # for i in range(1):
# #     vmax = max([np.abs(total_abs_iterrecords_totalparams_results_list[ttt][ttt, i, :]).max() for ttt in range(figure_num_v)])
# #     for j in range(1):
# for i in range(figure_num_h):
#     vmax = max([np.abs(total_abs_iterrecords_totalparams_results_list[ttt][ttt, i, :]).max() for ttt in range(figure_num_v)])
#     for j in range(figure_num_v):
#         fig = plt.figure(figsize=(0.5, 0.5), dpi=300)
#         # fig.suptitle(f'Magnitude of eigenmodes\nof jacobian', fontsize=14)
#         ax = {}
#         gs = gridspec.GridSpec(1, 1)
#         gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.2)
#         # gs.update(left=0.14, right=0.95, top=0.8, bottom=0.05, hspace=0.05, wspace=0.2)
#         ax[0] = fig.add_subplot(gs[0, 0])
#         _nii_eigenvectors_real_selected = total_abs_iterrecords_totalparams_results_list[j][j, i, :]
#         norm = Normalize(vmin=0,
#                  vmax=vmax)
#         # norm = Normalize(vmin=0,
#         #                  vmax=np.abs(_nii_eigenvectors_real_selected).max())
#         # norm = Normalize(vmin=np.abs(_nii_eigenvectors_real_selected).min(),
#         #          vmax=np.abs(_nii_eigenvectors_real_selected).max())
#         _scatter_ax = ax[0].scatter(full_coordinate_selected_slice[:, 0], full_coordinate_selected_slice[:, 1], s=1, c=_nii_eigenvectors_real_selected, cmap='Reds', marker='.', edgecolors='none',
#                                     norm=norm)
#         # colorbar = fig.colorbar(_scatter_ax, ax=ax[0], shrink=0.9)
#         # desired_number_of_ticks = 3
#         # colorbar.locator = ticker.MaxNLocator(desired_number_of_ticks)
#         # colorbar.update_ticks()
#         ax[0].set_axis_off()
#         # if i == 0:
#         #     ax[0].set_title(title[j], fontsize=12)
#         # ax[0].text(-40, 55, ylabels_abs[i], ha='center', va='center', fontsize=12)
#         plt.axis('equal')
#         # fig.savefig(os.path.join(_abs_save_dir, 'colorbar', f"colorbar_{i}_{j}.pdf"),
#         #             dpi=100)
#         fig.savefig(os.path.join(_abs_save_dir, f"eigenvectors_abs_total_jffdata_withsubcortical_renew_{i}_{j}.png"),
#                     dpi=300)
#         plt.show()

N = 256
segmented_cmap = make_segmented_cmap()
flat_huslmap = col.ListedColormap(sns.color_palette('husl',N))
# for i in range(1):
#     for j in range(1):
for i in range(figure_num_h):
    for j in range(figure_num_v):
        fig = plt.figure(figsize=(0.5, 0.5), dpi=300)
        # fig.suptitle(f'Phase of eigenmodes\nof jacobian', fontsize=14)
        ax = {}
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05, wspace=0.2)
        ax[0] = fig.add_subplot(gs[0, 0])
        _nii_eigenvectors_imag_selected = total_angle_iterrecords_totalparams_results_list[j][j, i, :]
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        _scatter_ax = ax[0].scatter(full_coordinate_selected_slice[:, 0], full_coordinate_selected_slice[:, 1], s=1, c=_nii_eigenvectors_imag_selected, cmap=segmented_cmap,marker='.', edgecolors='none',
                                    norm=norm)
        # colorbar = fig.colorbar(_scatter_ax, ax=ax[0], shrink=0.9)
        # desired_number_of_ticks = 3
        # colorbar.locator = ticker.MaxNLocator(desired_number_of_ticks)
        # colorbar.update_ticks()
        # colorbar.set_ticks([-np.pi, 0, np.pi])  # 设置刻度位置
        # colorbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
        print(ylabels_angle[i])
        ax[0].set_axis_off()
        # if i == 0:
        #     ax[0].set_title(title[j], fontsize=12)
        # ax[0].text(-40, 55, ylabels_angle[i], ha='center', va='center', fontsize=12)
        plt.axis('equal')
        # fig.savefig(
        #     os.path.join(_angle_save_dir, f"colorbar.pdf"),
        # dpi=200)
        fig.savefig(
            os.path.join(_angle_save_dir, f"eigenvectors_angle_total_jffdata_withsubcortical_renew_{i}_{j}.png"),
            dpi=300)
        # plt.show()


# fig = plt.figure(figsize=(10, 10), dpi=200)
# # fig = plt.figure(figsize=(2, 12 * 91 / 101), dpi=200)
# fig.suptitle(f'Magnitude of eigenmodes\nof jacobian', fontsize=44)
# ax = {}
# gs = gridspec.GridSpec(figure_num_h, figure_num_v)
# gs.update(left=0.14, right=0.95, top=0.8, bottom=0.05, hspace=0.05, wspace=0.25)
# for i in range(figure_num_h):
#     print('i', i)
#     for j in range(figure_num_v):
#         ax[i*figure_num_h+j] = fig.add_subplot(gs[i, j])
#         _nii_eigenvectors_real_selected = total_abs_iterrecords_totalparams_results_list[j][j, i, :]
#         norm = Normalize(vmin=0,
#                          vmax=np.abs(_nii_eigenvectors_real_selected).max())
#         # norm = Normalize(vmin=np.abs(_nii_eigenvectors_real_selected).min(),
#         #          vmax=np.abs(_nii_eigenvectors_real_selected).max())
#         _scatter_ax = ax[i*figure_num_h+j].scatter(full_coordinate_selected_slice[:, 0], full_coordinate_selected_slice[:, 1], s=1, c=_nii_eigenvectors_real_selected, cmap='Reds',
#                                     norm=norm)
#         colorbar = fig.colorbar(_scatter_ax, ax=ax[i*figure_num_h+j], shrink=0.9)
#         desired_number_of_ticks = 3
#         colorbar.locator = ticker.MaxNLocator(desired_number_of_ticks)
#         colorbar.update_ticks()
#         ax[i*figure_num_h+j].set_axis_off()
#         if i == 0:
#             ax[j].set_title(title[j], fontsize=32)
#     ax[i * figure_num_h].text(-40, 55, ylabels_abs[i], ha='center', va='center', fontsize=32)
# # fig.savefig(os.path.join(_base_save_dir, f"eigenvectors_abs_total_jffdata_withsubcortical_renew.png"),
# #             dpi=200)
# # fig.savefig(os.path.join(_base_save_dir, f"eigenvectors_abs_total_jffdata_withsubcortical_renew.pdf"),
# #             dpi=200)
# plt.axis('equal')
# plt.show()
#
# N = 256
# segmented_cmap = make_segmented_cmap()
# flat_huslmap = col.ListedColormap(sns.color_palette('husl',N))
#
# fig = plt.figure(figsize=(10, 10), dpi=200)
# fig.suptitle(f'Phase of eigenmodes\nof jacobian', fontsize=44)
# ax = {}
# gs = gridspec.GridSpec(figure_num_h, figure_num_v)
# gs.update(left=0.14, right=0.95, top=0.8, bottom=0.05, hspace=0.05, wspace=0.2)
# for i in range(figure_num_h):
#     print('i', i)
#     for j in range(figure_num_v):
#         ax[i*figure_num_h+j] = fig.add_subplot(gs[i, j])
#         _nii_eigenvectors_imag_selected = total_angle_iterrecords_totalparams_results_list[j][j, i, :]
#         norm = Normalize(vmin=-np.pi, vmax=np.pi)
#         _scatter_ax = ax[i*figure_num_h+j].scatter(full_coordinate_selected_slice[:, 0], full_coordinate_selected_slice[:, 1], s=1, c=_nii_eigenvectors_imag_selected, cmap=segmented_cmap,
#                                     norm=norm)
#         colorbar = fig.colorbar(_scatter_ax, ax=ax[i*figure_num_h+j], shrink=0.9)
#         desired_number_of_ticks = 3
#         colorbar.locator = ticker.MaxNLocator(desired_number_of_ticks)
#         colorbar.update_ticks()
#         colorbar.set_ticks([-np.pi, 0, np.pi])  # 设置刻度位置
#         colorbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
#         # ax[i*figure_num_h+j].set_axis_off()
#         if i == 0:
#             ax[j].set_title(title[j], fontsize=32)
#         # if i == figure_num_h-1:
#         #     ax[i*figure_num_h+j].text(0, 0, f'$\gamma={_gui_ampa_scale_values[relative_idxed[j]]}$', fontsize=32)
#     ax[i * figure_num_h].text(-40, 55, ylabels_angle[i], ha='center', va='center', fontsize=32)
# # fig.savefig(os.path.join(_base_save_dir, f"eigenvectors_angle_total_jffdata_withsubcortical_renew.png"),
# #             dpi=200)
# # fig.savefig(os.path.join(_base_save_dir, f"eigenvectors_angle_total_jffdata_withsubcortical_renew.pdf"),
# #             dpi=200)
# plt.axis('equal')
# plt.show()
