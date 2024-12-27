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


delta_t = 1
time_unit_converted = 1000
hp_update_time = 800
n_region = 378

fs = time_unit_converted / delta_t

_residual_time = 400

_gui_ampa_scale_values = np.linspace(0, 10, 10001)
_modified_J_EE = 0.05
_gui_ampa_scale_values = _gui_ampa_scale_values / _modified_J_EE

_SNN_load_dir_list = list(np.sort(np.concatenate((np.arange(0, 1901, 100), np.arange(1520, 1590, 20)))))
# _SNN_load_dir_list = list(np.arange(0, 191, 10))
_SNN_gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in _SNN_load_dir_list]
_SNN_gui_ampa_scale_values = np.array(_SNN_gui_ampa_scale_values)

_plot_start = 0
_plot_end = len(_SNN_load_dir_list)


delay = 3
hpc_label = np.arange(1, 379)

desired_number_of_ticks = 3
plt_color = ['royalblue', 'tomato', 'violet', 'mediumseagreen']
xticks_index = list(np.arange(0, 20, 5))
# xticks_index = list(np.arange(0, 19, 4))

_text_list = ['A', 'B', 'C', 'D']

_save_dir = 'data\SNN_coherence_new_meanhp'
coherence = np.load(os.path.join(_save_dir, "NEW_stat_whole_brain__downsample_1_more.npy"))[:, :, 1]

print(coherence.shape)
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
# fig.suptitle(r'Performance of the DTB for different inter-region coupling strength',
#              fontsize=16)
ax = {}
figure_num = 1
gs = gridspec.GridSpec(1, figure_num)
gs.update(left=0.32, right=0.98, top=0.82, bottom=0.16, hspace=0.15)
for i in range(figure_num):
    ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[i].plot(_SNN_gui_ampa_scale_values, coherence.mean(axis=1), color=plt_color[1], linewidth=1, alpha=0.7)
    ax[i].fill_between(_SNN_gui_ampa_scale_values, coherence.mean(axis=1) - coherence.std(axis=1), coherence.mean(axis=1) + coherence.std(axis=1), color=plt_color[1], alpha=0.2)
    ax[i].set_ylim([0, max(coherence.mean(axis=1) + coherence.std(axis=1))+0.05])
    legend = ax[i].legend([f'SNN'],loc='best'
                          )
    legend.legendHandles[0]._sizes = [12]
    locator = ax[i].yaxis.get_major_locator()
    ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
    ax[i].tick_params(axis='y', labelsize=13)
    ax[i].tick_params(axis='x', labelsize=13)
    ax[i].set_xticks(np.concatenate((np.round(_SNN_gui_ampa_scale_values[xticks_index], 0), np.array([40]))))

    ax[i].set_ylabel('Coherence', fontsize=16)
    ax[i].set_xlabel(r'$\gamma$', fontsize=16)
fig.savefig(os.path.join(_save_dir,
                         f"SNN_Coherence_{_plot_start}_{_plot_end}.pdf"),
            dpi=200)
plt.show()
