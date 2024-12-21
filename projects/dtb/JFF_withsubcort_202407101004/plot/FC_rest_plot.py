# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: FC_compute.py
# @time: 2024/1/8 16:51

import matplotlib.pyplot as plt
from matplotlib import gridspec, rc_file, rc, rcParams
import re
import os
import numpy as np
import h5py
from scipy import io
import matplotlib.ticker as ticker
from mpi4py import MPI
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr


from scipy.signal import hilbert


def xlz_kop(SIGNALS):
    ele_num, time_num = SIGNALS.shape

    # Obtain the instantaneous phase
    a_s = np.zeros((ele_num, time_num), dtype=complex)
    theta = np.zeros((ele_num, time_num))
    for NN in range(ele_num):
        a_s[NN, :] = hilbert(SIGNALS[NN, :])  # Analytical signals
        theta[NN, :] = np.angle(a_s[NN, :])  # Instantaneous phases

    phase = np.exp(1j * theta)  # The unit vector of phase

    # Calculate the Kuramoto order parameteres at each time point
    kop_complex = np.sum(phase, axis=0)

    r = np.abs(kop_complex) / ele_num
    return r, kop_complex, theta


def get_bold_signal(bold_path, b_min=None, b_max=None, lag=0, voxels_belong_to_region=None):
    bold_y = np.load(bold_path)[lag:]
    if b_max is not None:
        bold_y = b_min + (b_max - b_min) * (bold_y - bold_y[:, voxels_belong_to_region].min()) / (bold_y[:, voxels_belong_to_region].max() - bold_y[:, voxels_belong_to_region].min())
    return bold_y


def wjx_FC_diversity(data, bins=np.linspace(0, 1, 30)):
    b, _ = np.histogram(data, bins=bins)
    p = b / len(data)
    print("p.sum()", p.sum())
    assert np.abs(p.sum() - 1).sum() < 1e-8
    M = bins.shape[0]
    normalization_factor = 2 * (M - 1) / M
    diversity = 1 - np.sum(np.abs(p - 1/M)) / normalization_factor
    return diversity


def compute_theta_density(theta, theta_density_bins=np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/20),
                          theta_diff_density_bins=np.arange(0-np.pi/40, 2*np.pi+np.pi/40, np.pi/20)):

    ele_num, time_num = theta.shape

    theta_density = np.zeros((theta_density_bins.shape[0]-1, time_num))
    theta_diff_density = np.zeros((theta_diff_density_bins.shape[0]-1, time_num))
    # Compute histogram
    for i in range(time_num):
        theta_density[:, i] = np.histogram(theta[:, i], bins=theta_density_bins)[0]
        theta_diff_list = [abs(theta[j, i]-theta[jj, i]) for j in range(ele_num) for jj in range(j+1, ele_num)]
        theta_diff_density[:, i] = np.histogram(np.array(theta_diff_list), bins=theta_diff_density_bins)[0]

    return theta_density, theta_diff_density


def xlz_entropy(data, bins=np.linspace(0, 1, 30)):
    # Compute histogram
    b, _ = np.histogram(data, bins=bins)
    # Calculate probabilities
    p = b / len(data)
    # Count occurrences of zero probability
    zeroN = np.sum(p == 0)

    # Remove zero probabilities
    p = p[p != 0]

    # Compute entropy
    entropy = -np.sum(p * np.log2(p))

    return entropy, zeroN


def timeseriescorr_compare(res_path: str, save_path: str, raw_fc_matrix, sc_vec, load_idx, bold_exp, binnum=30, discard_time=50, delay=3, hpc_label=None, rela_load_idx=0):
    # file_nmae = re.compile(r"bold_.+_assim.npy")
    # # file_nmae = re.compile(r"firing_.+_assim_\d+.npy")  # compute fc using firing rate
    # blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    # assert len(blocks) == 1
    # bold_sim = np.load(os.path.join(res_path, blocks[0]))

    # bold_sim = np.load(os.path.join(res_path, f'bolds_out_record_{load_idx}.npy'))[:449, :hpc_label.shape[0]]
    bold_sim = np.load(os.path.join(res_path, f'BOLD_record_{load_idx}.npy'))[1:450, :hpc_label.shape[0]]
    # bold_sim = np.load(os.path.join(res_path, f'BOLD_record_{load_idx}.npy'))[:449, :hpc_label.shape[0]]

    assert bold_sim.shape[0] == 449

    # dt = 0.5
    # bold_sim = np.load(os.path.join(res_path, f'bolds_out_record_{rela_load_idx+5}_{load_idx:.2f}_{dt}.npy'))[:, :hpc_label.shape[0]]
    # bold_sim = np.load(os.path.join(res_path, f'bolds_out_record_{rela_load_idx+1}_{load_idx:.2f}_{dt}.npy'))[:, :hpc_label.shape[0]]
    # bold_sim = np.load(os.path.join(res_path, f'bolds_out_record_{rela_load_idx}_{load_idx:.2f}_{dt}.npy'))[:, :hpc_label.shape[0]]
    # bold_sim = np.load(os.path.join(res_path, f'bolds_out_recalcu.npy'))
    # bold_sim = np.load(os.path.join(res_path, f'FFreqs_region_divided_before_assim_{0}.npy')).sum(axis=1)
    # for i in range(1, 60):
    #     print("load_idx", load_idx, "i", i)
    #     bold_sim = np.concatenate((bold_sim, np.load(os.path.join(res_path, f'FFreqs_region_divided_before_assim_{i}.npy')).sum(axis=1),), axis=0)

    # bold_sim = bold_sim.reshape((-1, 360)).reshape((-1, 20*800, 360)).sum(axis=1)
    # bold_sim = bold_sim[discard_time:150 :]
    # bold_exp = bold_exp[discard_time:150, :]
    bold_sim = bold_sim[discard_time: :]
    bold_exp = bold_exp[discard_time:bold_sim.shape[0]+discard_time, :]

    total_region = hpc_label

    # HCPregion_name_label = '/public/home/ssct004t/project/zenglb/DetailedDTB/data/raw_data/HCPex_3mm_modified_label.csv'
    # HCPex_3mm_modified_label = pd.read_csv(HCPregion_name_label)
    # reorder_region_label_raw = np.array(HCPex_3mm_modified_label['Label'])
    # reorder_region_label_renew = np.concatenate((reorder_region_label_raw[180:360], reorder_region_label_raw[:180], np.arange(361, 379)))
    # reorder_region_label_renew = [_i for _i in reorder_region_label_renew if _i <= total_region.shape[0]]
    # reorder_region_label_renew = np.array(reorder_region_label_renew)
    reorder_region_label_renew = hpc_label

    raw_fc_matrix = raw_fc_matrix[reorder_region_label_renew-1, :][:, reorder_region_label_renew-1]
    bold_sim = bold_sim[:, reorder_region_label_renew-1]
    bold_exp = bold_exp[:, reorder_region_label_renew-1]

    print("bold_exp", bold_exp.shape)
    print("bold_sim", bold_sim.shape)

    T = len(bold_sim)

    C_total_self = []
    C_total = []
    for ii in range(bold_exp.shape[1]):
        x = np.vstack((bold_exp[:T - delay, ii], bold_sim[delay:, ii]))
        r = np.corrcoef(x)[0, 1]
        C_total.append(r)

        C_total_self.append(np.corrcoef(np.vstack((bold_exp[:T - delay, 0], bold_sim[delay:, ii])))[0, 1])

    C_total = np.array(C_total)
    C_total_self = np.array(C_total_self)

    thalamus_region = np.array([120, 300, 109, 111, 112, 289, 291, 292, 57, 59, 61, 62, 179, 180, 237, 239, 241, 242, 359, 360, 165, 166, 345, 346,
            361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378])
    thalamus_region_idx = np.isin(reorder_region_label_renew, thalamus_region).nonzero()[0]

    # thalamus_region = np.array([1, 181])
    wholebrain_series_corr_record = C_total
    thalamus_series_corr_record = C_total[thalamus_region_idx]
    C_mean_wholebrain_ = np.nanmean(wholebrain_series_corr_record)
    C_mean_thalamus_ = np.nanmean(thalamus_series_corr_record)

    C_mean_wholebrain_self = np.nanmean(C_total_self)

    assert bold_sim.shape[1] == raw_fc_matrix.shape[1]
    N = bold_sim.shape[1]

    assi_fc_matrix = np.corrcoef(bold_sim, rowvar=False)

    assimilated_region = thalamus_region
    other_region = [i for i in total_region if i not in assimilated_region]
    other_region = np.array(other_region)
    assimilated_region_idx = thalamus_region_idx
    other_region_idx = np.isin(reorder_region_label_renew, other_region).nonzero()[0]
    corr_from_assimilated_regions = assi_fc_matrix[assimilated_region_idx, :][:, other_region_idx].reshape(-1).mean()

    corr_within_assimilated_regions_ = assi_fc_matrix[assimilated_region_idx, :][:, assimilated_region_idx].reshape(-1)[:-1].reshape(assimilated_region_idx.shape[0] - 1, assimilated_region_idx.shape[0] + 1)[:, 1:].reshape(-1).mean()
    corr_within_nonassimilated_regions_ = assi_fc_matrix[other_region_idx, :][:, other_region_idx].reshape(-1)[:-1].reshape(other_region_idx.shape[0] - 1, other_region_idx.shape[0] + 1)[:, 1:].mean()

    nodaregion_series_corr_record = C_total[other_region_idx]
    C_mean_nodaregion_ = np.nanmean(nodaregion_series_corr_record)

    assi_fc_matrix_vec = assi_fc_matrix.reshape(-1)[:-1].reshape(assi_fc_matrix.shape[0] - 1, assi_fc_matrix.shape[0] + 1)[:, 1:].reshape(-1)
    raw_fc_matrix_vec = raw_fc_matrix.reshape(-1)[:-1].reshape(raw_fc_matrix.shape[0] - 1, raw_fc_matrix.shape[0] + 1)[:, 1:].reshape(-1)
    mask = ~np.isnan(assi_fc_matrix_vec)
    filtered_assi_fc_matrix_vec = assi_fc_matrix_vec[mask]
    filtered_raw_fc_matrix_vec = raw_fc_matrix_vec[mask]

    filtered_sc_vec = sc_vec[mask]

    print("filtered_assi_fc_matrix_vec", filtered_assi_fc_matrix_vec.shape)
    print("filtered_raw_fc_matrix_vec", filtered_raw_fc_matrix_vec.shape)
    fc_corr_value, fc_corr_p_value = pearsonr(filtered_assi_fc_matrix_vec, filtered_raw_fc_matrix_vec)
    # fc_corr_value = np.corrcoef(filtered_assi_fc_matrix_vec, filtered_raw_fc_matrix_vec)[0, 1]
    fc_dist_rela = np.linalg.norm(filtered_raw_fc_matrix_vec - filtered_assi_fc_matrix_vec) / np.linalg.norm(
        filtered_raw_fc_matrix_vec)
    fc_diversity_value = wjx_FC_diversity(data=np.abs(filtered_assi_fc_matrix_vec), bins=np.linspace(0, 1, binnum))
    fc_entropy_value, _ = xlz_entropy(data=np.abs(filtered_assi_fc_matrix_vec), bins=np.linspace(0, 1, binnum))
    mean_fc_value = filtered_assi_fc_matrix_vec.mean()
    fc_dti_corr_value = np.corrcoef(filtered_assi_fc_matrix_vec, filtered_sc_vec)[0, 1]

    # bold_exp = (bold_exp - bold_exp.mean(axis=0).reshape((1, bold_exp.shape[1]))) / bold_exp.std(axis=0).reshape((1, bold_exp.shape[1]))
    # bold_sim = (bold_sim - bold_sim.mean(axis=0).reshape((1, bold_sim.shape[1]))) / bold_sim.std(axis=0).reshape((1, bold_sim.shape[1]))
    #
    # steps = bold_sim.shape[0]
    # iteration = [i for i in range(steps)]
    # assert len(bold_sim.shape) == 2
    # for i in range(10):
    #     print("show_bold" + str(i))
    #     fig = plt.figure(figsize=(8, 4), dpi=500)
    #     ax1 = fig.add_subplot(1, 1, 1)
    #     ax1.plot(iteration, bold_exp[:steps, i], 'r-')
    #     ax1.plot(iteration, bold_sim[:steps, i], 'b-')
    #     # plt.ylim((0.0, 0.08))
    #     ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
    #     plt.savefig(os.path.join(save_path, f"{load_idx}bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
    #     # plt.savefig(os.path.join(path_out, "figure/bold" + str(i) + ".pdf"), bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)

    bold_sim_zscore = (bold_sim - bold_sim.mean(axis=0).reshape((1, bold_sim.shape[1]))) / bold_sim.std(axis=0).reshape(
        (1, bold_sim.shape[1]))
    value_simbold_kop, _, simbold_theta = xlz_kop(bold_sim_zscore.T)
    value_simbold_kop_mean = value_simbold_kop.mean()

    fig = plt.figure(figsize=(5, 5))
    ax = {}
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.15)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    ax[0].scatter(filtered_assi_fc_matrix_vec, filtered_raw_fc_matrix_vec, color='royalblue')
    ax[0].set_xlabel("Model FC")
    ax[0].set_ylabel("Empirical FC")
    ax[0].text(-0.1, 1.05, "A",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    ax[0].text(0.5, 1.2, f"Correlations of FC matrices={fc_corr_value:.2f}",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    plt.savefig(os.path.join(save_path, f"{load_idx}_trial{0}.png"), dpi=100)

    fig = plt.figure(figsize=(10, 5))
    ax = {}
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.15)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    ax[1] = fig.add_subplot(gs[0, 1], frameon=True)
    mappable = ax[0].imshow(assi_fc_matrix, cmap='RdBu_r', interpolation=None, vmin=-1, vmax=1)
    plt.colorbar(mappable, ax=ax[0], shrink=0.9)
    mappable = ax[1].imshow(raw_fc_matrix, cmap='RdBu_r', interpolation=None, vmin=-1, vmax=1)
    plt.colorbar(mappable, ax=ax[1], shrink=0.9)
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    ax[0].set_title("Model FC")
    ax[1].set_title("Empirical FC")
    ax[0].text(-0.1, 1.05, "A",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    ax[1].text(-0.1, 1.05, "B",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)
    ax[1].text(0.5, 1.2, f"Correlations of FC matrices={fc_corr_value:.2f}",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)
    ax[1].text(0.5, 1.1, f"F-norm distance of FC matrices={fc_dist_rela:.2f}",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)
    plt.savefig(os.path.join(save_path, f"FC{load_idx}_trial{0}.png"), dpi=100)
    return fc_corr_value, fc_dist_rela, fc_diversity_value, fc_entropy_value, mean_fc_value, fc_dti_corr_value, C_mean_thalamus_, C_mean_wholebrain_, value_simbold_kop_mean, C_total, C_mean_wholebrain_self, C_total_self, corr_from_assimilated_regions, C_mean_nodaregion_, corr_within_assimilated_regions_, corr_within_nonassimilated_regions_


if __name__ == "__main__":

    _discard_time = 60
    # _discard_time = 20
    delay = 3

    hpc_label = np.arange(1, 379)

    brain_file = h5py.File(
        r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])[:hpc_label.shape[0], :hpc_label.shape[0]]

    sc[np.diag_indices_from(sc)] = 0
    _sc_vec = sc.reshape(-1)[:-1].reshape(sc.shape[0] - 1, sc.shape[0] + 1)[:, 1:].reshape(-1)

    load_dir_list = list(np.sort(np.concatenate((np.arange(0, 201, 10), np.arange(201, 261)))))

    bold_path = r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/bold_process/202402102326_JFF_regionwise_rest_bold_after_zscore.npy'
    voxels_belong_to_region = np.isin(hpc_label, (120, 300, 109, 111, 112, 289, 291, 292, 57, 59, 61, 62, 179, 180, 237, 239, 241, 242, 359, 360, 165, 166, 345, 346,
            361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378)).nonzero()[0]
    bold_exp = get_bold_signal(bold_path, b_min=0.026, b_max=0.05, lag=0,
                                voxels_belong_to_region=voxels_belong_to_region)[:, :hpc_label.shape[0]]


    print("bold_exp.shape", bold_exp.shape)

    raw_fc_matrix = np.corrcoef(bold_exp, rowvar=False)
    print("raw_fc_matrix.sum()", raw_fc_matrix.sum())

    _base_load_dir = "/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739/save_data_0.01_360000_11/rest_intero_25000_500_378_0.01_360000_3200_hp57_378_500_378_0.25_0.05_0.55_0.4_0.1_0_260_20_False"
    # _base_load_dir = "/public/home/ssct004t/project/wangjiexiang/moment-neural-network/projects/dtb/JFF_withsubcort_202407101004/202407131739/save_data_0.05_360000"

    _base_save_dir = os.path.join(_base_load_dir, f'NEWrefine_BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')

    # _base_save_dir = os.path.join(_base_load_dir, f'test_NEWrefine_BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')
    # _base_save_dir = os.path.join(_base_load_dir, f'NEWrefine_BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')
    # _base_save_dir = os.path.join(_base_load_dir, f'refine_BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')
    # _base_save_dir = os.path.join(_base_load_dir, f'BOLD_region{hpc_label.shape[0]}_samelag{delay}_FC_pic{_discard_time}discard')
    os.makedirs(_base_save_dir, exist_ok=True)

    assert raw_fc_matrix.shape[1] == hpc_label.shape[0]

    _binnum = 30

    _gui_ampa_scale_values = np.linspace(0, 10, 1001)
    # _gui_ampa_scale_values = np.linspace(0, 10, 101)
    _gui_ampa_scale_values = [_gui_ampa_scale_values[i] for i in load_dir_list]
    _gui_ampa_scale_values = np.array(_gui_ampa_scale_values)

    # save_data_done = True
    save_data_done = False
    if not save_data_done:
        fc_corr_record = np.zeros((len(load_dir_list), 1))
        fc_dist_record = np.zeros((len(load_dir_list), 1))
        fc_diversity_record = np.zeros((len(load_dir_list), 1))
        fc_entropy_record = np.zeros((len(load_dir_list), 1))
        mean_fc_record = np.zeros((len(load_dir_list), 1))
        fc_dti_corr_record = np.zeros((len(load_dir_list), 1))
        C_thalamus_max_record = np.zeros((len(load_dir_list), 1))
        C_wholebrain_max_record = np.zeros((len(load_dir_list), 1))
        value_simbold_kop_mean_record = np.zeros((len(load_dir_list), 1))
        C_total_record = np.zeros((hpc_label.shape[0], len(load_dir_list)))
        C_total_self_record = np.zeros((hpc_label.shape[0], len(load_dir_list)))
        C_wholebrain_self_max_record = np.zeros((len(load_dir_list), 1))
        corr_from_assimilated_regions_record = np.zeros((len(load_dir_list), 1))
        C_mean_nodaregion_record = np.zeros((len(load_dir_list), 1))
        corr_within_assimilated_regions_record  = np.zeros((len(load_dir_list), 1))
        corr_within_nonassimilated_regions_record = np.zeros((len(load_dir_list), 1))
        for load_idx in range(len(load_dir_list)):
            _load_dir = _base_load_dir
            # _load_dir = os.path.join(_base_load_dir, f"{load_dir_list[load_idx]}")
            fc_corr_value, fc_dist_rela, fc_diversity_value, fc_entropy_value, mean_fc_value, fc_dti_corr_value, C_thalamus_max_, C_wholebrain_max_, value_simbold_kop_mean_, C_total_, C_mean_wholebrain_self_, C_total_self_, corr_from_assimilated_regions_, C_mean_nodaregion_, corr_within_assimilated_regions_, corr_within_nonassimilated_regions_ = timeseriescorr_compare(res_path=_load_dir, save_path=_base_save_dir, bold_exp=bold_exp, raw_fc_matrix=raw_fc_matrix, sc_vec=_sc_vec, load_idx=load_dir_list[load_idx], binnum=_binnum, discard_time=_discard_time, delay=delay, hpc_label=hpc_label, rela_load_idx=load_idx)
            # fc_corr_value, fc_dist_rela, fc_diversity_value, fc_entropy_value, mean_fc_value, fc_dti_corr_value, C_thalamus_max_, C_wholebrain_max_ = timeseriescorr_compare(res_path=_load_dir, save_path=_base_save_dir, bold_exp=bold_exp, raw_fc_matrix=raw_fc_matrix, sc_vec=_sc_vec, load_idx=load_dir_list[load_idx], binnum=_binnum, discard_time=_discard_time)
            fc_corr_record[load_idx, 0] = fc_corr_value
            fc_dist_record[load_idx, 0] = fc_dist_rela
            fc_diversity_record[load_idx, 0] = fc_diversity_value
            fc_entropy_record[load_idx, 0] = fc_entropy_value
            mean_fc_record[load_idx, 0] = mean_fc_value
            fc_dti_corr_record[load_idx, 0] = fc_dti_corr_value
            C_thalamus_max_record[load_idx, 0] = C_thalamus_max_
            C_wholebrain_max_record[load_idx, 0] = C_wholebrain_max_
            value_simbold_kop_mean_record[load_idx, 0] = value_simbold_kop_mean_
            C_total_record[:, load_idx] = C_total_
            C_total_self_record[:, load_idx] = C_total_self_
            C_wholebrain_self_max_record[load_idx, 0] = C_mean_wholebrain_self_
            corr_from_assimilated_regions_record[load_idx, 0] = corr_from_assimilated_regions_
            C_mean_nodaregion_record[load_idx, 0] = C_mean_nodaregion_
            corr_within_assimilated_regions_record[load_idx, 0] = corr_within_assimilated_regions_
            corr_within_nonassimilated_regions_record[load_idx, 0] = corr_within_nonassimilated_regions_

        np.save(os.path.join(_base_save_dir, 'fc_corr_record.npy'), fc_corr_record)
        np.save(os.path.join(_base_save_dir, 'fc_dist_record.npy'), fc_dist_record)
        np.save(os.path.join(_base_save_dir, 'fc_diversity_record.npy'), fc_diversity_record)
        np.save(os.path.join(_base_save_dir, 'fc_entropy_record.npy'), fc_entropy_record)
        np.save(os.path.join(_base_save_dir, 'mean_fc_record.npy'), mean_fc_record)
        np.save(os.path.join(_base_save_dir, 'fc_dti_corr_record.npy'), fc_dti_corr_record)
        np.save(os.path.join(_base_save_dir, 'C_thalamus_max_record.npy'), C_thalamus_max_record)
        np.save(os.path.join(_base_save_dir, 'C_wholebrain_max_record.npy'), C_wholebrain_max_record)
        np.save(os.path.join(_base_save_dir, 'value_simbold_kop_mean_record.npy'), value_simbold_kop_mean_record)
        np.save(os.path.join(_base_save_dir, 'C_total_record.npy'), C_total_record)
        np.save(os.path.join(_base_save_dir, 'C_total_self_record.npy'), C_total_self_record)
        np.save(os.path.join(_base_save_dir, 'C_wholebrain_self_max_record.npy'), C_wholebrain_self_max_record)
        np.save(os.path.join(_base_save_dir, 'corr_from_assimilated_regions_record.npy'), corr_from_assimilated_regions_record)
        np.save(os.path.join(_base_save_dir, 'C_mean_nodaregion_record.npy'), C_mean_nodaregion_record)
        np.save(os.path.join(_base_save_dir, 'corr_within_assimilated_regions_record.npy'), corr_within_assimilated_regions_record)
        np.save(os.path.join(_base_save_dir, 'corr_within_nonassimilated_regions_record.npy'), corr_within_nonassimilated_regions_record)
    else:
        fc_corr_record = np.load(os.path.join(_base_save_dir, 'fc_corr_record.npy'))
        fc_dist_record = np.load(os.path.join(_base_save_dir, 'fc_dist_record.npy'))
        fc_diversity_record = np.load(os.path.join(_base_save_dir, 'fc_diversity_record.npy'))
        fc_entropy_record = np.load(os.path.join(_base_save_dir, 'fc_entropy_record.npy'))
        mean_fc_record = np.load(os.path.join(_base_save_dir, 'mean_fc_record.npy'))
        fc_dti_corr_record = np.load(os.path.join(_base_save_dir, 'fc_dti_corr_record.npy'))
        C_thalamus_max_record = np.load(os.path.join(_base_save_dir, 'C_thalamus_max_record.npy'))
        C_wholebrain_max_record = np.load(os.path.join(_base_save_dir, 'C_wholebrain_max_record.npy'))
        value_simbold_kop_mean_record = np.load(os.path.join(_base_save_dir, 'value_simbold_kop_mean_record.npy'))
        C_total_record = np.load(os.path.join(_base_save_dir, 'C_total_record.npy'))
        C_total_self_record = np.load(os.path.join(_base_save_dir, 'C_total_self_record.npy'))
        C_wholebrain_self_max_record = np.load(os.path.join(_base_save_dir, 'C_wholebrain_self_max_record.npy'))
        corr_from_assimilated_regions_record = np.load(os.path.join(_base_save_dir, 'corr_from_assimilated_regions_record.npy'))
        C_mean_nodaregion_record = np.load(os.path.join(_base_save_dir, 'C_mean_nodaregion_record.npy'))
        corr_within_assimilated_regions_record = np.load(os.path.join(_base_save_dir, 'corr_within_assimilated_regions_record.npy'))
        corr_within_nonassimilated_regions_record = np.load(os.path.join(_base_save_dir, 'corr_within_nonassimilated_regions_record.npy'))

    _raw_fc_matrix_vec = raw_fc_matrix.reshape(-1)[:-1].reshape(raw_fc_matrix.shape[0] - 1, raw_fc_matrix.shape[0] + 1)[:, 1:].reshape(-1)
    _raw_fc_diversity_value = wjx_FC_diversity(data=np.abs(_raw_fc_matrix_vec), bins=np.linspace(0, 1, _binnum))
    _raw_fc_entropy_value, _ = xlz_entropy(data=np.abs(_raw_fc_matrix_vec), bins=np.linspace(0, 1, _binnum))
    _raw_mean_fc_value = _raw_fc_matrix_vec.mean()
    _raw_fc_dti_corr_value = fc_dti_corr_value = np.corrcoef(_raw_fc_matrix_vec, _sc_vec)[0, 1]

    _plot_start = 0
    # _plot_end = 18
    # _plot_end = len(load_dir_list) - 1
    # _plot_end = 52
    # _plot_end = 43
    # _plot_end = 45
    _plot_end = len(load_dir_list)

    # _plot_end = 23
    # _plot_end = 9
    # _plot_end = 31
    # _plot_end = 23
    # _plot_end = 50
    # _plot_end = 21
    # _plot_end = 32
    # _plot_end = 22
    # _plot_end = 21
    # _plot_end = 15
    _gui_ampa_scale_values = _gui_ampa_scale_values[_plot_start:_plot_end]
    fc_corr_record = fc_corr_record[_plot_start:_plot_end, :]
    fc_dist_record = fc_dist_record[_plot_start:_plot_end, :]
    fc_diversity_record = fc_diversity_record[_plot_start:_plot_end, :]
    fc_entropy_record = fc_entropy_record[_plot_start:_plot_end, :]
    mean_fc_record = mean_fc_record[_plot_start:_plot_end, :]
    fc_dti_corr_record = fc_dti_corr_record[_plot_start:_plot_end, :]
    C_thalamus_max_record = C_thalamus_max_record[_plot_start:_plot_end, :]
    C_wholebrain_max_record = C_wholebrain_max_record[_plot_start:_plot_end, :]
    value_simbold_kop_mean_record = value_simbold_kop_mean_record[_plot_start:_plot_end, :]
    C_total_record = C_total_record[:, _plot_start:_plot_end]

    C_mean_nodaregion_record = C_mean_nodaregion_record[_plot_start:_plot_end, :]
    corr_within_assimilated_regions_record = corr_within_assimilated_regions_record[_plot_start:_plot_end, :]
    corr_within_nonassimilated_regions_record = corr_within_nonassimilated_regions_record[_plot_start:_plot_end, :]

    C_wholebrain_self_max_record = C_wholebrain_self_max_record[_plot_start:_plot_end, :]
    C_total_self_record = C_total_self_record[:, _plot_start:_plot_end]
    corr_from_assimilated_regions_record = corr_from_assimilated_regions_record[_plot_start:_plot_end, :]

    print("C_wholebrain_max_record", C_wholebrain_max_record)

    '0 40 80 120 130 140 150 160 161 162 163 164 165 166 167 168 169 170 180 190 200 210 220'
    plt_color = ['royalblue', 'tomato', 'violet', 'mediumseagreen']
    desired_number_of_ticks = 3
    # xticks_index = list(np.arange(0, 21, 4))
    xticks_index = list(np.sort(np.concatenate((np.arange(0, 21, 4), np.arange(60, _plot_end, 40)))))
    # xticks_index = list(np.sort(np.concatenate((np.arange(0, 21, 4), np.arange(51, len(load_dir_list), 40)))))
    # xticks_index = list(np.sort(np.concatenate((np.arange(0, 41, 4), np.arange(53, len(load_dir_list), 40)))))
    # xticks_index = list(range(0, len(load_dir_list), 10))
    # xticks_index = [0, 4, 8, 12, 16]
    # xticks_index = [0, 4, 8, 12, 16]
    # xticks_index = [0, 1, 2, 3, 43, 83]
    # xticks_index = list(range(3, len(load_dir_list), 4))
    # xticks_index = list(range(0, len(load_dir_list), 4))
    # xticks_index = [0, 4, 8, 12, 25]
    # xticks_index = [0, 4, 8, 12, 16, 20, 24, 28]
    # xticks_index = [0, 1, 2, 3, 7, 20]
    # xticks_index = [0, 4, 8, 12, 16, 20]
    # xticks_index = [0, 4, 8, 12, 16, 20, 24, 64]
    # xticks_index = [0, 4, 8]

    fig = plt.figure(figsize=(10, 5), dpi=200)
    fig.suptitle(r'Performance of the DTB for different inter-region coupling strength',
                 fontsize=16)
    ax = {}
    figure_num = 2
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.09, right=0.95, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[0], alpha=0.25)
    ax[1].errorbar(_gui_ampa_scale_values,  fc_dist_record.mean(axis=1), yerr=fc_dist_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_dist_record.reshape(-1), color=plt_color[1], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('Correlations between FC matrices', fontsize=12)
    ax[1].set_ylabel('F-norm distance for FC matrices', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"FCcorr_dist_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=100)
    plt.show()

    modified_parameter_tao_ui = np.array([4., 4., 10., 50.])
    all_scale = 0.25
    scale_factor = 2500
    _J_EE = all_scale * 2
    _J_EE_scaled = _J_EE / np.sqrt(scale_factor)
    _gui_ampa_scale_values = _gui_ampa_scale_values * _J_EE_scaled / modified_parameter_tao_ui[0] * 1000

    figure_num = 3
    fig = plt.figure(figsize=(figure_num*5, 5), dpi=200)
    fig.suptitle(r'Performance of the DTB for different inter-region coupling strength',
                 fontsize=16)
    ax = {}
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.09, right=0.95, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  fc_diversity_record.mean(axis=1), yerr=fc_diversity_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  fc_diversity_record.reshape(-1), color=plt_color[0], alpha=0.25)
    ax[1].errorbar(_gui_ampa_scale_values,  fc_entropy_record.mean(axis=1), yerr=fc_entropy_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[1])
    ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_entropy_record.reshape(-1), color=plt_color[1], alpha=0.25)
    ax[2].errorbar(_gui_ampa_scale_values,  mean_fc_record.mean(axis=1), yerr=mean_fc_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    ax[2].scatter(_gui_ampa_scale_values.repeat(1),  mean_fc_record.reshape(-1), color=plt_color[2], alpha=0.25)
    # ax[3].errorbar(_gui_ampa_scale_values,  fc_dti_corr_record.mean(axis=1), yerr=fc_dti_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[3])
    # ax[3].scatter(_gui_ampa_scale_values.repeat(1),  fc_dti_corr_record.reshape(-1), color=plt_color[3], alpha=0.25)

    print("fc_dti_corr_record", fc_dti_corr_record)
    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    print("_raw_fc_dti_corr_value", _raw_fc_dti_corr_value)

    ax[0].set_ylabel('FC diversity', fontsize=12)
    ax[1].set_ylabel('FC entropy', fontsize=12)
    ax[2].set_ylabel('Mean PCC of FC', fontsize=12)
    # ax[3].set_ylabel('PCC between FC and SC', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[2].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    # ax[3].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[0].axhline(y=_raw_fc_diversity_value, color=plt_color[0], linestyle='--')
    ax[1].axhline(y=_raw_fc_entropy_value, color=plt_color[1], linestyle='--')
    ax[2].axhline(y=_raw_mean_fc_value, color=plt_color[2], linestyle='--')
    # ax[3].axhline(y=_raw_fc_dti_corr_value, color=plt_color[3], linestyle='--')
    fig.savefig(os.path.join(_base_save_dir, f"FCstat_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=100)
    plt.show()

    fig = plt.figure(figsize=(10, 5), dpi=200)
    fig.suptitle(r'Performance of the DTB for different inter-region coupling strength',
                 fontsize=16)
    ax = {}
    figure_num = 2
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.09, right=0.95, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  C_wholebrain_max_record.mean(axis=1), yerr=C_wholebrain_max_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  C_wholebrain_max_record.reshape(-1), color=plt_color[0], alpha=0.25)
    ax[1].errorbar(_gui_ampa_scale_values,  C_thalamus_max_record.mean(axis=1), yerr=C_thalamus_max_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    ax[1].scatter(_gui_ampa_scale_values.repeat(1),  C_thalamus_max_record.reshape(-1), color=plt_color[1], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('PCC of BOLD in the whole brain', fontsize=12)
    ax[1].set_ylabel('PCC of BOLD in V1', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"timeseriescorr_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=100)
    plt.show()

    fig = plt.figure(figsize=(10, 5), dpi=200)
    fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
                 fontsize=16)
    ax = {}
    figure_num = 2
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  C_wholebrain_max_record.mean(axis=1), yerr=C_wholebrain_max_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  C_wholebrain_max_record.reshape(-1), color=plt_color[0], alpha=0.25)
    ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('PCC between BOLD in the whole brain', fontsize=12)
    ax[1].set_ylabel('PCC between FC matrices', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"FCandtimeseriescorr_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    bold_exp_zscore = (bold_exp - bold_exp.mean(axis=0).reshape((1, bold_exp.shape[1]))) / bold_exp.std(axis=0).reshape(
        (1, bold_exp.shape[1]))
    raw_value_expbold_kop_mean = xlz_kop(bold_exp_zscore[_discard_time:, :].T)[0].mean()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
                 fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.07, right=0.97, top=0.9, bottom=0.1, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  value_simbold_kop_mean_record.mean(axis=1), yerr=value_simbold_kop_mean_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  value_simbold_kop_mean_record.reshape(-1), color=plt_color[0], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].axhline(y=raw_value_expbold_kop_mean, color=plt_color[0], linestyle='--')
    ax[0].set_ylabel('Mean KOP of region-wise BOLD', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"kop_mean_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    fig = plt.figure(figsize=(10, 5), dpi=300)
    figure_num = 1
    ax = {}
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.15)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    mappable = ax[0].imshow(C_total_record, cmap='RdBu_r', aspect="auto", origin="lower", interpolation=None, vmin=-1, vmax=1)
    plt.colorbar(mappable, ax=ax[0], shrink=0.9)
    for i in range(figure_num):
        ax[i].tick_params(axis='y', labelsize=10)
        ax[i].tick_params(axis='x', labelsize=10)
        # ax[i].set_yticks([0, 1])
        # ax[i].set_yticklabels([1, 2])
        ax[i].set_xlabel('Inter-region coupling strength (nS)', fontsize=12)
        ax[i].set_xticks(xticks_index)
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))
    ax[0].set_title("PCC between BOLD in the whole brain", fontsize=12)
    ax[0].set_ylabel("Brain region", fontsize=12)
    ax[0].text(-0.1, 1.05, "A",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    fig.savefig(os.path.join(_base_save_dir, f"PCC_BOLD_wholebrain_{_plot_start}_{_plot_end}.png"), dpi=300)

    fig = plt.figure(figsize=(10, 5), dpi=300)
    figure_num = 1
    ax = {}
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.15)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    mappable = ax[0].imshow(C_total_self_record, cmap='RdBu_r', aspect="auto", origin="lower", interpolation=None, vmin=-1, vmax=1)
    plt.colorbar(mappable, ax=ax[0], shrink=0.9)
    for i in range(figure_num):
        ax[i].tick_params(axis='y', labelsize=10)
        ax[i].tick_params(axis='x', labelsize=10)
        # ax[i].set_yticks([0, 1])
        # ax[i].set_yticklabels([1, 2])
        # ax[i].set_yticks([0, 180, 360])
        # ax[i].set_yticklabels([1, 181, 361])
        ax[i].set_xlabel('Inter-region coupling strength (nS)', fontsize=12)
        ax[i].set_xticks(xticks_index)
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))
    ax[0].set_title("PCC between BOLD in the whole brain", fontsize=12)
    ax[0].set_ylabel("Brain region", fontsize=12)
    ax[0].text(-0.1, 1.05, "A",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    fig.savefig(os.path.join(_base_save_dir, f"self_PCC_BOLD_wholebrain_{_plot_start}_{_plot_end}.png"), dpi=300)

    fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
    # fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
    #              fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.15, right=0.97, top=0.94, bottom=0.15, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  C_wholebrain_self_max_record.mean(axis=1), yerr=C_wholebrain_self_max_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  C_wholebrain_self_max_record.reshape(-1), color=plt_color[0], alpha=0.25)
    # ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    # ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('PCC between BOLD in the whole brain', fontsize=12)
    # ax[1].set_ylabel('PCC between FC matrices', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    # ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"self_smalltimeseriescorr_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    # fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
    #              fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.2, right=0.97, top=0.94, bottom=0.15, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  corr_from_assimilated_regions_record.mean(axis=1), yerr=corr_from_assimilated_regions_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  corr_from_assimilated_regions_record.reshape(-1), color=plt_color[0], alpha=0.25)
    # ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    # ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('Mean PCC between simulated BOLD in the\nassimilated regions and other regions', fontsize=12)
    # ax[1].set_ylabel('PCC between FC matrices', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    # ax[1].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"corr_from_assimilated_timeseriescorr_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    # fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
    #              fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.2, right=0.97, top=0.94, bottom=0.15, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  C_mean_nodaregion_record.mean(axis=1), yerr=C_mean_nodaregion_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  C_mean_nodaregion_record.reshape(-1), color=plt_color[0], alpha=0.25)
    # ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    # ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('PCC of BOLD in the non-assimilated regions', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"C_mean_nodaregion_record_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    # fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
    #              fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.2, right=0.97, top=0.94, bottom=0.15, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  corr_within_assimilated_regions_record.mean(axis=1), yerr=corr_within_assimilated_regions_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  corr_within_assimilated_regions_record.reshape(-1), color=plt_color[0], alpha=0.25)
    # ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    # ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('Mean PCC between simulated BOLD within assimilated regions', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"corr_within_assimilated_regions_record_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    # fig.suptitle(r'Performance of the DTB at task for different inter-region coupling strength',
    #              fontsize=16)
    ax = {}
    figure_num = 1
    gs = gridspec.GridSpec(1, figure_num)
    gs.update(left=0.2, right=0.97, top=0.94, bottom=0.15, hspace=0.2, wspace=0.25)
    for i in range(figure_num):
        ax[i] = fig.add_subplot(gs[0, i], frameon=True)

    ax[0].errorbar(_gui_ampa_scale_values,  corr_within_nonassimilated_regions_record.mean(axis=1), yerr=corr_within_nonassimilated_regions_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[0])
    ax[0].scatter(_gui_ampa_scale_values.repeat(1),  corr_within_nonassimilated_regions_record.reshape(-1), color=plt_color[0], alpha=0.25)
    # ax[1].errorbar(_gui_ampa_scale_values,  fc_corr_record.mean(axis=1), yerr=fc_corr_record.std(axis=1), fmt='-o', capsize=6, color=plt_color[2])
    # ax[1].scatter(_gui_ampa_scale_values.repeat(1),  fc_corr_record.reshape(-1), color=plt_color[2], alpha=0.25)

    for i in range(figure_num):
        locator = ax[i].yaxis.get_major_locator()
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(desired_number_of_ticks))
        # ax[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax[i].tick_params(axis='y', labelsize=9)
        ax[i].set_xticks(np.round(_gui_ampa_scale_values[xticks_index], 3))
        ax[i].set_xticklabels(np.round(_gui_ampa_scale_values[xticks_index], 3))

    ax[0].set_ylabel('Mean PCC between simulated BOLD within non-assimilated regions', fontsize=12)
    ax[0].set_xlabel(r'Inter-region coupling strength (nS)', fontsize=12)
    fig.savefig(os.path.join(_base_save_dir, f"corr_within_nonassimilated_regions_record_varying_Inter-region_synaptic_strength_{_plot_start}_{_plot_end}.png"), dpi=200)
    plt.show()
