# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:54:44 2024

@author: qiyangku
"""

import torch
import numpy as np
from projects.dtb.JFF_withsubcort_202407291634.ma_conductance_withhp_float64_202407291634 import Moment_Activation_Cond
from projects.dtb.get_connectivity import calculate_degree
from utils.helpers import torch_2_numpy, numpy2torch
from models.bold_model_pytorch import BOLD
from default_params import bold_params
from scipy.integrate import solve_ivp


class Cond_MNN_DTB():
    def __init__(self, config_for_Cond_MNN_DTB, config_for_Moment_exc_activation=None, config_for_Moment_inh_activation=None):
        self.is_cuda = config_for_Cond_MNN_DTB['is_cuda']  # use cuda if is_cuda == True
        if config_for_Cond_MNN_DTB['is_cuda']:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # initialize activation functions
        self.exc_activation = Moment_Activation_Cond(config_for_Moment_exc_activation)
        self.inh_activation = Moment_Activation_Cond(config_for_Moment_inh_activation)

        self._EI_ratio = config_for_Cond_MNN_DTB['_EI_ratio']
                
        # initializing DTI-constrained connectivity in-degree 
        K_EE, K_EI, K_IE, K_II, K_EE_long, degree_, block_size = calculate_degree(
            single_voxel_size=config_for_Cond_MNN_DTB['single_voxel_size'],
            degree=config_for_Cond_MNN_DTB['degree'],
            n_region=config_for_Cond_MNN_DTB['n_region'],
            _EI_ratio=self._EI_ratio, use_cluster=True, kunshan_cluster=True
            # _EI_ratio=self._EI_ratio, use_cluster=False
        )
        # assert np.unique(degree_)[0] == config_for_Cond_MNN_DTB['degree']
        self.N = K_EE.shape[0] # number of brain regions

        
        self.K_EE = torch.tensor(K_EE, dtype=torch.float64, device=self.device).unsqueeze(1) # dimensionality: Nx1
        self.K_EI = torch.tensor(K_EI, dtype=torch.float64, device=self.device).unsqueeze(1)
        self.K_IE = torch.tensor(K_IE, dtype=torch.float64, device=self.device).unsqueeze(1)
        self.K_II = torch.tensor(K_II, dtype=torch.float64, device=self.device).unsqueeze(1)
        self.K_EE_long = torch.tensor(K_EE_long, dtype=torch.float64, device=self.device)

        self.K_EE_numpy = K_EE.reshape((-1, 1)) # dimensionality: Nx1
        self.K_EI_numpy = K_EI.reshape((-1, 1))
        self.K_IE_numpy = K_IE.reshape((-1, 1))
        self.K_II_numpy = K_II.reshape((-1, 1))
        self.K_EE_long_numpy = K_EE_long.astype(np.float32)

        self.block_size = torch.tensor(block_size, dtype=torch.float64, device=self.device).unsqueeze(1) # dimensionality: Nx1
        self.block_size_numpy = block_size.astype(np.float32) # dimensionality: Nx1

        self.dt_mnn = config_for_Cond_MNN_DTB['dt_mnn']  # simulation time step for MNN, default 0.1 ms
        self.mnn_membrane_constant = config_for_Cond_MNN_DTB['mnn_membrane_constant']  # simulation time step for MNN, default 15 ms

        # # calculate background current stats
        # self.bg_mean_to_exc = config_for_Cond_MNN_DTB['bg_mean_to_exc'] #nA/uS; output is mV
        # self.bg_mean_to_inh = config_for_Cond_MNN_DTB['bg_mean_to_inh']
        # self.bg_std_to_exc = config_for_Cond_MNN_DTB['bg_std_to_exc']
        # self.bg_std_to_inh = config_for_Cond_MNN_DTB['bg_std_to_inh']

        self.xregion_gain_for_E_popu = torch.tensor(config_for_Cond_MNN_DTB['xregion_gain_for_E_popu'], dtype=torch.float64, device=self.device) # cross regional exc connection gain
        self.local_w_EE = torch.tensor(config_for_Cond_MNN_DTB['local_w_EE'], dtype=torch.float64, device=self.device) # cross regional exc connection gain
        self.local_w_EI = torch.tensor(config_for_Cond_MNN_DTB['local_w_EI'], dtype=torch.float64, device=self.device) # cross regional exc connection gain
        self.local_w_IE = torch.tensor(config_for_Cond_MNN_DTB['local_w_IE'], dtype=torch.float64, device=self.device) # cross regional exc connection gain
        self.local_w_II = torch.tensor(config_for_Cond_MNN_DTB['local_w_II'], dtype=torch.float64, device=self.device) # cross regional exc connection gain

        self.xregion_gain_for_E_popu_numpy = config_for_Cond_MNN_DTB['xregion_gain_for_E_popu']
        self.local_w_EE_numpy = config_for_Cond_MNN_DTB['local_w_EE']
        self.local_w_EI_numpy = config_for_Cond_MNN_DTB['local_w_EI']
        self.local_w_IE_numpy = config_for_Cond_MNN_DTB['local_w_IE']
        self.local_w_II_numpy = config_for_Cond_MNN_DTB['local_w_II']

        # self.batchsize = config_for_Cond_MNN_DTB['batchsize'] # not used now

        self.conduct_scaling_factor = torch.tensor(config_for_Cond_MNN_DTB['degree'] / degree_, dtype=torch.float64, device=self.device).unsqueeze(1) # dimensionality: Nx1
        self.conduct_scaling_factor_numpy = (config_for_Cond_MNN_DTB['degree'] / degree_).reshape((-1, 1)) # dimensionality: Nx1

        self.hp = torch.tensor(config_for_Cond_MNN_DTB['hp'], dtype=torch.float64, device=self.device)
        self.hp_numpy = config_for_Cond_MNN_DTB['hp'].astype(np.float32)
        self._assi_region = config_for_Cond_MNN_DTB['_assi_region'] # np.array([1])
        self.hp_update_time = config_for_Cond_MNN_DTB['hp_update_time'] # default 800 (ms)

        self.time_unit_converted = 1000  # convert (ms) into (s)
        self.bold = BOLD(delta_t=self.dt_mnn / self.time_unit_converted, **bold_params)

        self.init_time = config_for_Cond_MNN_DTB.get("init_time", round(0.8 * 1000 * 4))  # round(0.8 * 1000 * 4)
        self.T = config_for_Cond_MNN_DTB.get("T", round(0.8 * 1000 * 449))  # round(0.8 * 1000 * 449)

        self.save_dt = config_for_Cond_MNN_DTB.get("save_dt", self.dt_mnn)

        # TODO
        # 现在self.xregion_gain_for_E_popu是个数，但self.conduct_scaling_factor是个向量
        # 要对self.exc_activation.sE self.exc_activation.sI self.inh_activation.sI self.inh_activation.sI进行修改
        # 要加超参
        # ！！！超参电流和背景电流都要小心有没有除以g_L

    def synaptic_summation(self, ue, se, ui, si):
        assert len(ue.shape) == 2
        exc_input_mean_for_Epopu = (torch.mul(ue, self.K_EE.T) * self.local_w_EE + torch.mm(ue,
                                                                                  self.K_EE_long.T) * self.xregion_gain_for_E_popu) * self.conduct_scaling_factor.T
        inh_input_mean_for_Epopu = torch.mul(ui, self.K_EI.T) * self.local_w_EI * self.conduct_scaling_factor.T
        exc_input_std_for_Epopu = torch.mul(se * se, self.K_EE.T) * self.local_w_EE * self.local_w_EE * self.conduct_scaling_factor.T * self.conduct_scaling_factor.T \
                                  + torch.mm(se * se, self.K_EE_long.T) * self.xregion_gain_for_E_popu * self.xregion_gain_for_E_popu \
                                  * self.conduct_scaling_factor.T * self.conduct_scaling_factor.T
        exc_input_std_for_Epopu = exc_input_std_for_Epopu.pow(0.5)

        inh_input_std_for_Epopu = torch.mul(si * si, self.K_EI.T).pow(0.5) * self.local_w_EI * self.conduct_scaling_factor.T

        exc_input_mean_for_Ipopu = torch.mul(ue, self.K_IE.T) * self.local_w_IE * self.conduct_scaling_factor.T
        inh_input_mean_for_Ipopu = torch.mul(ui, self.K_II.T) * self.local_w_II * self.conduct_scaling_factor.T
        exc_input_std_for_Ipopu = torch.mul(se * se, self.K_IE.T).pow(0.5) * self.local_w_IE * self.conduct_scaling_factor.T
        inh_input_std_for_Ipopu = torch.mul(si * si, self.K_II.T).pow(0.5) * self.local_w_II * self.conduct_scaling_factor.T

        return exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu, \
               exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu

    def effective_current_stat(self, exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                               exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu, external_current_forEpopu_):
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu = self.exc_activation.cond2curr_includingbgOU_withhp(exc_input_mean=exc_input_mean_for_Epopu,
                                                                                                    exc_input_std=exc_input_std_for_Epopu,
                                                                                                    inh_input_mean=inh_input_mean_for_Epopu,
                                                                                                    inh_input_std=inh_input_std_for_Epopu,
                                                                                                    external_current=external_current_forEpopu_)

        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.inh_activation.cond2curr_includingbgOU_withhp(exc_input_mean=exc_input_mean_for_Ipopu,
                                                                                                    exc_input_std=exc_input_std_for_Ipopu,
                                                                                                    inh_input_mean=inh_input_mean_for_Ipopu,
                                                                                                    inh_input_std=inh_input_std_for_Ipopu,
                                                                                                    external_current=torch.zeros_like(exc_input_mean_for_Epopu).type_as(exc_input_mean_for_Epopu))
        return eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu

    def synaptic_summation_and_effective_current(self, ue, se, ui, si, _external_current_forEpopu):
        #  no hyperparameter involved
        #  for convenience of calculation of Jacobian matrix
        assert len(ue.shape) == 1

        ue = ue.reshape((1, -1))
        se = se.reshape((1, -1))
        ui = ui.reshape((1, -1))
        si = si.reshape((1, -1))

        exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu, \
        exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu = self.synaptic_summation(ue, se, ui, si)

        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = \
            self.effective_current_stat(exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                                        exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu, external_current_forEpopu_=_external_current_forEpopu)

        #  for convenience of calculation of Jacobian matrix
        eff_input_mean_for_Epopu = eff_input_mean_for_Epopu.reshape(-1)
        eff_input_std_for_Epopu = eff_input_std_for_Epopu.reshape(-1)
        tau_eff_for_Epopu = tau_eff_for_Epopu.reshape(-1)
        eff_input_mean_for_Ipopu = eff_input_mean_for_Ipopu.reshape(-1)
        eff_input_std_for_Ipopu = eff_input_std_for_Ipopu.reshape(-1)
        tau_eff_for_Ipopu = tau_eff_for_Ipopu.reshape(-1)
        return eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu

    def synaptic_summation_for_convenience_of_calculation_of_Jacobian(self, ue, se, ui, si):
        #  no hyperparameter involved
        #  for convenience of calculation of Jacobian matrix
        assert len(ue.shape) == 1

        ue = ue.reshape((1, -1))
        se = se.reshape((1, -1))
        ui = ui.reshape((1, -1))
        si = si.reshape((1, -1))

        exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu, \
        exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu = self.synaptic_summation(ue, se, ui, si)

        #  for convenience of calculation of Jacobian matrix
        exc_input_mean_for_Epopu = exc_input_mean_for_Epopu.reshape(-1)
        exc_input_std_for_Epopu = exc_input_std_for_Epopu.reshape(-1)
        inh_input_mean_for_Epopu = inh_input_mean_for_Epopu.reshape(-1)
        inh_input_std_for_Epopu = inh_input_std_for_Epopu.reshape(-1)
        exc_input_mean_for_Ipopu = exc_input_mean_for_Ipopu.reshape(-1)
        exc_input_std_for_Ipopu = exc_input_std_for_Ipopu.reshape(-1)
        inh_input_mean_for_Ipopu = inh_input_mean_for_Ipopu.reshape(-1)
        inh_input_std_for_Ipopu = inh_input_std_for_Ipopu.reshape(-1)
        return (exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu)

    def effective_current_for_convenience_of_calculation_of_Jacobian(self, exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                                                 exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu, _external_current_forEpopu):
        #  no hyperparameter involved
        #  for convenience of calculation of Jacobian matrix
        assert len(exc_input_mean_for_Epopu.shape) == 1

        exc_input_mean_for_Epopu = exc_input_mean_for_Epopu.reshape((1, -1))
        exc_input_std_for_Epopu = exc_input_std_for_Epopu.reshape((1, -1))
        inh_input_mean_for_Epopu = inh_input_mean_for_Epopu.reshape((1, -1))
        inh_input_std_for_Epopu = inh_input_std_for_Epopu.reshape((1, -1))
        exc_input_mean_for_Ipopu = exc_input_mean_for_Ipopu.reshape((1, -1))
        exc_input_std_for_Ipopu = exc_input_std_for_Ipopu.reshape((1, -1))
        inh_input_mean_for_Ipopu = inh_input_mean_for_Ipopu.reshape((1, -1))
        inh_input_std_for_Ipopu = inh_input_std_for_Ipopu.reshape((1, -1))

        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = \
            self.effective_current_stat(exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                                        exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu, external_current_forEpopu_=_external_current_forEpopu)

        #  for convenience of calculation of Jacobian matrix
        eff_input_mean_for_Epopu = eff_input_mean_for_Epopu.reshape(-1)
        eff_input_std_for_Epopu = eff_input_std_for_Epopu.reshape(-1)
        tau_eff_for_Epopu = tau_eff_for_Epopu.reshape(-1)
        eff_input_mean_for_Ipopu = eff_input_mean_for_Ipopu.reshape(-1)
        eff_input_std_for_Ipopu = eff_input_std_for_Ipopu.reshape(-1)
        tau_eff_for_Ipopu = tau_eff_for_Ipopu.reshape(-1)
        return eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu



    def synaptic_current_stat(self, ue, se, ui, si):
        ue = ue.reshape((1, -1))
        se = se.reshape((1, -1))
        ui = ui.reshape((1, -1))
        si = si.reshape((1, -1))

        exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu, \
        exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu = self.synaptic_summation(ue, se, ui, si)

        exc_channel_current_mean_for_Epopu, exc_channel_current_std_for_Epopu, \
        inh_channel_current_mean_for_Epopu, inh_channel_current_std_for_Epopu = \
            self.exc_activation.synaptic_current_stat_nohp(exc_input_mean=exc_input_mean_for_Epopu, exc_input_std=exc_input_std_for_Epopu,
                                                           inh_input_mean=inh_input_mean_for_Epopu, inh_input_std=inh_input_std_for_Epopu)

        exc_channel_current_mean_for_Ipopu, exc_channel_current_std_for_Ipopu, \
        inh_channel_current_mean_for_Ipopu, inh_channel_current_std_for_Ipopu = \
            self.inh_activation.synaptic_current_stat_nohp(exc_input_mean=exc_input_mean_for_Ipopu, exc_input_std=exc_input_std_for_Ipopu,
                                                           inh_input_mean=inh_input_mean_for_Ipopu, inh_input_std=inh_input_std_for_Ipopu)

        return exc_channel_current_mean_for_Epopu, exc_channel_current_std_for_Epopu, \
               inh_channel_current_mean_for_Epopu, inh_channel_current_std_for_Epopu, \
               exc_channel_current_mean_for_Ipopu, exc_channel_current_std_for_Ipopu, \
               inh_channel_current_mean_for_Ipopu, inh_channel_current_std_for_Ipopu


    def backward_for_MA(self, eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu):
        grad_uu_for_Epopu, grad_us_for_Epopu, grad_utau_for_Epopu, grad_su_for_Epopu, grad_ss_for_Epopu, grad_stau_for_Epopu = self.exc_activation.backward_for_MA(eff_input_mean=eff_input_mean_for_Epopu, eff_input_std=eff_input_std_for_Epopu, tau_eff=tau_eff_for_Epopu, is_cuda=self.is_cuda)
        grad_uu_for_Ipopu, grad_us_for_Ipopu, grad_utau_for_Ipopu, grad_su_for_Ipopu, grad_ss_for_Ipopu, grad_stau_for_Ipopu = self.inh_activation.backward_for_MA(eff_input_mean=eff_input_mean_for_Ipopu, eff_input_std=eff_input_std_for_Ipopu, tau_eff=tau_eff_for_Ipopu, is_cuda=self.is_cuda)

        assert len(eff_input_mean_for_Epopu.shape) == 1
        grad_uu_for_Epopu = np.diag(grad_uu_for_Epopu)
        grad_us_for_Epopu = np.diag(grad_us_for_Epopu)
        grad_utau_for_Epopu = np.diag(grad_utau_for_Epopu)
        grad_su_for_Epopu = np.diag(grad_su_for_Epopu)
        grad_ss_for_Epopu = np.diag(grad_ss_for_Epopu)
        grad_stau_for_Epopu = np.diag(grad_stau_for_Epopu)
        grad_uu_for_Ipopu = np.diag(grad_uu_for_Ipopu)
        grad_us_for_Ipopu = np.diag(grad_us_for_Ipopu)
        grad_utau_for_Ipopu = np.diag(grad_utau_for_Ipopu)
        grad_su_for_Ipopu = np.diag(grad_su_for_Ipopu)
        grad_ss_for_Ipopu = np.diag(grad_ss_for_Ipopu)
        grad_stau_for_Ipopu = np.diag(grad_stau_for_Ipopu)

        _np_zeros = np.zeros_like(grad_uu_for_Epopu)

        _jacobian_for_MA_1row = np.concatenate((grad_uu_for_Epopu, grad_us_for_Epopu, grad_utau_for_Epopu, _np_zeros, _np_zeros, _np_zeros), axis=1)
        _jacobian_for_MA_2row = np.concatenate((grad_su_for_Epopu, grad_ss_for_Epopu, grad_stau_for_Epopu, _np_zeros, _np_zeros, _np_zeros), axis=1)
        _jacobian_for_MA_3row = np.concatenate((_np_zeros, _np_zeros, _np_zeros, grad_uu_for_Ipopu, grad_us_for_Ipopu, grad_utau_for_Ipopu), axis=1)
        _jacobian_for_MA_4row = np.concatenate((_np_zeros, _np_zeros, _np_zeros, grad_su_for_Ipopu, grad_ss_for_Ipopu, grad_stau_for_Ipopu), axis=1)

        _jacobian_for_MA = np.concatenate((_jacobian_for_MA_1row, _jacobian_for_MA_2row, _jacobian_for_MA_3row, _jacobian_for_MA_4row), axis=0)
        return _jacobian_for_MA

    def backward_for_onestep_forexternalcurrent_fixedFanoFactor(self, ue, ui, _external_current_forEpopu, fixedFanoFactor=1):
        assert _external_current_forEpopu.shape == (self.N, )
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func = lambda _external_current_forEpopu: self.synaptic_summation_and_effective_current(ue, torch.sqrt(ue*fixedFanoFactor), ui, torch.sqrt(ui*fixedFanoFactor), _external_current_forEpopu=_external_current_forEpopu.reshape(1, self.N))
        tuple_tuple_jacobian_summation_and_effective_current = torch.autograd.functional.jacobian(lambda_func, (_external_current_forEpopu, ))

        list_jacobian_summation_and_effective_current = \
            [torch.cat(tuple_tuple_jacobian_summation_and_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_summation_and_effective_current))]
        _jacobian_summation_and_effective_current = torch.cat(list_jacobian_summation_and_effective_current, dim=0)
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.synaptic_summation_and_effective_current(ue, torch.sqrt(ue*fixedFanoFactor), ui, torch.sqrt(ui*fixedFanoFactor), _external_current_forEpopu=_external_current_forEpopu.reshape(1, self.N))
        _jacobian_summation_and_effective_current = torch_2_numpy(_jacobian_summation_and_effective_current, is_cuda=self.is_cuda)

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                               eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                               tau_eff_for_Epopu=tau_eff_for_Epopu,
                                               eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                               eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                               tau_eff_for_Ipopu=tau_eff_for_Ipopu)

        outputmean_idx = np.concatenate((np.arange(self.N), np.arange(self.N*2, self.N*3)))
        _jacobian_for_MA_onlyoutputmean = _jacobian_for_MA[outputmean_idx, :]
        _jacobian_onestep_forexternalcurrent_onlyoutputmean = _jacobian_for_MA_onlyoutputmean @ _jacobian_summation_and_effective_current
        return _jacobian_onestep_forexternalcurrent_onlyoutputmean


    def backward_for_onestep_forexternalcurrent(self, ue, se, ui, si, _external_current_forEpopu):
        assert _external_current_forEpopu.shape == (self.N, )
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func = lambda _external_current_forEpopu: self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu.reshape(1, self.N))
        tuple_tuple_jacobian_summation_and_effective_current = torch.autograd.functional.jacobian(lambda_func, (_external_current_forEpopu, ))

        # print(type(tuple_tuple_jacobian_summation_and_effective_current), len(tuple_tuple_jacobian_summation_and_effective_current))
        # print(type(tuple_tuple_jacobian_summation_and_effective_current[0]), len(tuple_tuple_jacobian_summation_and_effective_current[0]))
        # print(type(tuple_tuple_jacobian_summation_and_effective_current[0][0]), tuple_tuple_jacobian_summation_and_effective_current[0][0].shape)

        list_jacobian_summation_and_effective_current = \
            [torch.cat(tuple_tuple_jacobian_summation_and_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_summation_and_effective_current))]
        _jacobian_summation_and_effective_current = torch.cat(list_jacobian_summation_and_effective_current, dim=0)
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu.reshape(1, self.N))
        _jacobian_summation_and_effective_current = torch_2_numpy(_jacobian_summation_and_effective_current, is_cuda=self.is_cuda)

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                               eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                               tau_eff_for_Epopu=tau_eff_for_Epopu,
                                               eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                               eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                               tau_eff_for_Ipopu=tau_eff_for_Ipopu)
        # print("np.any(np.isnan(_jacobian_for_MA))", np.any(np.isnan(_jacobian_for_MA)))
        # print("_jacobian_for_MA", _jacobian_for_MA.shape)
        _jacobian_onestep_forexternalcurrent = _jacobian_for_MA @ _jacobian_summation_and_effective_current
        return _jacobian_onestep_forexternalcurrent

    def backward_for_onestep_forexternalcurrent_mapping_variance(self, ue, se, ui, si, _external_current_forEpopu):
        assert _external_current_forEpopu.shape == (self.N, )
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        _jacobian_onestep_forexternalcurrent = self.backward_for_onestep_forexternalcurrent(ue, se, ui, si, _external_current_forEpopu)
        _new_ue, _new_se, _new_ui, _new_si = self.forward(ue.reshape(1, -1),
                                                          se.reshape(1, -1),
                                                          ui.reshape(1, -1),
                                                          si.reshape(1, -1),
                                                          external_current_forEpopu=_external_current_forEpopu.reshape(1, self.N))
        _new_ue = _new_ue.reshape(-1)
        _new_se = _new_se.reshape(-1)
        _new_ui = _new_ui.reshape(-1)
        _new_si = _new_si.reshape(-1)

        lambda_func_reconvert_sigma_to_variance = lambda ue, se, ui, si: (ue, se**2, ui, si**2)
        tuple_tuple_jacobian_reconvert_sigma_to_variance = torch.autograd.functional.jacobian(lambda_func_reconvert_sigma_to_variance, (_new_ue, _new_se, _new_ui, _new_si))
        list_jacobian_reconvert_sigma_to_variance = \
            [torch.cat(tuple_tuple_jacobian_reconvert_sigma_to_variance[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_reconvert_sigma_to_variance))]
        _jacobian_reconvert_sigma_to_variance = torch.cat(list_jacobian_reconvert_sigma_to_variance, dim=0)
        _jacobian_reconvert_sigma_to_variance = torch_2_numpy(_jacobian_reconvert_sigma_to_variance, is_cuda=self.is_cuda)

        _jacobian_onestep_forexternalcurrent_mapping_variance = _jacobian_reconvert_sigma_to_variance @ _jacobian_onestep_forexternalcurrent

        return _jacobian_onestep_forexternalcurrent_mapping_variance

    def backward_for_MA_and_summation_and_effective_current(self, ue, se, ui, si, _external_current_forEpopu=None):
        if _external_current_forEpopu is None:
            _external_current_forEpopu = torch.zeros(1, self.N).type_as(self.K_EE)
        assert _external_current_forEpopu.shape == (1, self.N)
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func = lambda ue, se, ui, si: self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu)
        tuple_tuple_jacobian_summation_and_effective_current = torch.autograd.functional.jacobian(lambda_func, (ue, se, ui, si))
        list_jacobian_summation_and_effective_current = \
            [torch.cat(tuple_tuple_jacobian_summation_and_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_summation_and_effective_current))]
        _jacobian_summation_and_effective_current = torch.cat(list_jacobian_summation_and_effective_current, dim=0)
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu)
        _jacobian_summation_and_effective_current = torch_2_numpy(_jacobian_summation_and_effective_current, is_cuda=self.is_cuda)

        # print("np.any(np.isnan(_jacobian_summation_and_effective_current))", np.any(np.isnan(_jacobian_summation_and_effective_current)))

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                               eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                               tau_eff_for_Epopu=tau_eff_for_Epopu,
                                               eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                               eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                               tau_eff_for_Ipopu=tau_eff_for_Ipopu)
        return _jacobian_for_MA, _jacobian_summation_and_effective_current

    def backward_for_onestep_fixedFanoFactor(self, ue, ui, _external_current_forEpopu=None, fixedFanoFactor=1):
        if _external_current_forEpopu is None:
            _external_current_forEpopu = torch.zeros(1, self.N).type_as(self.K_EE)
        assert _external_current_forEpopu.shape == (1, self.N)
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        assert fixedFanoFactor >= 0

        if fixedFanoFactor == 0:
            lambda_func = lambda ue, ui: self.synaptic_summation_and_effective_current(ue, torch.zeros_like(ue).type_as(self.K_EE), ui, torch.zeros_like(ui).type_as(self.K_EE), _external_current_forEpopu=_external_current_forEpopu)
        else:
            lambda_func = lambda ue, ui: self.synaptic_summation_and_effective_current(ue, torch.sqrt(ue*fixedFanoFactor), ui, torch.sqrt(ui*fixedFanoFactor), _external_current_forEpopu=_external_current_forEpopu)
        # lambda_func = lambda ue, ui: self.synaptic_summation_and_effective_current(ue, torch.sqrt(ue*fixedFanoFactor), ui, torch.sqrt(ui*fixedFanoFactor), _external_current_forEpopu=_external_current_forEpopu)
        tuple_tuple_jacobian_summation_and_effective_current = torch.autograd.functional.jacobian(lambda_func, (ue, ui))
        list_jacobian_summation_and_effective_current = \
            [torch.cat(tuple_tuple_jacobian_summation_and_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_summation_and_effective_current))]
        _jacobian_summation_and_effective_current = torch.cat(list_jacobian_summation_and_effective_current, dim=0)
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.synaptic_summation_and_effective_current(ue, torch.sqrt(ue*fixedFanoFactor), ui, torch.sqrt(ui*fixedFanoFactor), _external_current_forEpopu=_external_current_forEpopu)
        _jacobian_summation_and_effective_current = torch_2_numpy(_jacobian_summation_and_effective_current, is_cuda=self.is_cuda)

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                               eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                               tau_eff_for_Epopu=tau_eff_for_Epopu,
                                               eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                               eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                               tau_eff_for_Ipopu=tau_eff_for_Ipopu)
        outputmean_idx = np.concatenate((np.arange(self.N), np.arange(self.N*2, self.N*3)))
        _jacobian_for_MA_onlyoutputmean = _jacobian_for_MA[outputmean_idx, :]
        _jacobian_onestep_fixedFanoFactor = _jacobian_for_MA_onlyoutputmean @ _jacobian_summation_and_effective_current
        # print("np.any(np.isnan(_jacobian_for_MA_onlyoutputmean))", np.any(np.isnan(_jacobian_for_MA_onlyoutputmean)))
        # print("np.any(np.isnan(_jacobian_summation_and_effective_current))", np.any(np.isnan(_jacobian_summation_and_effective_current)))

        return _jacobian_onestep_fixedFanoFactor


    def backward_for_onestep(self, ue, se, ui, si, _external_current_forEpopu=None):
        if _external_current_forEpopu is None:
            _external_current_forEpopu = torch.zeros(1, self.N).type_as(self.K_EE)
        assert _external_current_forEpopu.shape == (1, self.N)
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func = lambda ue, se, ui, si: self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu)
        tuple_tuple_jacobian_summation_and_effective_current = torch.autograd.functional.jacobian(lambda_func, (ue, se, ui, si))
        list_jacobian_summation_and_effective_current = \
            [torch.cat(tuple_tuple_jacobian_summation_and_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_summation_and_effective_current))]
        _jacobian_summation_and_effective_current = torch.cat(list_jacobian_summation_and_effective_current, dim=0)
        eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu, \
        eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu = self.synaptic_summation_and_effective_current(ue, se, ui, si, _external_current_forEpopu=_external_current_forEpopu)
        _jacobian_summation_and_effective_current = torch_2_numpy(_jacobian_summation_and_effective_current, is_cuda=self.is_cuda)

        # print("np.any(np.isnan(_jacobian_summation_and_effective_current))", np.any(np.isnan(_jacobian_summation_and_effective_current)))

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                               eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                               tau_eff_for_Epopu=tau_eff_for_Epopu,
                                               eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                               eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                               tau_eff_for_Ipopu=tau_eff_for_Ipopu)
        # print("np.any(np.isnan(_jacobian_for_MA))", np.any(np.isnan(_jacobian_for_MA)))
        # print("_jacobian_for_MA", _jacobian_for_MA.shape)
        _jacobian_onestep = _jacobian_for_MA @ _jacobian_summation_and_effective_current
        return _jacobian_onestep

    def backward_for_onestep_mapping_variance(self, ue, var_e, ui, var_i, _external_current_forEpopu=None):
        if _external_current_forEpopu is None:
            _external_current_forEpopu = torch.zeros(1, self.N).type_as(self.K_EE)
        assert _external_current_forEpopu.shape == (1, self.N)
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func_convert_variance_to_sigma = lambda ue, var_e, ui, var_i: (ue, torch.sqrt(var_e), ui, torch.sqrt(var_i))
        tuple_tuple_jacobian_convert_variance_to_sigma = torch.autograd.functional.jacobian(lambda_func_convert_variance_to_sigma, (ue, var_e, ui, var_i))
        list_jacobian_convert_variance_to_sigma = \
            [torch.cat(tuple_tuple_jacobian_convert_variance_to_sigma[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_convert_variance_to_sigma))]
        _jacobian_convert_variance_to_sigma = torch.cat(list_jacobian_convert_variance_to_sigma, dim=0)
        _jacobian_convert_variance_to_sigma = torch_2_numpy(_jacobian_convert_variance_to_sigma, is_cuda=self.is_cuda)

        ue_convert, se_convert, ui_convert, si_convert = lambda_func_convert_variance_to_sigma(ue, var_e, ui, var_i)
        _jacobian_onestep = self.backward_for_onestep(ue_convert.reshape(-1), se_convert.reshape(-1),
                                                      ui_convert.reshape(-1), si_convert.reshape(-1),
                                                      _external_current_forEpopu=_external_current_forEpopu)

        _new_ue, _new_se, _new_ui, _new_si = self.forward(ue_convert.reshape(1, -1),
                                                          se_convert.reshape(1, -1),
                                                          ui_convert.reshape(1, -1),
                                                          si_convert.reshape(1, -1),
                                                          external_current_forEpopu=_external_current_forEpopu)
        _new_ue = _new_ue.reshape(-1)
        _new_se = _new_se.reshape(-1)
        _new_ui = _new_ui.reshape(-1)
        _new_si = _new_si.reshape(-1)

        lambda_func_reconvert_sigma_to_variance = lambda ue, se, ui, si: (ue, se**2, ui, si**2)
        tuple_tuple_jacobian_reconvert_sigma_to_variance = torch.autograd.functional.jacobian(lambda_func_reconvert_sigma_to_variance, (_new_ue, _new_se, _new_ui, _new_si))
        list_jacobian_reconvert_sigma_to_variance = \
            [torch.cat(tuple_tuple_jacobian_reconvert_sigma_to_variance[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_reconvert_sigma_to_variance))]
        _jacobian_reconvert_sigma_to_variance = torch.cat(list_jacobian_reconvert_sigma_to_variance, dim=0)
        _jacobian_reconvert_sigma_to_variance = torch_2_numpy(_jacobian_reconvert_sigma_to_variance, is_cuda=self.is_cuda)
        _jacobian_onestep_mapping_variance = _jacobian_reconvert_sigma_to_variance @ _jacobian_onestep @ _jacobian_convert_variance_to_sigma

        return _jacobian_onestep_mapping_variance

    def backward_for_MA_and_summation_and_effective_current_return3_mappingvariance(self, ue, var_e, ui, var_i,
                                                                                    _external_current_forEpopu=None):
        if _external_current_forEpopu is None:
            _external_current_forEpopu = torch.zeros(1, self.N).type_as(self.K_EE)
        assert _external_current_forEpopu.shape == (1, self.N)
        assert isinstance(_external_current_forEpopu, torch.Tensor)
        assert _external_current_forEpopu.device.type == self.device

        lambda_func_convert_variance_to_sigma = lambda ue, var_e, ui, var_i: (ue, torch.sqrt(var_e), ui, torch.sqrt(var_i))
        tuple_tuple_jacobian_convert_variance_to_sigma = torch.autograd.functional.jacobian(lambda_func_convert_variance_to_sigma, (ue, var_e, ui, var_i))
        list_jacobian_convert_variance_to_sigma = \
            [torch.cat(tuple_tuple_jacobian_convert_variance_to_sigma[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_convert_variance_to_sigma))]
        _jacobian_convert_variance_to_sigma = torch.cat(list_jacobian_convert_variance_to_sigma, dim=0)
        _jacobian_convert_variance_to_sigma = torch_2_numpy(_jacobian_convert_variance_to_sigma, is_cuda=self.is_cuda)
        ue_convert, se_convert, ui_convert, si_convert = lambda_func_convert_variance_to_sigma(ue, var_e, ui, var_i)

        lambda_func_synaptic_summation = lambda ue, se, ui, si: self.synaptic_summation_for_convenience_of_calculation_of_Jacobian(ue, se, ui, si)
        tuple_tuple_jacobian_synaptic_summation = torch.autograd.functional.jacobian(lambda_func_synaptic_summation, (ue_convert, se_convert, ui_convert, si_convert))
        list_jacobian_synaptic_summation = \
            [torch.cat(tuple_tuple_jacobian_synaptic_summation[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_synaptic_summation))]
        _jacobian_synaptic_summation = torch.cat(list_jacobian_synaptic_summation, dim=0)
        _jacobian_synaptic_summation = torch_2_numpy(_jacobian_synaptic_summation, is_cuda=self.is_cuda)

        (exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
         exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu) = (
            self.synaptic_summation_for_convenience_of_calculation_of_Jacobian(ue_convert, se_convert, ui_convert, si_convert))

        lambda_func_effective_current = lambda _exc_input_mean_for_Epopu, _exc_input_std_for_Epopu, _inh_input_mean_for_Epopu, _inh_input_std_for_Epopu, _exc_input_mean_for_Ipopu, _exc_input_std_for_Ipopu, _inh_input_mean_for_Ipopu, _inh_input_std_for_Ipopu: self.effective_current_for_convenience_of_calculation_of_Jacobian(
            _exc_input_mean_for_Epopu, _exc_input_std_for_Epopu, _inh_input_mean_for_Epopu, _inh_input_std_for_Epopu,
            _exc_input_mean_for_Ipopu, _exc_input_std_for_Ipopu, _inh_input_mean_for_Ipopu, _inh_input_std_for_Ipopu, _external_current_forEpopu=_external_current_forEpopu)
        tuple_tuple_jacobian_effective_current = torch.autograd.functional.jacobian(lambda_func_effective_current,
                                                                                    (exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                                                                                     exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu))
        list_jacobian_effective_current = \
            [torch.cat(tuple_tuple_jacobian_effective_current[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_effective_current))]
        _jacobian_effective_current = torch.cat(list_jacobian_effective_current, dim=0)
        _jacobian_effective_current = torch_2_numpy(_jacobian_effective_current, is_cuda=self.is_cuda)

        (eff_input_mean_for_Epopu, eff_input_std_for_Epopu, tau_eff_for_Epopu,
         eff_input_mean_for_Ipopu, eff_input_std_for_Ipopu, tau_eff_for_Ipopu) = (
            self.effective_current_for_convenience_of_calculation_of_Jacobian(exc_input_mean_for_Epopu, exc_input_std_for_Epopu, inh_input_mean_for_Epopu, inh_input_std_for_Epopu,
                                                                              exc_input_mean_for_Ipopu, exc_input_std_for_Ipopu, inh_input_mean_for_Ipopu, inh_input_std_for_Ipopu,
                                                                              _external_current_forEpopu=_external_current_forEpopu
                                                                              ))

        _jacobian_for_MA = self.backward_for_MA(eff_input_mean_for_Epopu=eff_input_mean_for_Epopu,
                                                eff_input_std_for_Epopu=eff_input_std_for_Epopu,
                                                tau_eff_for_Epopu=tau_eff_for_Epopu,
                                                eff_input_mean_for_Ipopu=eff_input_mean_for_Ipopu,
                                                eff_input_std_for_Ipopu=eff_input_std_for_Ipopu,
                                                tau_eff_for_Ipopu=tau_eff_for_Ipopu)

        _new_ue, _new_se, _new_ui, _new_si = self.forward(ue_convert.reshape(1, -1),
                                                          se_convert.reshape(1, -1),
                                                          ui_convert.reshape(1, -1),
                                                          si_convert.reshape(1, -1),
                                                          external_current_forEpopu=_external_current_forEpopu)
        _new_ue = _new_ue.reshape(-1)
        _new_se = _new_se.reshape(-1)
        _new_ui = _new_ui.reshape(-1)
        _new_si = _new_si.reshape(-1)
        lambda_func_reconvert_sigma_to_variance = lambda ue, se, ui, si: (ue, se**2, ui, si**2)
        tuple_tuple_jacobian_reconvert_sigma_to_variance = torch.autograd.functional.jacobian(lambda_func_reconvert_sigma_to_variance, (_new_ue, _new_se, _new_ui, _new_si))
        list_jacobian_reconvert_sigma_to_variance = \
            [torch.cat(tuple_tuple_jacobian_reconvert_sigma_to_variance[i], dim=1)
             for i in range(len(tuple_tuple_jacobian_reconvert_sigma_to_variance))]
        _jacobian_reconvert_sigma_to_variance = torch.cat(list_jacobian_reconvert_sigma_to_variance, dim=0)
        _jacobian_reconvert_sigma_to_variance = torch_2_numpy(_jacobian_reconvert_sigma_to_variance, is_cuda=self.is_cuda)

        return _jacobian_reconvert_sigma_to_variance @ _jacobian_for_MA, _jacobian_effective_current, _jacobian_synaptic_summation @ _jacobian_convert_variance_to_sigma

    def forward_numpy(self, ue, se, ui, si, external_current_forEpopu):
        assert isinstance(ue, np.ndarray)
        assert isinstance(external_current_forEpopu, np.ndarray)
        assert ue.shape == (1, self.N)
        assert se.shape == (1, self.N)
        assert ui.shape == (1, self.N)
        assert si.shape == (1, self.N)
        assert external_current_forEpopu.shape == (1, self.N)

        exc_input_mean = (np.multiply(ue, self.K_EE_numpy.T) * self.local_w_EE_numpy + np.dot(ue,
                                                                                  self.K_EE_long_numpy.T) * self.xregion_gain_for_E_popu_numpy) * self.conduct_scaling_factor_numpy.T
        inh_input_mean = np.multiply(ui, self.K_EI_numpy.T) * self.local_w_EI_numpy * self.conduct_scaling_factor_numpy.T
        exc_input_std = np.multiply(se * se,
                                  self.K_EE_numpy.T) * self.local_w_EE_numpy * self.local_w_EE_numpy * self.conduct_scaling_factor_numpy.T * self.conduct_scaling_factor_numpy.T \
                        + np.dot(se * se,
                                   self.K_EE_long_numpy.T) * self.xregion_gain_for_E_popu_numpy * self.xregion_gain_for_E_popu_numpy * self.conduct_scaling_factor_numpy.T * self.conduct_scaling_factor_numpy.T
        exc_input_std = np.sqrt(exc_input_std)
        inh_input_std = np.sqrt(np.multiply(si * si, self.K_EI_numpy.T)) * self.local_w_EI_numpy * self.conduct_scaling_factor_numpy.T

        # excitatory population
        # convert conductance input to current
        # !!!! The eff_input_mean, eff_input_std, tau_eff already takes the background OU current and hyperparameter_current into consideration
        eff_input_mean, eff_input_std, tau_eff = self.exc_activation.cond2curr_includingbgOU_withhp_numpy(
            exc_input_mean=exc_input_mean, exc_input_std=exc_input_std, inh_input_mean=inh_input_mean,
            inh_input_std=inh_input_std, external_current=external_current_forEpopu)

        # calculate moment activation
        exc_mean_out, exc_std_out = self.exc_activation.activate_numpy(eff_input_mean, eff_input_std, tau_eff,)

        # inhibitory population
        exc_input_mean = np.multiply(ue, self.K_IE_numpy.T) * self.local_w_IE_numpy * self.conduct_scaling_factor_numpy.T
        inh_input_mean = np.multiply(ui, self.K_II_numpy.T) * self.local_w_II_numpy * self.conduct_scaling_factor_numpy.T
        exc_input_std = np.sqrt(np.multiply(se * se, self.K_IE_numpy.T)) * self.local_w_IE_numpy * self.conduct_scaling_factor_numpy.T
        inh_input_std = np.sqrt(np.multiply(si * si, self.K_II_numpy.T)) * self.local_w_II_numpy * self.conduct_scaling_factor_numpy.T

        # !!!! The eff_input_mean, eff_input_std, tau_eff already takes the background OU current and hyperparameter_current into consideration
        eff_input_mean, eff_input_std, tau_eff = self.inh_activation.cond2curr_includingbgOU_withhp_numpy(exc_input_mean,
                                                                                                    exc_input_std,
                                                                                                    inh_input_mean,
                                                                                                    inh_input_std,
                                                                                                    external_current=np.zeros_like(
                                                                                                        ue).astype(np.float32))

        inh_mean_out, inh_std_out = self.inh_activation.activate_numpy(eff_input_mean, eff_input_std, tau_eff)

        return exc_mean_out, exc_std_out, inh_mean_out, inh_std_out


    def forward(self, ue, se, ui, si, external_current_forEpopu):
        '''
        ue, se: mean/std of excitatory neurons  dims: batchsize x num of neurons
        ui, si: mean/std of inhibitory neurons
        '''
        assert ue.shape == (1, self.N)
        assert se.shape == (1, self.N)
        assert ui.shape == (1, self.N)
        assert si.shape == (1, self.N)
        assert external_current_forEpopu.shape == (1, self.N)

        exc_input_mean = (torch.mul(ue, self.K_EE.T) * self.local_w_EE + torch.mm(ue, self.K_EE_long.T) * self.xregion_gain_for_E_popu) * self.conduct_scaling_factor.T
        inh_input_mean = torch.mul(ui, self.K_EI.T) * self.local_w_EI * self.conduct_scaling_factor.T
        exc_input_std = torch.mul(se*se, self.K_EE.T) * self.local_w_EE * self.local_w_EE * self.conduct_scaling_factor.T * self.conduct_scaling_factor.T\
                        + torch.mm(se*se, self.K_EE_long.T) * self.xregion_gain_for_E_popu * self.xregion_gain_for_E_popu\
                        * self.conduct_scaling_factor.T * self.conduct_scaling_factor.T
        exc_input_std = exc_input_std.pow(0.5)

        inh_input_std = torch.mul(si*si, self.K_EI.T).pow(0.5) * self.local_w_EI * self.conduct_scaling_factor.T
        
        # excitatory population
        # convert conductance input to current
        # !!!! The eff_input_mean, eff_input_std, tau_eff already takes the background OU current and hyperparameter_current into consideration
        eff_input_mean, eff_input_std, tau_eff = self.exc_activation.cond2curr_includingbgOU_withhp(exc_input_mean=exc_input_mean, exc_input_std=exc_input_std, inh_input_mean=inh_input_mean, inh_input_std=inh_input_std, external_current=external_current_forEpopu)

        # calculate moment activation
        exc_mean_out, exc_std_out = self.exc_activation.activate(eff_input_mean, eff_input_std, tau_eff, is_cuda=self.is_cuda)
        
        # inhibitory population
        exc_input_mean = torch.mul(ue, self.K_IE.T) * self.local_w_IE * self.conduct_scaling_factor.T
        inh_input_mean = torch.mul(ui, self.K_II.T) * self.local_w_II * self.conduct_scaling_factor.T
        exc_input_std = torch.mul(se*se, self.K_IE.T).pow(0.5) * self.local_w_IE * self.conduct_scaling_factor.T
        inh_input_std = torch.mul(si*si, self.K_II.T).pow(0.5) * self.local_w_II * self.conduct_scaling_factor.T
        
        # !!!! The eff_input_mean, eff_input_std, tau_eff already takes the background OU current and hyperparameter_current into consideration
        eff_input_mean, eff_input_std, tau_eff = self.inh_activation.cond2curr_includingbgOU_withhp(exc_input_mean, exc_input_std, inh_input_mean, inh_input_std, external_current=torch.zeros_like(ue).type_as(ue))
        
        inh_mean_out, inh_std_out = self.inh_activation.activate(eff_input_mean, eff_input_std, tau_eff, is_cuda=self.is_cuda)
        
        return exc_mean_out, exc_std_out, inh_mean_out, inh_std_out

    def newton_fixedpoint_iter_onestep_fixedFanoFactor(self, init_point, external_current_forEpopu_=None, stepsize=1, fixedFanoFactor=1):
        assert init_point.shape == (self.N*2, 1)
        assert isinstance(init_point, np.ndarray)
        if external_current_forEpopu_ is None:
            external_current_forEpopu_ = torch.zeros(1, self.N).type_as(self.K_EE)
        assert external_current_forEpopu_.shape == (1, self.N)
        assert isinstance(external_current_forEpopu_, torch.Tensor)
        assert external_current_forEpopu_.device.type == self.device

        exc_mean_out, _, inh_mean_out, _ = \
            self.forward(numpy2torch(init_point[:self.N, :], is_cuda=self.is_cuda).reshape(1, -1),
                         torch.sqrt(numpy2torch(init_point[:self.N, :], is_cuda=self.is_cuda).reshape(1, -1)*fixedFanoFactor),
                         numpy2torch(init_point[self.N:self.N * 2, :], is_cuda=self.is_cuda).reshape(1, -1),
                         torch.sqrt(numpy2torch(init_point[self.N:self.N * 2, :], is_cuda=self.is_cuda).reshape(1, -1)*fixedFanoFactor),
                         external_current_forEpopu=external_current_forEpopu_)
        fun_eval = np.concatenate((exc_mean_out, inh_mean_out), axis=1)
        assert fun_eval.shape == (1, self.N*2)
        fun_eval = fun_eval.reshape((self.N*2, 1))
        assert fun_eval.shape == (self.N*2, 1)
        fun_eval = - init_point + fun_eval
        assert fun_eval.shape == (self.N*2, 1)
        # print("np.any(np.isnan(fun_eval))", np.any(np.isnan(fun_eval)))
        _jacobian_onestep = self.backward_for_onestep_fixedFanoFactor(numpy2torch(init_point[:self.N, :], is_cuda=self.is_cuda).reshape(-1),
                                                      numpy2torch(init_point[self.N:self.N * 2, :], is_cuda=self.is_cuda).reshape(-1),
                                                      _external_current_forEpopu=external_current_forEpopu_, fixedFanoFactor=fixedFanoFactor)
        assert _jacobian_onestep.shape == (self.N*2, self.N*2)
        # print("np.any(np.isnan(_jacobian_onestep))", np.any(np.isnan(_jacobian_onestep)))
        _jacobian_onestep = _jacobian_onestep - np.eye(self.N * 2)
        assert _jacobian_onestep.shape == (self.N*2, self.N*2)

        newton_direction = np.linalg.solve(_jacobian_onestep, fun_eval)
        nextpoint = init_point - newton_direction * stepsize
        return nextpoint
        
    def newton_fixedpoint_iter_onestep(self, init_point, external_current_forEpopu_=None, stepsize=1):
        assert init_point.shape == (self.N*4, 1)
        assert isinstance(init_point, np.ndarray)
        if external_current_forEpopu_ is None:
            external_current_forEpopu_ = torch.zeros(1, self.N).type_as(self.K_EE)
        assert external_current_forEpopu_.shape == (1, self.N)
        assert isinstance(external_current_forEpopu_, torch.Tensor)
        assert external_current_forEpopu_.device.type == self.device

        exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = \
            self.forward(numpy2torch(init_point[:self.N, :], is_cuda=self.is_cuda).reshape(1, -1),
                         numpy2torch(init_point[self.N:self.N * 2, :], is_cuda=self.is_cuda).reshape(1, -1),
                         numpy2torch(init_point[self.N * 2:self.N * 3, :], is_cuda=self.is_cuda).reshape(1, -1),
                         numpy2torch(init_point[self.N * 3:self.N * 4, :], is_cuda=self.is_cuda).reshape(1, -1),
                         external_current_forEpopu=external_current_forEpopu_)
        fun_eval = np.concatenate((exc_mean_out, exc_std_out, inh_mean_out, inh_std_out), axis=1)
        assert fun_eval.shape == (1, self.N*4)
        fun_eval = fun_eval.reshape((self.N*4, 1))
        assert fun_eval.shape == (self.N*4, 1)
        fun_eval = - init_point + fun_eval
        assert fun_eval.shape == (self.N*4, 1)
        # print("np.any(np.isnan(fun_eval))", np.any(np.isnan(fun_eval)))
        _jacobian_onestep = self.backward_for_onestep(numpy2torch(init_point[:self.N, :], is_cuda=self.is_cuda).reshape(-1),
                                                      numpy2torch(init_point[self.N:self.N * 2, :], is_cuda=self.is_cuda).reshape(-1),
                                                      numpy2torch(init_point[self.N * 2:self.N * 3, :], is_cuda=self.is_cuda).reshape(-1),
                                                      numpy2torch(init_point[self.N * 3:self.N * 4, :], is_cuda=self.is_cuda).reshape(-1),
                                                      _external_current_forEpopu=external_current_forEpopu_)
        assert _jacobian_onestep.shape == (self.N*4, self.N*4)
        # print("np.any(np.isnan(_jacobian_onestep))", np.any(np.isnan(_jacobian_onestep)))
        _jacobian_onestep = _jacobian_onestep - np.eye(self.N * 4)
        assert _jacobian_onestep.shape == (self.N*4, self.N*4)

        # _jacobian_onestep = np.eye(self.N * 4)
        newton_direction = np.linalg.solve(_jacobian_onestep, fun_eval)
        # print("np.any(np.isnan(newton_direction))", np.any(np.isnan(newton_direction)))
        nextpoint = init_point - newton_direction * stepsize
        return nextpoint

    def run_newton_fixedpoint_iter_fixedFanoFactor(self, init_point, nsteps, _external_current_forEpopu, stepsize=1, fixedFanoFactor=1):
        assert init_point.shape == (self.N*2, 1)
        current_point = init_point
        point_record = np.zeros((nsteps+1, self.N*2)) #
        point_record[0, :] = current_point.reshape(-1)

        for i in range(1, nsteps+1):
            current_point = self.newton_fixedpoint_iter_onestep_fixedFanoFactor(init_point=current_point, external_current_forEpopu_=_external_current_forEpopu, stepsize=stepsize, fixedFanoFactor=fixedFanoFactor)
            current_point[current_point >= 1] = 1
            current_point[current_point <= 0] = 1e-4
            point_record[i, :] = current_point.reshape(-1)
            # if (i) % 10 == 0:
            #     print(f"_idx: {i}")
        return point_record

    def run_newton_fixedpoint_iter(self, init_point, nsteps, _external_current_forEpopu, stepsize=1):
        assert init_point.shape == (self.N*4, 1)
        current_point = init_point
        point_record = np.zeros((nsteps+1, self.N*4)) #
        point_record[0, :] = current_point.reshape(-1)

        # _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        # _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        for i in range(1, nsteps+1):
            current_point = self.newton_fixedpoint_iter_onestep(init_point=current_point, external_current_forEpopu_=_external_current_forEpopu, stepsize=stepsize)
            current_point[current_point >= 1] = 1
            current_point[current_point <= 0] = 1e-4
            point_record[i, :] = current_point.reshape(-1)
            # if (i) % 10 == 0:
            #     print(f"_idx: {i}")
        return point_record

    def ode_system(self, t, y, ):
        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        if t < self.init_time:
            _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))
        else:
            _idx = int((t-self.init_time) / (self.hp_update_time * self.time_unit_converted))
            _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = \
            self.forward(numpy2torch(y[:self.N], is_cuda=self.is_cuda).reshape(1, -1),
            numpy2torch(y[self.N:self.N * 2], is_cuda=self.is_cuda).reshape(1, -1),
            numpy2torch(y[self.N * 2:self.N * 3], is_cuda=self.is_cuda).reshape(1, -1),
            numpy2torch(y[self.N * 3:self.N * 4], is_cuda=self.is_cuda).reshape(1, -1),
                         external_current_forEpopu=_external_current_forEpopu)

        output_y = torch.cat((exc_mean_out, exc_std_out, inh_mean_out, inh_std_out), dim=1).reshape(-1)
        output_y = torch_2_numpy(output_y, is_cuda=self.is_cuda)
        dydt = (-y + output_y) / self.mnn_membrane_constant

        return dydt

    def solveode_useRK(self, init_point=None):
        if init_point is None:
            init_point = np.zeros(self.N * 4)
        assert init_point.shape == (self.N*4,)
        assert isinstance(init_point, np.ndarray)

        t_span = (0, self.init_time + self.T)
        solution = solve_ivp(fun=self.ode_system, t_span=t_span, y0=init_point,
                             t_eval=np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/self.dt_mnn)+1),
                             method='RK45')
        return solution.t, solution.y


    def run_with_noisestimulus_mapping_variance_FixedFanoFactor(self, T=None, init_time=None, init_point=None, stimulus_strength=100, bg_noise_strength=0, stimulus_region=np.array([1, 181]), stimulus_strength_step=1, savebold=True, fixedFanoFactor=0):
        print("run_with_noisestimulus_mapping_variance")
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*2)
            ue = init_point[:, :self.N]
            se = torch.sqrt(ue.reshape(1, -1) * fixedFanoFactor)
            ui = init_point[:, self.N:self.N*2]
            si = torch.sqrt(ui.reshape(1, -1) * fixedFanoFactor)
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = torch.sqrt(ue.reshape(1, -1) * fixedFanoFactor)
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = torch.sqrt(ui.reshape(1, -1) * fixedFanoFactor)

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)
        assert stimulus_strength_step <= init_iter_steps

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        for i in range(init_iter_steps):
            if i < init_iter_steps - stimulus_strength_step:
                exc_mean_out, _, inh_mean_out, _ = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)
            else:
                impulse_stimulus_forEpopu = torch.randn(1, self.N, dtype=torch.float64, device=self.device) * bg_noise_strength
                impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength * torch.randn(stimulus_region.shape[0], dtype=torch.float64, device=self.device)
                exc_mean_out, _, inh_mean_out, _ = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt(ue.reshape(1, -1) * fixedFanoFactor)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt(ui.reshape(1, -1) * fixedFanoFactor)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            impulse_stimulus_forEpopu = torch.randn(1, self.N, dtype=torch.float64, device=self.device) * bg_noise_strength
            # impulse_stimulus_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength * torch.randn(stimulus_region.shape[0], dtype=torch.float64, device=self.device)
            exc_mean_out, _, inh_mean_out, _ = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt(ue.reshape(1, -1) * fixedFanoFactor)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt(ui.reshape(1, -1) * fixedFanoFactor)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record


    def run_with_noisestimulus_mapping_variance(self, T=None, init_time=None, init_point=None, stimulus_strength=100, bg_noise_strength=0, stimulus_region=np.array([1, 181]), stimulus_strength_step=1, savebold=True):
        print("run_with_noisestimulus_mapping_variance")
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = ue.clone()
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = ui.clone()

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)
        assert stimulus_strength_step <= init_iter_steps

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        for i in range(init_iter_steps):
            if i < init_iter_steps - stimulus_strength_step:
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)
            else:
                impulse_stimulus_forEpopu = torch.randn(1, self.N, dtype=torch.float64, device=self.device) * bg_noise_strength
                impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength * torch.randn(stimulus_region.shape[0], dtype=torch.float64, device=self.device)
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            impulse_stimulus_forEpopu = torch.randn(1, self.N, dtype=torch.float64, device=self.device) * bg_noise_strength
            # impulse_stimulus_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength * torch.randn(stimulus_region.shape[0], dtype=torch.float64, device=self.device)
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record


    def run_with_impulse_stimulus_mapping_variance(self, T=None, init_time=None, init_point=None, stimulus_strength=100, stimulus_region=np.array([1, 181]), stimulus_strength_step=1, savebold=True):
        print("run_with_impulse_stimulus_mapping_variance")
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = ue.clone()
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = ui.clone()

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)
        assert stimulus_strength_step <= init_iter_steps

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        impulse_stimulus_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength

        for i in range(init_iter_steps):
            if i < init_iter_steps - stimulus_strength_step:
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)
            else:
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record

    def run_with_impulse_stimulus(self, T=None, init_time=None, init_point=None, stimulus_strength=100, stimulus_region=np.array([1, 181]), stimulus_strength_step=1, savebold=True):
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = ue.clone()
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = ui.clone()

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)
        assert stimulus_strength_step <= init_iter_steps

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        impulse_stimulus_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        impulse_stimulus_forEpopu[:, stimulus_region - 1] = stimulus_strength

        for i in range(init_iter_steps):
            if i < init_iter_steps - stimulus_strength_step:
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)
            else:
                exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu + impulse_stimulus_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se + self.dt_mnn * (-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si + self.dt_mnn * (-si + inh_std_out) / self.mnn_membrane_constant

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue+ self.dt_mnn*(-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se+ self.dt_mnn*(-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui+ self.dt_mnn*(-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si+ self.dt_mnn*(-si + inh_std_out) / self.mnn_membrane_constant

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record


    def run_mapping_var(self, T=None, init_time=None, init_point=None, savebold=True):
        print("run_mapping_var")
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        # init_point is about ue, se, ui, si but not variance
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = ue.clone()
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = ui.clone()

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        for i in range(init_iter_steps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue+ self.dt_mnn*(-ue + exc_mean_out) / self.mnn_membrane_constant
            se = torch.sqrt((se ** 2) + self.dt_mnn * (-(se ** 2) + (exc_std_out ** 2)) / self.mnn_membrane_constant)

            ui = ui+ self.dt_mnn*(-ui + inh_mean_out) / self.mnn_membrane_constant
            si = torch.sqrt((si ** 2) + self.dt_mnn * (-(si ** 2) + (inh_std_out ** 2)) / self.mnn_membrane_constant)

            if torch.isnan(se).any() or torch.isnan(si).any() or torch.isinf(se).any() or torch.isinf(si).any():
                raise 'has_nan_or_inf'

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record


    def run(self, T=None, init_time=None, init_point=None, savebold=True):
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            se = ue.clone()
            ui = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
            si = ui.clone()

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)

        assert iter_nsteps <= self.hp.shape[0] * round(self.hp_update_time / self.dt_mnn)

        ue_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device) #
        se_record = ue_record.clone()
        ui_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        si_record = ui_record.clone()

        if savebold:
            voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
            BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = torch.zeros(1, self.N, dtype=torch.float64, device=self.device)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp[0, :].reshape((1, -1))

        for i in range(init_iter_steps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se + self.dt_mnn * (-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si + self.dt_mnn * (-si + inh_std_out) / self.mnn_membrane_constant

            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue+ self.dt_mnn*(-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se+ self.dt_mnn*(-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui+ self.dt_mnn*(-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si+ self.dt_mnn*(-si + inh_std_out) / self.mnn_membrane_constant
            if savebold:
                unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
                _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
                                       unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
                bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                if savebold:
                    voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                if savebold:
                    BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp[_idx, :].reshape((1, -1))

        ue_record = torch_2_numpy(ue_record, is_cuda=self.is_cuda)
        se_record = torch_2_numpy(se_record, is_cuda=self.is_cuda)
        ui_record = torch_2_numpy(ui_record, is_cuda=self.is_cuda)
        si_record = torch_2_numpy(si_record, is_cuda=self.is_cuda)
        if savebold:
            voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
            BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)
            return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record
        else:
            return ue_record, se_record, ui_record, si_record


    def run_numpy(self, T=None, init_time=None, init_point=None):
        if init_time is None:
            init_time = self.init_time
        if T is None:
            T = self.T

        # initial condition
        if init_point is not None:
            assert init_point.shape == (1, self.N*4)
            ue = init_point[:, :self.N]
            se = init_point[:, self.N:self.N*2]
            ui = init_point[:, self.N*2:self.N*3]
            si = init_point[:, self.N*3:self.N*4]
        else:
            ue = np.zeros((1, self.N)).astype(np.float32)
            se = np.zeros((1, self.N)).astype(np.float32)
            ui = np.zeros((1, self.N)).astype(np.float32)
            si = np.zeros((1, self.N)).astype(np.float32)

        init_iter_steps = round(init_time / self.dt_mnn)
        init_record_steps = round(init_time / self.save_dt)
        iter_nsteps = round(T/self.dt_mnn)
        record_steps = round(T/self.save_dt)


        assert iter_nsteps <= self.hp_numpy.shape[0] * round(self.hp_update_time / self.dt_mnn)

        ue_record = np.zeros((record_steps+init_record_steps, self.N)).astype(np.float32) #
        se_record = np.zeros((record_steps+init_record_steps, self.N)).astype(np.float32)
        ui_record = np.zeros((record_steps+init_record_steps, self.N)).astype(np.float32)
        si_record = np.zeros((record_steps+init_record_steps, self.N)).astype(np.float32)

        # voxelwise_firingrate_record = torch.zeros(record_steps+init_record_steps, self.N, dtype=torch.float64, device=self.device)
        # BOLD_record = torch.zeros(record_steps // round(self.hp_update_time / self.save_dt), self.N, dtype=torch.float64, device=self.device)

        _external_current_forEpopu = np.zeros((1, self.N)).astype(np.float32)
        _external_current_forEpopu[:, self._assi_region - 1] = self.hp_numpy[0, :].reshape((1, -1))

        for i in range(init_iter_steps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward_numpy(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue + self.dt_mnn * (-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se + self.dt_mnn * (-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui + self.dt_mnn * (-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si + self.dt_mnn * (-si + inh_std_out) / self.mnn_membrane_constant

            # unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
            # _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
            #                        unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
            # self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))
            if (i+1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.save_dt / self.dt_mnn) - 1

                ue_record[_idx, :] = ue[0, :]
                se_record[_idx, :] = se[0, :]
                ui_record[_idx, :] = ui[0, :]
                si_record[_idx, :] = si[0, :]
                # voxelwise_firingrate_record[_idx, :] = _voxelwise_firingrate[0, :]

        print(f'Init: {init_time/1000:.2f}s')

        for i in range(iter_nsteps):
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward_numpy(ue, se, ui, si, external_current_forEpopu=_external_current_forEpopu)

            ue = ue+ self.dt_mnn*(-ue + exc_mean_out) / self.mnn_membrane_constant
            se = se+ self.dt_mnn*(-se + exc_std_out) / self.mnn_membrane_constant

            ui = ui+ self.dt_mnn*(-ui + inh_mean_out) / self.mnn_membrane_constant
            si = si+ self.dt_mnn*(-si + inh_std_out) / self.mnn_membrane_constant

            # unit_noise = torch.randn((1, self.N), dtype=torch.float64, device=self.device)
            # _voxelwise_firingrate = ue * self._EI_ratio + ui * (1 - self._EI_ratio) + \
            #                        unit_noise * torch.sqrt(self._EI_ratio * se ** 2 + (1 - self._EI_ratio) * si ** 2) / np.sqrt(self.dt_mnn) / torch.sqrt(self.block_size).reshape((1, -1))
            # bold_out = self.bold.run(torch.max(_voxelwise_firingrate[0, :].type_as(ue_record), torch.tensor([1e-05]).type_as(ue_record)))

            if (i + 1) % round(self.save_dt / self.dt_mnn) == 0:
                _idx = (i + 1) // round(self.save_dt / self.dt_mnn) - 1
                ue_record[_idx+init_record_steps, :] = ue[0, :]
                se_record[_idx+init_record_steps, :] = se[0, :]
                ui_record[_idx+init_record_steps, :] = ui[0, :]
                si_record[_idx+init_record_steps, :] = si[0, :]
                # voxelwise_firingrate_record[_idx+init_record_steps, :] = _voxelwise_firingrate[0, :]

            if (i+1) % round(self.hp_update_time / self.dt_mnn) == 0:
                _idx = (i+1) // round(self.hp_update_time / self.dt_mnn)
                print(f"_idx: {_idx*self.hp_update_time/1000:.2f}s")
                # BOLD_record[_idx-1, :] = bold_out.reshape((1, -1))
                if i < iter_nsteps-1:
                    _external_current_forEpopu[:, self._assi_region - 1] = self.hp_numpy[_idx, :].reshape((1, -1))


        # voxelwise_firingrate_record = torch_2_numpy(voxelwise_firingrate_record, is_cuda=self.is_cuda)
        # BOLD_record = torch_2_numpy(BOLD_record, is_cuda=self.is_cuda)

        return ue_record, se_record, ui_record, si_record
        # return ue_record, se_record, ui_record, si_record, voxelwise_firingrate_record, BOLD_record

# if __name__=='__main__':
#     torch.set_default_dtype(torch.float64)
#
#     device='cpu'
#     batchsize = 21
#     config ={
#         'batchsize': batchsize,
#         'bg_mean_to_exc': 0.4, #nA/uS; output is mV; 50 leak conductance in nS
#         'bg_mean_to_inh': 0.13,
#         'bg_std_to_exc': 0.0,
#         'bg_std_to_inh': 0.0,
#         'xregion_gain_for_E_popu': torch.linspace(0,2,batchsize, device=device).unsqueeze(1) # shape is batchsize x 1
#         }
#         #Le = 0.05 # leak conductance of exc neuron; unit in micro S
#         #Li = 0.1 # leak conductance of inh neuron
#     T = 10
#
#     mnn = Cond_MNN_DTB(config_for_Cond_MNN_DTB=config, device=device)
#     ue, se, ui, si = mnn.run(T=T)
#
#     plt.close('all')
#
#     plt.plot(ue.flatten().cpu().numpy())
#
#
#     # extent = (config['xregion_gain_for_E_popu'][0,0],config['xregion_gain_for_E_popu'][-1,0], 1, ue.shape[1])
#
#     # plt.figure(figsize=(3.5,3))
#     # plt.imshow(ue.T.cpu().numpy().round(3), aspect='auto', origin='lower', extent=extent)
#     # plt.ylabel('Region index')
#     # plt.xlabel('Cross-region gain')
#     # plt.title('Mean firing rate (sp/ms)')
#     # plt.colorbar()
#     # plt.tight_layout()
#
#
#
#