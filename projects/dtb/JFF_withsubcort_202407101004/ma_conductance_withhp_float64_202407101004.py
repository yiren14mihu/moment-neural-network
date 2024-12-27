# moment activation for condutance-based neurons

from mnn.mnn_core.mnn_utils import Mnn_Core_Func, Param_Container
import numpy as np
from matplotlib import pyplot as plt
import torch
from utils.helpers import numpy2torch, torch_2_numpy


class Moment_Activation_Cond(Mnn_Core_Func):
    def __init__(self, config=None):
        super().__init__()
        ''' strictly follow PRE 2024'''
        
        if config==None: 
            self.tau_L = 20 # membrane time constant
            self.tau_E = 4 # excitatory synaptic time scale (ms)
            self.tau_I = 10 # inhibitory synaptic time scale (ms)
            self.VL = -60 # leak reverseal potential
            self.VE = 0 # excitatory reversal potential
            self.VI = -80 # inhibitory reversal potential
            self.vol_th = -50 # firing threshold
            self.vol_rest = -60 # reset potential
            self.t_ref = 2 # ms; NB inhibitory neuron has different value
            self.bgOU_mean = 0.36
            self.bgOU_std = 1
            self.bgOU_tau = 4
        else:    
            self.tau_L = config['tau_L'] # membrane time constant
            self.tau_E = config['tau_E'] # excitatory synaptic time scale (ms)
            self.tau_I = config['tau_I'] # inhibitory synaptic time scale (ms)
            self.VL = config['VL'] # leak reverseal potential
            self.VE = config['VE'] # excitatory reversal potential
            self.VI = config['VI'] # inhibitory reversal potential
            self.vol_th = config['Vth'] # firing threshold
            self.vol_rest = config['Vres'] # reset potential
            self.t_ref = config['Tref'] # ms; NB inhibitory neuron has different value
            self.bgOU_mean = config['bgOU_mean']
            self.bgOU_std = config['bgOU_std']
            self.bgOU_tau = config['bgOU_tau']

        self.sE = 1.0  # modifier to conductance (can be voltage dependent)
        self.sI = 1.0
        self.L = 1/self.tau_L
        
        # TODO: horrible scalability; consider a list of g, input_mean, input_std
        # so it supports any number of channels

    def synaptic_current_stat_nohp(self, exc_input_mean, exc_input_std, inh_input_mean, inh_input_std):
        tau_eff = self.tau_L / (
                    1 + self.sE * exc_input_mean * self.tau_E + self.sI * inh_input_mean * self.tau_I)  # effective time constant
        V_eff = tau_eff / self.tau_L * (self.VL + self.sE * exc_input_mean * self.tau_E * self.VE
                                        + self.sI * inh_input_mean * self.tau_I * self.VI + self.bgOU_mean)  # effective reversal potential

        h_E = self.tau_E / self.tau_L * self.sE * (self.VE - V_eff) * exc_input_std
        h_I = self.tau_I / self.tau_L * self.sI * (self.VI - V_eff) * inh_input_std

        exc_channel_current_mean = self.sE * (self.VE - V_eff) * self.tau_E * exc_input_mean / self.tau_L
        inh_channel_current_mean = self.sI * (self.VI - V_eff) * self.tau_I * inh_input_mean / self.tau_L
        exc_channel_current_std = (tau_eff / (tau_eff + self.tau_E) * h_E * h_E).pow(0.5)
        inh_channel_current_std = (tau_eff / (tau_eff + self.tau_I) * h_I * h_I).pow(0.5)

        return exc_channel_current_mean, exc_channel_current_std, inh_channel_current_mean, inh_channel_current_std


    def cond2curr_includingbgOU_withhp(self, exc_input_mean, exc_input_std, inh_input_mean, inh_input_std, external_current):
        ''' This step should be called synaptic activation (vs neuronal activation)
        map conductance-based to current-based spiking neuron
        using the effective time constant approximation'''

        tau_eff = self.tau_L / (
                    1 + self.sE * exc_input_mean * self.tau_E + self.sI * inh_input_mean * self.tau_I)  # effective time constant
        V_eff = tau_eff / self.tau_L * (self.VL + self.sE * exc_input_mean * self.tau_E * self.VE
                                        + self.sI * inh_input_mean * self.tau_I * self.VI + self.bgOU_mean + external_current)  # effective reversal potential
        # print("V_eff", V_eff.min(), V_eff.max())

        # approximating multiplicative noise;
        # h_E = np.sqrt(self.tau_E)*self.tau_E/tau_L*self.gE*(self.VE-V_eff)*exc_input_std
        # h_I = np.sqrt(self.tau_I)*self.tau_I/tau_L*self.gI*(self.VI-V_eff)*inh_input_std

        h_E = self.tau_E / self.tau_L * self.sE * (self.VE - V_eff) * exc_input_std
        h_I = self.tau_I / self.tau_L * self.sI * (self.VI - V_eff) * inh_input_std
        h_bgOU = self.bgOU_std * np.sqrt(2 * self.bgOU_tau) / self.tau_L

        # effective input current mean/std
        # NB: in PRE 2024 this is voltage mean/std
        # so a conversion factor of 1/tau_eff is applied
        eff_input_mean = V_eff / tau_eff
        tmp = tau_eff / (tau_eff + self.tau_E) * h_E * h_E
        tmp = tmp + tau_eff / (tau_eff + self.tau_I) * h_I * h_I
        tmp = tmp + tau_eff / (tau_eff + self.bgOU_tau) * h_bgOU * h_bgOU
        eff_input_std = tmp.pow(0.5)
        # eff_input_std = np.power(tmp, 0.5)

        return eff_input_mean, eff_input_std, tau_eff


    def forward_fast_mean(self, ubar, sbar, tau_eff):
        '''Calculates the mean output firing rate given the mean & std of input firing rate'''
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th /tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        mean_out = np.zeros(ubar.shape)
        
        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        
        # test_correction = -0.6
        # ub += test_correction
        # lb += test_correction
        
        temp_mean = 2 *tau_eff[indx2] * (self.Dawson1.int_fast(ub) - self.Dawson1.int_fast(lb))

        mean_out[indx2] = 1 / (temp_mean + self.t_ref)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.vol_th /tau_eff)
        indx4 = np.logical_and(~indx0, ubar > self.vol_th /tau_eff)
        mean_out[indx3] = 0.0
        mean_out[indx4] = 1 / (self.t_ref -  tau_eff[indx4] * np.log(1 - self.vol_th /tau_eff[indx4] / ubar[indx4]))

        return mean_out

    def backward_fast_mean(self, ubar, sbar, tau_eff, u_a):
        indx0 = sbar > 0
        indx1 = (self.vol_th /tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        grad_uu = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))

        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        grad_uu[indx2] = u_a[indx2] * u_a[indx2] / sbar[indx2] * delta_g * 2 * tau_eff[indx2] * np.sqrt(tau_eff[indx2])

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx6 = np.logical_and(~indx0, ubar <= self.vol_th / tau_eff)
        indx4 = np.logical_and(~indx0, ubar > self.vol_th / tau_eff)

        grad_uu[indx6] = 0.0
        grad_uu[indx4] = self.vol_th * u_a[indx4] * u_a[indx4] / ubar[indx4] / (ubar[indx4] - self.vol_th / tau_eff[indx4])
        # grad_uu[indx4] = self.vol_th * tau_eff[indx4] * u_a[indx4] * u_a[indx4] / ubar[indx4] / (ubar[indx4] * tau_eff[indx4] - self.vol_th)

        # ---------------

        grad_us = np.zeros(ubar.shape)
        temp = self.Dawson1.dawson1(ub) * ub - self.Dawson1.dawson1(lb) * lb
        grad_us[indx2] = u_a[indx2] * u_a[indx2] / sbar[indx2] * temp * 2 * tau_eff[indx2]

        grad_utau = np.zeros(ubar.shape)
        delta_G = self.Dawson1.int_fast(ub) - self.Dawson1.int_fast(lb)
        temp1 = 2 * delta_G
        temp2 = self.Dawson1.dawson1(ub) * (- self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))\
                - self.Dawson1.dawson1(lb) * (- self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        grad_utau[indx2] = - u_a[indx2] * u_a[indx2] * (temp1 + temp2)
        grad_utau[indx6] = 0.0

        grad_utau[indx4] = u_a[indx4] * u_a[indx4] * \
                           (np.log(1 - self.vol_th / tau_eff[indx4] / ubar[indx4]) + self.vol_th / tau_eff[indx4] / (ubar[indx4] - self.vol_th / tau_eff[indx4]))

        return grad_uu, grad_us, grad_utau

    def forward_fast_std(self, ubar, sbar, tau_eff, u_a):
        '''Calculates the std of output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th /tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        fano_factor = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        
        # cached mean used
        varT = 8 *tau_eff[indx2]*tau_eff[indx2] * (self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb))
        fano_factor[indx2] = varT * u_a[indx2] * u_a[indx2]

        # Region 2 is calculated with analytical limit as sbar --> 0
        fano_factor[~indx0] = (ubar[~indx0] < self.vol_th /tau_eff[~indx0]) + 0.0

        std_out = np.sqrt(fano_factor * u_a)
        return std_out

    def backward_fast_std(self, ubar, sbar, tau_eff, u_a, s_a, grad_utau):
        '''Calculates the gradient of the std of the firing rate with respect to the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th / tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))

        grad_su = np.zeros(ubar.shape)

        delta_g = self.Dawson1.dawson1(ub) - self.Dawson1.dawson1(lb)
        delta_h = self.Dawson2.dawson2(ub) - self.Dawson2.dawson2(lb)
        delta_H = self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb)

        temp1 = 3 * tau_eff[indx2] * np.sqrt(tau_eff[indx2]) * s_a[indx2] / sbar[indx2] * u_a[indx2] * delta_g
        temp2 = - 1 / 2 * np.sqrt(tau_eff[indx2]) * s_a[indx2] / sbar[indx2] * delta_h / delta_H

        grad_su[indx2] = temp1 + temp2

        grad_ss = np.zeros(ubar.shape)

        temp_dg = self.Dawson1.dawson1(ub) * ub - self.Dawson1.dawson1(lb) * lb
        temp_dh = self.Dawson2.dawson2(ub) * ub - self.Dawson2.dawson2(lb) * lb

        grad_ss[indx2] = 3 * tau_eff[indx2] * s_a[indx2] / sbar[indx2] * u_a[indx2] * temp_dg \
                         - 1 / 2 * s_a[indx2] / sbar[indx2] * temp_dh / delta_H

        indx4 = np.logical_and(~indx0, ubar > self.vol_th * self.L)

        grad_ss[indx4] = 1 * np.sqrt(tau_eff[indx4] / 2) * np.power(u_a[indx4], 1.5) \
                         * np.sqrt(1 / (self.vol_th /tau_eff[indx4] - ubar[indx4]) / (self.vol_th /tau_eff[indx4] - ubar[indx4]) - 1 / ubar[indx4] / ubar[indx4])  # TODO ！！这里要小心，因为self.vol_th * self.L=1所以略去了

        grad_stau = np.zeros(ubar.shape)
        temp1_stau = 3 / 2 * grad_utau[indx2] * s_a[indx2] / u_a[indx2]
        temp2_stau = s_a[indx2] / tau_eff[indx2]
        temp3_stau = s_a[indx2] / 2 / delta_H * \
                     (self.Dawson2.dawson2(ub) * (- self.vol_th /tau_eff[indx2] - ubar[indx2]) / (2 * sbar[indx2] * np.sqrt(tau_eff[indx2]))
                      - self.Dawson2.dawson2(lb) * (- self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (2 * sbar[indx2] * np.sqrt(tau_eff[indx2])))

        grad_stau[indx2] = temp1_stau + temp2_stau + temp3_stau

        return grad_su, grad_ss, grad_stau
    
    def activate(self, eff_input_mean, eff_input_std, tau_eff, is_cuda=True):
        # for pytorch
        # device = eff_input_mean.device
        
        eff_input_mean = torch_2_numpy(eff_input_mean, is_cuda=is_cuda)
        eff_input_std = torch_2_numpy(eff_input_std, is_cuda=is_cuda)
        tau_eff = torch_2_numpy(tau_eff, is_cuda=is_cuda)
        
        mean_out = self.forward_fast_mean(eff_input_mean, eff_input_std, tau_eff)
        std_out = self.forward_fast_std(eff_input_mean, eff_input_std, tau_eff, mean_out)

        mean_out = numpy2torch(mean_out, is_cuda=is_cuda)
        std_out = numpy2torch(std_out, is_cuda=is_cuda)
        # mean_out = torch.tensor(mean_out, dtype=torch.float64, device=device)
        # std_out = torch.tensor(std_out, dtype=torch.float64, device=device)
        return mean_out, std_out

    def backward_for_MA(self, eff_input_mean, eff_input_std, tau_eff, is_cuda=True):
        eff_input_mean = torch_2_numpy(eff_input_mean, is_cuda=is_cuda)
        eff_input_std = torch_2_numpy(eff_input_std, is_cuda=is_cuda)
        tau_eff = torch_2_numpy(tau_eff, is_cuda=is_cuda)

        mean_out = self.forward_fast_mean(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff)
        std_out = self.forward_fast_std(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff, u_a=mean_out)

        grad_uu, grad_us, grad_utau = self.backward_fast_mean(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff, u_a=mean_out)  # ubar, sbar, tau_eff, u_a
        grad_su, grad_ss, grad_stau = self.backward_fast_std(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff, u_a=mean_out, s_a=std_out, grad_utau=grad_utau)  # ubar, sbar, tau_eff, u_a, s_a, grad_utau
        # # grad_su, grad_ss, _ = self.backward_fast_std(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff, u_a=mean_out, s_a=std_out, grad_utau=grad_utau)  # ubar, sbar, tau_eff, u_a, s_a, grad_utau
        #
        # _delta_scalar = 0.00001
        # std_out_delta_taueff = self.forward_fast_std(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff+_delta_scalar*np.ones_like(tau_eff), u_a=mean_out)
        # # std_out_delta_taueff = self.forward_fast_std(ubar=eff_input_mean, sbar=eff_input_std, tau_eff=tau_eff+_delta_scalar*np.ones_like(tau_eff), u_a=mean_out)
        # # grad_stau = (std_out_delta_taueff-std_out) / _delta_scalar
        # # print("grad_stau", grad_stau)
        # # print("(std_out_delta_taueff-std_out) / _delta_scalar", (std_out_delta_taueff-std_out) / _delta_scalar)

        grad_uu = numpy2torch(grad_uu, is_cuda=is_cuda)
        grad_us = numpy2torch(grad_us, is_cuda=is_cuda)
        grad_utau = numpy2torch(grad_utau, is_cuda=is_cuda)
        grad_su = numpy2torch(grad_su, is_cuda=is_cuda)
        grad_ss = numpy2torch(grad_ss, is_cuda=is_cuda)
        grad_stau = numpy2torch(grad_stau, is_cuda=is_cuda)
        return grad_uu, grad_us, grad_utau, grad_su, grad_ss, grad_stau


def plot_the_map():
    config = {}
    
    ma = Moment_Activation_Cond()
    # print all properties and methods of the class
    print(vars(ma))

    n = 51
    exc_rate = np.linspace(0,2,n) # firng rate in kHz
    inh_rate = np.linspace(0,0.1,n)
    
    X, Y = np.meshgrid(exc_rate, inh_rate, indexing='xy')
    
    eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(X,X,Y,Y)
    #add external input current here if needed

    mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
    std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)

    # print('Output spike stats:')
    # print(mean_out)
    # print(std_out)

    extent = (exc_rate[0],exc_rate[-1],inh_rate[0],inh_rate[-1])
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(eff_input_mean*tau_eff, origin='lower', extent=extent, aspect='auto')
    plt.ylabel('Inh input rate (sp/ms)')
    plt.title('Eff. rev. pot.')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(eff_input_std, origin='lower', extent=extent, aspect='auto')
    plt.title('Eff. input std')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(mean_out, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('Exc input rate (sp/ms)')
    plt.ylabel('Inh input rate (sp/ms)')
    plt.title('Output mean')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(std_out, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('Exc input rate (sp/ms)')
    plt.title('Output std')
    plt.tight_layout()
    plt.colorbar()

    plt.savefig('projects/dtb/temp.png',dpi=300)


if __name__=='__main__':
    plot_the_map()
