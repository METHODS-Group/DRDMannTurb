

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
import pathlib
# import numpy as np
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
from pylab import *
import os

from .common import MannEddyLifetime
from .OnePointSpectra import OnePointSpectra
from .SpectralCoherence import SpectralCoherence

from typing import Dict, Any


"""
==================================================================================================================
Loss funtion for calibration
=================================== ===============================================================================
"""


class LossFunc:

    def __init__(self, **kwargs):
        pass

    def __call__(self, model, target, pen=None):
        # loss = 0.5*torch.mean( (model * torch.log(model.abs()) - target * torch.log(target.abs()) )**2 )
        loss = 0.5*torch.mean((torch.log(torch.abs(model/target)))**2)
        # loss = 0.5*torch.mean( ( (model-target)/(1.e-6 + target) )**2 )
        if pen:
            loss = loss + pen
        return loss


"""
==================================================================================================================
Calibration problem class
==================================================================================================================
"""


class CalibrationProblem:

    def __init__(self, **kwargs: Dict[str, Any]):
        # stringify the activation functions used; for manual bash only 
        self.activfuncstr = str(kwargs.get('activations', ['relu', 'relu']))
        print(self.activfuncstr)

        self.input_size = kwargs.get('input_size', 3)
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 0)
        self.init_with_noise = kwargs.get('init_with_noise', False)
        self.noise_magnitude = kwargs.get('noise_magnitude', 1.e-3)

        self.OPS = OnePointSpectra(**kwargs)
        self.init_device()
        if self.init_with_noise:
            self.initialize_parameters_with_noise()

        self.vdim = 3
        self.output_directory = kwargs.get('output_folder','/home/gdeskos/WindGenerator/')
        self.fg_coherence = kwargs.get("fg_coherence", False)
        if self.fg_coherence:
            self.Coherence = SpectralCoherence(**kwargs)

    # enable gpu device
    def init_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.OPS.to(device)

    # =========================================

    @property
    def parameters(self):
        NN_parameters = parameters_to_vector(self.OPS.parameters())
        with torch.no_grad():
            param_vec = NN_parameters.cpu().numpy()
        return param_vec

    @parameters.setter
    def parameters(self, param_vec):
        assert(len(param_vec) >= 1)
        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(param_vec, dtype=torch.float64)
        vector_to_parameters(param_vec, self.OPS.parameters())

    def update_parameters(self, param_vec):
        self.parameters = param_vec

    def initialize_parameters_with_noise(self):
        noise = torch.tensor(self.noise_magnitude * torch.randn(*self.parameters.shape), dtype=torch.float64)
        # self.update_parameters(noise**2)
        vector_to_parameters(noise.abs(), self.OPS.parameters())
        try:
            vector_to_parameters(noise, self.OPS.tauNet.parameters())
        except:
            pass
        try:
            vector_to_parameters(noise.abs(), self.OPS.Corrector.parameters())
        except:
            pass

    # =========================================

    def __call__(self, k1):
        return self.eval(k1)

    def eval(self, k1):
        Input = self.format_input(k1)
        with torch.no_grad():
            Output = self.OPS(Input)
        return self.format_output(Output)

    def eval_grad(self, k1):
        self.OPS.zero_grad()
        Input = self.format_input(k1)
        self.OPS(Input).backward()
        grad = torch.cat([param.grad.view(-1)
                         for param in self.OPS.parameters()])
        return self.format_output(grad)

    def format_input(self, k1):
        # TODO: it seems like these are not being hit. e.g. np call below
        #   would error since numpy was not originally imported...
        # if k1.ndim == 0:
        #     return torch.tensor([k1], dtype=torch.float64)
        # else:
        return torch.tensor(k1, dtype=torch.float64)

    def format_output(self, out):
        return out.cpu().numpy()

    # -----------------------------------------
    # Calibration method
    # -----------------------------------------

    def calibrate(self, **kwargs: Dict[str, Any]):
        print('\nCallibrating MannNet...')

        DataPoints, DataValues = kwargs.get('Data')
        OptimizerClass = kwargs.get('OptimizerClass', torch.optim.LBFGS)
        lr = kwargs.get('lr',  1e-1)
        tol = kwargs.get('tol', 1e-3)
        nepochs = kwargs.get('nepochs', 100)
        show = kwargs.get('show', False)
        self.curves = kwargs.get('curves', [0, 1, 2,3])

        alpha_pen = kwargs.get('penalty', 0)
        alpha_reg = kwargs.get('regularization', 0)

        self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[
            :, 0].squeeze()

        # create a single numpy.ndarray with numpy.array() and then convert to a porch tensor
        # single_data_array=torch.tensor( [DataValues[:, i, i] for i in range(
        #     3)] + [DataValues[:, 0, 2]])
        # self.kF_data_vals = torch.tensor(single_data_array, dtype=torch.float64)
        self.kF_data_vals = torch.cat((DataValues[:, 0, 0], DataValues[:, 1, 1], DataValues[:, 2, 2], DataValues[:, 0, 2]))

        k1_data_pts, y_data0 = self.k1_data_pts, self.kF_data_vals
        # self.x, self.y, self.y_data = k1_data_pts

        y = self.OPS(k1_data_pts)
        y_data = torch.zeros_like(y)
        print(y_data0.shape)
        y_data[:4, ...] = y_data0.view(4, y_data0.shape[0] // 4)


        ### The case with the coherence
        ### formatting the data
        ### DataPoints_coh = (k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts) - tuple of 3 one-dimensional arrays (axes f, Delta_y, Delatz)
        ### DataValues_coh - 3D array of coherence values at the data points
        if self.fg_coherence:
            DataPoints_coh, DataValues_coh = kwargs.get('Data_Coherence')
            k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts = DataPoints_coh
            k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts = torch.meshgrid(k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts)
            y_coh      = self.Coherence(k1_data_pts, Delta_y_data_pts, Delta_z_data_pts)
            y_coh_data = torch.zeros_like(y_coh)
            y_coh_data[:] = DataValues_coh

        self.loss_fn = LossFunc()
        # self.loss_fn = torch.nn.MSELoss(reduction='mean')

        ##############################
        # Optimization
        ##############################
        if OptimizerClass == torch.optim.LBFGS: 
            optimizer = OptimizerClass(
                self.OPS.parameters(), lr=lr, line_search_fn='strong_wolfe', max_iter=50, history_size=nepochs) 
        else: 
            optimizer = OptimizerClass(self.OPS.parameters(), lr=lr)

#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        softplus = torch.nn.Softplus()
        logk1 = torch.log(self.k1_data_pts).detach()
        h1 = torch.diff(logk1)
        h2 = torch.diff(0.5*(logk1[:-1]+logk1[1:]))
        h3 = torch.diff(0.5*(self.k1_data_pts[:-1]+self.k1_data_pts[1:]))
        h4 = torch.diff(self.k1_data_pts)
        D = logk1.max() - logk1.min()

        def PenTerm(y):
            """
            TODO: are these embedded functions necessary?
            """
            logy = torch.log(torch.abs(y))
            d2logy = torch.diff(torch.diff(logy, dim=-1)/h1, dim=-1)/h2
            f = torch.relu(d2logy).square()
            # pen = torch.sum( f * h2 ) / D
            pen = torch.mean(f)
            return pen

        def RegTerm():
            """
            TODO: are these embedded functions necessary?
            """
            reg = 0
            if self.OPS.type_EddyLifetime == 'tauNet':
                theta_NN = parameters_to_vector(
                    self.OPS.tauNet.NN.parameters())
                reg = theta_NN.square().mean()
            return reg

        def loss_fn(model, target,weights):
            """
            TODO: are these embedded functions necessary?
            """
            #y = torch.abs((model-target)).square()
            y = torch.log(torch.abs(model/target)).square()
            # y = ( (model-target)/(target) ).square()
            # y = 0.5*(y[...,:-1]+y[...,1:])
            # loss = 0.5*torch.sum( y * h4 )
            # loss = torch.sum( y * h1 )
            loss = torch.mean(y)
            return loss

        # self.loss_fn = LossFunc()
        self.loss_fn = loss_fn
        w = torch.abs(y)/torch.sum(torch.abs(y[:,0]))
        self.loss = self.loss_fn(y[self.curves], y_data[self.curves], w[self.curves])

        self.loss_history_total = []
        self.loss_history_epochs = []

        print('Initial loss: ', self.loss.item())
        self.loss_history_total.append(self.loss.item())
        self.loss_history_epochs.append(self.loss.item())

        for i in (0,):  # range(len(self.curves),0,-1):
            def closure():
                optimizer.zero_grad()
                y = self.OPS(k1_data_pts)
                w = k1_data_pts*y_data/torch.sum(k1_data_pts*y_data)
                self.loss = self.loss_fn(y[self.curves[i:]], y_data[self.curves[i:]],w[self.curves[i:]])
                if self.fg_coherence:
                    w1, w2 = 1, 1 ### weights to balance the coherence misfit and others
                    y_coh     = self.Coherence(k1_data_pts, Delta_y_data_pts, Delta_z_data_pts)
                    loss_coh  = self.loss_fn(y_coh, y_coh_data)
                    self.loss = w1*self.loss + w2*loss_coh
                self.loss_only = 1.*self.loss.item()
                self.loss_history_total.append(self.loss_only)
                if alpha_pen:
                    pen = alpha_pen * PenTerm(y[self.curves[i:]])
                    self.loss = self.loss + pen
                    # print('pen = ', pen.item())
                if alpha_reg:
                    reg = alpha_reg * RegTerm()
                    self.loss = self.loss + reg
                    # print('reg = ', reg.item())
                self.loss.backward()
                print('loss  = ', self.loss.item())
                # if hasattr(self.OPS, 'tauNet'):
                #     if hasattr(self.OPS.tauNet.Ra.nu, 'item'):
                #         print('-> nu = ', self.OPS.tauNet.Ra.nu.item())
                self.kF_model_vals = y.clone().detach()
                # self.plot(**kwargs, plt_dynamic=True,
                #           model_vals=self.kF_model_vals.cpu().detach().numpy() if torch.is_tensor(self.kF_model_vals) else self.kF_model_vals)
                return self.loss

            for epoch in range(nepochs):
                print('\n=================================')
                print('-> Epoch {0:d}'.format(epoch))
                print('=================================\n')
                optimizer.step(closure)
                # TODO: refactor the scheduler things, plateau requires loss 
                # scheduler.step(self.loss) #if scheduler
                scheduler.step()
                self.print_grad()
                print('---------------------------------\n')
                self.print_parameters()
                print('=================================\n')
                self.loss_history_epochs.append(self.loss_only)
                if self.loss.item() < tol:
                    break

        print('\n=================================')
        print('Calibration terminated.')
        print('=================================\n')
        print('loss = {0}'.format(self.loss.item()))
        print('tol  = {0}'.format(tol))
        self.print_parameters()
        self.plot(plt_dynamic=False)

        return self.parameters

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def print_parameters(self):
        # print(('Optimal NN parameters = [' + ', '.join(['{}'] *
        #       len(self.parameters)) + ']\n').format(*self.parameters))
        pass

    def print_grad(self):
        # self.grad = torch.cat([param.grad.view(-1)
        #                       for param in self.OPS.parameters()]).detach().numpy()
        # print('grad = ', self.grad)
        pass

    def plot(self, **kwargs:Dict[str, Any]):
        """
        Handles all plotting
        """
        plt_dynamic = kwargs.get('plt_dynamic', False)
        if plt_dynamic:
            ion()
        else:
            ioff()

        Data = kwargs.get('Data')
        if Data is not None:
            DataPoints, DataValues = Data
            self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[
                :, 0].squeeze()
            # create a single numpy.ndarray with numpy.array() and then convert to a porch tensor
            # single_data_array=np.array( [DataValues[:, i, i] for i in range(
            # 3)] + [DataValues[:, 0, 2]])
            # self.kF_data_vals = torch.tensor(single_data_array, dtype=torch.float64)
            self.kF_data_vals = torch.cat((DataValues[:, 0, 0], DataValues[:, 1, 1], DataValues[:, 2, 2], DataValues[:, 0, 2]))

        k1 = self.k1_data_pts
        k = torch.stack([0*k1, k1, 0*k1], dim=-1)

        plt_tau = kwargs.get('plt_tau', True)
        if plt_tau:
            k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
            k_1 = torch.stack([k_gd, 0*k_gd, 0*k_gd], dim=-1)
            k_2 = torch.stack([0*k_gd, k_gd, 0*k_gd], dim=-1)
            k_3 = torch.stack([0*k_gd, 0*k_gd, k_gd], dim=-1)
            k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3**(1/2)
            # k_norm = torch.norm(k, dim=-1)

        self.kF_model_vals = kwargs.get('model_vals', None)
        if self.kF_model_vals is None:
            self.kF_model_vals = self.OPS(k1).cpu().detach().numpy()

        if not hasattr(self, 'fig'):
            nrows = 1
            ncols = 2 if plt_tau else 1
            self.fig, self.ax = subplots(
                nrows=nrows, ncols=ncols, num='Calibration', clear=True, figsize=[10, 5])
            if not plt_tau:
                self.ax = [self.ax]

            # Subplot 1: One-point spectra
            self.ax[0].set_title('One-point spectra')
            self.lines_SP_model = [None]*(self.vdim+1)
            self.lines_SP_data = [None]*(self.vdim+1)
            clr=['red','blue','green','magenta']
            for i in range(self.vdim):
                self.lines_SP_model[i], = self.ax[0].plot(
                    k1.cpu().detach().numpy(), self.kF_model_vals[i], color=clr[i],label=r'$F{0:d}$ model'.format(i+1)) #'o-'

            print(f"k1.size: {k1.size()}   self.kF_data_vals: {self.kF_data_vals.size()}")
            
            s = self.kF_data_vals.shape[0]

            for i in range(self.vdim):
                self.lines_SP_data[i],  = self.ax[0].plot(
                    k1.cpu().detach().numpy(), self.kF_data_vals.view(4,s//4)[i].cpu().detach().numpy(), '--', color=clr[i], label=r'$F{0:d}$ data'.format(i+1))
            if 3 in self.curves:
                self.lines_SP_model[self.vdim], = self.ax[0].plot(
                    k1.cpu().detach().numpy(), -self.kF_model_vals[self.vdim], 'o-',color=clr[3], label=r'$-F_{13}$ model')
                self.lines_SP_data[self.vdim],  = self.ax[0].plot(
                    k1.cpu().detach().numpy(), -self.kF_data_vals.view(4,s//4)[self.vdim].cpu().detach().numpy(), '--',color=clr[3],label=r'$-F_{13}$ data')
            self.ax[0].legend()
            self.ax[0].set_xscale('log')
            self.ax[0].set_yscale('log')
            self.ax[0].set_xlabel(r'$k_1$')
            self.ax[0].set_ylabel(r'$k_1 F_i /u_*^2$')
            self.ax[0].grid(which='both')
            #self.ax[0].set_aspect(1/2)

            if plt_tau:
                # Subplot 2: Eddy Lifetime
                self.ax[1].set_title('Eddy lifetime')
                self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
                self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
                self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
                self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()
                # self.tau_model1m= self.OPS.EddyLifetime(-k_1).detach().numpy()
                # self.tau_model2m= self.OPS.EddyLifetime(-k_2).detach().numpy()
                # self.tau_model3m= self.OPS.EddyLifetime(-k_3).detach().numpy()
                self.tau_ref = 3.9 * \
                    MannEddyLifetime(0.59 * k_gd).cpu().detach().numpy()
                self.lines_LT_model1, = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(), self.tau_model1, '-', label=r'$\tau_{model}(k_1)$')
                self.lines_LT_model2, = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(), self.tau_model2, '-', label=r'$\tau_{model}(k_2)$')
                self.lines_LT_model3, = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(), self.tau_model3, '-', label=r'$\tau_{model}(k_3)$')
                self.lines_LT_model4, = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(), self.tau_model4, '-', label=r'$\tau_{model}(k,k,k)$')
                # self.lines_LT_model1m, = self.ax[1].plot(k_gd, self.tau_model1m, '-', label=r'$\tau_{model}(-k_1)$')
                # self.lines_LT_model2m, = self.ax[1].plot(k_gd, self.tau_model2m, '-', label=r'$\tau_{model}(-k_2)$')
                # self.lines_LT_model3m, = self.ax[1].plot(k_gd, self.tau_model3m, '-', label=r'$\tau_{model}(-k_3)$')
                self.lines_LT_ref,   = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(), self.tau_ref,  '--', label=r'$\tau_{ref}=$Mann')
                self.ax[1].legend()
                self.ax[1].set_xscale('log')
                self.ax[1].set_yscale('log')
                self.ax[1].set_xlabel(r'$k$')
                self.ax[1].set_ylabel(r'$\tau$')
                self.ax[1].grid(which='both')

                # plt.show()

            # TODO clean up plotting things? 
            self.fig.canvas.draw()
            # TODO: comment next out if to save 
            self.fig.canvas.flush_events()

        for i in range(self.vdim):
            self.lines_SP_model[i].set_ydata(self.kF_model_vals[i])
        if 3 in self.curves:
            self.lines_SP_model[self.vdim].set_ydata(
                -self.kF_model_vals[self.vdim])
        #self.ax[0].set_aspect(1)

        if plt_tau:
            self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
            self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
            self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
            self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()
            # self.tau_model1m= self.OPS.EddyLifetime(-k_1).detach().numpy()
            # self.tau_model2m= self.OPS.EddyLifetime(-k_2).detach().numpy()
            # self.tau_model3m= self.OPS.EddyLifetime(-k_3).detach().numpy()
            self.lines_LT_model1.set_ydata(self.tau_model1)
            self.lines_LT_model2.set_ydata(self.tau_model2)
            self.lines_LT_model3.set_ydata(self.tau_model3)
            self.lines_LT_model4.set_ydata(self.tau_model4)
            # self.lines_LT_model1m.set_ydata(self.tau_model1m)
            # self.lines_LT_model2m.set_ydata(self.tau_model2m)
            # self.lines_LT_model3m.set_ydata(self.tau_model3m)


            # plt.show()
            
        if plt_dynamic:
            for ax in self.ax:
                ax.relim()
                ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            pass
            # TODO: uncomment next!
            # print("="*30)
            # print("SAVING FINAL SOLUTION RESULTS TO " + f'{self.output_directory+"/" + self.activfuncstr +"final_solution.png"}')

            # self.fig.savefig(self.output_directory+"/" + self.activfuncstr + "final_solution.png", format='png', dpi=100)

            # plt.savefig(self.output_directory.resolve()+'Final_solution.png',format='png',dpi=100)

            
        #self.fig.savefig(self.output_directory, format='png', dpi=100)
        # self.fig.savefig(self.output_directory.resolve()+"final_solution.png", format='png', dpi=100)


############################################################################
############################################################################
