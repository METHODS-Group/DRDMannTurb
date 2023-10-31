"""
This module implements and exposes the various terms that can be
added and scaled in a generic loss function
"""
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from drdmannturb import LossParameters

writer = SummaryWriter()


class LossAggregator:
    def __init__(self, params: LossParameters, k1space: torch.Tensor):
        """Combines all loss functions and evaluates each term for the optimizer.
        The loss function for spectra fitting is determined by the following minimization problem:

        .. math::
            \min _{\boldsymbol{\theta}}\left\{\operatorname{MSE}[\boldsymbol{\theta}]+\alpha \operatorname{Pen}[\boldsymbol{\theta}]+\beta \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]\right\}

        where the loss terms are defined as follows:


        where :math:`L` is the number of data points :math:`f_j \in \mathcal{D}`. The data :math:`J_i (f_j)` is evaluated using the Kaimal spectra. The second-order penalization term is defined as

        .. math::
            \operatorname{Pen}_2[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|} \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial^2 \log \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\left(\partial \log k_1\right)^2}\right)\right\|_{\mathcal{D}}^2,

        and the first-order penalization term is defined as

        .. math::
            \operatorname{Pen}_1[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|} \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial \log \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\partial \log k_1}\right)\right\|_{\mathcal{D}}^2.

        Note that the norm :math:`||\cdot||_{\mathcal{D}}` is defined as

        .. math::
            \|g\|_{\mathcal{D}}^2:=\int_{\mathcal{D}}|g(f)|^2 \mathrm{~d}(\log f).

        The regularization term is defined as

        .. math::
            \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]:=\frac{1}{N} \sum_{i=1}^N \theta_{\mathrm{NN}, i}^2.

                Parameters
                ----------
                params : LossParameters
                    Dataclass with parameters that determine the loss function
                k1space : torch.Tensor
                    _description_
        """
        self.params = params
        self.k1space = k1space
        self.logk1 = torch.log(self.k1space)
        self.h1 = torch.diff(self.logk1)
        self.h2 = torch.diff(0.5 * (self.logk1[:-1] + self.logk1[1:]))

        t_alphapen1 = (
            lambda y, theta_NN, epoch: self.params.alpha_pen1
            * self.Pen1stOrder(y, epoch)
            if self.params.alpha_pen1
            else 0.0
        )

        t_alphapen2 = (
            lambda y, theta_NN, epoch: self.params.alpha_pen2
            * self.Pen2ndOrder(y, epoch)
            if self.params.alpha_pen2
            else 0.0
        )

        t_beta_reg = (
            lambda y, theta_NN, epoch: self.params.beta_reg
            * self.Regularization(theta_NN, epoch)
            if self.params.beta_reg
            else 0.0
        )

        self.loss_func = (
            lambda y, theta_NN, epoch: t_alphapen1(y, theta_NN, epoch)
            + t_alphapen2(y, theta_NN, epoch)
            + t_beta_reg(y, theta_NN, epoch)
        )

    def MSE_term(self, model: torch.Tensor, target: torch.Tensor, epoch: int):
        """Evaluates the loss term
        .. math::
            \operatorname{MSE}[\boldsymbol{\theta}]:=\frac{1}{L} \sum_{i=1}^4 \sum_{j=1}^L\left(\log \left|J_i\left(f_j\right)\right|-\log \left|\widetilde{J}_i\left(f_j, \boldsymbol{\theta}\right)\right|\right)^2,

                Parameters
                ----------
                model : torch.Tensor
                    _description_
                target : torch.Tensor
                    _description_
                epoch : int
                    Current epoch number.

                Returns
                -------
                _type_
                    _description_
        """

        mse_loss = torch.mean(torch.log(torch.abs(model / target)).square())
        writer.add_scalar("MSE Loss", mse_loss, epoch)

        return mse_loss

    def Pen2ndOrder(self, y: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        y : torch.Tensor
            _description_
        epoch : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        logy = torch.log(torch.abs(y))
        d2logy = torch.diff(torch.diff(logy, dim=-1) / self.h1, dim=-1) / self.h2

        pen2ndorder_loss = torch.mean(torch.relu(d2logy).square())

        writer.add_scalar("2nd Order Penalty", pen2ndorder_loss, epoch)

        return pen2ndorder_loss

    def Pen1stOrder(self, y: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        y : torch.Tensor
            _description_
        epoch : int
            _description_

        Returns
        -------
        _type_
            _description_
        """

        logy = torch.log(torch.abs(y))
        d1logy = torch.diff(logy, dim=-1) / self.h1

        pen1storder_loss = torch.mean(torch.relu(d1logy).square())
        writer.add_scalar("1st Order Penalty", pen1storder_loss, epoch)

        return pen1storder_loss

    def Regularization(self, theta_NN: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        theta_NN : torch.Tensor
            _description_
        epoch : int
            _description_

        Returns
        -------
        _type_
            _description_
        """

        reg_loss = theta_NN.square().mean()

        writer.add_scalar("Regularization", reg_loss, epoch)

        return reg_loss

    def eval(
        self,
        y: torch.Tensor,
        y_data: torch.Tensor,
        theta_NN: Optional[torch.Tensor],
        epoch: int,
    ):
        """_summary_

        Parameters
        ----------
        y : torch.Tensor
            _description_
        y_data : torch.Tensor
            _description_
        theta_NN : Optional[torch.Tensor]
            _description_
        epoch : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        total_loss = self.MSE_term(y, y_data, epoch) + self.loss_func(
            y, theta_NN, epoch
        )

        writer.add_scalar("Total Loss", total_loss, epoch)

        return total_loss
