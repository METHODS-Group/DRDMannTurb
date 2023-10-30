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
    def __init__(
        self, params: LossParameters, k1space: torch.Tensor, max_epochs: torch.Tensor
    ):
        """_summary_

        Parameters
        ----------
        params : LossParameters
            _description_
        k1space : torch.Tensor
            _description_
        """
        self.params = params
        self.k1space = k1space
        self.logk1 = torch.log(self.k1space)
        self.h1 = torch.diff(self.logk1)
        self.h2 = torch.diff(0.5 * (self.logk1[:-1] + self.logk1[1:]))

        self.loss_func = lambda y, theta_NN, epoch: 0.0
        if self.params.alpha_pen1:
            t_alphapen1 = (
                lambda y, theta_NN, epoch: self.params.alpha_pen1
                * self.Pen1stOrder(y, epoch)
            )
            # self.loss_func = (
            # lambda y, theta_NN, epoch: self.loss_func(y, theta_NN, epoch) + self.params.alpha_pen1 * self.Pen1StOrder(y, epoch)
            # )
        if self.params.alpha_pen2:
            t_alphapen2 = (
                lambda y, theta_NN, epoch: self.params.alpha_pen2
                * self.Pen2ndOrder(y, epoch)
            )
            # self.loss_func = (
            # lambda y, theta_NN, epoch: self.loss_func(y, theta_NN, epoch) + self.params.alpha_pen2 * self.Pen2ndOrder(y, epoch)
            # )
        if self.params.beta_reg:
            t_beta_reg = (
                lambda y, theta_NN, epoch: self.params.beta_reg
                * self.Regularization(theta_NN, epoch)
            )
            # self.loss_func = (
            # lambda y, theta_NN, epoch: self.loss_func(y, theta_NN, epoch) + self.params.beta_reg
            # * self.Regularization(theta_NN, epoch)
            # )

    def MSE_term(self, model: torch.Tensor, target: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        model : torch.Tensor
            _description_
        target : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # print(f"model tensor: {model}")
        # print(f"target tensor: {target}")
        mse_loss = torch.mean(torch.log(torch.abs(model / target)).square())
        writer.add_scalar("MSE Loss", mse_loss, epoch)

        print(f"mse loss: {mse_loss}")

        return mse_loss
        # return torch.mean(torch.log(torch.abs(model/ target)).square())

    def Pen2ndOrder(self, y: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        y : torch.Tensor
            _description_
        """
        logy = torch.log(torch.abs(y))
        d2logy = torch.diff(torch.diff(logy, dim=-1) / self.h1, dim=-1) / self.h2

        pen2ndorder_loss = torch.mean(torch.relu(d2logy).square())

        print(f"2nd order pen loss: {pen2ndorder_loss}")

        writer.add_scalar("2nd Order Penalty", pen2ndorder_loss, epoch)

        return pen2ndorder_loss

        # return torch.mean(torch.relu(d2logy).square())

    def Pen1stOrder(self, y: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        y : torch.Tensor
            _description_
        """
        logy = torch.log(torch.abs(y))
        d1logy = torch.diff(logy, dim=-1) / self.h1

        pen1storder_loss = torch.mean(torch.relu(d1logy).square())
        writer.add_scalar("1st Order Penalty", pen1storder_loss, epoch)

        print(f"1st order pen loss: {pen1storder_loss}")

        return pen1storder_loss
        # return torch.mean(torch.relu(d1logy).square())

    def Regularization(self, theta_NN: torch.Tensor, epoch: int):
        """_summary_

        Parameters
        ----------
        theta_NN : torch.Tensor
            _description_
        """
        reg_loss = theta_NN.square().mean()

        writer.add_scalar("Regularization", reg_loss, epoch)

        print(f"regularization loss: {reg_loss}")

        return reg_loss

        # return theta_NN.square().mean()

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
        """
        total_loss = self.MSE_term(y, y_data, epoch)  # + self.loss_func(
        # y, theta_NN, epoch
        # )

        writer.add_scalar("Total Loss", total_loss, epoch)
        return total_loss
        # return self.MSE_term(y, y_data, epoch) + self.loss_func(y, theta_NN, epoch)


# def PenTerm(y):
#     """
#     TODO: are these embedded functions necessary?
#     """
#     logy = torch.log(torch.abs(y))
#     d2logy = torch.diff(torch.diff(logy, dim=-1) / h1, dim=-1) / h2
#     # f = torch.nn.GELU()(d2logy) ** 2
#     f = torch.relu(d2logy).square()
#     # pen = torch.sum( f * h2 ) / D
#     pen = torch.mean(f)
#     return pen


# def PenTerm1stO(y):
#     """
#     TODO: are these embedded functions necessary?
#     """
#     logy = torch.log(torch.abs(y))
#     d1logy = torch.diff(logy, dim=-1) / h1
#     # d2logy = torch.diff(torch.diff(logy, dim=-1)/h1, dim=-1)/h2
#     # f = torch.nn.GELU()(d1logy) ** 2
#     f = torch.relu(d1logy).square()
#     # pen = torch.sum( f * h2 ) / D
#     pen = torch.mean(f)
#     return pen


# def RegTerm():
#     """
#     TODO: are these embedded functions necessary?
#     """
#     reg = 0
#     if self.OPS.type_EddyLifetime == "tauNet":
#         theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())
#         reg = theta_NN.square().mean()
#     return reg


# def loss_fn(model, target, weights):
#     """
#     TODO: are these embedded functions necessary?
#     """
#     # y = torch.abs((model-target)).square()

#     y = torch.log(torch.abs(model / target)).square()

#     # y = ( (model-target)/(target) ).square()
#     # y = 0.5*(y[...,:-1]+y[...,1:])
#     # loss = 0.5*torch.sum( y * h4 )
#     # loss = torch.sum( y * h1 )

#     loss = torch.mean(y)
#     return loss
