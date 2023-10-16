"""
This module implements and exposes the various terms that can be
added and scaled in a generic loss function
"""

import torch
import numpy as np


def PenTerm(y):
    """
    TODO: are these embedded functions necessary?
    """
    logy = torch.log(torch.abs(y))
    d2logy = torch.diff(torch.diff(logy, dim=-1) / h1, dim=-1) / h2
    # f = torch.nn.GELU()(d2logy) ** 2
    f = torch.relu(d2logy).square()
    # pen = torch.sum( f * h2 ) / D
    pen = torch.mean(f)
    return pen


def PenTerm1stO(y):
    """
    TODO: are these embedded functions necessary?
    """
    logy = torch.log(torch.abs(y))
    d1logy = torch.diff(logy, dim=-1) / h1
    # d2logy = torch.diff(torch.diff(logy, dim=-1)/h1, dim=-1)/h2
    # f = torch.nn.GELU()(d1logy) ** 2
    f = torch.relu(d1logy).square()
    # pen = torch.sum( f * h2 ) / D
    pen = torch.mean(f)
    return pen

def RegTerm():
    """
    TODO: are these embedded functions necessary?
    """
    reg = 0
    if self.OPS.type_EddyLifetime == "tauNet":
        theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())
        reg = theta_NN.square().mean()
    return reg


def loss_fn(model, target, weights):
    """
    TODO: are these embedded functions necessary?
    """
    # y = torch.abs((model-target)).square()

    y = torch.log(torch.abs(model / target)).square()

    # y = ( (model-target)/(target) ).square()
    # y = 0.5*(y[...,:-1]+y[...,1:])
    # loss = 0.5*torch.sum( y * h4 )
    # loss = torch.sum( y * h1 )

    loss = torch.mean(y)
    return loss