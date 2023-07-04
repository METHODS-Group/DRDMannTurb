
import torch
import torch.nn as nn

from .LearnableFunctions import Rational, SimpleNN, CustomMLP

"""
==================================================================================================================
Learnable Eddy Liftime class
==================================================================================================================
"""

class ResNetBlock(nn.Module): 
    def __init__(self, inlayer=3, outlayer=3): 
        super(ResNetBlock, self).__init__() 

        self.fc1 = nn.Sequential(
            nn.Linear(inlayer, outlayer, bias=False).double(), 
            # nn.LayerNorm(outlayer), 
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(outlayer, outlayer, bias=False).double(), 
            # nn.LayerNorm(outlayer),             
            # nn.ReLU()
        )

        self.outlayer = outlayer
        self.relu = nn.ReLU()


    def _forward_impl(self, x): 
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        # print(f"@@@@BLOCK forward_impl type(x) = {type(x)}")

        residual = x 
        # print(f"@@@@BLOCK forward_impl type(residual) = {type(residual)}")

        output = self.fc1(x) 
        # print(f"@@@@BLOCK forward_impl after fc1 type(output) = {type(output)}")

        output = self.fc2(output)
        # print(f"@@@@BLOCK forward_impl after fc2 type(output) = {type(output)}")

        output += residual 
        output = self.relu(output)
        # output = nn.ReLU()(output) 

        return output 

    def forward(self, x): 

        return self._forward_impl(x) 


class ResNet(nn.Module): 
    def __init__(self, hlayers, inlayer=3, outlayer=3) -> None:
        super(ResNet, self).__init__()
        self.indims = 10 # not of the data but after the first layer upward

        # this serves as a substitute for the initial conv 
        # present in resnets for image-based tasks
        self.layer0 = nn.Sequential(                    
            nn.Linear(inlayer, self.indims, bias=False).double(), 
            nn.ReLU()
        )

        # TODO: need to downsample if not 4...??????
        self.block1 = self._make_layer(hlayers[0], self.indims)
        self.block2 = self._make_layer(hlayers[1], self.indims)

        self.fc = nn.Linear(self.indims, outlayer).double()

    def _make_layer(self, blocks, indims): 
        layers = [] 
        layers.append(ResNetBlock(self.indims, indims))

        self.indims = indims

        for _ in range(1, blocks): 
            layers.append(ResNetBlock(inlayer=self.indims, outlayer=indims))

        return nn.Sequential(*layers) 

    def _forward_impl(self, x): 
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # print(f"@@@@RESNSET forward_impl type(x) = {type(x)}")
        x = self.layer0(x) 
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.fc(x) 

        return x 

    def forward(self, x): 
        return self._forward_impl(x)


class tauResNet(nn.Module): 
    def __init__(self, **kwargs): 
        super(tauResNet, self).__init__() 

        self.hlayers = kwargs.get('hlayers', [10, 10])
        # self.activations = kwargs.get('activations', [nn.ReLU(), nn.ReLU()])

        self.nModes            = kwargs.get('nModes', 10)
        self.fg_learn_nu       = kwargs.get('learn_nu', True)

        # TODO: change activations list here and propagate through to resnet blocks
        # self.NN = CustomMLP(hlayers=self.hlayers, activations=self.activations, inlayer=3, outlayer=3)
        self.NN = ResNet(hlayers=self.hlayers, inlayer=3, outlayer=3)
        self.Ra = Rational(nModes=self.nModes, learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()


    def sym(self, f, k):
        return 0.5*(f(k) + f(k*self.sign))


    def forward(self, k):
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau   = self.Ra(k_mod)

        return tau


class tauNet(nn.Module):
    def __init__(self, **kwargs):
        super(tauNet, self).__init__()        
        self.nlayers = kwargs.get('nlayers', 2)
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 3)

        self.nModes            = kwargs.get('nModes', 10)
        self.fg_learn_nu       = kwargs.get('learn_nu', True)

        self.NN = SimpleNN(nlayers=self.nlayers, inlayer=3, hlayer=self.hidden_layer_size, outlayer=3)
        self.Ra = Rational(nModes=self.nModes, learn_nu=self.fg_learn_nu)

        # self.T = nn.Linear(3,3,bias=False).double()
        

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(self, f, k):
        return 0.5*(f(k) + f(k*self.sign))


    def forward(self, k):
        # NN    = self.sym(self.NN, k)
        # NN    = self.NN(k**2)
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau   = self.Ra(k_mod)
        return tau

        # k2 = k.norm(dim=-1,keepdim=True).square()
        # l = k/k2
        # tau = self.NN(l) + self.NN(l*self.sign)
        # # k_mod = k.norm(dim=-1) + NN.squeeze(-1)
        # # tau   = self.Ra(k_mod)
        # return tau.squeeze(-1)

        # k_mod = self.T(k).norm(dim=-1)
        # tau   = self.Ra(k_mod)
        # return tau

class customNet(nn.Module): 
    def __init__(self, **kwargs): 
        super().__init__() 

        self.hlayers = kwargs.get('hlayers', [10, 10])
        self.activations = kwargs.get('activations', [nn.ReLU(), nn.ReLU()])

        self.nModes            = kwargs.get('nModes', 10)
        self.fg_learn_nu       = kwargs.get('learn_nu', True)

        self.NN = CustomMLP(hlayers=self.hlayers, activations=self.activations, inlayer=3, outlayer=3)
        self.Ra = Rational(nModes=self.nModes, learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(self, f, k):
        return 0.5*(f(k) + f(k*self.sign))

    def forward(self, k):
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau   = self.Ra(k_mod)
        return tau
    
