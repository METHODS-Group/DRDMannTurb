import torch
from fracturbulence import tauNet
from torchview import draw_graph
from torchviz import make_dot

torch.set_default_tensor_type("torch.cuda.FloatTensor")

if __name__ == "__main__":
    config = {"hlayers": [2, 4], "nModes": 10, "learn_nu": False}
    model = tauNet.tauResNet(**config)

    x = torch.ones((3,)).double()
    y = model(x)

    # make_dot(y, params=dict(list(model.named_parameters()))).render('resnet_basic', format='png')

    model_graph = draw_graph(
        model, input_size=(3,), expand_nested=True, dtypes=[torch.float64]
    )
    model_graph.visual_graph.render(filename="resnet_arch", format="png")
