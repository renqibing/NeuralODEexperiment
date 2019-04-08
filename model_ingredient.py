"""Ingredient for making a ODEnet model for MNIST"""

import torch
from torch import nn
from sacred import Ingredient
from examples.module_updated import ODEnet,ODEnetRandTime
# from examples.module_updated import ResBlock,ResNet
from examples.resnet_18 import ResidualBlock,ResNet

model_ingredient = Ingredient('model')

@model_ingredient.config
def model_config():
    """Config for model"""
    a_tol = 1e-6
    r_tol = 1e-6
    min_end_time = 1
    max_end_time = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    isgpu = True
    isrand = False
@model_ingredient.capture
def make_model(a_tol,
               r_tol,
               min_end_time,
               max_end_time,
               device,
               isgpu,
               isrand,
               _log):
    """Create ODEnet model from config"""
    if isrand:
        ode_model = ODEnetRandTime(min_end_time= min_end_time, max_end_time= max_end_time,
                                   atol= a_tol, rtol= r_tol)
    else:
        ode_model = ODEnet(atol= a_tol,
                           rtol=r_tol)
    # if isinstance(device, list):
    #     model = nn.DataParallel(ode_model, device_ids=device).to(device[0])
    # else:
    #     model = ode_model.cuda()
    if isgpu :
        model = ode_model.cuda()
    else:
        model = ode_model.cpu()

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = len(params)
    _log.info("Created ODEnetRandTime model with {num_params} parameters \
    on 'cuda'".format(num_params = num_params))

    ode_params = torch.nn.utils.parameters_to_vector(
        ode_model.odefunc.parameters()).shape[0]
    _log.info("ODE function has {ode_params} parameters".format(ode_params = ode_params))
    return model

@model_ingredient.config
def resnet_config():
    layers = [2,2,2,2]
    isgpu = True

@model_ingredient.capture
def make_resnet(layers,isgpu,_log):
    if isgpu:
        resnet = ResNet(ResidualBlock).cuda()
    else:
        resnet = ResNet(ResidualBlock)
    resnet_params = torch.nn.utils.parameters_to_vector(resnet.parameters()).shape[0]
    _log.info("Resnet-18 has {} parameters".format(resnet_params))
    return resnet






