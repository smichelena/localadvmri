import os
import torch
import pickle
import config
from networks import IterativeNet, Tiramisu
from operators import (
    Fourier,
    LearnableInverter,
    RadialMaskFunc,
    unprep_fft_channel,
)

path_results = os.path.join(config.path_to_ellipses,"results")
path_tiramisu = os.path.join(path_results,"Fourier_Tiramisu_jitter_v3_train_phase_2")
path_tiramisu_ee = os.path.join(path_results,"Fourier_Tiramisu_ee_jitter_v7_train_phase_2")

# from config_robustness_fourier (can't be imported if some networks are missing)
def _load_net(path, subnet, subnet_params, it_net_params):
    subnet = subnet(**subnet_params).to(config.device)
    it_net = IterativeNet(subnet, **it_net_params).to(config.device)
    it_net.load_state_dict(torch.load(path, map_location=torch.device(config.device)))
    it_net.freeze()
    it_net.eval()
    return it_net

mask_name = "radial40"
mask = pickle.load(open(os.path.join(config.path_to_resources,"mask_%s.p"%mask_name),"rb"))
mask = mask.to(config.device)
mask = mask.reshape((1,1)+mask.shape) # add batch and complex dimensions
n = mask.shape[-1]
OpA = Fourier(mask)

tiramisu_params = {
    "in_channels": 2,
    "out_channels": 2,
    "drop_factor": 0.0,
    "down_blocks": (5, 7, 9, 12, 15),
    "up_blocks": (15, 12, 9, 7, 5),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 20,
    "growth_rate": 16,
    "out_chans_first_conv": 16,
}

tiramisu_it_net_params = {
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "operator": OpA,
    "inverter": LearnableInverter((n,n), mask, learnable=False),
}

tiramisu_ee_it_net_params = {
    "num_iter": 1,
    "lam": 0.0,
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "operator": OpA,
    "inverter": LearnableInverter((n,n), mask, learnable=True),
}


def get_tiramisu():
    return _load_net(os.path.join(path_tiramisu,"model_weights.pt"),
                Tiramisu,
                tiramisu_params,
                tiramisu_it_net_params)

def get_tiramisu_ee():
    return _load_net(os.path.join(path_tiramisu_ee,"model_weights.pt"),
                Tiramisu,
                tiramisu_params,
                tiramisu_ee_it_net_params)

