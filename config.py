

import sys
import os
import torch

path_to_robust_nets = "../robust-nets"
path_to_ellipses = os.path.join(path_to_robust_nets,"ellipses")
sys.path.append(path_to_ellipses)
def path_to_gs_params(mask_name):
    append = "_" + mask_name
    if mask_name == "radial40":
        append = "" # this is the default mask
    return os.path.join(path_to_ellipses,"results","grid_search_l1"+append,"grid_search_l1_fourier_all.pkl.cpu.pkl")

def path_to_gs_params_wavelets(mask_name):
    append = "_" + mask_name
    if mask_name == "radial40":
        append = "" # this is the default mask
    return os.path.join(path_to_ellipses,"results","grid_search_l1"+append,"grid_search_l1_wavelet_all.pkl")

path_to_resources = os.path.join("resources")
path_to_results = os.path.join("results")
path_to_figs = os.path.join(path_to_results,"figs")

if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)

if not os.path.exists(path_to_figs):
    os.makedirs(path_to_figs)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

