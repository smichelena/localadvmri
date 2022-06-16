import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pickle
import pandas as pd

import config
from config_grid_attack import (OpA, spatial_grid_attack,
                                select_loc, get_ratios,
                                reconstruct_full, mask_name,
                                reconstruct_batch_dims
                                )

from data_management import IPDataset
from operators import unprep_fft_channel

###### Define networks - comment if not trained ########
from network_utils import get_tiramisu, get_tiramisu_ee
tira = get_tiramisu().to(config.device)
tira_ee = get_tiramisu_ee().to(config.device)

method = tira
method_name = "tira"
########################################################

print("Using device: " + str(config.device))

# Prepare data
batch_size = 5
torch.manual_seed(0)
test_data = IPDataset("test", os.path.join(config.path_to_ellipses,"raw_data"))
test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

# stop after this many batches
n_batches = 1

Noise_Rel = [0.0, 0.01] # test experiment
Noise_Rel = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1] # extended experiment
attack_iter = 30
width_loc = 10
n_locs = (8,8)
exp_name = "exp_grid_attack_%s_quant_%s_width%d"%(method_name,mask_name,int(width_loc))
results_path = os.path.join(config.path_to_results,exp_name+".p")

if os.path.exists(results_path):
    results = results = pd.read_pickle(results_path)
    Noise_Rel = np.setdiff1d(Noise_Rel,results.index.to_numpy())
    for noise_rel in Noise_Rel:
        results.loc[noise_rel] = np.nan
    results.loc[Noise_Rel,"width_loc"] = width_loc
    results.loc[Noise_Rel,"attack_iter"] = attack_iter
    results.sort_index(inplace=True)
else:
    results = pd.DataFrame(columns=["noise_rel","Xrec","Yadv","Xadv","Rho0","sel_locs","width_loc","n_locs","attack_iter"])
    results.noise_rel = Noise_Rel
    results = results.set_index("noise_rel")
    results.width_loc = width_loc
    results.attack_iter = attack_iter


for i_batch, batch in enumerate(test_loader):
    if i_batch >= n_batches:
        break
    print("\nPerturbing batch %d"%(i_batch))
    # create complex tensor
    X = torch.cat( (batch[0], torch.zeros_like(batch[0])), dim=-3 ).to(config.device)
    Y = OpA(X)
    for noise_rel in Noise_Rel:
        print("\n\t at %.1f%% relative noise."%(noise_rel*100))
        Xrec = method(Y)
        if noise_rel > 0:
            Yadv = spatial_grid_attack(Y,noise_rel,width=width_loc,n_locs=n_locs,iter=attack_iter,method=method)
            Xadv = reconstruct_batch_dims(Yadv,method)
            # Select the perturbation with the largest (in inf-norm) reconstruction artifact.
            idx = select_loc(Xadv - X,p=np.inf)
            sel_locs = torch.stack((idx[0],idx[1]),dim=-1)
            # The coordinates of loc for image number im_no can be retrieved as follows:
            # grid = get_grid(n_locs,X.shape[-2:])
            # loc = grid[tuple(sel_locs[im_no])]

            # discard less interesting perturbations
            Yadv = Yadv[idx]
            Xadv = Xadv[idx]
            Pert = Yadv - Y
            Rho0 = method(Pert)[0] # reconstruction artifact for zero measurements
        else:
            Yadv = Y
            Xadv = Xrec
            Rho0 = torch.zeros_like(X)
            sel_locs = torch.zeros(len(X),2)
        # Save as cpu tensors for portability.
        # Use at[] instead of loc[] to avoid ValueError.
        if i_batch == 0:
            results.at[noise_rel,"n_locs"] = n_locs
            results.at[noise_rel,"Xrec"] = Xrec.cpu()
            results.at[noise_rel,"Xadv"] = Xadv.cpu()
            results.at[noise_rel,"Yadv"] = Yadv.cpu()
            results.at[noise_rel,"Rho0"] = Rho0.cpu()
            results.at[noise_rel,"sel_locs"] = sel_locs.cpu()
        else:
            results.at[noise_rel,"n_locs"] = n_locs
            results.at[noise_rel,"Xrec"] = torch.cat( (results.at[noise_rel,"Xrec"], Xrec.cpu()), 0 )
            results.at[noise_rel,"Xadv"] = torch.cat( (results.at[noise_rel,"Xadv"], Xadv.cpu()), 0 )
            results.at[noise_rel,"Yadv"] = torch.cat( (results.at[noise_rel,"Yadv"], Yadv.cpu()), 0 )
            results.at[noise_rel,"Rho0"] = torch.cat( (results.at[noise_rel,"Rho0"], Rho0.cpu()), 0 )
            results.at[noise_rel,"sel_locs"] = torch.cat( (results.at[noise_rel,"sel_locs"], sel_locs.cpu()), 0 )

    results.to_pickle(results_path)

