import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pickle
import pandas as pd

import config
from plot_utils import plot_complex
from operators import Fourier, Wavelet
from reconstruction_methods import admm_l1_rec


###### Define networks - comment if not trained ########
#from network_utils import get_tiramisu, get_tiramisu_ee
#tira = get_tiramisu()
#tira_ee = get_tiramisu_ee()
########################################################

torch.manual_seed(0)

mask_name = "radial40"
mask = pickle.load(open(os.path.join(config.path_to_resources,"mask_%s.p"%mask_name),"rb"))
mask = mask.to(config.device)
mask = mask.reshape((1,1)+mask.shape) # add batch and complex dimensions

n = mask.shape[-1]
m = mask.bool().sum().item()

X = pickle.load(open(os.path.join(config.path_to_resources,"ell_example.p"),"rb"))
# create complex tensor
X = torch.cat( (X, torch.zeros_like(X)), dim=-3 )

OpA = Fourier(mask)
OpW = Wavelet((n,n),device=config.device, level=4)

gs_params = pd.read_pickle(config.path_to_gs_params(mask_name))
def _get_gs_param(noise_rel):
    idx = (gs_params.noise_rel - noise_rel).abs().to_numpy().argmin()
    return gs_params.grid_param[idx]["lam"], gs_params.grid_param[idx]["rho"]

rec_iterations = 150

def reconstruct(y, noise_rel):
    lam, rho = _get_gs_param(noise_rel)
    x, _ = admm_l1_rec(
        y,
        OpA,
        OpW,
        OpA.adj(y),
        OpW(OpA.adj(y)),
        lam,
        rho,
        iter=rec_iterations,
        silent=False,
    )
    return x


Y = OpA(X).to(config.device)
X = X.to(config.device)
noise_rel = 0.05
method = "l1"
# method = tira
# method = tira_ee
if method == "l1":
    Xhat = reconstruct(Y,noise_rel)
else:
    Xhat = method(Y)
R = Xhat - X

# should be <= noise_lvl
err = (OpA(Xhat) - Y).detach().cpu()
err = err.flatten(start_dim=1).norm(dim=-1)
Y = Y.detach().cpu()
relerr = err/Y.norm(dim=(-2,-1))
print("Relative errors: ")
print(relerr.cpu().numpy())

N_plots = 3
fig, axs = plt.subplots(N_plots,3,sharex=True,sharey=True,figsize=(12,3*N_plots))
axs[0,0].set_title("$X$")
axs[0,1].set_title(r"$\hat{X}=\mathrm{rec}(X)$")
axs[0,2].set_title(r"$\hat{X}-X$ (scaled)")
for im_no, ax in enumerate(axs):
    plot_complex(X[im_no].detach().cpu() ,ax=ax[0])
    plot_complex(Xhat[im_no].detach().cpu() ,ax=ax[1])
    plot_complex(R[im_no].detach().cpu() ,ax=ax[2],vmax="max")
if method == "l1":
    plt.suptitle("Reconstruction demo using %d iterations"%(rec_iterations))
else:
    plt.suptitle("Reconstruction demo using DNN")
path_fig = os.path.join(config.path_to_results,"demo_reconstruction_wavelet.pdf")
plt.savefig(path_fig,dpi=n)
print("Figure saved in: " + path_fig)




