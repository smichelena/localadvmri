import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pickle

from plot_utils import plot_complex, add_zoom
import config
from config_grid_attack_wavelets import (
    OpA,
    get_grid,
    spatial_grid_attack,
    select_loc,
    get_ratios,
    reconstruct_full,
    reconstruct_batch_dims,
)


###### Define networks - comment if not trained ########
# from network_utils import get_tiramisu, get_tiramisu_ee
# tira = get_tiramisu()
# tira_ee = get_tiramisu_ee()
########################################################


X = pickle.load(open(os.path.join(config.path_to_resources, "ell_example.p"), "rb"))
# create complex tensor
X = torch.cat((X, torch.zeros_like(X)), dim=-3)

N_small = 3
X = X[0:3].to(config.device)
Y = OpA(X).to(config.device)

print("shape of big X: ", X.shape)
print("shape of big Y: ", Y.shape)

noise_rel = 0.05
# try only two locations with a fixed y-coordinate.
n_locs = (1, 2)
width = 8

method = "l1"
# method = tira
# method = tira_ee

Yadv = spatial_grid_attack(
    Y, noise_rel, width=width, n_locs=n_locs, iter=1, method=method
)

# Some extra processing needed due to wavelet operator implementation

Yadv = Yadv.squeeze(0)
Yadv = torch.unbind(Yadv, dim=0)
out = []
for i in range(len(Yadv)):
    temp = torch.squeeze(Yadv[i], dim=0)
    x, _ = reconstruct_full(temp, noise_rel)
    out.append(x.unsqueeze(0))

Yadv = torch.stack(Yadv, dim=0)
Yadv = Yadv.unsqueeze(0)

if method == "l1":
    Xadv = torch.stack(out, dim=1)
else:
    Xadv = reconstruct_batch_dims(Yadv, method)


# =======================================================

idx = select_loc(Xadv - X, p=np.inf)

Xadv = Xadv[idx]
Yadv = Yadv[idx]

R = OpA.adj((Yadv - Y))  # perturbation orthogonal to ker(OpA)
Rho = Xadv - X  # reconstruction artifact

Xpert = X + R
Xpert = Xpert.detach().cpu()

Xadv = Xadv.detach().cpu()
Yadv = Yadv.detach().cpu()
X = X.detach().cpu()
R = R.detach().cpu()
Rho = Rho.detach().cpu()

ratios = get_ratios(Rho, R)
print("Ratio of inf norms of artifacts to perturbations:")
print(ratios.numpy())

N_plots = N_small
fig, axs = plt.subplots(N_plots, 3, sharex=True, sharey=True, figsize=(12, 3 * N_plots))
axs[0, 0].set_title("$X$")
axs[0, 1].set_title(r"$X+R$")
axs[0, 2].set_title(r"$\mathrm{rec}(A(X+R))$")
grid = get_grid(n_locs, X.shape[-2:])
for im_no, ax in enumerate(axs):
    plot_complex(X[im_no], ax=ax[0])
    plot_complex(Xpert[im_no], ax=ax[1])
    plot_complex(Xadv[im_no], ax=ax[2])
    loc = grid[idx[0][im_no], idx[1][im_no]]
    add_zoom(Xpert[im_no], ax[1], loc, width)
    add_zoom(Xadv[im_no], ax[2], loc, width)
plt.suptitle("Grid attack demo using %.1f%% relative noise" % (100 * noise_rel))
path_fig = os.path.join(config.path_to_results, "demo_grid_attack_wavelets.pdf")
plt.savefig(path_fig, dpi=X.shape[-1])
print("Figure saved in: " + path_fig)
