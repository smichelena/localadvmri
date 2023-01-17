import os
import numpy as np
import torch
import pickle
import pandas as pd

import config

# import
from find_adversarial import PAdam, untargeted_attack
from operators import Fourier, Wavelet, rotate_real, proj_l2_ball, im2vec
from reconstruction_methods import admm_l1_rec

torch.manual_seed(1)

mask_name = "radial40"
mask = pickle.load(
    open(os.path.join(config.path_to_resources, "mask_%s.p" % mask_name), "rb")
)
mask = mask.to(config.device)
mask = mask.reshape((1, 1) + mask.shape)  # add batch and complex dimensions

# dimensions
n = mask.shape[-1]
m = mask.bool().sum().item()

OpA = Fourier(mask)
OpW = Wavelet((n, n), device=config.device, level=4)

# ------ reconstruction method ----------

# Need far less iterations for speed

rec_full_iter = 10
rec_adv_iter = 5
rec_init_iter = 5
perturb_iter = 10
attack_iter = 5

gs_params = pd.read_pickle(config.path_to_gs_params_wavelets(mask_name))


def _get_gs_param(noise_rel):
    if torch.is_tensor(noise_rel):
        noise_rel = noise_rel.numpy()
    idx = (gs_params.noise_rel - noise_rel).abs().to_numpy().argmin()
    return gs_params.grid_param[idx]["lam"], gs_params.grid_param[idx]["rho"]


def reconstruct_full(y, noise_rel, iter=rec_full_iter):
    lam, rho = _get_gs_param(noise_rel)
    return admm_l1_rec(
        y,
        OpA,
        OpW,
        OpA.adj(y),
        OpW(OpA.adj(y)),
        lam,
        rho,
        iter=iter,
        silent=False,
    )


def reconstruct_adv(y, lam, rho, x0, z0, iter=rec_adv_iter):
    x, _ = admm_l1_rec(
        y,
        OpA,
        OpW,
        x0,
        z0,
        lam,
        rho,
        iter=iter,
        silent=True,
    )
    return x


# ------ attack method ----------

MSELoss = torch.nn.MSELoss(reduction="sum")


def complexMSE(ref, pred):
    """
    Requires ref and pred to be of shape (...,2,n1,n2).
    """
    return MSELoss(rotate_real(ref)[:, 0:1, ...], rotate_real(pred)[:, 0:1, ...])


def perturb(y0, noise_rel, loss=complexMSE, iter=perturb_iter, init=None, method="l1"):
    """
    y0 has shape (...,2,m)
    """
    if noise_rel == 0.0:
        return y0

    if init is None:
        if method == "l1":
            # calculate initial guess for reconstruct_adv()
            x0_adv, z0_adv = reconstruct_full(y0, noise_rel, iter=rec_init_iter)
        else:
            x0_adv = method(y0)
    else:
        x0_adv, z0_adv = init

    if method == "l1":
        lam, rho = _get_gs_param(noise_rel)
        method = lambda y: reconstruct_adv(y, lam, rho, x0_adv, z0_adv)

    noise_lvl = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    adv_init_fac = 3.0 * noise_lvl
    adv_param = {
        "codomain_dist": loss,
        "domain_dist": None,
        "mixed_dist": None,
        "weights": (1.0, 1.0, 1.0),
        "optimizer": PAdam,
        "projs": [lambda y: proj_l2_ball(y, y0, noise_lvl)],
        "iter": iter,
        "stepsize": 5e0,
    }
    yadv = y0.clone().detach() + (
        adv_init_fac / np.sqrt(np.prod(y0.shape[-2:]))
    ) * torch.randn_like(y0)
    yadv.requires_grad_(True)

    yadv = untargeted_attack(method, yadv, y0, t_out_ref=x0_adv, **adv_param).detach()

    return yadv


class LocalLoss:
    def __init__(self, shape, loc, width, loss=complexMSE, device="cpu"):
        """
        LL = LocalLoss(shape,loc,width) takes as input tensors X
        such that X.shape == (...,shape[0],shape[1]).

        loc is expressed in (x,y) coordinates, i.e.
            loc[0] relates to the column dimension of X: X.shape[-1],
            loc[1] relates to the row dimension of X: X.shape[-2],

        """
        rows, cols = shape
        x = torch.arange(0.0, cols)
        y = torch.arange(0.0, rows)
        yy, xx = torch.meshgrid(y, x)
        dist2 = (xx - loc[0]) ** 2 + (yy - loc[1]) ** 2
        self.locmask = (dist2 <= (width / 2.0) ** 2).to(device)
        if self.locmask.sum() < 1:
            print("Warning: localizing on empty slice.")
        self.loss = loss

    def __call__(self, x, y):
        flatmask = im2vec(self.locmask)
        # Localize x and y according to self.locmask.
        # The result is a "flat" tensor (ignoring complex and batch dimensions),
        # of shape (...,k), k=self.locmask.sum().
        # The flat tensor is artifically turned into a 2-D tensor of shape (...,k,1) in order
        # to make it compatible with complexMSE (or any image-specific complex loss function).
        x = im2vec(x)[..., flatmask].unsqueeze(-1)
        y = im2vec(y)[..., flatmask].unsqueeze(-1)
        return self.loss(x, y)


def get_grid(n_locs, shape):
    """
    n_locs = (N1,N2)
    shape = (n1,n2)

    grid = get_grid(n_locs,shape) a tensor of shape (N1,N2,2),
    representing a regular grid for [0,n2-1]*[0,n1-1].

    NOTE: loc=grid[i,j] is interpreted such that loc[0] is the x-coordinate
    (tensor columns) and loc[1] is the y-coordinate (tensor rows)!
    Indexing is thus reversed.
    """
    grid1 = (torch.arange(0, n_locs[0]) + 0.5) * shape[0] / n_locs[0]
    grid2 = (torch.arange(0, n_locs[1]) + 0.5) * shape[1] / n_locs[1]
    return torch.stack(torch.meshgrid(grid1, grid2), -1).flip(-1)


def spatial_grid_attack(y0, noise_rel, width=1, n_locs=(1, 1), iter=attack_iter, method="l1"):
    device = y0.device
    shape = OpA.n
    if method == "l1":
        init = reconstruct_full(y0, noise_rel, iter=rec_init_iter)
    else:
        init = (method(y0), None)
    grid = get_grid(n_locs, shape)
    Yadv = torch.zeros(n_locs + y0.shape, device=device)
    for ii in range(n_locs[0]):
        for jj in range(n_locs[1]):
            print(
                "Perturbing at location\t(%d,%d) / (%d,%d)."
                % ((ii + 1, jj + 1) + n_locs)
            )
            loc = grid[ii, jj]
            locLoss = LocalLoss(shape, loc, width, device=device)
            Yadv[ii, jj] = perturb(
                y0, noise_rel, loss=locLoss, iter=iter, init=init, method=method
            )
    return Yadv


def spike_attack(y0, noise_rel, n_locs=(1, 1)):
    device = y0.device
    shape = OpA.n
    spike_grid = torch.zeros(*n_locs, 2, *shape, device=device)
    grid = get_grid(n_locs, shape).round().to(int)
    for ii in range(n_locs[0]):
        for jj in range(n_locs[1]):
            loc = grid[ii, jj]
            # spike only in real part
            spike_grid[ii, jj, 0, loc[1], loc[0]] = 1
    pert_grid = OpA(spike_grid)
    pert_grid = pert_grid / pert_grid.norm(p=2, dim=(-2, -1), keepdim=True)

    noise_lvl = noise_rel * y0.norm(p=2, dim=(-2, -1), keepdim=True)
    # two new dims for grid shape
    scaled_pert_grid = noise_lvl[..., None, None] * pert_grid
    Yadv = y0[..., None, None, :, :] + scaled_pert_grid

    # complicated way to get grid dimensions in fornt of unknown batch dimensions
    dims = range(Yadv.ndim)
    return Yadv.permute(dims[-4], dims[-3], *dims[:-4], -2, -1)


def select_loc(R, p=np.inf):
    """
    idx = select_loc(R,p)

    R has shape (N1,N2,b,2,n1,n2)
    For each k=0,...,b-1 finds the i,j in {0,...,N1-1}*{0,...,N2-1}
    which maximizes |R[i,j,k]|_p.
    The selected entries are R[idx].
    """
    # first calculate the modulus of R,
    modulus = (R**2).sum(-3).sqrt()  # faster than R.norm(p=2,dim=-3)
    # then the p-norm of the modulus
    norms = modulus.norm(p=p, dim=(-2, -1))
    N1, N2, b = norms.shape
    maxnorms, ind = norms.flatten(end_dim=1).max(dim=0)
    # invert torch.flatten
    ind1 = ind // N2
    ind2 = ind % N2
    return (ind1, ind2, range(b))


def complex_norm(X, p=np.inf):
    return (X**2).sum(-3).sqrt().norm(p=p, dim=(-2, -1))


def get_ratios(R1, R2, p=np.inf):
    norms1 = complex_norm(R1, p=p)
    norms2 = complex_norm(R2, p=p)
    return norms1 / norms2


def reconstruct_batch_dims(Y, method, batch_size=5):
    """
    Y has shape (...,2,m)
    X should have shape (...,2,n,n)
    """
    Yflat = Y.reshape((-1,) + Y.shape[-2:])
    X = []
    n = len(Yflat)
    i = 0
    while i < n:
        X.append(method(Yflat[i : i + batch_size]))
        i += batch_size
    X = torch.cat(X, dim=0)
    return X.reshape(Y.shape[:-2] + X.shape[-3:])
