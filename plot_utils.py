import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino"],
    'mathtext.fontset': 'stix',
    "pdf.fonttype": 42
})

def modulus(x):
    # x has shape (...,2,n,n)
    return (x**2).sum(dim=-3).sqrt()

def reduce_complex(x,reduction):
    if reduction is None:
        # then x is not complex and should have shape (n,n)
        return x
    elif reduction == "modulus":
        return modulus(x)
    elif reduction == "real":
        return x[...,0,:,:]
    elif reduction == "imag":
        return x[...,1,:,:]
    else:
        raise ValueError("reduction \"%s\" not implemented"%str(reduction))

def plot_complex(x,ax=None,vmin=0,vmax=1,reduction="modulus",cmap="gray",**kwargs):
    """
    x has shape (2,n,n) or (n,n) (if reduction is None)
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,sharex=True,sharey=True)
    x = reduce_complex(x,reduction)
    if vmin == "min":
        vmin = x.min()
    if vmax == "max":
        vmax = x.max()
    # find a better way to deal with norm vs vmin/vmax
    if "norm" in kwargs.keys():
        ax.imshow(x,cmap=cmap,origin="lower",interpolation=None,**kwargs)
    else:
        ax.imshow(x,vmin=vmin,vmax=vmax,cmap=cmap,origin="lower",interpolation=None,**kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def add_zoom(x,ax,loc,width,**kwargs):
    """
    x has shape (2,n,n)
    """
    n = x.shape[-1]
    radius = 1.1*width/2.

    # the zoomed view should fill less than a quarter of the image
    zoom = n/(4.4*radius)
    # if little or no zoom, don't bother
    if zoom < 1.2:
        print("Box too wide. No zoom added.")
        return

    locx, locy = loc
    x1, x2 = max(locx-radius,0), min(locx+radius,n)
    y1, y2 = max(locy-radius,0), min(locy+radius,n)
    # set zoomloc appropriately
    if locx <= n/2:
        if locy <= n/2:
            # upper right
            zoomloc = 1
            con1 = 2
            con2 = 4
        else:
            # lower right
            zoomloc = 4
            con1 = 1
            con2 = 3
    else:
        if locy <= n/2:
            # upper left
            zoomloc = 2
            con1 = 1
            con2 = 3
        else:
            # lower left
            zoomloc = 3
            con1 = 2
            con2 = 4
    axins = zoomed_inset_axes(ax, zoom, loc=zoomloc)
    plot_complex(x,ax=axins,**kwargs)
    # sub region of the original image
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.spines['right'].set_color('red')
    axins.spines['left'].set_color('red')
    axins.spines['top'].set_color('red')
    axins.spines['bottom'].set_color('red')

    # pink: "#ff908c"
    mark_inset(ax, axins, loc1=con1, loc2=con2, fc="none", ec="#ff908c")

    return axins

def combine_cmaps(cmap1,cmap2,vmin=0,vmax=1,vsplit=0.5):
    cmap1 = cm.get_cmap(cmap1)
    cmap2 = cm.get_cmap(cmap2)
    n = 256
    n1 = int( (vsplit-vmin)/(vmax-vmin)*256 )
    n2 = n - n1
    x1 = np.linspace(0,1,n1)
    x2 = np.linspace(0,1,n2)
    colors = np.vstack((cmap1(x1),cmap2(x2)))
    return ListedColormap(colors)


# https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

# not in use:
def add_clipping(x,ax,lower_lim=0.0,upper_lim=1.0,vmax=None,vmin=None,reduction="modulus",cmap=None,**kwargs):

    x = reduce_complex(x,reduction)

    x_hi = np.ma.masked_where(x <= upper_lim, x)
    x_lo = np.ma.masked_where(x >= lower_lim, x)

    if vmax is None:
        vmax = x_hi.max()
    if vmin is None:
        vmin = x_lo.min()

    if not cmap is None:
        cmap_hi = cmap
        cmap_lo = cmap
    else:
        cmap_hi = "Reds"
        cmap_lo = "winter_r"


    if not x_hi.mask.all():
        plot_complex(x_hi,ax=ax,vmin=upper_lim,vmax=vmax,reduction=None,cmap=cmap_hi,**kwargs)
    if not x_lo.mask.all():
        plot_complex(x_lo,ax=ax,vmin=vmin,vmax=lower_lim,reduction=None,cmap=cmap_lo,**kwargs)












