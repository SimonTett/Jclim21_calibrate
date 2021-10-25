"""
Plot the control SST for  the cases used in Tett et al, 2021

"""

import pandas as pd
import PaperLib
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import iris
import iris.plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import pathlib
import xarray

gmConstraint = iris.Constraint(site_number=3.0)  # constraint for g-m
def get_ts(exper, file='ts_sst.pp', dfols=False):
    """
    Read SST cube from experiment
    :param exper: experiment name to read in.
    :return:
    """

    if dfols:
        dir = PaperLib.DFOLSpath / 'DFOLS_SIMS'
    else:
        dir = PaperLib.OptClimPath / 'grl17_coupled/'

    dir = dir / f'HadCM3/ctl/{exper}/A/{exper}.000100'


    ts = PaperLib.readPP(dir, file)
    return xarray.DataArray.from_iris(ts) # return it as an dataarray


lookup = pd.read_excel('OptClim_lookup.xlsx', index_col=0)
sst = {}
ocn_temp={}
for name, row in lookup.iterrows():
    if not (name.startswith('HadAM3-7#') or name.startswith('HadAM3-DFO14#') or (name == 'Standard')):
        continue
    print(row.Status)
    if row.Status != 'Succeeded':
        continue
    dfols = name.startswith('HadAM3-DFO14#')
    exper_name = pathlib.Path(row.Control).name
    sst[name] = get_ts(exper_name, dfols=dfols)
    ocn_temp[name] = get_ts(exper_name,file='ts_ot.pp',dfols=dfols)

## now to plot the data..
fig, axis = plt.subplots(nrows=2, ncols=1,  clear=True, figsize=(8.6, 11), num='SST_VAOT_ts')
for name in sst.keys():
    (col, ls, m) = lookup.loc[name, ['Colour', 'Linestyle', 'Marker']]
    ts_sst = sst[name].sel(site_number=3).rolling(time=11,center=True).mean()
    ts_sst.plot(color=col, x='year', marker=m, ms=8, markevery=20, ax=axis[0])
    ocn_temp[name].plot(x='year', color=col, marker=m, ms=8, markevery=20, axes=axis[1])

for ax,title in zip(axis,['SST (C)','Volume Avg Temp (C)']):
    ax.axvline(40.5,color='grey',linewidth=4,alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel('C')

fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)
