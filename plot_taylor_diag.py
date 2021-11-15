"""
Q: Really needed


Plot taylor diagrams for CMIP6, CMIP5 & Calibrated HadAM3


Outline of what is needed
1) Regrid   CMIP5 & CMIP6 data to N48 grid. Data has already been regridded to N216. cdo remapcon
2) Regrid  obs data to N48 grid.
3) Loop over each variable    OLR, RSR, Land-Tas>= 60S, Land-Pr >=60S , SLP, RH@500, T@500 --- 7 variables.
    Read in obs field
    Remove area avg from obs field
    Compute obs field stdev (field - area_avg)
    Remove  area mean from field.
    Compute field stddev and ratio with obs stdev.
    Compute correlation between field and obs

    But looks like there is a package (skillMetrics) that could do much of this. Though still need to regrid & mask,
    No it does not look like it does any area weighting... So will not bother using it. stastsmodels would let me do that.

For DFOLS cases have the data.
For CE7 will need to reuse old values..
"""
import PaperLib
import string
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

CMIP6_taylor = pd.read_csv(PaperLib.dataPath / 'CMIP6_taylor.csv', header=[0], index_col=[0])
CMIP5_taylor = pd.read_csv(PaperLib.dataPath / 'CMIP5_taylor.csv', header=[0], index_col=[0])
CMIP6_taylor.loc[:, 'Ensemble'] = 'CMIP6'
CMIP6_taylor.loc[:,'zorder'] = 2.0
CMIP6_taylor.loc['CMIP6-MM','zorder']=0
CMIP5_taylor.loc[:, 'Ensemble'] = 'CMIP5'
CMIP5_taylor.loc[:,'zorder'] = 1.0
CMIP5_taylor.loc['CMIP5-MM','zorder']=0
# get in the taylor info  for DFOLS

DFOLS_taylor = pd.read_csv(PaperLib.dataPath / 'DFOLS_taylor.csv', header=[0], index_col=[0])
DFOLS_taylor.loc[:, 'Ensemble'] = 'DF14'
DFOLS_taylor.loc[:,'zorder'] = 10
DFOLS_taylor.loc['DFOLS-MM','zorder'] = 0
# and the standard values
Standard_taylor = pd.read_csv(PaperLib.dataPath / 'Standard_taylor.csv', header=[0], index_col=[0]).T
Standard_taylor.loc[:,'Ensemble']='ICE'
Standard_taylor.loc[:,'zorder'] =20
df_all = pd.concat([DFOLS_taylor, CMIP5_taylor, CMIP6_taylor,Standard_taylor])
## plot the DOLFS values
fig = plt.figure('taylor_diag', figsize=[7, 5], clear=True)
taylor_diags = []

pinfo=(
        (dict(air_pressure_at_mean_sea_level='^'), 'Surface', 221),
        (dict(air_temperature_tas='*',precipitation_flux='D'),'Land Surface',222),
        (dict(relative_humidity='s',air_temperature_500='*'),'Mid Troposphere',223),
        (dict(toa_outgoing_longwave_flux='D',toa_outgoing_shortwave_flux='v'),'TOA Radiation',224)
)

for info in pinfo:
    dia = PaperLib.TaylorDiagram(rect=info[-1], sdConts=[0.75, 1.0, 1.25],
                                 theta_range=[0,np.arccos(0.7)],sd_range=[0.0,1.25])
    taylor_diags.append(dia)
    dia._ax.set_title(info[-2],size=14,position=(0.3,0.0))



sym_col = PaperLib.sym_colors(df_all).rename('sym_colour')
df_all = df_all.merge(sym_col, right_index=True, left_index=True)
ms = 7  # marker size
alpha = 0.6

# info for the Standard labels
std_col = 'grey'
std_alpha = 0.6
bbox = dict(boxstyle="round", fc=std_col, ec="none", alpha=std_alpha,fill=False,pad=0.0)
el = mpl.patches.Ellipse((2, -1), 0.5, 0.5)
arrowprops = dict(arrowstyle="wedge,tail_width=1.", alpha=std_alpha,
                  fc=std_col, ec="none",
                  patchA=None, patchB=el, relpos=(0.0, 0.5))

arrowprops=dict(arrowstyle='fancy',color='grey')

Shortnames = dict(air_temperature_tas='LAT',
                  air_pressure_at_mean_sea_level='SLP',
                  precipitation_flux='LP',
                  air_temperature_500='T500',
                  relative_humidity='q500',
                  toa_outgoing_longwave_flux='OLR',
                  toa_outgoing_shortwave_flux='RSR')

for dia, info in zip(taylor_diags, pinfo):
    # plot markers for all models
    for var, marker in info[0].items():
        for name, row in df_all.iterrows():
            marker_size = ms
            alpha_plot = alpha
            if name.startswith('CMIP') or (name == 'DFOLS-MM'):  # multi-model mean
                marker_size = 11
                alpha_plot = 1
            dia.add_sample(row.loc[var + '_sd'], row.loc[var + '_corr'], color=row.sym_colour,
                           marker=marker, markersize=marker_size, alpha=alpha_plot,zorder=row.zorder)
        # Name std case.
        c = np.array([float(Standard_taylor.loc[:,var + "_" + x].values) for x in ['corr', 'sd']])
        c[0] = np.arccos(c[0])
        dia.ax.annotate(Shortnames[var], c, c + [0.05, -0.2], color='black', alpha=1, fontsize=11,
                        weight='bold', ha='right', bbox=bbox, arrowprops=arrowprops)

lab = PaperLib.plotLabel(fontdict=dict(size=12,weight='12'))
for dia  in taylor_diags:
    contours = dia.add_contours(5, colors='gray', linestyles=':')
    plt.clabel(contours, inline=1, fontsize=12, fmt='%2.2f')
    lab.plot(dia.ax)
fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)

##
