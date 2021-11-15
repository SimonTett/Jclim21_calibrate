## plot simple figure for proposal. Show COST for CMIP6 & DF14 & T140 for the same.
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats



import PaperLib
import StudyConfig

def readData(tsData, jsonFile, scale=None, dfols=False):
    """

    :param tsData: dataframe containing information on various simulations. This function only uses the index.
    :param jsonFile: example json file -- used to define obs and other things.
    :return: dataFrame with useful information from reading all JSON files in the tsData json file.
    """

    files = []  # files to read
    names = []  # labels to give them.
    cols = ['Atmosphere Run#1', 'Atmosphere Run#2']

    for ind in cols:
        L = tsData.loc[:, ind].notnull()
        dfols_cases = tsData.Ensemble == 'DF14'
        files.extend([PaperLib.DFOLSpath / d / 'observations.nc' for d in tsData[L].loc[dfols_cases, ind]])
        names.extend(tsData[L & dfols_cases].index)  # index names for the dfols_cases
        files.extend([os.path.join(PaperLib.rootData, d, 'observations.nc') for d in tsData[L].loc[~dfols_cases, ind]])
        names.extend(tsData[L & ~dfols_cases].index)  # index names for the non dfols cases
    newDF = PaperLib.NCtoDF(files, json_file, labels=names, scale=scale)
    newDF = pd.concat([tsData, newDF], axis=1, join='outer', verify_integrity=True)
    return newDF


def plotValues(df, index, col, ax, normalise=None):
    """
    Plot boxes and text
    :return: nada
    """

    series = df.loc[:, col]
    colors = sym_colors(df)
    ok = ~series.isnull()
    series = series[ok]
    colors = colors[ok]
    if normalise:
        series = series / df.loc[ok, normalise]
    # indx = series.index
    yvalues = np.random.uniform(-0.5, 0.5, len(series)) + index
    text = df.loc[ok, 'shortText']
    L = df.index == 'Standard'
    yvalues[L] = index  # no jitter for the std model

    fontDir = dict(size=11, weight='bold')
    bboxDir = dict(linewidth=2, alpha=0.5, pad=0)
    for color, name, x, y in zip(colors.values, text, series.values, yvalues):  # iterate over rows.
        bboxDir['color'] = color
        ax.text(x, y, name, fontdict=fontDir, va='center', rotation='horizontal', ha='center', bbox=bboxDir)
    # add some fake points so matplotlib does the range properly...
    ax.plot(series.values, yvalues, '.', ms=0.1, alpha=0)
    return None



def sym_colors(df, text=False):
    """
    Return a list of colours
    :param df -- the dataframe  -- needs to have Optimised & Ensemble and set
    :param text if True (Default is False) return text colour.
    """

    ens_colours = {'CMIP5': 'black', 'CMIP6': 'cornflowerblue', 'CE7': 'grey', 'DF14': 'orange',
                   'CE14': 'indianred',
                   'ICE': 'grey'}
    ens_text_colours = {'CMIP5': 'black', 'CMIP6': 'darkblue', 'CE7': 'black', 'DF14': 'black',
                        'CE14': 'black',
                        'ICE': 'black'}

    optimise = dict(y='lightblue', n='red')  # force optimise to lowercase.

    if text:
        lookup_colours = ens_text_colours
        stdColour = 'black'
    else:
        lookup_colours = ens_colours
        stdColour = 'grey'
    colours = dict()  # list of colors
    for name, row in df.iterrows():
        colour = lookup_colours.get(row.Ensemble)
        if colour is None:
            colour = optimise.get(str(row.Optimised).lower(), None)
        if name == 'Standard':
            colour = stdColour
        colours[name] = colour

    return pd.Series(colours, name='Colour')


def comp_sd(var, variance):
    """
    Compute the standard deviation for specified variable.
    :param var: variable wanted
    :param variance: variance dataframe.
    :return: variance of requested variable.
    """
    translate_var = {'TCR': 'ts_t15_yr70', 'T140': 'ts_t15_yr140',
                     'CTL_ts_t15': 'ts_t15_yr140',
                     'ts_ice_extent4': 'ts_ice_extent_yr140', 'ts_nathcstrength4': 'ts_nathcstrength_yr140',
                     'ts_rh504': 'ts_rh50_yr140', 'ts_t504': 'ts_t50_yr140',
                     'ts_cloud4': 'ts_cloud_yr140'
                     }  # name translation for variances

    sd = np.sqrt(2 * variance.loc['Standard', translate_var.get(var, var)])
    if var.startswith('CTL'):  # control data.
        sd = sd / np.sqrt(2.)
    return sd

def plot_var(dataframe, variance, ensembles, ax, var, title, xtitle, label=None, normalise=None, jitter=0.5, error=True,simple=False):
    """

    :param dataframe:
    :param var: variable to plot
    :param ax:axis on what to plot
    :param error (default is True). If True plot error bar
    :param simple (default is False). If True plot simple version.
    :param jitter (default is 0.5). How much jitter wanted (seperation between rows is 1)
    :return:
    """
    # whiskerprops = dict(linewidth=2)
    medianprops = dict(color='white', linewidth=2)

    sizes = dict(CMIP5=12, CMIP6=12)
    markers = {'ICE': '+'}

    for index, ensemble in enumerate(ensembles):
        data = dataframe.query(f'Ensemble == "{ensemble}"')
        series = data.loc[:, var]
        colors = sym_colors(data)
        boxprops = dict(facecolor=colors[0], alpha=0.5)
        whiskerprops = dict(linewidth=2, color=colors[0])
        size = sizes.get(ensemble, 12)
        marker = markers.get(ensemble, 'd')
        textColor = sym_colors(data, text=True)
        ok = ~series.isnull()  # where values are OK
        series = series[ok]
        colors = colors[ok]
        textColor = textColor[ok]
        if ensemble in ['CE7', 'CE14', 'DF14']:
            labels = [f"{ii:1d}" for ii in range(0, len(series))]
            labels = np.array(labels)
        else:
            labels = data.identifier

        labels = labels[ok]
        if normalise is not None:
            series = series / data.loc[ok, normalise]

        v = series.values
        yv = np.random.uniform(-jitter, jitter, len(v)) + index  # jittering the points
        L = series.index == 'Standard'
        yv[L] = index  # no jittering for std case.
        if ensemble == 'SS':  # sens case -- with lots of special code.
            plotValues(data, index, var, ax, normalise=normalise)
            if var.startswith('ECS') or var.startswith('lambda') or var.startswith('F_'):
                # use JMG values to get SD
                iceDF = dataframe.query("Ensemble=='ICE'")
                icev = iceDF[var]  # get the init cond ensemble
                if normalise is not None:
                    icev = icev / iceDF[normalise]
                sd = icev.std()
            elif var not in ['COST']:  # list of variables which don't have sampling errors..
                sd = comp_sd(var, variance)  # extract error from ctl variance.
                if normalise is not None:
                    sd = sd / dataframe.loc['Standard', normalise]  # scale by std value.
            else:
                sd = None
            # now plot errors
            if (sd is not None) and error:  # have a sd so plot it.
                ax.errorbar(series.Standard, index, xerr=2 * sd,
                            capthick=2, fmt='x', color='black', capsize=10, elinewidth=3, zorder=100)
                #ax.errorbar(series.Standard, index, xerr=np.sqrt(2.) * 2. * sd, # 2*sqrt(2) sigma error
                #            capthick=1.5, fmt='x', color='grey', capsize=10, elinewidth=1.5, zorder=100)
        else:  # everything else.
            if ensemble not in ('CMIP5', 'CMIP6'):
                ax.scatter(v, yv, marker=marker, color=colors.values.tolist(), s=size ** 2)
            else:
                ax.plot(v, yv, marker='.', ms=0, color='white',
                        linestyle='None')  # update the limits as annotation does not.

        if ensemble in ('CMIP5', 'CMIP6'):  # plot box/whisker plot
            boxes = ax.boxplot(v, positions=[index], whiskerprops=whiskerprops, medianprops=medianprops,
                               vert=False, whis=[5, 95],
                               patch_artist=True, boxprops=boxprops, showfliers=False)  #
    # axis decorators
    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel("Ensemble")
    ax.set_yticks(range(0, index + 1))
    ax.set_yticklabels(ensembles)
    ax.set_ylim(-0.5, index + 0.5)
    print(ax.get_xlim())
    if label is not None:
        label.plot(ax=ax)


## read in data.
rename_col = {'ctl_ts_t15_yr181': 'CTL_ts_t15', 'TCR4': 'T140', 'ts_t15_yr140': 'T140',
              'ts_t15_yr70': 'TCR'}  # rename things!
json_file = os.path.join(PaperLib.rootData, "7param", "vierzehn", "vierzehn_final.json")
config = StudyConfig.readConfig(json_file)
allData = PaperLib.read_data().rename(columns=rename_col)  # rename for consistency with CMIP/AMIP runs
allData = readData(allData, json_file)

## read in CMIP5 & 6 data
CMIP5 = PaperLib.read_CMIP(json_file, 'CMIP5')
CMIP6 = PaperLib.read_CMIP(json_file, 'CMIP6')
# overwite with data from Mark Ringer
CMIP6 = CMIP6.drop(columns=['TCR', 'T140', 'ECS_2xCO2', 'ECS_4xCO2'])
mrData = pd.read_csv('data/tcr_cmip6.csv', header=0, index_col=0)
mrData = mrData.drop('Mean').drop(columns=['ratio'])
# mrData.loc[:,'Source']='Ringer et al, 2020'
# CMIP6 = pd.merge(CMIP6,mrData)
# and then get the ECS data
mrData2 = pd.read_csv('data/gregory_plot_cmip6.csv', header=0, index_col=0)
mrData2 = mrData2.drop('Mean').loc[:, ['ECS']]
mrData2 = mrData2.rename(columns=dict(ECS='ECS_2xCO2'))
mrData2.loc[:, 'ECS_4xCO2'] = 2 * mrData2.loc[:, 'ECS_2xCO2']
mrData = mrData.merge(mrData2, left_index=True, right_index=True)
CMIP6 = CMIP6.merge(mrData, how='outer', left_index=True, right_index=True)
CMIP6.loc[mrData.index, 'Source'] = 'Ringer et al, 2020'

CMIP6.loc[:, 'identifier'] = list(string.ascii_letters[0:CMIP6.shape[0]])
CMIP5.loc[:, 'identifier'] = list(string.ascii_letters[0:CMIP5.shape[0]])
CMIP6.loc[:, 'Ensemble'] = 'CMIP6'
CMIP5.loc[:, 'Ensemble'] = 'CMIP5'
allData = allData.append([CMIP5, CMIP6])
## get in ICE ensemble
JMGensData = pd.read_csv(os.path.join(PaperLib.dataPath, 'JMG_ens_data.csv'), index_col=0)
# overwrite index!
index = [f"JMG#{ind:02}" for ind in range(len(JMGensData.index))]
JMGensData.index = index
JMG2xCO2ensData = pd.read_csv(os.path.join(PaperLib.dataPath, 'JMG_2xCO2_ens_data.csv'), index_col=0)
JMG2xCO2ensData.index = index
ice = pd.concat([JMGensData, JMG2xCO2ensData], axis=1)  # merge the IC ensemble data ran by JMG
# ice= JMGensData # 2xCO2 data short runs for some reason...
ice.loc[:, 'Ensemble'] = 'ICE'
allData = allData.append(ice)  # add in the ICE ensemble
# get variance estimates from long runs
variance_ctl = pd.read_csv(os.path.join(PaperLib.dataPath, 'internalVar.csv'), index_col=0)
# fix the ice stuff to be in million km^2
ice_vars = [v for v in variance_ctl.columns if 'ice' in v]
variance_ctl[ice_vars] = variance_ctl[ice_vars] / 1e24

# convert clouds to %
cols = [c for c in allData.columns if 'ts_cloud' in c]
allData.loc[:, cols] *= 100  # convert to %
cols = [c for c in variance_ctl.columns if 'ts_cloud' in c]
variance_ctl.loc[:, cols] *= 100 ** 2  # convert to %

# read in mean and sd's from compECS4covar
mn = pd.read_csv(PaperLib.dataPath/'mn_change.csv',header=0,index_col=0)
sd = pd.read_csv(PaperLib.dataPath/'sd_change.csv',header=0,index_col=0)
## plot data
fig, axis = plt.subplots(nrows=1, ncols=2, num="simp_uncert", sharey=True,figsize=[7.5,5], clear=True)
ensembles=['CMIP6','DF14']
for var, title, xtitle, ax in zip(['COST',  'T140' ],
                                  ['2001 - 2005 AMIP',  'Warming at 4xCO$_2$'],
                                  ['Mistfit', 'Warming (K)'],
                                  axis.flatten()):
    plot_var(allData, variance_ctl, ensembles, ax, var, title, xtitle,simple=True,jitter=0.2)
    # put the standard value on as a black spot...
    ax.plot(allData.loc['Standard',var],1.0,marker='o',ms=7,color='black')

# add the linear estimated mn and uncertainty to the TCR plot.
axis[-1].errorbar(mn.loc['SigPR','sat_TCR4'],1.1,xerr=2*sd.loc['SigPR','sat_TCR4'],
                  ecolor='grey',elinewidth=2, capthick=5,
                  marker='o',color='grey',ms=7)

fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)