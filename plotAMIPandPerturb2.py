"""
Plot the AMIP and perturbed Physics runs from atmosphere only cases
"""

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import PaperLib
import StudyConfig


## functions
def short_names(obsNames):
    """
    Produce  short names with "sensible" sorting
    :param obsNames:
    :return:
    """
    shrtNamesTranslate = PaperLib.shortNames(obsNames=obsNames)  # get short names
    # and sort them..
    sortK = {'LAT': 0, 'LP': 1, 'SLP': 2, 'RSR': 3, 'OLR': 4, 'T500': 5, 'q500': 6}  # sorting for SHORT names
    sortfn = lambda var: sortK.get(var, 10)  # fn for sorting use of.get gives default value for sorting
    shrtNames = sorted(shrtNamesTranslate.values(), key=sortfn)

    return shrtNames, shrtNamesTranslate


def plotSimDelta2(deltaDF, ax, colors, obsNames, pos=None):
    """
    Make barplots for each normalised data difference.
    :param deltaDF:
    :param colors:
    :param ax:
    :return: nada
    """
    use_pos = pos
    if use_pos is None:
        ndf = len(deltaDF)
        use_pos = np.linspace(0.1, 1, ndf + 1)

    width = np.min(use_pos[1:] - use_pos[0:-1]) * 0.8
    rot = 30.
    shrtNames, shrtNamesTranslate = short_names(obsNames)
    nx = len(shrtNames)

    manage_xticks = True
    for df, color, p in zip(deltaDF, colors, use_pos):

        shrt = df.loc[:, obsNames].rename(columns=shrtNamesTranslate)
        box = shrt.plot.box(ax=ax, rot=rot, fontsize='small', color=color, whis=[5, 95],
                            positions=p + np.arange(0, nx),
                            patch_artist=True, manage_ticks=manage_xticks,
                            widths=width, return_type='dict', showfliers=False)
        manage_xticks = False
        for b in box['medians']:
            b.set_linewidth(1)
            if color != 'black':
                b.set_color('black')
            else:
                b.set_color('white')
        for b in box['whiskers']:
            b.set_color(color)
            b.set_linestyle('solid')
            b.set_linewidth(2)

    ax.set_xlim(-0.5, nx)
    ax.set_ylim(-20, 30)
    ax.set_yscale('symlog', linthresh=4)
    # plot NH, Tropics, SH
    for indx, rgn in enumerate(['NHEx', 'Tropics', 'SHEx']):
        ax.axvline((indx + 1) * 7 - 0.1 - (rgn == 'SHEx'), linestyle='dashed', linewidth=2, color='green')
        ax.text(indx * (nx // 3) + 0.5, -13, rgn, color='black', fontsize='large', ha='left')
    ax.axhline(0.0, linewidth=2, linestyle='solid', color='black')
    return


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%3.1f' % (x)


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


def scatter_values(df, xvar, yvar, ax):
    """
    Plot boxes and text
    :return: nada
    """

    colours = sym_colors(df)
    text = df.loc[:, 'shortText']
    optimised = df.loc[:, 'Optimised']
    translate = {'Y': 'lightblue'}
    fontDir = dict(size=12, weight='bold')
    bboxDir = dict(linewidth=0, alpha=0.5, pad=0)
    for colour, (name, row) in zip(colours.values,
                                   df.iterrows()):  # x,y,name,tt,o in zip(df.loc[:,xvar],df.loc[:,yvar],df.index,text,optimised): # iterate over rows.
        bboxDir['color'] = colour
        ax.text(row.loc[xvar], row.loc[yvar], row.shortText, fontdict=fontDir, va='center', rotation='horizontal',
                ha='center', bbox=bboxDir)
    # add some fake points so matplotlib does the range properly...
    ax.scatter(df.loc[:, xvar], df.loc[:, yvar], marker='.', s=1)
    return None


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


def half_var(dataframe, var, normalise=None):
    """
    Compute the mean of the input data frame and the appropriately scaled value
    :param dataframe: dataframe continaing data
    :param var: column name to compute mean
    :param normalise: If not None the column to normalise by rather than scaling
    :param scale (default 2). How much to scale other variable by.
    :return: mean for specified column and appropriately scaled other column/
    Example half_car
    """
    half_vars = {'T140': 'TCR', 'ECS_4xCO2': 'ECS_2xCO2',
                 'ts_ice_extent4': 'ts_ice_extent', 'ts_nathcstrength4': 'ts_nathcstrength',
                 'ts_rh504': 'ts_rh50', 'ts_t504': 'ts_t50', 'ts_cloud4': 'ts_cloud',
                 'lambda_4xCO2_SW': 'lambda_2xCO2_SW', 'lambda_4xCO2_LW': 'lambda_2xCO2_LW',
                 'lambda_4xCO2_SWC': 'lambda_2xCO2_SWC', 'lambda_4xCO2_LWC': 'lambda_2xCO2_LWC',
                 'F_4xCO2': 'F_2xCO2', 'lambda_4xCO2': 'lambda_2xCO2'}  # variable and matching var for 1/2 response
    scale = 2
    if var.startswith('lambda'):
        scale = 1.
    data = dataframe[var]
    ok = ~data.isnull()
    if var in half_vars:

        if normalise is not None:
            mn = data / dataframe[ok].loc[:, normalise]
            mn = mn.mean()
            t = dataframe[ok].loc[:, half_vars[var]]
            t = t / dataframe[ok].loc[:, half_vars[normalise]]  # normalise var.
            mn_scale = t.mean()
        else:
            mn = data.mean()
            mn_scale = dataframe[ok].loc[:, half_vars[var]].mean() * scale

        return (mn, mn_scale)
    else:
        return None


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


def plot_var(dataframe, variance, ensembles, ax, var, title, xtitle, label=None, normalise=None, error=True):
    """

    :param dataframe:
    :param var: variable to plot
    :param ax:axis on what to plot
    :param error (default is True). If True plot error bar
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
        yv = np.random.uniform(-0.5, 0.5, len(v)) + index  # jittering the points
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
            for l, x, y, color in zip(labels, v, yv, textColor.values):
                ax.annotate(l, (x, y), color=color, ha='center', va='center', fontsize=size, fontweight='bold')

            vars = half_var(data, var, normalise=normalise)
            try:
                for val, marker_mn, alpha in zip(vars, ['*', 'h'], [1, 0.5]):
                    ax.plot(val, index, color=colors.iloc[0], marker=marker_mn, ms=18,
                            linestyle='None', markeredgewidth=2, mec='black', alpha=alpha)
            except TypeError:
                ax.plot(v.mean(), index, color=colors.iloc[0], marker='*', ms=18, linestyle='None', markeredgewidth=2,
                        mec='black')

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


## start of main code -- defns first
PPsym = 'd'
AMIPsym = 's'
PPsize = 70
AMIPsize = 60
PPcolor = 'black'
AMIPcolor = 'black'

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

"""ens_colours = {'CMIP5': 'black', 'CMIP6':'cornflowerblue','CE7': 'grey', 'DF14': 'orange', 'SS': 'lightblue', 'CE14': 'indianred',
               'ICE': 'grey'}
ens_text_colours = {'CMIP5': 'black', 'CMIP6':'darkblue','CE7': 'black', 'DF14': 'black', 'SS': 'black', 'CE14': 'black',
               'ICE': 'black'}"""
## plot distributions for main paper.
labelV = PaperLib.plotLabel()
ensembles = ['SS', 'DF14', 'CE7', 'CMIP6', 'CMIP5']  # ensembles to plot.
fig, axis = plt.subplots(2, 2, sharey=True, num="distributions", figsize=[8.2,7], clear=True)
for var, title, xtitle, ax in zip(['COST', 'CTL_ts_t15', 'T140', 'ECS_4xCO2', ],
                                  ['2001 - 2005 Cost', 'Control GMSAT', 'T140', 'ECS4'],
                                  ['Cost', 'GMSAT (K)', 'Warming (K)', 'Warming (K)'],
                                  axis.flatten()):
    plot_var(allData, variance_ctl, ensembles, ax, var, title, xtitle, label=labelV)

best = 273.15 + 13.6
err = 0.5  ## 3 sigma error
axis[0, 1].axvspan(best - err, best + err, alpha=0.3, color='grey')
axis[0, 1].axvline(best, linewidth=2, color='black', linestyle='dashed')

plt.tight_layout()
fig.show()
PaperLib.saveFig(fig, "AMIP_CMIP_dist")
## and print out some min, mean & max values and latex tables for paper
for var in ['COST', 'CTL_ts_t15']:
    for ens in [CMIP5, CMIP6]:
        series = ens.loc[:, var]
        print(
            f"Name: {ens.Ensemble.unique()} {var}:  {series.mean(): 4.1f} ({series.min(): 4.1f} - {series.max(): 4.1f})")
        print("Small/large", series.idxmin(), series.idxmax())

# print out values as Latex Tables
vars = ['COST', 'ECS_2xCO2', 'TCR', 'T140', 'CTL_ts_t15', 'Source']
rewrite = dict(ECS_2xCO2='ECS', CTL_ts_t15='GMSAT', identifier='ID', Natmos='N\textsubscript{atmos}',
               Ncoup='N\textsubscript{coup}')
source_rewrites = {'Ringer *et *al.*$': '\\\\citet{ringer20cmip}',
                   'Gregory *et *al.*$': '\\\\citet{gregory15tcr}',
                   'Stocker *et *al.*$': '\\\\citet{stocker2013ipcc}',
                   '.*Zelinka.*$': '\\\\citet{Zelinka2020sens}'}
import re

capt_CMIP5 = r'''\small{Summary properties for CMIP5 models. 
 ID is the label used in Fig.~\ref{fig:amipCMIP} and other plots. N\textsubscript{atmos} and N\textsubscript{coup} are the sizes of the atmospheric and coupled ensembles.  
 COST is the dimensionless value of the cost-function.
  Shown in K are the Equilibrium Climate Sensitivity (ECS), Transient Climate Response (TCR), Transient Climate Response (T140) at $4\times$CO\textsubscript{2} and the 
  pre-industrial control global mean surface air temperature (GMSAT). 
   Source shows where ECS/TCR/T140 values came from and MM Mean shows the multi-model mean of the ensemble. Other values are defined in the main text.}'''
capt_CMIP6 = r'Summary properties for CMIP6 models with details as table~\ref{tab:cmip5}'
for ens, outfile, caption in zip([CMIP5, CMIP6], ['cmip5.tex', 'cmip6.tex'], [capt_CMIP5, capt_CMIP6]):
    ens_name = outfile.split('.')[0]
    ens.index.name = 'Model'
    tt = ens.loc[:,
         ['identifier', 'COST', 'Natmos', 'ECS_2xCO2', 'TCR', 'T140', 'CTL_ts_t15', 'Ncoup', 'Source']].rename(
        columns=rewrite)
    # rewrite the sources...
    src = tt.Source.fillna('--')
    for k, v in source_rewrites.items():
        src = [re.sub(k, v, s) for s in src]
    tt.loc[:, 'Source'] = src
    # add in the mean
    mn_vars = ['COST', 'ECS', 'TCR', 'T140', 'GMSAT']
    tt.loc['MM Mean', mn_vars] = tt.loc[:, mn_vars].mean()
    # need to replace the '<NA>' string from missing integer data. Alas to_latex does not (yet) do that.
    # so we need to explicitly write the data out rather than using the buffer option..
    with open(PaperLib.dataPath / outfile, mode='w') as file:
        tlabel = f'tab:{ens_name}'
        file.write(tt.to_latex(float_format='%4.1f', na_rep='--', escape=False, longtable=False, label=tlabel,
                               caption=caption).replace('<NA>', '--'))

## then print out correlations
import scipy.stats

for ens in [CMIP5, CMIP6]:
    for var in ['COST', 'CTL_ts_t15']:
        for var2 in ['T140', 'ECS_4xCO2']:
            L = (ens.loc[:, var].notnull()) & (ens.loc[:, var2].notnull())
            r, p = scipy.stats.pearsonr(ens.loc[L, var], ens.loc[L, var2])
            print(f"pearson {var} {var2} {r: 3.2f} {p:3.2f}")
            r, p = scipy.stats.spearmanr(ens.loc[L, var], ens.loc[L, var2])
            print(f"spearman {var} {var2} {r: 3.2f} {p:3.2f}")
    print("-------------------------------------")
## print out the variability etc.
for ensemble in ['CMIP5', 'CMIP6', 'CE7', 'DF14', 'ICE']:
    strTable = ''  # avoid using str as a variable name,,,
    data = allData.query(f'Ensemble == "{ensemble}"')
    print(f"{ensemble:6s}", end='&')
    for var in ['ECS_2xCO2', 'ECS_4xCO2', 'TCR', 'T140']:
        if ((ensemble == 'ICE') and (var in ['TCR', 'T140'])):
            sd = comp_sd(var, variance_ctl)
            mn = allData.loc['Standard', var]
        else:
            sd = data[var].std()
            mn = data[var].mean()
        cv = sd / mn * 100
        strTable += f"${mn:4.2g} \\pm {sd:4.1g} ({cv:3.0f}\%)$ &"
    print(strTable[:-1] + r'\\')  # print string except for last element -- the trailing & and add in \\

## plot  feedbacks change
labelV = PaperLib.plotLabel()
ensembles = ['SS', 'DF14', 'CE7']  # ensembles to plot.
Standard = allData.loc['Standard']
names = ['CE7', 'DF14', 'SS', 'ICE']
fig, axis = plt.subplots(3, 1, num="PP_ECS", figsize=[8.6,8], clear=True)

linestyle = dict(color='black', linewidth=2, linestyle='dashed')  # linestyle ctls
fill = dict(color='grey', alpha=0.3)  # fill styles
fill2 = dict(color='grey', alpha=0.2)  # fill styles
fixVar = dict(lambda_4xCO2_SW='lambda_4xCO2', lambda_4xCO2_SWC='lambda_4xCO2_C', lambda_4xCO2='ECS_4xCO2')
for (varx, vary), title, (xtitle, ytitle), ax in zip(
        [('lambda_4xCO2', 'F_4xCO2'), ('lambda_4xCO2_SW', 'lambda_4xCO2_LW'), ('lambda_4xCO2_SWC', 'lambda_4xCO2_LWC')],
        # vars
        [r'$\lambda$ vs F(4xCO$_2$) ',
         r'$\lambda_{SW}$ vs  $\lambda_{LW}$ ', r'$\lambda_{SWC}$ vs  $\lambda_{LWC}$'],  # titles
        [
            (r'$\lambda$ (Wm$^{-2}$K$^{-1}$)', r'F (Wm$^{-2}$)'),
            (r'$\lambda_{SW}$ (Wm$^{-2}$K$^{-1}$)', r'$\lambda_{LW}$ (Wm$^{-2}$K$^{-1}$)'),
            (r'$\lambda_{SWC}$ (Wm$^{-2}$K$^{-1}$)', r'$\lambda_{LWC}$ (Wm$^{-2}$K$^{-1}$)')  # xy titles
        ]
        , axis.flatten()
):
    if varx.startswith(r'$\lambda') and vary.startswith(r'$\lambda'):  # want proportional axis and line of equal total
        ax.set_aspect('equal')

    for ensemble in names:  # loop over ensembles
        # color = ens_colours[ensemble]
        size = 11
        marker = 'd'
        data = allData.query(f'Ensemble == "{ensemble}"')
        L = ~data.loc[:, varx].isnull()
        d = data[L]  # removed all missing data
        colors = sym_colors(d)
        if ensemble == 'SS':
            scatter_values(d, varx, vary, ax)
        elif ensemble != 'ICE':
            d.plot.scatter(varx, vary, color=colors.values.tolist(), marker=marker, s=(size ** 2), ax=ax)

            for ii, xy in enumerate(zip(d.loc[:, varx], d.loc[:, vary])):
                ax.annotate(f"{ii:1d}", xy, ha='center', va='center', fontsize=size,
                            fontweight='bold')  # only need 1 dp
            varx_half = half_var(data, varx)[0:1]
            vary_half = half_var(data, vary)[0:1]  # hack to only show the 4xCO2 result
            for vx, vy, marker_mn in zip(varx_half, vary_half, ['*', 'h']):
                ax.plot(vx, vy, color=colors[0], marker=marker_mn, ms=20, linestyle='None', markeredgewidth=2,
                        mec='black', alpha=0.7)
        else:  # ICE data.
            cov = np.cov(d.loc[:, varx], d.loc[:, vary])
            elip = PaperLib.plot_cov_ellipse(Standard.loc[[varx, vary]], cov, ax=ax, edgecolor='black',
                                             facecolor='None', linewidth=2)
            ax.errorbar(Standard.loc[varx], Standard.loc[vary],
                        xerr=d.loc[:, varx].std() * 2, yerr=d.loc[:, vary].std() * 2, linewidth=2, color='black')

    ax.set_title(title)
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    labelV.plot(ax)
    ax.autoscale_view()
    # plot some lines

    xr = ax.get_xlim()
    x = np.linspace(xr[0], xr[1])


    fixVariable = fixVar[varx]
    std_value = Standard.loc[fixVariable]
    sd_fix = 2*np.sqrt(2)*allData.query("Ensemble=='ICE'").loc[:, fixVariable].std()
    if varx in ['lambda_4xCO2_SW', 'lambda_4xCO2_SWC']:  # plot equal total feedback
        y = std_value - x
        pos = (0.9, 0.9)
        neg = (0.1, 0.1)
        # work out range.
        y_upper = std_value + sd_fix - x
        y_lower = std_value - sd_fix - x
    elif (vary == 'F_4xCO2') and (varx == 'lambda_4xCO2'):
        y = x * std_value  # assume constant ECS4; F = ECS*lambda
        pos = (0.1, 0.9)
        neg = (0.9, 0.1)
        y_upper = x*(std_value + sd_fix)
        y_lower = x*(std_value - sd_fix)
    elif vary == 'F_4xCO2':
        #x = y / Standard.lambda_4xCO2  # assume constant lambda
        raise Exception("Reimplement this...")
    elif vary == 'lambda_4xCO2':
        #x = Standard.F_4xCO2 / y  # assume constant Forcing
        raise Exception("Reimplement this...")
    else:
        pass
    # bit of a hack as assume have x & y..but plot error

    ax.fill_between(x, y_lower, y_upper, **fill2)
    ax.plot(x, y, **linestyle)
    ax.text(pos[0], pos[1], 'H', fontsize='xx-large', transform=ax.transAxes,va='center')
    ax.text(neg[0], neg[1], 'L', fontsize='xx-large', transform=ax.transAxes,va='center')
plt.tight_layout()
fig.show()
PaperLib.saveFig(fig, "PP_ECS4")

## plot  cld, T@500, RH@500. Ice extent &

fig, axis = plt.subplots(2, 2, num="PP_response", figsize=[8.2,6], clear=True)
labelV = PaperLib.plotLabel()
vars_want = ['ts_cloud4', 'ts_t504', 'ts_rh504', 'ts_ice_extent4']
for var, title, xtitle, ax in zip(vars_want,
                                  ['Cloud Fraction', '500 hPa Temp', '500 hPa RH', 'Ice extent'],
                                  ['%/K', 'K/K', '%/K', r'$10^{12}$ m$^2$/K', ],
                                  axis.flatten()):

    plot_var(allData, variance_ctl, ensembles, ax, var, title, xtitle, label=labelV, normalise='T140', error=False)
    sd = comp_sd(var, variance_ctl) / Standard.loc['T140']
    yr = ax.get_ylim()
    y = np.linspace(yr[0], yr[1])
    for s, f in zip([1, np.sqrt(2)], [fill, fill2]):
        err = s * 2 * sd
        mn = Standard.loc[var] / Standard.loc['T140']
        ax.fill_betweenx(y, mn - err, mn + err, **f)



plt.tight_layout()
fig.show()
PaperLib.saveFig(fig, "PP_response")

## Compute normalised differences
delta = []
tgt = config.targets(scale=True)
cov = config.Covariances(obsNames=tgt.index, scale=True)['CovTotal']
sd = pd.Series(np.sqrt(np.diag(cov)), index=cov.columns)
for ensemble in ['CMIP5', 'CMIP6', 'CE7', 'DF14']:
    d = allData.query(f'Ensemble=="{ensemble}"').copy()
    d.loc[:, tgt.index] = (d.loc[:, tgt.index] - tgt) / sd
    delta.append(d)

## plot the normalised differences
f, ax = plt.subplots(1, 1, num='AMIP_NORM', clear=True, figsize=[8.2,6])
obsNames = config.obsNames()
plotSimDelta2(delta, ax, ['silver', 'cornflowerblue', 'black', 'orange'], obsNames)
# add plus or minus 2 lines
for v in [-2, 2]:
    ax.axhline(v, color='black', linestyle='dashed', linewidth=2)

# put the sens cases on top.
shrt, translate = short_names(obsNames)
sensCases = allData.query(f'Ensemble=="SS"').copy()
sensCases.loc[:, tgt.index] = (sensCases.loc[:, tgt.index] - tgt) / sd
for name, series in sensCases.loc[:, obsNames].rename(columns=translate).iterrows():
    if sensCases.loc[name, 'Optimised'] == 'N':
        color = 'red'
    else:
        color = 'lightblue'
    plotName = sensCases.loc[name, 'shortText']
    x = np.arange(len(series.values)) - 0.25 + np.random.uniform(0, 0.5)
    ax.plot(x, series.values, marker='o', color=color, linestyle='none', zorder=10)

ax.set_ylabel(r'$\Delta$')
ax.set_title('Normalised Observations - Target')
ax.set_ylim(-25, 25)
ax.set_yticks([-20, -10, -5, -4, -2, 0, 2, 4, 5, 10, 20])
formatter = plt.FuncFormatter(log_10_product)
ax.yaxis.set_major_formatter(formatter)
f.show()
PaperLib.saveFig(f, "AMIP_scaled")

## print out "table" for Sens studies
data = allData.query("Ensemble=='SS'")
data = data.append(allData.loc['HadAM3-7#05', :])
data = data.rename(
    columns=dict(ECS_2xCO2='ECS', ECS_4xCO2='ECS4', CTL_ts_t15='GMSAT', shortText='ID', Comments='Description'))
df = data.loc[:, ['ID', 'COST', 'ECS', 'ECS4', 'TCR', 'T140', 'GMSAT', 'Description']]
# (latex) escape any # we find in the index. The text should already be latex formatted..

df.index = [d.replace('#', r'\#') for d in df.index]

with open(PaperLib.dataPath / 'sens_table.tex', mode='w') as file:
    with pd.option_context('max_colwidth', None):  # no limit on column width
        df.to_latex(file, float_format='%3.1f', na_rep='--', column_format='c cccc ccc p{0.3\linewidth}', escape=False)
