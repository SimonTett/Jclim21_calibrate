import pandas as pd
import PaperLib
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as gridspec

# import study

import StudyConfig
import argparse

import itertools

import pathlib

parser = argparse.ArgumentParser(description="Plot parameters")
parser.add_argument("-defaultCov", "--verbose", help="Provide verbose output",
                    action="count", default=0)
parser.add_argument("-skipread", help='Skip reading', action='store_true')
args = parser.parse_args()  # and parse the arguments

nStudies = 0
if not args.skipread:

    studies = {}
    allTs7Data = PaperLib.read_data(sevenParam=True)
    allTs14Data = PaperLib.read_data(fourteenParam=True)
    allTsSens = PaperLib.read_data(sensCases=True)
    allTSdfols = PaperLib.read_data(dfols=True)
    runs_to_fake = []  # runs we need to fake.
    refCase = StudyConfig.readConfig(os.path.join(PaperLib.rootData, allTs7Data.iloc[0, :].JsonFile), ordered=True)

    for l, exper in itertools.chain(allTs7Data.iterrows(),
                                    allTsSens.iterrows()):  # iterate over experiments
        if exper.loc['JsonFile'] is np.nan:  # nothing defined
            runs_to_fake.append(l)
        else:
            studies[l] = StudyConfig.readConfig(os.path.join(PaperLib.rootData, exper.loc['JsonFile']), ordered=True)
        nStudies += 1  # increment no of studies

    # need to add DFOLS cases

    for l, exper in allTSdfols.iterrows():
        studies[l] = StudyConfig.readConfig(PaperLib.DFOLSpath / 'studies' / exper.loc['JsonFile'], ordered=True)
        nStudies += 1

## plot "fingerprints" of parameters
orderNames = ['CT', 'ENTCOEF', 'RHCRIT', 'EACF', 'VF1', 'ALPHAM', 'CW_LAND', 'KAY_GWAVE', 'CHARNOCK', 'ICE_SIZE',
              'ASYM_LAMBDA', 'G0', 'DYNDIFF', 'Z0FSEA']
# ordered in ECS
fig = plt.figure("params", figsize=[8, 6], clear=True)

gs = gridspec.GridSpec(nStudies, 6)  # 1 extra for standard config

paramNames = studies[allTSdfols.index[-1]].paramNames()
stdParam = refCase.standardParam(paramNames=paramNames, scale=True)
sevenParams = refCase.paramNames()[:]
stdParam = stdParam.reindex(orderNames)
stdParam = stdParam.rename(PaperLib.shortNames(paramNames=paramNames, mark=sevenParams), copy=True)
# keys = allTs7Data.index.tolist() + allTs14Data.index.tolist() + allTSdfols.index.tolist() + allTsSens.index.tolist()
keys = allTs7Data.index.tolist() + allTSdfols.index.tolist() + allTsSens.index.tolist()
colour = 'grey'
indexV = {}  # set up as empty dict
# loop over ensembles...
for indx, k in enumerate(keys):  # plot finger print for each study
    if 'HadAM3-7' in k:
        colour = 'black'
        ensemble = 'CE7'

    elif 'HadAM3-14' in k:
        colour = 'indianred'
        ensemble = 'CE14'
    elif 'DFO' in k:
        colour = 'orange'
        ensemble = 'DF14'
    else:  # sens cases
        colour = 'lightblue'
        ensemble = None

    ax = plt.subplot(gs[indx, :])
    try:
        simuln = studies[k].optimumParams(paramNames=paramNames, normalise=True)

    except KeyError:
        print("Using std values for %s" % (k))
        simuln = refCase.standardParam(paramNames=paramNames, scale=True)
        colour = 'red'
        # fix Perturb-Ice.
        if k == 'Perturb Ice':
            simuln.ALPHAM = 1.0
    if ensemble is not None:
        indx = indexV.get(ensemble, 0)
        indexV[ensemble] = indx + 1
        name = f"{ensemble:s}-{indx}"
    else:
        name = k
    simln = simuln.loc[paramNames]
    simln = simln.reindex(orderNames)
    simuln = simln.rename(PaperLib.shortNames(paramNames=paramNames, mark=sevenParams), copy=True)
    simln.plot.bar(ax=ax, color=colour)
    stdParam.plot(marker='o', color='grey', linestyle='None')  # plot the standard values

    ax.get_yaxis().set_visible(False)
    ax.set_ylabel(k, rotation=0)  # , fontsize='x-small')
    ax.text(-3., 0.5, name, bbox=dict(facecolor='grey', alpha=0.5))
    ax.set_xlim(-3, 13.5)
    if indx == (len(keys) - 1):
        ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=30)
    else:
        ax.set_xticklabels('' * 30)  # blank labels

fig.suptitle('Parameter Values')
ax.get_xaxis().set_visible(True)
fig.tight_layout()
fig.show()
PaperLib.saveFig(fig, "params")
# print out the stdev of the parameters for the 7 parameter cases and genetae
p7 = pd.DataFrame(
    [s.optimumParams(paramNames=paramNames, normalise=True).rename(k) for k, s in studies.items() if ('HadAM3-7' in k)])
print("Norm param std dev \n", p7.std())
print("Norm mean - std \n", (p7.mean() - refCase.standardParam(paramNames=paramNames, scale=True)) / p7.std())
pall = pd.DataFrame(
    [s.optimumParams(paramNames=paramNames, normalise=True).rename(k) for k, s in studies.items() if k != 'Long Control'])

pDFOLS = pd.DataFrame(
    [s.optimumParams(paramNames=paramNames, normalise=True).rename(k) for k, s in studies.items() if k.startswith('HadAM3-DFO14')])

## print out number of cases where near boundaries..
edges = [0.005,0.995]
DFOLS_edge = (((pDFOLS <edges[0]) | (pDFOLS >edges[1])).sum()/5).rename('DF14')
CE7_edge = (((p7[sevenParams] <edges[0]) | (p7[sevenParams] >edges[1])).sum()/10).rename('CE7')
ds=pd.DataFrame([CE7_edge,DFOLS_edge])
print(ds.T)
